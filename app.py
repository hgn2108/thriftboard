import json
import os
import re
import time
import uuid
import zlib
from pathlib import Path

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")  # SERPAPI_KEY locally; Cloud Run sets it as an env var

from google.api_core.exceptions import GoogleAPIError  # noqa: E402

from tools import TOOLS, _firestore, run_tool  # noqa: E402

# --- Config ---

SYSTEM_PROMPT = """You are Thriftboard, a secondhand personal shopper. Users share a Pinterest board of
outfits they love, and you help them find those pieces secondhand at a price they're happy with.

When to use each tool:
- read_pinterest_board: whenever the user shares a Pinterest board link. Never describe a board
  you haven't read with this tool.
- search_listings: when the user wants to find or shop a piece. For a board piece, pass its query
  and its broad_query as fallback_query. Pass max_price and size whenever the user has given them,
  including earlier in the conversation.
- price_verdict: when the user asks if something is a good deal, worth it, or fairly priced.
- style_match: when the user asks which listing fits their board or style best, or which to pick.
- shop_the_pin: when the user wants one specific pin or look recreated ("get me pin 6", "how much
  for this outfit"), or pastes a link to a single pin (pass it as pin_url; no board needed).
  Prefer it over several search_listings calls for a whole outfit. A pin.it link can be a board
  or a pin: try read_pinterest_board, and if it says it's a single pin, call shop_the_pin.
- board_unlock: when the user gives a budget for the board or asks what to buy first. Pass the
  numbers of any pieces they've said they already own; shop_the_pin then skips those too.
- watch_item: when the user asks to watch, track, or be alerted. Watch a search (query +
  target_price) to catch new listings under a price, or specific listings (listing_ids) to follow
  price drops and sell-outs. If it's unclear which they want, ask.
- check_watchlist: whenever the user asks about their watchlist, what's new, or price drops.
  Always call it; never assume the watchlist is empty or unchanged.
- unwatch_item: when the user wants to stop watching something.
Don't call tools for general styling chat you can answer from what's already in the conversation.
When the user refers to "the second piece" or "that skirt", resolve it from the numbered lists
you already gave.

How to answer:
- After reading a board, open with the style in your own words, then the pieces as a short
  numbered list so the user can refer to them by number. End by suggesting next steps: shop a
  pin, or plan purchases with a budget.
- For shop_the_pin, lead with the outfit total and savings, then one line per piece.
- For board_unlock, open with its headline, bold and word for word, then each listing to buy
  (price, site, id) and why that piece carries the board, then the next best buy. Every number
  must come from the tool result; never recompute or round differently.
- When showing listings, give the best 3-5 with price, site, and the listing id (L1, L2...).
  The app shows every listing as a card with photo and link, so don't paste URLs.
- Keep replies short and specific. Be honest when a listing doesn't really match.
- The watchlist doesn't send notifications: after watching a piece, tell the user to come back
  and ask "what's new on my watchlist?" to see new listings and price drops.
- If a tool returns an error, explain it plainly and tell the user exactly what to do next.
- Never invent listings, prices, links, or scores."""
MAX_TOOL_ROUNDS = 8  # a turn can chain several tools, e.g. search, then verdict, then style match

# --- The Harness ---


def run_agent(messages: list[dict], state: dict) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    `state` is the session's scratchpad, passed to every tool by the harness, never by the model.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
            num_retries=3,  # Vertex's shared quota returns occasional 429s; retry with backoff
        ).choices[0].message

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content, tool_calls

        # The harness runs each tool and appends the result
        for call in reply.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                result = run_tool(call.function.name, args, state)
            except json.JSONDecodeError:
                # Every tool call must get a result, or the next completion rejects the history
                args, result = {}, json.dumps({"error": "The arguments were not valid JSON. Call the tool again with a JSON object."})
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages, and session_id -> tool state (last board read, last listings
# shown, pieces the user owns). Kept in memory for speed and saved to Firestore after every turn:
# Cloud Run stops idle instances and may run several, and a conversation must survive both.
sessions: dict[str, list] = {}
states: dict[str, dict] = {}


def load_session(session_id: str) -> bool:
    """Bring a saved session back into memory. False if there is none (or Firestore is unreachable)."""
    try:
        doc = _firestore().collection("sessions").document(session_id).get()
    except GoogleAPIError:
        return False
    if not doc.exists:
        return False
    saved = json.loads(zlib.decompress(doc.get("data")))
    sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}] + saved["messages"][1:]
    states[session_id] = {**saved["state"], "owned": set(saved["state"].get("owned", []))}
    return True


def save_session(session_id: str) -> None:
    state = {**states[session_id], "owned": sorted(states[session_id].get("owned", set()))}
    blob = zlib.compress(json.dumps({"messages": sessions[session_id], "state": state}, default=str).encode())
    if len(blob) > 900_000:  # Firestore's limit is 1 MB a document; a session that long stays in memory only
        return
    try:
        _firestore().collection("sessions").document(session_id).set({"data": blob, "updated": time.time()})
    except GoogleAPIError:
        pass

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    # no-cache: browsers revalidate each visit, so a redeploy shows up without a hard refresh
    return FileResponse(Path(__file__).parent / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session. The id becomes a database key, so only accept ids we could have issued.
    session_id = request.session_id if re.fullmatch(r"[0-9a-f-]{36}", request.session_id or "") else str(uuid.uuid4())
    if session_id not in sessions and not load_session(session_id):
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        states[session_id] = {"session_id": session_id}

    # Append user's message to the context
    sessions[session_id] += [{"role": "user", "content": request.message}]

    try:
        response, tool_calls = run_agent(sessions[session_id], states[session_id])
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    save_session(session_id)
    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    states.pop(session_id, None)
    if session_id and re.fullmatch(r"[0-9a-f-]{36}", session_id):
        try:
            _firestore().collection("sessions").document(session_id).delete()
        except GoogleAPIError:
            pass
    return {"status": "ok"}


if __name__ == "__main__":
    # Cloud Run sets PORT and needs 0.0.0.0; locally this is still http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

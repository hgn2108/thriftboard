"""The tools the harness can run, and the JSON that describes them to the model."""

import json
import re
import xml.etree.ElementTree as ET
from typing import Literal
from urllib.parse import urlparse

import litellm
import requests
from cachetools import TTLCache
from pydantic import BaseModel

# Pinterest blocks the default python-requests user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
VISION_MODEL = "vertex_ai/gemini-3.5-flash-lite"
MAX_PINS = 25  # Pinterest's board RSS only ever returns the ~25 most recent pins anyway

# Successful board reads, so follow-up turns don't re-run the vision call. Errors are never cached.
board_cache = TTLCache(maxsize=32, ttl=600)

# --- read_pinterest_board ---

Category = Literal["tops", "bottoms", "dresses", "outerwear", "shoes", "bags", "accessories", "jewelry"]


class Piece(BaseModel):
    pin: int
    query: str
    broad_query: str
    category: Category


class BoardRead(BaseModel):
    style: str
    palette: list[str]
    pieces: list[Piece]


VISION_PROMPT = """These are {n} pins from a Pinterest board, numbered 1 to {n} in order.
Act as a personal stylist who shops secondhand.

1. style: one or two sentences naming the board's overall aesthetic, specific enough that the
   owner feels seen (e.g. "90s minimalist: slip skirts, oversized blazers, black and camel").
2. palette: the 3-6 colors that recur most.
3. pieces: every distinct clothing item or accessory worth hunting for. For each:
   - pin: the number of the pin it appears in
   - query: a 3-6 word search a secondhand marketplace would match, as color + material + garment
     (e.g. "black satin slip midi skirt"). Include a brand only if a logo is clearly visible.
   - broad_query: the same piece in 2-3 words, for when the specific search finds nothing
     (e.g. "satin midi skirt")
   - category
   Skip pins with no clothing (food, rooms, quotes). If the same piece appears in several pins,
   list it once. At most 12 pieces, favoring the ones that define the style.
"""


def _parse_board_url(board_url: str) -> tuple[str, str] | str:
    """Return (user, board) or an error message the model can act on."""
    url = board_url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    host = urlparse(url).netloc.lower()

    # pin.it short links redirect to the real board or pin URL
    if host.endswith("pin.it"):
        try:
            url = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True).url
        except requests.RequestException:
            return "Could not expand the pin.it short link. Ask the user for the full pinterest.com board URL."
        host = urlparse(url).netloc.lower()

    if "pinterest." not in host:
        return f"'{board_url}' is not a Pinterest link. Ask the user for a board URL like pinterest.com/<user>/<board>/."

    parts = [p for p in urlparse(url).path.split("/") if p]
    if parts and parts[0] == "pin":
        return "That is a single pin, not a board. Ask the user for the board URL: pinterest.com/<user>/<board>/."
    if len(parts) < 2:
        return "That is a profile, not a board. Ask the user which board to use: pinterest.com/<user>/<board>/."
    return parts[0], parts[1]


def read_pinterest_board(board_url: str) -> str:
    """Read a public Pinterest board and turn its outfits into searchable secondhand pieces."""
    parsed = _parse_board_url(board_url)
    if isinstance(parsed, str):
        return json.dumps({"error": parsed})
    user, board = parsed
    if (user, board) in board_cache:
        return board_cache[user, board]

    # Every public board has an RSS feed of its latest pins. No API key or login needed.
    try:
        resp = requests.get(f"https://www.pinterest.com/{user}/{board}.rss", headers=HEADERS, timeout=15)
        items = ET.fromstring(resp.content).findall("./channel/item") if "xml" in resp.headers.get("content-type", "") else []
    except (requests.RequestException, ET.ParseError) as e:
        return json.dumps({"error": f"Pinterest could not be reached ({type(e).__name__}). Tell the user to try again in a minute."})
    if not items:
        return json.dumps({"error": f"No pins found for board '{user}/{board}'. It may be secret, empty, or misspelled. Ask the user to check the board is public and the URL is right."})

    pins = []
    for item in items[:MAX_PINS]:
        img = re.search(r'src="(https://i\.pinimg\.com/[^"]+)"', item.findtext("description", ""))
        if img:
            # The feed links small 236px thumbnails; the same path at 736x is sharp enough to read fabrics.
            pins += [{"image": img.group(1).replace("/236x/", "/736x/"), "link": item.findtext("link", "")}]

    # Label every image: unlabeled, the model miscounts long image lists and pin numbers drift.
    content = [{"type": "text", "text": VISION_PROMPT.format(n=len(pins))}]
    for i, p in enumerate(pins, start=1):
        content += [{"type": "text", "text": f"Pin {i}:"}, {"type": "image_url", "image_url": {"url": p["image"]}}]
    try:
        reply = litellm.completion(
            model=VISION_MODEL,
            vertex_location="global",
            messages=[{"role": "user", "content": content}],
            response_format=BoardRead,  # constrained decoding: always valid JSON in this shape
            num_retries=3,  # Vertex's shared quota returns occasional 429s; retry with backoff
        ).choices[0].message.content
        read = BoardRead.model_validate_json(reply)
    except litellm.RateLimitError:
        return json.dumps({"error": "The image model is busy (rate limited). Tell the user to wait about 30 seconds and send the board again."})
    except Exception as e:
        return json.dumps({"error": f"Could not analyze the pin images ({type(e).__name__}). Tell the user to try again."})

    # Attach each piece's pin image and link so the UI can show what it came from
    pieces = []
    for piece in read.pieces:
        pin = pins[piece.pin - 1] if 1 <= piece.pin <= len(pins) else {}
        pieces += [{**piece.model_dump(), "image": pin.get("image"), "pin_link": pin.get("link")}]

    board_cache[user, board] = json.dumps({
        "board": f"{user}/{board}",
        "pins_read": len(pins),
        "style": read.style,
        "palette": read.palette,
        "pieces": pieces,
    })
    return board_cache[user, board]


# What the model sees: the "set notes" in the screenplay.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_pinterest_board",
            "description": (
                "Read a public Pinterest board's latest pins (up to 25) and turn the outfits in them into a "
                "style summary, a color palette, and a list of concrete clothing pieces, each with a specific "
                "secondhand search query and a broader fallback query. Call this whenever the user shares a "
                "Pinterest board link or asks what to shop for from their board. Results are cached, so call "
                "it again rather than guessing if you need the pieces after a long conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "board_url": {
                        "type": "string",
                        "description": (
                            "The board's URL exactly as the user gave it, e.g. "
                            "'https://www.pinterest.com/jane/summer-fits/' or a 'pin.it/...' short link."
                        ),
                    },
                },
                "required": ["board_url"],
            },
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {"read_pinterest_board": read_pinterest_board}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"})
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})

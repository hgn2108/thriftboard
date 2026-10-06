"""The tools the harness can run, and the JSON that describes them to the model.

Every tool takes a keyword-only `state`: the session's scratchpad (session id, last board read, last
search results). The harness passes it in; the model never sees or fills it.
"""

import json
import os
import re
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

import litellm
import requests
from cachetools import TTLCache
from google.api_core.exceptions import GoogleAPIError
from google.cloud import firestore
from pydantic import BaseModel

# Pinterest blocks the default python-requests user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
VISION_MODEL = "vertex_ai/gemini-3.5-flash-lite"
MAX_PINS = 25  # Pinterest's board RSS only ever returns the ~25 most recent pins anyway
MAX_LISTINGS = 8  # listings shown to the model per search; the rest still count as price comparables
MAX_WATCHED = 6  # each watched item costs one search per check, and SerpAPI's free tier is 250/month

# Successful board reads and searches, so follow-up turns don't redo slow or metered calls.
# Errors are never cached.
board_cache = TTLCache(maxsize=32, ttl=3600)
search_cache = TTLCache(maxsize=256, ttl=1800)


def _error(message: str) -> str:
    return json.dumps({"error": message})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="minutes")


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


BOARD_PROMPT = """These are {n} pins from a Pinterest board, numbered 1 to {n} in order.
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


def _read_board(board_url: str) -> str:
    parsed = _parse_board_url(board_url)
    if isinstance(parsed, str):
        return _error(parsed)
    user, board = parsed
    if (user, board) in board_cache:
        return board_cache[user, board]

    # Every public board has an RSS feed of its latest pins. No API key or login needed.
    try:
        resp = requests.get(f"https://www.pinterest.com/{user}/{board}.rss", headers=HEADERS, timeout=15)
        items = ET.fromstring(resp.content).findall("./channel/item") if "xml" in resp.headers.get("content-type", "") else []
    except (requests.RequestException, ET.ParseError) as e:
        return _error(f"Pinterest could not be reached ({type(e).__name__}). Tell the user to try again in a minute.")
    if not items:
        return _error(f"No pins found for board '{user}/{board}'. It may be secret, empty, or misspelled. Ask the user to check the board is public and the URL is right.")

    pins = []
    for item in items[:MAX_PINS]:
        img = re.search(r'src="(https://i\.pinimg\.com/[^"]+)"', item.findtext("description", ""))
        if img:
            # The feed links small 236px thumbnails; the same path at 736x is sharp enough to read fabrics.
            pins += [{"image": img.group(1).replace("/236x/", "/736x/"), "link": item.findtext("link", "")}]

    # Label every image: unlabeled, the model miscounts long image lists and pin numbers drift.
    content = [{"type": "text", "text": BOARD_PROMPT.format(n=len(pins))}]
    for i, p in enumerate(pins, start=1):
        content += [{"type": "text", "text": f"Pin {i}:"}, {"type": "image_url", "image_url": {"url": p["image"]}}]
    try:
        reply = litellm.completion(
            model=VISION_MODEL,
            vertex_location="global",
            messages=[{"role": "user", "content": content}],
            response_format=BoardRead,  # constrained decoding: always valid JSON in this shape
            temperature=0,  # the same board should read the same way every time
            num_retries=3,  # Vertex's shared quota returns occasional 429s; retry with backoff
        ).choices[0].message.content
        read = BoardRead.model_validate_json(reply)
    except litellm.RateLimitError:
        return _error("The image model is busy (rate limited). Tell the user to wait about 30 seconds and send the board again.")
    except Exception as e:
        return _error(f"Could not analyze the pin images ({type(e).__name__}). Tell the user to try again.")

    # Attach each piece's pin image and link so the UI can show what it came from
    pieces = []
    for n, piece in enumerate(read.pieces, start=1):
        pin = pins[piece.pin - 1] if 1 <= piece.pin <= len(pins) else {}
        pieces += [{"number": n, **piece.model_dump(), "image": pin.get("image"), "pin_link": pin.get("link")}]

    board_cache[user, board] = json.dumps({
        "board": f"{user}/{board}",
        "pins_read": len(pins),
        "style": read.style,
        "palette": read.palette,
        "pieces": pieces,
    })
    return board_cache[user, board]


def read_pinterest_board(board_url: str, *, state: dict) -> str:
    """Read a public Pinterest board and turn its outfits into searchable secondhand pieces."""
    result = _read_board(board_url)
    if "error" not in json.loads(result):
        state["board"] = json.loads(result)  # style_match compares listings against this
    return result


# --- Listing search (SerpAPI) ---

Source = Literal["marketplaces", "ebay"]


def _serpapi(params: dict) -> dict | str:
    """One SerpAPI search, cached. Returns the response, or an error message the model can act on."""
    key = os.environ.get("SERPAPI_KEY", "").strip()
    if not key:
        return "Listing search is not configured on this server. Tell the user live search is unavailable; you can still read boards."
    cache_key = tuple(sorted(params.items()))
    if cache_key in search_cache:
        return search_cache[cache_key]

    try:
        # Most searches take 1-2s, but a fresh Google Shopping search occasionally takes ~30s
        data = requests.get("https://serpapi.com/search.json", params={**params, "api_key": key}, timeout=60).json()
    except requests.Timeout:
        # SerpAPI keeps working after we give up and caches the result, so a retry is usually instant
        return "The search is running slowly. Tell the user to ask again in a few seconds; the retry is usually instant."
    except (requests.RequestException, ValueError) as e:
        return f"The listing search service could not be reached ({type(e).__name__}). Tell the user to try again in a minute."
    if "error" in data:
        if "run out of searches" in data["error"]:
            return "This month's search quota is used up. Tell the user live search is unavailable; you can still read boards and talk through pieces."
        if "hasn't returned any results" not in data["error"]:  # zero results is an answer, not a failure
            return f"Listing search failed: {data['error']}"

    search_cache[cache_key] = data
    return data


def _search(query: str, size: str | None, source: Source) -> list[dict] | str:
    """Search one source and normalize every priced result. Returns listings or an error message."""
    q = f"{query} size {size}" if size else query
    listings = []

    if source == "ebay":
        data = _serpapi({"engine": "ebay", "_nkw": q, "LH_ItemCondition": "3000"})  # 3000 = used
        if isinstance(data, str):
            return data
        for r in data.get("organic_results", []):
            price = (r.get("price") or {}).get("extracted")  # price ranges (multi-size listings) have none
            item = re.search(r"/itm/(\d+)", r.get("link", ""))
            if price is not None and item:
                listings += [{
                    "key": f"ebay_{item.group(1)}", "title": r.get("title"), "price": price, "site": "eBay",
                    "condition": r.get("condition") or "Pre-Owned", "link": r["link"].split("?")[0], "image": r.get("thumbnail"),
                }]
    else:
        # Google Shopping, biased to used: covers Depop, Poshmark, Mercari, Etsy, and eBay sellers in one search
        data = _serpapi({"engine": "google_shopping", "q": f"{q} used", "gl": "us", "hl": "en"})
        if isinstance(data, str):
            return data
        for r in data.get("shopping_results", []):
            if r.get("extracted_price") is not None:
                listings += [{
                    "key": f"gs_{r.get('product_id')}", "title": r.get("title"), "price": r["extracted_price"],
                    "site": (r.get("source") or "unknown").split(" - ")[0],
                    "condition": r.get("second_hand_condition") or "unspecified",
                    "link": r.get("product_link"), "image": r.get("thumbnail"),
                }]

    return listings


def search_listings(
    query: str,
    max_price: float | None = None,
    size: str | None = None,
    source: Source = "marketplaces",
    fallback_query: str | None = None,
    *,
    state: dict,
) -> str:
    """Find real secondhand listings for a piece."""
    found = _search(query, size, source)
    if isinstance(found, str):
        return _error(found)

    # A very specific query can come back thin; the broader one usually doesn't
    if len(found) < 3 and fallback_query:
        broader = _search(fallback_query, size, source)
        if isinstance(broader, list) and len(broader) > len(found):
            found, query = broader, fallback_query

    if not found:
        return _error(f"No secondhand listings for '{query}'. Retry with a shorter, broader query (the piece's broad_query), or without the size.")

    prices = sorted(l["price"] for l in found)
    matches = [l for l in found if max_price is None or l["price"] <= max_price][:MAX_LISTINGS]
    for i, listing in enumerate(matches, start=1):
        listing["id"] = f"L{i}"

    # Remember what was searched so price_verdict, style_match and watch_item can refer back to it
    state["last_search"] = {"query": query, "size": size, "source": source}
    state["listings"] = {l["id"]: l for l in matches}

    result = {
        "query": query,
        "size": size,
        "source": source,
        "results_scanned": len(found),
        "price_summary": {"low": prices[0], "median": statistics.median(prices), "high": prices[-1]},
        "listings": [{k: l[k] for k in ("id", "title", "price", "site", "condition", "link", "image")} for l in matches],
    }
    if not matches:
        result["note"] = (
            f"Nothing at or under ${max_price:g}. The cheapest of {len(found)} listings was ${prices[0]:g}. "
            "Offer to raise the budget, or to watch this piece at the user's price with watch_item."
        )
    return json.dumps(result)


# --- price_verdict ---


def _verdict(price: float, comps: list[float]) -> dict:
    """Where a price sits among comparable listings. Quartiles keep one wild outlier from skewing it."""
    p25, median, p75 = statistics.quantiles(comps, n=4) if len(comps) >= 2 else (comps[0],) * 3
    cheaper_than = round(100 * sum(c > price for c in comps) / len(comps))
    verdict = "steal" if price <= p25 else "fair" if price <= p75 else "overpriced"
    return {"verdict": verdict, "cheaper_than_pct_of_comparables": cheaper_than, "vs_median": round(price - median, 2)}


def price_verdict(
    listing_ids: list[str] | None = None,
    price: float | None = None,
    query: str | None = None,
    *,
    state: dict,
) -> str:
    """Judge prices against every comparable secondhand listing, not just the few shown."""
    if listing_ids:
        listings = state.get("listings", {})
        unknown = [i for i in listing_ids if i not in listings]
        if unknown:
            return _error(f"Unknown listing ids {unknown}. Valid ids from the last search: {list(listings) or 'none (call search_listings first)'}.")
        last = state["last_search"]
        to_judge = [(i, listings[i]["title"], listings[i]["price"]) for i in listing_ids]
    elif price is not None and query:
        last = {"query": query, "size": None, "source": "marketplaces"}
        to_judge = [("user", query, price)]
    else:
        return _error("Pass listing_ids from the last search_listings results, or both price and query for a listing the user found elsewhere.")

    # The same search the listings came from: cached, so this costs no extra quota
    comps = _search(last["query"], last["size"], last["source"])
    if isinstance(comps, str):
        return _error(comps)
    prices = [c["price"] for c in comps]
    if len(prices) < 5:
        return _error(f"Only {len(prices)} comparable listings for '{last['query']}', too few to judge. Retry with a broader query.")

    p25, median, p75 = statistics.quantiles(prices, n=4)
    return json.dumps({
        "market": {
            "query": last["query"],
            "comparable_listings": len(prices),
            "median": round(median, 2),
            "typical_range": [round(p25, 2), round(p75, 2)],
        },
        "verdicts": [{"id": i, "title": t, "price": p, **_verdict(p, prices)} for i, t, p in to_judge],
    })


# --- style_match ---


class StyleScore(BaseModel):
    id: str
    score: int
    reason: str


class StyleScores(BaseModel):
    scores: list[StyleScore]


STYLE_PROMPT = """You are a stylist judging secondhand listings against a client's Pinterest board.
Board style: {style}
Board palette: {palette}
The client is hunting for: {piece}

For each listing below, give a score from 0 to 100 for how well it would fit this board (not just
whether it matches the search words: silhouette, fabric, color and vibe all count), and a reason of
at most 12 words naming what fits or what's off (e.g. "right camel wool, but cropped where pins are long").
"""


def style_match(listing_ids: list[str] | None = None, *, state: dict) -> str:
    """Score how well each listing's photo fits the board's style."""
    board = state.get("board")
    if not board:
        return _error("No board has been read in this conversation. Ask the user for their Pinterest board and call read_pinterest_board first.")
    listings = state.get("listings")
    if not listings:
        return _error("No listings to score. Call search_listings first.")
    ids = listing_ids or list(listings)
    unknown = [i for i in ids if i not in listings]
    if unknown:
        return _error(f"Unknown listing ids {unknown}. Valid ids from the last search: {list(listings)}.")

    # If the search came from a board piece, show the model the pin it came from
    query = state["last_search"]["query"]
    piece = next((p for p in board["pieces"] if query in (p["query"], p["broad_query"])), None)

    content = [{"type": "text", "text": STYLE_PROMPT.format(style=board["style"], palette=", ".join(board["palette"]), piece=query)}]
    if piece and piece.get("image"):
        content += [{"type": "text", "text": "The pin this piece came from:"}, {"type": "image_url", "image_url": {"url": piece["image"]}}]
    for i in ids:
        l = listings[i]
        content += [{"type": "text", "text": f"Listing {i}: {l['title']} (${l['price']:g}, {l['site']})"}]
        if l.get("image"):
            content += [{"type": "image_url", "image_url": {"url": l["image"]}}]

    try:
        reply = litellm.completion(
            model=VISION_MODEL,
            vertex_location="global",
            messages=[{"role": "user", "content": content}],
            response_format=StyleScores,
            num_retries=3,
        ).choices[0].message.content
        scores = StyleScores.model_validate_json(reply).scores
    except litellm.RateLimitError:
        return _error("The image model is busy (rate limited). Tell the user to wait about 30 seconds and ask again.")
    except Exception as e:
        return _error(f"Could not score the listing photos ({type(e).__name__}). Tell the user to try again.")

    scored = [
        {"id": s.id, "title": listings[s.id]["title"], "price": listings[s.id]["price"], "score": max(0, min(100, s.score)), "reason": s.reason}
        for s in scores if s.id in listings
    ]
    return json.dumps({"board_style": board["style"], "scores": sorted(scored, key=lambda s: -s["score"])})


# --- Watchlist (Firestore) ---

_db = None


def _watchlist(state: dict):
    """This session's watchlist collection. The session id comes from the harness, never the model."""
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db.collection("watchlists").document(state["session_id"]).collection("items")


def _slug(*parts) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(str(p) for p in parts if p).lower()).strip("-")


def watch_item(
    query: str,
    target_price: float,
    size: str | None = None,
    source: Source = "marketplaces",
    *,
    state: dict,
) -> str:
    """Save a piece and target price; check_watchlist later reports what's new or cheaper."""
    found = _search(query, size, source)
    if isinstance(found, str):
        return _error(found)

    try:
        items = _watchlist(state)
        if len(list(items.limit(MAX_WATCHED).stream())) >= MAX_WATCHED:
            return _error(f"The watchlist is full ({MAX_WATCHED} pieces). Tell the user to drop a piece first; this keeps checks fast.")
        items.document(_slug(query, size, source)).set({
            "query": query, "size": size, "source": source, "target_price": target_price,
            # Every listing seen so far and its price: the baseline for "new" and "price drop"
            "seen": {l["key"]: l["price"] for l in found},
            "created": _now(), "last_checked": _now(),
        })
        count = len(list(items.stream()))
    except GoogleAPIError as e:
        return _error(f"The watchlist database is unavailable ({type(e).__name__}). Tell the user to try again shortly.")

    under = sorted((l for l in found if l["price"] <= target_price), key=lambda l: l["price"])[:3]
    return json.dumps({
        "watching": query,
        "size": size,
        "target_price": target_price,
        "listings_tracked": len(found),
        "already_under_target": [{k: l[k] for k in ("title", "price", "site", "link", "image")} for l in under],
        "pieces_on_watchlist": count,
    })


def check_watchlist(*, state: dict) -> str:
    """Re-search every watched piece and report new listings under target and price drops."""
    try:
        docs = list(_watchlist(state).limit(MAX_WATCHED).stream())
    except GoogleAPIError as e:
        return _error(f"The watchlist database is unavailable ({type(e).__name__}). Tell the user to try again shortly.")
    if not docs:
        return json.dumps({"pieces": [], "note": "The watchlist is empty. Offer to watch a piece with watch_item."})

    report = []
    for doc in docs:
        item = doc.to_dict()
        found = _search(item["query"], item["size"], item["source"])
        if isinstance(found, str):
            report += [{"query": item["query"], "error": found}]
            continue

        seen, target = item["seen"], item["target_price"]
        new = [l for l in found if l["key"] not in seen and l["price"] <= target]
        drops = [{**l, "was": seen[l["key"]]} for l in found if l["key"] in seen and l["price"] < seen[l["key"]]]
        doc.reference.update({"seen": {**seen, **{l["key"]: l["price"] for l in found}}, "last_checked": _now()})

        fields = ("title", "price", "site", "link", "image")
        report += [{
            "query": item["query"],
            "size": item["size"],
            "target_price": target,
            "previous_check": item["last_checked"],
            "new_under_target": [{k: l[k] for k in fields} for l in sorted(new, key=lambda l: l["price"])[:5]],
            "price_drops": [{**{k: l[k] for k in fields}, "was": l["was"]} for l in drops[:5]],
            "cheapest_now": min((l["price"] for l in found), default=None),
        }]

    return json.dumps({"checked_at": _now(), "pieces": report})


# What the model sees: the "set notes" in the screenplay.
SOURCE_PARAM = {
    "type": "string",
    "enum": ["marketplaces", "ebay"],
    "description": (
        "'marketplaces' (default) searches used listings across Depop, Poshmark, Mercari, Etsy and eBay "
        "via Google Shopping. 'ebay' searches eBay only, with direct listing links."
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_pinterest_board",
            "description": (
                "Read a public Pinterest board's latest pins (up to 25) and turn the outfits in them into a "
                "style summary, a color palette, and a numbered list of concrete clothing pieces, each with a "
                "specific secondhand search query and a broader fallback query. Call this whenever the user "
                "shares a Pinterest board link. Results are cached, so call it again rather than guessing if "
                "you need the pieces after a long conversation."
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
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": (
                "Search live secondhand listings for one clothing piece. Returns up to 8 real listings (each with "
                "an id like 'L1', title, price in USD, site, condition, link, image) plus a price summary across "
                "every result scanned. Call this when the user wants to find, shop, or price a piece. For a board "
                "piece, pass its query as query and its broad_query as fallback_query. Only ever quote listings, "
                "prices and links that this tool returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, 3-6 words: color + material + garment, e.g. 'brown plaid pleated mini skirt'."},
                    "max_price": {"type": "number", "description": "Only return listings at or under this price in USD, e.g. 60. Omit if the user gave no budget."},
                    "size": {"type": "string", "description": "Size as the user said it, e.g. 'M', '8', '28'. Omit if not given."},
                    "source": SOURCE_PARAM,
                    "fallback_query": {"type": "string", "description": "A broader 2-3 word version of query, searched only if query finds fewer than 3 listings, e.g. 'plaid mini skirt'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "price_verdict",
            "description": (
                "Judge whether prices are a steal, fair, or overpriced by ranking them against every comparable "
                "secondhand listing (median, typical range, and what share of listings each one is cheaper than). "
                "Call this when the user asks if something is a good deal or worth it. Either pass listing_ids from "
                "the last search_listings results, or pass price and query for a listing the user found elsewhere."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_ids": {"type": "array", "items": {"type": "string"}, "description": "Ids from the last search_listings results, e.g. ['L1', 'L3']."},
                    "price": {"type": "number", "description": "A price in USD the user found elsewhere, e.g. 40. Use with query."},
                    "query": {"type": "string", "description": "What that listing is, 3-6 words, e.g. 'black leather knee high boots'. Use with price."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "style_match",
            "description": (
                "Look at the photos of listings from the last search_listings results and score each 0-100 on how "
                "well it fits the user's Pinterest board (silhouette, fabric, color, vibe), with a short reason. "
                "Call this when the user asks which listing best matches their board or style, or which to pick. "
                "Needs a board read and a search earlier in this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_ids": {"type": "array", "items": {"type": "string"}, "description": "Ids to score, e.g. ['L1', 'L2']. Omit to score every listing from the last search."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "watch_item",
            "description": (
                "Add a piece to the user's watchlist with a target price. It records every current listing as a "
                "baseline, so check_watchlist can later report only what's new or cheaper. Also returns any listings "
                "already at or under the target. Call this when the user asks to watch, track, or be alerted about a "
                "piece. The watchlist holds up to 6 pieces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The piece to watch, 3-6 words, e.g. 'brown leather knee high boots'. Reuse the query the user already searched."},
                    "target_price": {"type": "number", "description": "The most the user wants to pay, in USD, e.g. 45."},
                    "size": {"type": "string", "description": "Size as the user said it, e.g. 'M'. Omit if not given."},
                    "source": SOURCE_PARAM,
                },
                "required": ["query", "target_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_watchlist",
            "description": (
                "Re-search every piece on the user's watchlist and report, per piece, new listings at or under the "
                "target price and listings whose price dropped since the previous check, plus the cheapest price now. "
                "Call this when the user asks what's new, to check their watchlist, or whether anything dropped."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "read_pinterest_board": read_pinterest_board,
    "search_listings": search_listings,
    "price_verdict": price_verdict,
    "style_match": style_match,
    "watch_item": watch_item,
    "check_watchlist": check_watchlist,
}


def run_tool(name: str, args: dict, state: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return _error(f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}")
    args.pop("state", None)  # session state comes from the harness only
    try:
        return TOOL_MAP[name](**args, state=state)
    except TypeError as e:
        return _error(f"Bad arguments for {name}: {e}")

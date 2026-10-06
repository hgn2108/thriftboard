"""The tools the harness can run, and the JSON that describes them to the model.

Every tool takes a keyword-only `state`: the session's scratchpad (session id, last board read, last
listings shown). The harness passes it in; the model never sees or fills it.
"""

import hashlib
import json
import os
import re
import statistics
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
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
MAX_PIN_PIECES = 4  # pieces shop_the_pin will hunt for in one outfit
OPTIONS_PER_PIECE = 3  # per piece, the stylist model sees this many most-relevant AND this many cheapest listings
UNLOCK_CANDIDATES = 6  # pieces board_unlock prices: one search each, and 2^6 combinations to try
PIECE_TIMEOUT = 25  # per search when hunting several pieces at once, so one slow search can't stall the rest
VALUE_MARGIN = 10  # a listing within this many match points of the best one is "as good"; then the cheapest wins
MIN_MATCH = 60  # below this, a listing isn't really the piece, however cheap

# A pin counts as "recreated" once you have its garments; bags and jewelry are optional extras.
GARMENTS = {"tops", "bottoms", "dresses", "outerwear", "shoes"}

# Successful board reads and searches, so follow-up turns don't redo slow or metered calls.
# Errors are never cached.
board_cache = TTLCache(maxsize=32, ttl=3600)
search_cache = TTLCache(maxsize=256, ttl=1800)
cache_lock = threading.Lock()  # shop_the_pin and board_unlock search in parallel threads


def _error(message: str) -> str:
    return json.dumps({"error": message})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="minutes")


_db = None


def _firestore():
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


# Saved in Firestore too, so results survive Cloud Run restarts: the same board reads the same way
# for a day, and repeat searches don't spend SerpAPI's 250/month quota.
BOARD_TTL = 24 * 3600
SEARCH_TTL = 6 * 3600


def _saved(kind: str, key: str, max_age: int) -> str | None:
    """A previously saved result, if it's fresh. The cache is an optimization: any failure is a miss."""
    try:
        doc = _firestore().collection(f"cache_{kind}").document(hashlib.sha1(key.encode()).hexdigest()).get()
        if doc.exists and time.time() - doc.get("at") < max_age:
            return doc.get("value")
    except GoogleAPIError:
        pass
    return None


def _save(kind: str, key: str, value: str) -> None:
    try:
        _firestore().collection(f"cache_{kind}").document(hashlib.sha1(key.encode()).hexdigest()).set({"key": key, "value": value, "at": time.time()})
    except GoogleAPIError:
        pass


def _vision(content: list, schema: type[BaseModel]) -> BaseModel | str:
    """One Gemini call over text and images, constrained to `schema`. Returns it, or an error message."""
    try:
        reply = litellm.completion(
            model=VISION_MODEL,
            vertex_location="global",
            messages=[{"role": "user", "content": content}],
            response_format=schema,  # constrained decoding: always valid JSON in this shape
            temperature=0,  # the same images should read the same way every time
            num_retries=3,  # Vertex's shared quota returns occasional 429s; retry with backoff
        ).choices[0].message.content
        return schema.model_validate_json(reply)
    except litellm.RateLimitError:
        return "The image model is busy (rate limited). Tell the user to wait about 30 seconds and ask again."
    except Exception as e:
        return f"Could not analyze the images ({type(e).__name__}). Tell the user to try again."


# --- read_pinterest_board ---

Category = Literal["tops", "bottoms", "dresses", "outerwear", "shoes", "bags", "accessories", "jewelry"]


class Piece(BaseModel):
    pins: list[int]
    query: str
    broad_query: str
    category: Category


class BoardRead(BaseModel):
    style: str
    palette: list[str]
    outfit_pins: list[int]
    pieces: list[Piece]


BOARD_PROMPT = """These are {n} pins from a Pinterest board, numbered 1 to {n} in order.
Act as a personal stylist who shops secondhand.

1. style: one or two sentences naming the board's overall aesthetic, specific enough that the
   owner feels seen (e.g. "90s minimalist: slip skirts, oversized blazers, black and camel").
2. palette: the 3-6 colors that recur most.
3. outfit_pins: the numbers of every pin that shows wearable clothing. Skip food, rooms, quotes.
4. pieces: the distinct clothing items and accessories worth hunting for. For each:
   - pins: every pin number where this piece, or one close enough to stand in for it, appears.
     Reuse a piece across pins rather than listing near-duplicates.
   - query: a 3-6 word search a secondhand marketplace would match, as color + material + garment
     (e.g. "black satin slip midi skirt"). Include a brand only if a logo is clearly visible.
   - broad_query: the same piece in 2-3 words, for when the specific search finds nothing
     (e.g. "satin midi skirt")
   - category
   At most 15 pieces. Prefer covering the main garments (top, bottom or dress, outerwear, shoes)
   of as many outfit pins as possible over listing every accessory.
"""


def _pinterest_path(link: str) -> list[str] | str:
    """The path parts of a Pinterest link, expanding pin.it short links. Or an error message."""
    url = link.strip()
    if not url.startswith("http"):
        url = "https://" + url
    host = urlparse(url).netloc.lower()

    # pin.it short links redirect to the real board or pin URL
    if host == "pin.it" or host.endswith(".pin.it"):
        try:
            url = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True).url
        except requests.RequestException:
            return "Could not expand the pin.it short link. Ask the user for the full pinterest.com link."
        host = urlparse(url).netloc.lower()

    if "pinterest." not in host:
        return f"'{link}' is not a Pinterest link. Ask the user for a board link (pinterest.com/<user>/<board>/) or a pin link (pinterest.com/pin/<id>/)."
    return [p for p in urlparse(url).path.split("/") if p]


def _parse_board_url(board_url: str) -> tuple[str, str] | str:
    """Return (user, board) or an error message the model can act on."""
    parts = _pinterest_path(board_url)
    if isinstance(parts, str):
        return parts
    if parts and parts[0] == "pin":
        return "That is a single pin, not a board. To recreate that one outfit, call shop_the_pin with it as pin_url."
    if len(parts) < 2:
        return "That is a profile page, not a board. Ask the user which of their boards to use: pinterest.com/<user>/<board>/."
    return parts[0], parts[1]


def _read_board(board_url: str) -> str:
    parsed = _parse_board_url(board_url)
    if isinstance(parsed, str):
        return _error(parsed)
    user, board = parsed
    if (user, board) in board_cache:
        return board_cache[user, board]
    saved = _saved("board", f"{user}/{board}", BOARD_TTL)
    if saved:
        board_cache[user, board] = saved
        return saved

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
            pins += [{"number": len(pins) + 1, "image": img.group(1).replace("/236x/", "/736x/"), "link": item.findtext("link", "")}]

    # Label every image: unlabeled, the model miscounts long image lists and pin numbers drift.
    content = [{"type": "text", "text": BOARD_PROMPT.format(n=len(pins))}]
    for p in pins:
        content += [{"type": "text", "text": f"Pin {p['number']}:"}, {"type": "image_url", "image_url": {"url": p["image"]}}]
    read = _vision(content, BoardRead)
    if isinstance(read, str):
        return _error(read)

    # Number the pieces so the user can say "piece 3", and drop pin numbers the model made up
    valid = {p["number"] for p in pins}
    pieces = []
    for n, piece in enumerate(read.pieces, start=1):
        on_pins = [i for i in piece.pins if i in valid] or [1]
        pieces += [{"number": n, **piece.model_dump(), "pins": on_pins, "image": pins[on_pins[0] - 1]["image"]}]

    board_cache[user, board] = json.dumps({
        "board": f"{user}/{board}",
        "pins_read": len(pins),
        "style": read.style,
        "palette": read.palette,
        "outfit_pins": [i for i in read.outfit_pins if i in valid],
        "pieces": pieces,
        "pins": pins,
    })
    _save("board", f"{user}/{board}", board_cache[user, board])
    return board_cache[user, board]


def _read_pin(pin_url: str, board: dict | None) -> tuple[dict, int | None] | str:
    """A one-pin "board" for shop_the_pin: the pin's number on the user's board if it's there,
    otherwise its photo read the same way a board is. Returns (board, pin number) or an error message."""
    parts = _pinterest_path(pin_url)
    if isinstance(parts, str):
        return parts
    if len(parts) >= 2 and parts[0] != "pin":
        return "That is a board link, not a single pin. Call read_pinterest_board with it instead."
    if len(parts) < 2:
        return "That is a profile page, not a pin or a board. Ask the user which board (pinterest.com/<user>/<board>/) or pin (pinterest.com/pin/<id>/) they mean."
    pin_id = parts[1]

    # Already on the board the user shared: same as tapping it in the grid
    for pin in (board or {}).get("pins", []):
        if f"/pin/{pin_id}/" in pin["link"]:
            return board, pin["number"]

    saved = board_cache.get(("pin", pin_id)) or _saved("pin", pin_id, BOARD_TTL)
    if not saved:
        # A pin's public page carries its full-size image in the og:image tag. No login needed.
        try:
            html = requests.get(f"https://www.pinterest.com/pin/{pin_id}/", headers=HEADERS, timeout=15).text
        except requests.RequestException as e:
            return f"Pinterest could not be reached ({type(e).__name__}). Tell the user to try again in a minute."
        image = re.search(r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', html) or re.search(r'<meta[^>]+content="([^"]+)"[^>]+property="og:image"', html)
        if not image:
            return "That pin couldn't be opened. It may be private or deleted. Ask the user for another pin."
        pin = {"number": 1, "image": image.group(1), "link": f"https://www.pinterest.com/pin/{pin_id}/"}
        read = _vision([{"type": "text", "text": BOARD_PROMPT.format(n=1)}, {"type": "text", "text": "Pin 1:"},
                        {"type": "image_url", "image_url": {"url": pin["image"]}}], BoardRead)
        if isinstance(read, str):
            return read
        pieces = [{"number": n, **p.model_dump(), "pins": [1], "image": pin["image"]} for n, p in enumerate(read.pieces, start=1)]
        saved = json.dumps({"style": read.style, "palette": read.palette, "outfit_pins": [1], "pieces": pieces, "pins": [pin]})
        _save("pin", pin_id, saved)
    board_cache["pin", pin_id] = saved
    return json.loads(saved), None


def read_pinterest_board(board_url: str, *, state: dict) -> str:
    """Read a public Pinterest board and turn its outfits into searchable secondhand pieces."""
    result = _read_board(board_url)
    if "error" not in json.loads(result):
        state["board"] = json.loads(result)  # every later tool works from this
    return result


def _board_or_error(state: dict) -> dict | str:
    return state.get("board") or _error(
        "No board has been read in this conversation. Ask the user for their Pinterest board and call read_pinterest_board first."
    )


# --- Listing search (SerpAPI) ---

Source = Literal["marketplaces", "ebay"]


def _serpapi(params: dict, timeout: int = 60) -> dict | str:
    """One SerpAPI search, cached. Returns the response, or an error message the model can act on."""
    key = os.environ.get("SERPAPI_KEY", "").strip()
    if not key:
        return "Listing search is not configured on this server. Tell the user live search is unavailable; you can still read boards."
    cache_key = tuple(sorted(params.items()))
    with cache_lock:
        if cache_key in search_cache:
            return search_cache[cache_key]

    try:
        # Most searches take 1-2s, but a fresh Google Shopping search occasionally takes 30s or more
        data = requests.get("https://serpapi.com/search.json", params={**params, "api_key": key}, timeout=timeout).json()
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

    with cache_lock:
        search_cache[cache_key] = data
    return data


def _search(query: str, size: str | None = None, source: Source = "marketplaces", timeout: int = 60) -> list[dict] | str:
    """_fetch_listings, but served from the saved copy when one is fresh."""
    key = json.dumps([query, size, source])
    saved = _saved("search", key, SEARCH_TTL)
    if saved:
        listings = json.loads(saved)
        for l in listings:
            if "size" not in l:  # saved before titles were cleaned
                l["title"], l["size"] = _clean_title(l["title"])
        return listings
    listings = _fetch_listings(query, size, source, timeout)
    if isinstance(listings, list):
        _save("search", key, json.dumps(listings))
    return listings


def _clean_title(title: str | None) -> tuple[str, str | None]:
    """Poshmark-style titles arrive as 'Brand Category | Title | Color: X | Size: Y | Seller's Closet'.
    Keep the real title and pull out the size, which is worth showing on its own."""
    parts = [p.strip() for p in (title or "").split(" | ")]
    size = next((p.split(":", 1)[1].strip() for p in parts if p.lower().startswith("size:")), None)
    kept = [p for p in parts if not re.match(r"(?i)(color|size):", p) and not p.lower().endswith("closet")]
    return (max(kept, key=len) if kept else title or ""), size


def _fetch_listings(query: str, size: str | None, source: Source, timeout: int = 60) -> list[dict] | str:
    """Search one source and normalize every priced result. Returns listings or an error message.

    Each listing remembers the search it came from, so later tools can re-run it (from cache) to get
    price comparables or to re-check a watched listing.
    """
    q = f"{query} size {size}" if size else query
    search = {"query": query, "size": size, "source": source}
    listings = []

    if source == "ebay":
        data = _serpapi({"engine": "ebay", "_nkw": q, "LH_ItemCondition": "3000"}, timeout)  # 3000 = used
        if isinstance(data, str):
            return data
        for r in data.get("organic_results", []):
            price = (r.get("price") or {}).get("extracted")  # price ranges (multi-size listings) have none
            item = re.search(r"/itm/(\d+)", r.get("link", ""))
            if price is not None and item:
                listings += [{
                    "key": f"ebay_{item.group(1)}", "title": r.get("title"), "price": price, "site": "eBay",
                    "condition": r.get("condition") or "Pre-Owned", "link": r["link"].split("?")[0],
                    "image": r.get("thumbnail"), "search": search,
                }]
    else:
        # Google Shopping, biased to used: covers Depop, Poshmark, Mercari, Etsy, and eBay sellers in one search
        data = _serpapi({"engine": "google_shopping", "q": f"{q} used", "gl": "us", "hl": "en"}, timeout)
        if isinstance(data, str):
            return data
        for r in data.get("shopping_results", []):
            if r.get("extracted_price") is not None:
                title, listed_size = _clean_title(r.get("title"))
                listings += [{
                    "key": f"gs_{r.get('product_id')}", "title": title, "size": listed_size, "price": r["extracted_price"],
                    "site": (r.get("source") or "unknown").split(" - ")[0],
                    "condition": r.get("second_hand_condition") or "unspecified",
                    "link": r.get("product_link"), "image": r.get("thumbnail"), "search": search,
                }]

    return listings


def _search_with_fallback(query: str, broad_query: str | None, size: str | None = None, source: Source = "marketplaces", timeout: int = 60) -> list[dict] | str:
    """A very specific query can come back thin; the broader one usually doesn't."""
    found = _search(query, size, source, timeout)
    if isinstance(found, list) and len(found) < 3 and broad_query:
        broader = _search(broad_query, size, source, timeout)
        if isinstance(broader, list) and len(broader) > len(found):
            return broader
    return found


def _search_pieces(pieces: list[dict]) -> list[list[dict] | str]:
    """Search several board pieces at once: the wait is the slowest search, not the sum of them.

    One slow search shouldn't sink the whole answer, so each gets a short timeout and one retry
    (SerpAPI finishes the search after we give up, so the retry often comes straight from its cache).
    Pieces that still fail come back as error strings for the caller to skip and mention.
    """
    def one(piece: dict) -> list[dict] | str:
        found = _search_with_fallback(piece["query"], piece["broad_query"], timeout=PIECE_TIMEOUT)
        if isinstance(found, str) and "slowly" in found:
            found = _search_with_fallback(piece["query"], piece["broad_query"], timeout=PIECE_TIMEOUT)
        return found

    with ThreadPoolExecutor(max_workers=len(pieces) or 1) as pool:
        return list(pool.map(one, pieces))


def _show(listings: list[dict], state: dict) -> list[dict]:
    """Give listings short ids (L1, L2...) the model can refer back to, and remember them for this session."""
    for i, listing in enumerate(listings, start=1):
        listing["id"] = f"L{i}"
    state["listings"] = {l["id"]: l for l in listings}
    return [_card(l) for l in listings]


def _card(listing: dict) -> dict:
    """The fields the model and the UI need about one listing."""
    return {k: listing.get(k) for k in ("id", "title", "price", "site", "size", "condition", "link", "image")}


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
    found = _search_with_fallback(query, fallback_query, size, source)
    if isinstance(found, str):
        return _error(found)
    if not found:
        return _error(f"No secondhand listings for '{query}'. Retry with a shorter, broader query (the piece's broad_query), or without the size.")

    prices = sorted(l["price"] for l in found)
    matches = [l for l in found if max_price is None or l["price"] <= max_price][:MAX_LISTINGS]
    result = {
        "query": found[0]["search"]["query"],
        "size": size,
        "source": source,
        "results_scanned": len(found),
        "price_summary": {"low": prices[0], "median": statistics.median(prices), "high": prices[-1]},
        "listings": _show(matches, state),
    }
    if not matches:
        result["note"] = (
            f"Nothing at or under ${max_price:g}. The cheapest of {len(found)} listings was ${prices[0]:g}. "
            "Offer to raise the budget, or to watch this piece at the user's price with watch_item."
        )
    return json.dumps(result)


def _listings_or_error(listing_ids: list[str] | None, state: dict) -> list[dict] | str:
    shown = state.get("listings")
    if not shown:
        return _error("No listings to work with yet. Call search_listings or shop_the_pin first.")
    unknown = [i for i in listing_ids or [] if i not in shown]
    if unknown:
        return _error(f"Unknown listing ids {unknown}. Valid ids from the last results: {list(shown)}.")
    return [shown[i] for i in listing_ids or shown]


# --- price_verdict ---


def _market(prices: list[float]) -> dict:
    p25, median, p75 = statistics.quantiles(prices, n=4)
    return {"comparable_listings": len(prices), "median": round(median, 2), "typical_range": [round(p25, 2), round(p75, 2)]}


def _verdict(price: float, prices: list[float]) -> dict:
    """Where a price sits among comparable listings. Quartiles keep one wild outlier from skewing it."""
    market = _market(prices)
    p25, p75 = market["typical_range"]
    return {
        "verdict": "steal" if price <= p25 else "fair" if price <= p75 else "overpriced",
        "cheaper_than_pct_of_comparables": round(100 * sum(c > price for c in prices) / len(prices)),
        "vs_median": round(price - market["median"], 2),
    }


def price_verdict(
    listing_ids: list[str] | None = None,
    price: float | None = None,
    query: str | None = None,
    *,
    state: dict,
) -> str:
    """Judge prices against every comparable secondhand listing, not just the few shown."""
    if listing_ids:
        listings = _listings_or_error(listing_ids, state)
        if isinstance(listings, str):
            return listings
        to_judge = [(l["id"], l["title"], l["price"], l["search"]) for l in listings]
    elif price is not None and query:
        to_judge = [("user", query, price, {"query": query, "size": None, "source": "marketplaces"})]
    else:
        return _error("Pass listing_ids from the last results, or both price and query for a listing the user found elsewhere.")

    markets, verdicts = {}, []
    for listing_id, title, p, search in to_judge:
        # The same search the listing came from: cached, so this costs no extra quota
        comps = _search(**search)
        if isinstance(comps, str):
            return _error(comps)
        prices = [c["price"] for c in comps]
        if len(prices) < 5:
            verdicts += [{"id": listing_id, "title": title, "price": p, "verdict": "unknown", "why": f"only {len(prices)} comparable listings"}]
            continue
        markets[search["query"]] = _market(prices)
        verdicts += [{"id": listing_id, "title": title, "price": p, "compared_with": search["query"], **_verdict(p, prices)}]

    return json.dumps({"markets": markets, "verdicts": verdicts})


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

For each listing below, give a score from 0 to 100 for how well it would fit this board (not just
whether it matches the search words: silhouette, fabric, color and vibe all count), and a reason of
at most 12 words naming what fits or what's off (e.g. "right camel wool, but cropped where pins are long").
"""


def _piece_for(board: dict, query: str) -> dict | None:
    return next((p for p in board["pieces"] if query in (p["query"], p["broad_query"])), None)


def style_match(listing_ids: list[str] | None = None, *, state: dict) -> str:
    """Score how well each listing's photo fits the board's style."""
    board = _board_or_error(state)
    if isinstance(board, str):
        return board
    listings = _listings_or_error(listing_ids, state)
    if isinstance(listings, str):
        return listings

    content = [{"type": "text", "text": STYLE_PROMPT.format(style=board["style"], palette=", ".join(board["palette"]))}]
    # If the listings all hunt for one board piece, show the model the pin it came from
    queries = {l["search"]["query"] for l in listings}
    piece = _piece_for(board, queries.pop()) if len(queries) == 1 else None
    if piece:
        content += [{"type": "text", "text": f"The client is hunting for: {piece['query']}. The pin it came from:"},
                    {"type": "image_url", "image_url": {"url": piece["image"]}}]
    for l in listings:
        content += [{"type": "text", "text": f"Listing {l['id']} (searched as '{l['search']['query']}'): {l['title']} (${l['price']:g}, {l['site']})"}]
        if l.get("image"):
            content += [{"type": "image_url", "image_url": {"url": l["image"]}}]

    scores = _vision(content, StyleScores)
    if isinstance(scores, str):
        return _error(scores)

    by_id = {l["id"]: l for l in listings}
    scored = [
        {"id": s.id, "title": by_id[s.id]["title"], "price": by_id[s.id]["price"], "score": max(0, min(100, s.score)), "reason": s.reason}
        for s in scores.scores if s.id in by_id
    ]
    return json.dumps({"board_style": board["style"], "scores": sorted(scored, key=lambda s: -s["score"])})


# --- Value picks: the real listing to buy for each piece (shared by shop_the_pin and board_unlock) ---


class OptionScore(BaseModel):
    piece: int
    option: int
    score: int
    reason: str


class OptionScores(BaseModel):
    scores: list[OptionScore]


PICK_PROMPT = """You are a stylist helping a client shop secondhand for pieces from their Pinterest board.
Board style: {style}

{context}For each piece below there are numbered secondhand options. Score EVERY option from 0 to 100
on how well it would stand in for that piece on this board: silhouette, fabric, color and vibe.
Score below 50 if it is a different garment, a kids' item, or clearly not what's wanted. Ignore price.
Give a reason of at most 12 words for each.
"""


def _shortlist(found: list[dict]) -> list[dict]:
    """The most relevant few plus the cheapest few (among on-topic results). The cheapest alone are
    often the wrong garment; the most relevant alone miss the deals. The value rule picks between them."""
    cheapest = sorted(found[:15], key=lambda l: l["price"])[:OPTIONS_PER_PIECE]
    return list({l["key"]: l for l in found[:OPTIONS_PER_PIECE] + cheapest}.values())


def _value_picks(board: dict, pieces: list[dict], options: dict[int, list[dict]], pin: dict | None = None) -> dict:
    """Pick one real listing per piece: the cheapest of the strong matches.

    The stylist model scores every option in one vision call; code then applies the rule, so the
    choice is consistent and explainable: within VALUE_MARGIN points of the best match (and at
    least MIN_MATCH), price decides. Returns piece number -> {listing, match, reason}.
    """
    context = "The outfit being recreated is the pin below.\n" if pin else ""
    content = [{"type": "text", "text": PICK_PROMPT.format(style=board["style"], context=context)}]
    if pin:
        content += [{"type": "image_url", "image_url": {"url": pin["image"]}}]
    for piece in pieces:
        for k, l in enumerate(options.get(piece["number"], []), start=1):
            content += [{"type": "text", "text": f"Piece {piece['number']} ({piece['query']}), option {k}: {l['title']}"}]
            if l.get("image"):
                content += [{"type": "image_url", "image_url": {"url": l["image"]}}]

    scored = _vision(content, OptionScores)
    scores = {}
    if isinstance(scored, OptionScores):
        for s in scored.scores:
            if 1 <= s.option <= len(options.get(s.piece, [])):
                scores[s.piece, s.option] = (max(0, min(100, s.score)), s.reason)

    picks = {}
    for number, opts in options.items():
        rated = [(scores[number, k][0], scores[number, k][1], l) for k, l in enumerate(opts, start=1) if (number, k) in scores]
        if not rated:
            # The style check failed: fall back to the cheapest relevant listing, and say so
            picks[number] = {"listing": opts[0], "match": None, "reason": "style check unavailable; cheapest relevant listing"}
            continue
        top = max(r[0] for r in rated)
        if top < MIN_MATCH:
            continue  # nothing here is really the piece
        match, reason, listing = min((r for r in rated if r[0] >= top - VALUE_MARGIN), key=lambda r: r[2]["price"])
        picks[number] = {"listing": listing, "match": match, "reason": reason}
    return picks


def _hunt(board: dict, pieces: list[dict], pin: dict | None = None) -> tuple[dict, dict, list[str]]:
    """Search pieces in parallel and value-pick a listing for each.
    Returns (picks, typical secondhand price per piece, queries that came up empty or failed)."""
    options, typical, missed = {}, {}, []
    for piece, found in zip(pieces, _search_pieces(pieces)):
        if isinstance(found, str) or not found:
            missed += [piece["query"]]
            continue
        options[piece["number"]] = _shortlist(found)
        typical[piece["number"]] = statistics.median(l["price"] for l in found)
    picks = _value_picks(board, pieces, options, pin) if options else {}
    missed += [f"{p['query']} (no close match right now)" for p in pieces if p["number"] in options and p["number"] not in picks]
    return picks, typical, missed


# --- shop_the_pin ---


def shop_the_pin(pin_number: int | None = None, pin_url: str | None = None, *, state: dict) -> str:
    """Recreate one pin's whole outfit secondhand and total up what it would cost."""
    if pin_url:
        found = _read_pin(pin_url, state.get("board"))
        if isinstance(found, str):
            return _error(found)
        board, pin_number = found
        on_board = pin_number is not None
        pin_number = pin_number or 1
    elif pin_number is not None:
        board = _board_or_error(state)
        if isinstance(board, str):
            return board
        on_board = True
    else:
        return _error("Pass pin_number for a pin on the board you read, or pin_url for a pin link the user pasted.")

    pieces = [p for p in board["pieces"] if pin_number in p["pins"]]
    if not pieces:
        shoppable = sorted({n for p in board["pieces"] for n in p["pins"]})
        return _error(f"Pin {pin_number} has no shoppable pieces. Pins with pieces: {shoppable}. Ask the user to pick one of those.")
    # Skip what the user said they own (remembered by board_unlock), and put garments first:
    # they make the outfit; a bag or bracelet is the first thing to drop
    owned = [p for p in pieces if on_board and p["number"] in state.get("owned", set())]
    pieces = sorted((p for p in pieces if p not in owned), key=lambda p: p["category"] not in GARMENTS)[:MAX_PIN_PIECES]
    if not pieces:
        return _error(f"The user already owns every piece in pin {pin_number}. Tell them they can wear this look today.")

    pin = board["pins"][pin_number - 1]
    picks, typical, missed = _hunt(board, pieces, pin)
    if not picks:
        return _error(f"No good secondhand matches for the pieces in this pin right now ({', '.join(missed)}). Suggest another pin, or try again in a minute.")

    order = [p["number"] for p in pieces if p["number"] in picks]
    shown = _show([picks[n]["listing"] for n in order], state)  # lets the user price-check or watch any pick
    queries = {p["number"]: p["query"] for p in pieces}
    total = sum(l["price"] for l in shown)
    typical_total = sum(typical[n] for n in order)
    return json.dumps({
        "pin": pin_number if on_board else None,  # None: a pasted pin that isn't on the user's board
        "pin_image": pin["image"],
        "pin_link": pin["link"],
        "pieces": [
            {"piece": n, "looking_for": queries[n], "pick": card, "style_score": picks[n]["match"], "reason": picks[n]["reason"]}
            for n, card in zip(order, shown)
        ],
        "already_own": [p["query"] for p in owned],
        "not_found": missed,
        "total": round(total, 2),
        "typical_secondhand_total": round(typical_total, 2),
        "saved_vs_typical": round(typical_total - total, 2),
        "how_picked": f"For each piece: the cheapest listing within {VALUE_MARGIN} match points of the best match.",
    })


# --- board_unlock ---


def board_unlock(budget: float, owned_pieces: list[int] | None = None, *, state: dict) -> str:
    """Pick which real listings to buy within a budget to recreate the most outfits on the board."""
    board = _board_or_error(state)
    if isinstance(board, str):
        return board
    numbers = {p["number"] for p in board["pieces"]}
    if set(owned_pieces or []) - numbers:
        return _error(f"Unknown piece numbers {sorted(set(owned_pieces) - numbers)}. Valid piece numbers: {sorted(numbers)}.")
    # Remember what the user owns for the rest of the session, so shop_the_pin skips it too
    state["owned"] = owned = state.get("owned", set()) | set(owned_pieces or [])

    # What each outfit pin needs: its garments (accessories are optional)
    needs = {}
    for pin in board["outfit_pins"]:
        garments = {p["number"] for p in board["pieces"] if pin in p["pins"] and p["category"] in GARMENTS}
        if garments:
            needs[pin] = garments
    if not needs:
        return _error("No outfit pins with identifiable garments on this board, so there is nothing to unlock.")

    # Hunt only the pieces that appear in the most pins: they're the ones that can unlock outfits
    pieces = {p["number"]: p for p in board["pieces"]}
    reach = {n: sum(n in need for need in needs.values()) for n in numbers - owned if pieces[n]["category"] in GARMENTS}
    candidates = sorted((n for n in reach if reach[n]), key=lambda n: -reach[n])[:UNLOCK_CANDIDATES]

    # A real listing per piece, value-picked: the plan is priced from things you can actually buy
    picks, _, missed = _hunt(board, [pieces[n] for n in candidates])
    prices = {n: picks[n]["listing"]["price"] for n in picks}

    def recreated(have: set) -> set:
        return {pin for pin, need in needs.items() if need <= have}

    # Few enough pieces to try every combination: the true best plan, not a greedy guess.
    # Ties go to the cheaper plan.
    best, best_cost = (), 0.0
    best_pins = recreated(owned)
    for r in range(1, len(prices) + 1):
        for combo in combinations(prices, r):
            cost = sum(prices[n] for n in combo)
            pins = recreated(owned | set(combo))
            if cost <= budget and (len(pins), -cost) > (len(best_pins), -best_cost):
                best, best_cost, best_pins = combo, cost, pins

    # The single piece outside the plan that would recreate the most extra outfits
    have = owned | set(best)
    upgrades = [(len(recreated(have | {n}) - best_pins), -prices[n], n) for n in prices if n not in have]
    gain, _, upgrade = max(upgrades, default=(0, 0, None))

    # Show the plan's listings (and the next buy) with ids, so they can be watched or price-checked
    order = sorted(best, key=lambda n: -reach[n]) + ([upgrade] if gain else [])
    cards = dict(zip(order, _show([picks[n]["listing"] for n in order], state)))
    already = recreated(owned)
    return json.dumps({
        "headline": f"${best_cost:.0f} of your ${budget:g} recreates {len(best_pins)} of the {len(needs)} outfits on this board.",
        "outfits_recreated": len(best_pins),
        "outfits_on_board": len(needs),
        "outfits_recreated_by_what_you_own": len(already),
        "pct_of_board": round(100 * len(best_pins) / len(needs)),
        "budget": budget,
        "spend": round(best_cost, 2),
        "recreated_pins": sorted(best_pins),
        "buy": [
            {"piece": n, "looking_for": pieces[n]["query"], "listing": cards[n], "match": picks[n]["match"],
             "in_pins": sorted(p for p in needs if n in needs[p])}
            for n in best
        ],
        "next_best_buy": (
            {"piece": upgrade, "looking_for": pieces[upgrade]["query"], "listing": cards[upgrade], "would_recreate_more": gain}
            if gain else None
        ),
        "couldnt_find": missed,
        "how": (
            "Each piece is priced by a real listing: the cheapest one within "
            f"{VALUE_MARGIN} match points of the best match. A pin counts as recreated once you have all its garments. "
            "Every combination of pieces within budget was tried."
        ),
    })


# --- Watchlist (Firestore) ---

def _watchlist(state: dict):
    """This session's watchlist collection. The session id comes from the harness, never the model."""
    return _firestore().collection("watchlists").document(state["session_id"]).collection("items")


def _slug(*parts) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(str(p) for p in parts if p).lower()).strip("-")


CARD = ("title", "price", "site", "link", "image")


def watch_item(
    query: str | None = None,
    target_price: float | None = None,
    size: str | None = None,
    source: Source = "marketplaces",
    listing_ids: list[str] | None = None,
    *,
    state: dict,
) -> str:
    """Watch either a search for new listings under a target price, or specific listings for price drops."""
    try:
        items = _watchlist(state)
        existing = len(list(items.limit(MAX_WATCHED + 1).stream()))
    except GoogleAPIError as e:
        return _error(f"The watchlist database is unavailable ({type(e).__name__}). Tell the user to try again shortly.")

    if listing_ids:
        # Track exact listings: only these few listings' prices are stored
        listings = _listings_or_error(listing_ids, state)
        if isinstance(listings, str):
            return listings
        if existing + len(listings) > MAX_WATCHED:
            return _error(f"The watchlist holds {MAX_WATCHED} things and has {existing}. Ask the user what to drop with unwatch_item first.")
        for l in listings:
            items.document(_slug("listing", l["key"])).set({
                "kind": "listing", "key": l["key"], **{k: l[k] for k in CARD}, "search": l["search"],
                "created": _now(), "last_checked": _now(),
            })
        return json.dumps({"watching_listings": [{k: l[k] for k in CARD} for l in listings], "watchlist_size": existing + len(listings)})

    if not query or target_price is None:
        return _error("Pass query and target_price to watch for new listings under a price, or listing_ids to watch specific listings.")
    if existing >= MAX_WATCHED:
        return _error(f"The watchlist is full ({MAX_WATCHED} things). Ask the user what to drop with unwatch_item first.")
    found = _search(query, size, source)
    if isinstance(found, str):
        return _error(found)

    # Only the listings already under target are stored, so the next check reports just what's new
    under = sorted((l for l in found if l["price"] <= target_price), key=lambda l: l["price"])
    items.document(_slug(query, size, source)).set({
        "kind": "search", "query": query, "size": size, "source": source, "target_price": target_price,
        "reported": [l["key"] for l in under], "created": _now(), "last_checked": _now(),
    })
    return json.dumps({
        "watching_search": query, "size": size, "target_price": target_price,
        "already_under_target": [{k: l[k] for k in CARD} for l in under[:3]],
        "watchlist_size": existing + 1,
    })


def check_watchlist(*, state: dict) -> str:
    """Re-check everything watched: new listings under target, price drops, and listings that vanished."""
    try:
        docs = list(_watchlist(state).limit(MAX_WATCHED).stream())
    except GoogleAPIError as e:
        return _error(f"The watchlist database is unavailable ({type(e).__name__}). Tell the user to try again shortly.")
    if not docs:
        return json.dumps({"watching": [], "note": "The watchlist is empty. Offer to watch a piece with watch_item."})

    report = []
    for doc in docs:
        item = doc.to_dict()
        search = item.get("search") or {k: item[k] for k in ("query", "size", "source")}
        found = _search(**search)  # watched listings from one search share one (cached) lookup
        if isinstance(found, str):
            report += [{"watch_id": doc.id, "error": found}]
            continue

        if item["kind"] == "search":
            under = sorted((l for l in found if l["price"] <= item["target_price"]), key=lambda l: l["price"])
            new = [l for l in under if l["key"] not in item["reported"]]
            doc.reference.update({"reported": [l["key"] for l in under], "last_checked": _now()})
            report += [{
                "watch_id": doc.id, "kind": "search", "query": item["query"], "target_price": item["target_price"],
                "previous_check": item["last_checked"],
                "new_under_target": [{k: l[k] for k in CARD} for l in new[:5]],
                "under_target_total": len(under),
                "cheapest_now": min((l["price"] for l in found), default=None),
            }]
        else:
            now = next((l for l in found if l["key"] == item["key"]), None)
            if now is None:
                status = "gone from search results; it may have sold"
            elif now["price"] < item["price"]:
                status = f"price dropped from ${item['price']:g} to ${now['price']:g}"
                doc.reference.update({"price": now["price"]})
            else:
                status = f"still listed at ${now['price']:g}"
            doc.reference.update({"last_checked": _now()})
            report += [{"watch_id": doc.id, "kind": "listing", **{k: item[k] for k in CARD}, "status": status, "previous_check": item["last_checked"]}]

    return json.dumps({"checked_at": _now(), "watching": report})


def unwatch_item(watch_ids: list[str], *, state: dict) -> str:
    """Remove things from the watchlist."""
    try:
        items = _watchlist(state)
        current = {d.id for d in items.list_documents()}
        unknown = [w for w in watch_ids if w not in current]
        if unknown:
            return _error(f"Unknown watch ids {unknown}. Current ids: {sorted(current)}. Call check_watchlist to see what each one is.")
        for w in watch_ids:
            items.document(w).delete()
    except GoogleAPIError as e:
        return _error(f"The watchlist database is unavailable ({type(e).__name__}). Tell the user to try again shortly.")
    return json.dumps({"removed": watch_ids, "still_watching": sorted(current - set(watch_ids))})


# What the model sees: the "set notes" in the screenplay.
SOURCE_PARAM = {
    "type": "string",
    "enum": ["marketplaces", "ebay"],
    "description": (
        "'marketplaces' (default) searches used listings across Depop, Poshmark, Mercari, Etsy and eBay "
        "via Google Shopping. 'ebay' searches eBay only, with direct listing links."
    ),
}
LISTING_IDS = {"type": "array", "items": {"type": "string"}, "description": "Ids from the most recent listing results, e.g. ['L1', 'L3']."}


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []},
    }}


TOOLS = [
    _tool(
        "read_pinterest_board",
        "Read a public Pinterest board's latest pins (up to 25) and turn the outfits in them into a style summary, "
        "a color palette, and a numbered list of clothing pieces. Each piece has the pins it appears in, a specific "
        "secondhand search query and a broader fallback query. Call this whenever the user shares a Pinterest board "
        "link. Results are cached, so call it again rather than guessing if you need the pieces after a long conversation.",
        {"board_url": {"type": "string", "description": "The board's URL exactly as the user gave it, e.g. 'https://www.pinterest.com/jane/summer-fits/' or a 'pin.it/...' short link."}},
        ["board_url"],
    ),
    _tool(
        "search_listings",
        "Search live secondhand listings for one clothing piece. Returns up to 8 real listings (each with an id like "
        "'L1', title, price in USD, site, condition, link, image) plus a price summary across every result scanned. "
        "Call this when the user wants to find, shop, or price a single piece. For a board piece, pass its query as "
        "query and its broad_query as fallback_query. Only ever quote listings, prices and links that a tool returned.",
        {
            "query": {"type": "string", "description": "What to search for, 3-6 words: color + material + garment, e.g. 'brown plaid pleated mini skirt'."},
            "max_price": {"type": "number", "description": "Only return listings at or under this price in USD, e.g. 60. Omit if the user gave no budget."},
            "size": {"type": "string", "description": "Size as the user said it, e.g. 'M', '8', '28'. Omit if not given."},
            "source": SOURCE_PARAM,
            "fallback_query": {"type": "string", "description": "A broader 2-3 word version of query, searched only if query finds fewer than 3 listings, e.g. 'plaid mini skirt'."},
        },
        ["query"],
    ),
    _tool(
        "price_verdict",
        "Judge whether prices are a steal, fair, or overpriced by ranking each against every comparable secondhand "
        "listing (median, typical range, and what share of listings it is cheaper than). Call this when the user asks "
        "if something is a good deal or worth it. Pass listing_ids from the most recent results, or price and query "
        "for a listing the user found elsewhere.",
        {
            "listing_ids": LISTING_IDS,
            "price": {"type": "number", "description": "A price in USD the user found elsewhere, e.g. 40. Use with query."},
            "query": {"type": "string", "description": "What that listing is, 3-6 words, e.g. 'black leather knee high boots'. Use with price."},
        },
    ),
    _tool(
        "style_match",
        "Look at the photos of listings from the most recent results and score each 0-100 on how well it fits the "
        "user's Pinterest board (silhouette, fabric, color, vibe), with a short reason. Call this when the user asks "
        "which listing best matches their board or style, or which to pick. Needs a board read earlier in the conversation.",
        {"listing_ids": {**LISTING_IDS, "description": "Ids to score, e.g. ['L1', 'L2']. Omit to score every listing from the most recent results."}},
    ),
    _tool(
        "shop_the_pin",
        "Recreate one pin's whole outfit secondhand: searches every piece in that pin, has a stylist score the "
        "cheapest relevant listings against the pin's photo, and picks the best value per piece (the cheapest strong "
        "match), then totals the outfit against what those pieces typically sell for secondhand. Call this when the user wants a specific pin or look, e.g. 'get me pin 6', "
        "'how much to recreate this outfit', or when they paste a link to a single pin (no board needed). Pass "
        "exactly one of pin_number or pin_url. The picks get listing ids, so they can be price-checked or watched after.",
        {
            "pin_number": {"type": "integer", "description": "The pin's number from read_pinterest_board (pins are numbered from 1), e.g. 6."},
            "pin_url": {"type": "string", "description": "A link to one pin exactly as the user gave it, e.g. 'https://www.pinterest.com/pin/1093882197027642046/' or a 'pin.it/...' short link."},
        },
    ),
    _tool(
        "board_unlock",
        "Plan which pieces to buy secondhand, within a budget, to be able to recreate the most outfit pins on the "
        "board. For the pieces that appear in the most pins it finds a real listing to buy (the cheapest strong "
        "match), then tries every combination of those listings within budget. Returns a ready headline, the "
        "listings to buy, the outfits they recreate, and the single best next buy. Quote its headline numbers as given. Call this when the user "
        "has a budget for the whole board or asks what to buy first or where to start. Pieces the user already owns "
        "count toward outfits for free.",
        {
            "budget": {"type": "number", "description": "Total the user wants to spend, in USD, e.g. 100."},
            "owned_pieces": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of board pieces the user already owns, e.g. [1, 3]. Omit if none."},
        },
        ["budget"],
    ),
    _tool(
        "watch_item",
        "Add to the user's watchlist, in one of two ways. (1) To catch new listings for a piece under a price, pass "
        "query and target_price: it reports listings already under target now, and check_watchlist later reports "
        "only new ones. (2) To follow specific listings for price drops or selling out, pass listing_ids from the "
        "most recent results. Use whichever the user asked for; ask if unclear. The watchlist holds up to 6 things.",
        {
            "query": {"type": "string", "description": "Way 1: the piece to watch, 3-6 words. Reuse the query the user already searched, e.g. 'brown leather knee high boots'."},
            "target_price": {"type": "number", "description": "Way 1: the most the user wants to pay, in USD, e.g. 45."},
            "size": {"type": "string", "description": "Way 1: size as the user said it, e.g. 'M'. Omit if not given."},
            "source": SOURCE_PARAM,
            "listing_ids": {**LISTING_IDS, "description": "Way 2: ids of the specific listings to follow, e.g. ['L2']."},
        },
    ),
    _tool(
        "check_watchlist",
        "Re-check everything on the user's watchlist. Watched searches report new listings under target since the "
        "previous check; watched listings report a price drop, still listed, or gone (may have sold). Each entry has a "
        "watch_id. Call this when the user asks what's new, to check their watchlist, or whether anything dropped.",
        {},
    ),
    _tool(
        "unwatch_item",
        "Remove things from the user's watchlist. Call this when the user asks to stop watching something, or to make "
        "room when the watchlist is full. Use watch_ids from check_watchlist.",
        {"watch_ids": {"type": "array", "items": {"type": "string"}, "description": "watch_id values from check_watchlist, e.g. ['brown-plaid-pleated-mini-skirt-marketplaces']."}},
        ["watch_ids"],
    ),
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "read_pinterest_board": read_pinterest_board,
    "search_listings": search_listings,
    "price_verdict": price_verdict,
    "style_match": style_match,
    "shop_the_pin": shop_the_pin,
    "board_unlock": board_unlock,
    "watch_item": watch_item,
    "check_watchlist": check_watchlist,
    "unwatch_item": unwatch_item,
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
    except Exception as e:
        # A surprise from an outside service (odd data, a network blip) becomes an error the model can relay
        return _error(f"{name} failed unexpectedly ({type(e).__name__}). Tell the user to try again in a moment.")

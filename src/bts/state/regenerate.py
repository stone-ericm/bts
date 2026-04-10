"""Reconstruct BTS state from Bluesky post history + MLB API.

Used for disaster recovery: if the Fly machine loses its volume entirely,
or during migration between providers, this command rebuilds the full
state from authoritative external sources. Pre-cutoff state comes from
the committed initial-state.json snapshot.

The heart of this module is a post parser that must handle the human-readable
format produced by src/bts/posting.py's format_post() and format_skip_post()
functions. If those formats change, this parser must be updated in lockstep.
The unit tests deliberately import the formatters and feed real output to the
parser so that drift between the two halves of the round-trip is caught
immediately.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional


@dataclass
class ParsedPost:
    """A Bluesky post parsed into structured pick fields."""
    uri: str
    created_at: str
    text: str
    is_reply: bool
    # For pick posts
    is_skip: bool = False
    batter_name: Optional[str] = None
    team: Optional[str] = None
    is_double_down: bool = False
    double_down_batter: Optional[str] = None
    double_down_team: Optional[str] = None
    streak_at_time: Optional[int] = None
    # For result reply posts
    is_result: bool = False
    result: Optional[str] = None  # "hit" | "miss" | "skip"
    streak_after: Optional[int] = None


def _bluesky_client():
    """Lazy import wrapper so tests can mock it."""
    from atproto import Client
    return Client()


def fetch_bluesky_posts(
    handle: str,
    from_date: str,
    limit: int = 5000,
) -> list[ParsedPost]:
    """Fetch the author's post history via atproto get_author_feed.

    Returns posts in chronological order (oldest first). Filters to
    posts created on or after from_date.
    """
    client = _bluesky_client()
    all_posts: list = []
    cursor = None

    while True:
        response = client.get_author_feed(
            actor=handle,
            limit=100,
            cursor=cursor,
        )
        feed = response.feed
        if not feed:
            break
        all_posts.extend(feed)
        cursor = response.cursor
        if not cursor or len(all_posts) >= limit:
            break

    # Filter by date + parse
    parsed: list[ParsedPost] = []
    for entry in all_posts:
        record = entry.post.record
        created_at = record.created_at
        if created_at < f"{from_date}T00:00:00":
            continue
        is_reply = getattr(record, "reply", None) is not None
        parsed.append(ParsedPost(
            uri=entry.post.uri,
            created_at=created_at,
            text=record.text,
            is_reply=is_reply,
        ))

    parsed.sort(key=lambda p: p.created_at)
    return parsed


# Regex patterns for post format (see src/bts/posting.py format_post / format_skip_post).
#
# Real format examples:
#
#   Single pick:
#     Today's pick: Nico Hoerner (CHC)
#     vs Test Pitcher | 78.3%
#
#     Streak: 2
#
#   Double down:
#     Today's picks (double down):
#     Jose Altuve (HOU) vs Pitcher A — 82.0%
#     Kyle Tucker (HOU) vs Pitcher B — 80.0%
#     P(both): 65.6%
#
#     Streak: 5
#
#   Skip:
#     Sitting today out. Top pick: Top Batter (NYY) at 76.5% — below our threshold.
#
#     Streak holds at 3.
#
# The double-down second line may omit the team and/or pitcher (e.g.
# "Kyle Tucker — 80.0%"), so the team group is optional in _DOUBLE_RE.
_PICK_RE = re.compile(
    r"Today's pick:\s*(?P<name>.+?)\s*\((?P<team>[A-Za-z]{2,4})\)",
)
_DOUBLE_HEADER_RE = re.compile(r"Today's picks \(double down\):")
# First batter line of a double-down post. Anchored to the line after the header.
_DOUBLE_FIRST_RE = re.compile(
    r"Today's picks \(double down\):\s*\n"
    r"(?P<name>.+?)\s*\((?P<team>[A-Za-z]{2,4})\)\s*(?:vs\s+\S.*?)?\s*[—-]\s*\d",
)
# Second batter line — team is optional. We anchor on the line that ends with a percentage
# and is followed (eventually) by "P(both):".
_DOUBLE_SECOND_RE = re.compile(
    r"Today's picks \(double down\):\s*\n"
    r".+?\n"
    r"(?P<name>.+?)(?:\s*\((?P<team>[A-Za-z]{2,4})\))?(?:\s*vs\s+\S.*?)?\s*[—-]\s*\d.*?%\s*\n"
    r"P\(both\):",
    re.DOTALL,
)
_SKIP_RE = re.compile(
    r"Sitting today out\.\s*Top pick:\s*(?P<name>.+?)\s*\((?P<team>[A-Za-z]{2,4})\)",
)
# Matches both "Streak: 5" (active pick) and "Streak holds at 3" (skip post).
_STREAK_RE = re.compile(
    r"Streak(?:\s+holds\s+at)?[:\s]+(?P<streak>\d+)",
    re.IGNORECASE,
)


def parse_pick_from_post(text: str) -> Optional[ParsedPost]:
    """Parse a pick post's text into structured fields.

    Returns None if the post doesn't look like a BTS pick. Returns a
    ParsedPost with is_skip=True if it's a skip announcement.
    """
    # Skip post: "Sitting today out. Top pick: <name> (<team>) ..."
    skip_match = _SKIP_RE.search(text)
    if skip_match:
        parsed = ParsedPost(
            uri="", created_at="", text=text, is_reply=False,
            is_skip=True,
            batter_name=skip_match.group("name").strip(),
            team=skip_match.group("team").strip(),
        )
        streak_match = _STREAK_RE.search(text)
        if streak_match:
            parsed.streak_at_time = int(streak_match.group("streak"))
        return parsed

    # Double-down post: header on first line, two batter lines, P(both), Streak.
    if _DOUBLE_HEADER_RE.search(text):
        first = _DOUBLE_FIRST_RE.search(text)
        second = _DOUBLE_SECOND_RE.search(text)
        if first is None or second is None:
            return None
        parsed = ParsedPost(
            uri="", created_at="", text=text, is_reply=False,
            batter_name=first.group("name").strip(),
            team=first.group("team").strip(),
            is_double_down=True,
            double_down_batter=second.group("name").strip(),
            double_down_team=(second.group("team") or "").strip() or None,
        )
        streak_match = _STREAK_RE.search(text)
        if streak_match:
            parsed.streak_at_time = int(streak_match.group("streak"))
        return parsed

    # Single-pick post.
    match = _PICK_RE.search(text)
    if not match:
        return None

    parsed = ParsedPost(
        uri="", created_at="", text=text, is_reply=False,
        batter_name=match.group("name").strip(),
        team=match.group("team").strip(),
    )

    streak_match = _STREAK_RE.search(text)
    if streak_match:
        parsed.streak_at_time = int(streak_match.group("streak"))

    return parsed


# Regex for result replies. Based on real format_result_reply() output:
#   hit:  "\u2705 Streak: N"
#   miss: "\u274c Streak reset to 0"
# Use the emoji prefix to detect and the rest to extract details.
_HIT_EMOJI = "\u2705"   # white check mark
_MISS_EMOJI = "\u274c"  # cross mark
_STREAK_NUM_RE = re.compile(r"Streak[:\s]+(\d+)", re.IGNORECASE)
_STREAK_RESET_RE = re.compile(r"reset to (\d+)", re.IGNORECASE)


def parse_result_from_reply(text: str) -> ParsedPost:
    """Parse a Bluesky reply post as a BTS result announcement.

    Returns a ParsedPost with is_result=False if the text doesn't look
    like a BTS result reply.
    """
    is_hit = _HIT_EMOJI in text
    is_miss = _MISS_EMOJI in text
    if not (is_hit or is_miss):
        return ParsedPost(uri="", created_at="", text=text, is_reply=True, is_result=False)

    result = "hit" if is_hit and not is_miss else "miss"

    # Extract streak number — try both "Streak: N" and "reset to N" variants
    streak_after = None
    m = _STREAK_NUM_RE.search(text)
    if m:
        streak_after = int(m.group(1))
    else:
        m = _STREAK_RESET_RE.search(text)
        if m:
            streak_after = int(m.group(1))

    return ParsedPost(
        uri="", created_at="", text=text, is_reply=True,
        is_result=True,
        result=result,
        streak_after=streak_after,
    )


@dataclass
class HistoricalPickRecord:
    """A pick + its result as reconstructed from Bluesky."""
    date: str
    batter_name: str
    team: str
    is_double_down: bool
    double_down_batter: Optional[str]
    double_down_team: Optional[str]
    bluesky_uri: str
    result: Optional[str]
    streak_after: Optional[int]


@dataclass
class Timeline:
    """Full reconstructed timeline from Bluesky."""
    pick_records: list[HistoricalPickRecord] = field(default_factory=list)
    final_streak: int = 0
    saver_available_at_end: bool = True


def _date_from_created_at(created_at: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp."""
    return created_at[:10]


def reconstruct_pick_timeline(posts: list[ParsedPost]) -> Timeline:
    """Walk posts chronologically to reconstruct pick history + streak.

    Posts must be sorted oldest-first. Pairs pick posts with their result
    replies based on date proximity (result for day D is the first reply
    seen after the pick for day D with an is_result=True payload).
    """
    # Parse reply text where we haven't already
    for p in posts:
        if p.is_reply and not p.is_result:
            parsed = parse_result_from_reply(p.text)
            p.is_result = parsed.is_result
            p.result = parsed.result
            p.streak_after = parsed.streak_after

    records_by_date: dict[str, HistoricalPickRecord] = {}
    timeline_order: list[str] = []

    for post in posts:
        date = _date_from_created_at(post.created_at)
        if not post.is_reply:
            # Pick post
            if post.is_skip or post.batter_name is None:
                continue
            if date not in records_by_date:
                records_by_date[date] = HistoricalPickRecord(
                    date=date,
                    batter_name=post.batter_name,
                    team=post.team or "",
                    is_double_down=post.is_double_down,
                    double_down_batter=post.double_down_batter,
                    double_down_team=post.double_down_team,
                    bluesky_uri=post.uri,
                    result=None,
                    streak_after=None,
                )
                timeline_order.append(date)
        elif post.is_result:
            # Reply post carrying a result; attribute to most recent unresolved record
            for date_key in reversed(timeline_order):
                record = records_by_date[date_key]
                if record.result is None:
                    record.result = post.result
                    record.streak_after = post.streak_after
                    break

    pick_records = [records_by_date[d] for d in timeline_order]

    # Last known resolved streak is the final_streak
    final_streak = 0
    for r in reversed(pick_records):
        if r.streak_after is not None:
            final_streak = r.streak_after
            break

    return Timeline(
        pick_records=pick_records,
        final_streak=final_streak,
        saver_available_at_end=True,  # Conservative default; Task 5 refines
    )

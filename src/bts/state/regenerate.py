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

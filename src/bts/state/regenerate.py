"""Reconstruct BTS state from Bluesky post history + MLB API.

Used for disaster recovery: if the Fly machine loses its volume entirely,
or during migration between providers, this command rebuilds the full
state from authoritative external sources. Pre-cutoff state comes from
the committed initial-state.json snapshot.

The heart of this module is a post parser that must handle the human-readable
format produced by src/bts/posting.py's format_post() function. If that
format changes, this parser must be updated in lockstep.
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


# Regex patterns for post format (see src/bts/posting.py format_post)
_PICK_RE = re.compile(
    r"Today's BTS pick:\s*(?P<name>[^(]+?)\s*\((?P<team>[A-Z]{2,3})\)",
    re.MULTILINE,
)
_DOUBLE_RE = re.compile(
    r"Double down:\s*(?P<name>[^(]+?)\s*\((?P<team>[A-Z]{2,3})\)",
    re.MULTILINE,
)
_SKIP_RE = re.compile(r"SKIP", re.IGNORECASE)
_STREAK_RE = re.compile(r"Streak[:\s]+(?P<streak>\d+)", re.IGNORECASE)


def parse_pick_from_post(text: str) -> Optional[ParsedPost]:
    """Parse a pick post's text into structured fields.

    Returns None if the post doesn't look like a BTS pick. Returns a
    ParsedPost with is_skip=True if it's a skip announcement.
    """
    if _SKIP_RE.search(text) and "Today's BTS pick" in text:
        return ParsedPost(
            uri="", created_at="", text=text, is_reply=False,
            is_skip=True,
        )

    match = _PICK_RE.search(text)
    if not match:
        return None

    parsed = ParsedPost(
        uri="", created_at="", text=text, is_reply=False,
        batter_name=match.group("name").strip(),
        team=match.group("team").strip(),
    )

    double_match = _DOUBLE_RE.search(text)
    if double_match:
        parsed.is_double_down = True
        parsed.double_down_batter = double_match.group("name").strip()
        parsed.double_down_team = double_match.group("team").strip()

    streak_match = _STREAK_RE.search(text)
    if streak_match:
        parsed.streak_at_time = int(streak_match.group("streak"))

    return parsed

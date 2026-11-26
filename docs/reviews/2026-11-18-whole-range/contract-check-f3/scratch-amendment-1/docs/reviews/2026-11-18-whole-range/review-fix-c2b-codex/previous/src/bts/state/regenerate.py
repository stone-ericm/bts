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
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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

    # Saver: available until first miss at streak 10-15 (per MDP rules).
    # Walk through pick_records chronologically. On a miss, determine the
    # streak just before that miss; if it was in the saver phase [10, 15],
    # the saver was consumed.
    saver_available = True
    for idx, r in enumerate(pick_records):
        if r.result != "miss":
            continue
        # Determine streak just before this miss by looking at the previous record's streak_after
        streak_before = 0
        for prior in reversed(pick_records[:idx]):
            if prior.streak_after is not None:
                streak_before = prior.streak_after
                break
        # Saver consumed if streak_before in [10, 15] — MDP saver phase
        if 10 <= streak_before <= 15:
            saver_available = False
            break

    return Timeline(
        pick_records=pick_records,
        final_streak=final_streak,
        saver_available_at_end=saver_available,
    )


def compose_state_from_snapshot_and_timeline(
    snapshot_path: Path,
    timeline: Timeline,
    out_picks_dir: Path,
) -> None:
    """Write pick files + streak.json from a snapshot + Bluesky timeline.

    Pre-cutoff records come from the committed initial snapshot; post-cutoff
    records come from the Bluesky timeline. Writes pick files to
    out_picks_dir/{date}.json in the format the scheduler expects.
    """
    out_picks_dir.mkdir(parents=True, exist_ok=True)

    snapshot = json.loads(snapshot_path.read_text())

    # Write historical picks from snapshot
    for hist in snapshot.get("historical_picks", []):
        pick_file = _hist_to_pick_file(hist)
        out_path = out_picks_dir / f"{hist['date']}.json"
        out_path.write_text(json.dumps(pick_file, indent=2))

    # Write picks from Bluesky timeline (post-cutoff)
    cutoff = snapshot.get("cutoff_date", "0000-00-00")
    for record in timeline.pick_records:
        if record.date <= cutoff:
            continue  # Snapshot already wrote this
        pick_file = _record_to_pick_file(record)
        out_path = out_picks_dir / f"{record.date}.json"
        out_path.write_text(json.dumps(pick_file, indent=2))

    # Streak file: prefer the timeline's final_streak, fall back to snapshot
    streak_data = {
        "streak": (timeline.final_streak
                   if timeline.pick_records
                   else snapshot.get("streak_at_cutoff", 0)),
        "saver_available": (timeline.saver_available_at_end
                            if timeline.pick_records
                            else snapshot.get("saver_available", True)),
    }
    (out_picks_dir / "streak.json").write_text(json.dumps(streak_data, indent=2))


def _hist_to_pick_file(hist: dict) -> dict:
    """Convert a snapshot historical record back to a pick file shape.

    Backfills any Pick fields that may be missing from older snapshots
    so load_pick() can construct a valid DailyPick.
    """
    pick = hist["pick"]
    if pick is not None:
        pick = _backfill_pick_fields(pick, hist["date"])
    dd = hist.get("double_down")
    if dd is not None:
        dd = _backfill_pick_fields(dd, hist["date"])
    return {
        "date": hist["date"],
        "run_time": hist.get("run_time", f"{hist['date']}T12:00:00+00:00"),
        "pick": pick,
        "double_down": dd,
        "runner_up": None,
        "bluesky_posted": hist.get("bluesky_posted", True),
        "bluesky_uri": hist.get("bluesky_uri"),
        "result": hist.get("result"),
    }


def _backfill_pick_fields(pick: dict, date: str) -> dict:
    """Ensure all Pick dataclass fields are present with sensible defaults."""
    pick.setdefault("lineup_position", 0)
    pick.setdefault("flags", [])
    pick.setdefault("projected_lineup", False)
    pick.setdefault("pitcher_team", None)
    # Ensure batter_id is an int for Pick(**data) — older snapshots may have None
    if pick.get("batter_id") is None:
        pick["batter_id"] = 0
    if pick.get("pitcher_name") is None:
        pick["pitcher_name"] = "Unknown"
    if pick.get("p_game_hit") is None:
        pick["p_game_hit"] = 0.0
    if pick.get("game_pk") is None:
        pick["game_pk"] = 0
    if pick.get("game_time") is None:
        pick["game_time"] = f"{date}T19:00:00+00:00"
    return pick


def _record_to_pick_file(record: HistoricalPickRecord) -> dict:
    """Convert a regenerated HistoricalPickRecord to a pick file shape.

    Note: some fields (batter_id, pitcher info, p_game_hit) cannot be
    recovered from Bluesky alone and are left as None/0/[]. A follow-up
    pass could use the MLB API to backfill batter_id from name+team+date.

    All Pick dataclass fields are included so load_pick() can construct
    a valid DailyPick without crashing.
    """
    return {
        "date": record.date,
        "run_time": f"{record.date}T12:00:00+00:00",
        "pick": {
            "batter_name": record.batter_name,
            "batter_id": 0,
            "team": record.team,
            "lineup_position": 0,
            "pitcher_name": "Unknown",
            "pitcher_id": None,
            "p_game_hit": 0.0,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 0,
            "game_time": f"{record.date}T19:00:00+00:00",
            "pitcher_team": None,
        },
        "double_down": {
            "batter_name": record.double_down_batter,
            "batter_id": 0,
            "team": record.double_down_team or "",
            "lineup_position": 0,
            "pitcher_name": "Unknown",
            "pitcher_id": None,
            "p_game_hit": 0.0,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 0,
            "game_time": f"{record.date}T19:00:00+00:00",
            "pitcher_team": None,
        } if record.is_double_down else None,
        "runner_up": None,
        "bluesky_posted": True,
        "bluesky_uri": record.bluesky_uri,
        "result": record.result,
    }


def regenerate(
    snapshot_path: Path,
    bluesky_handle: str,
    out_picks_dir: Path,
) -> dict:
    """Full regeneration: fetch Bluesky, compose with snapshot, write pick files.

    Returns a summary dict with counts of regenerated picks and the
    final streak.
    """
    snapshot = json.loads(snapshot_path.read_text())
    cutoff = snapshot.get("cutoff_date", "0000-00-00")

    posts = fetch_bluesky_posts(handle=bluesky_handle, from_date=cutoff)

    # Parse pick posts (non-reply) through parse_pick_from_post
    parsed_posts: list[ParsedPost] = []
    for p in posts:
        if p.is_reply:
            parsed_posts.append(p)
            continue
        pick_parse = parse_pick_from_post(p.text)
        if pick_parse is None:
            continue
        pick_parse.uri = p.uri
        pick_parse.created_at = p.created_at
        pick_parse.is_reply = False
        parsed_posts.append(pick_parse)

    timeline = reconstruct_pick_timeline(parsed_posts)
    compose_state_from_snapshot_and_timeline(
        snapshot_path=snapshot_path,
        timeline=timeline,
        out_picks_dir=out_picks_dir,
    )

    return {
        "snapshot_cutoff": cutoff,
        "snapshot_picks": len(snapshot.get("historical_picks", [])),
        "bluesky_picks": len(timeline.pick_records),
        "final_streak": timeline.final_streak,
        "saver_available": timeline.saver_available_at_end,
    }

"""Bluesky posting for BTS picks."""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def format_post(
    batter: str,
    team: str,
    pitcher: str,
    p_game: float,
    streak: int,
    double: str | None = None,
    double_p_game: float | None = None,
) -> str:
    """Format Bluesky post text for a BTS pick."""
    if double and double_p_game is not None:
        p_both = p_game * double_p_game
        return (
            f"Today's picks: {batter} ({team}) + {double}\n"
            f"vs {pitcher} | P(both): {p_both:.1%}\n\n"
            f"Streak: {streak}"
        )
    return (
        f"Today's pick: {batter} ({team})\n"
        f"vs {pitcher} | {p_game:.1%}\n\n"
        f"Streak: {streak}"
    )


def get_bluesky_password() -> str:
    """Get Bluesky app password from macOS Keychain.

    Raises RuntimeError if the password is not found.
    """
    result = subprocess.run(
        ["security", "find-generic-password", "-a", "claude-cli",
         "-s", "bluesky-bts-app-password", "-w"],
        capture_output=True, text=True,
    )
    password = result.stdout.strip()
    if result.returncode != 0 or not password:
        raise RuntimeError("Bluesky app password not found in keychain")
    return password


def post_to_bluesky(text: str) -> str:
    """Post text to Bluesky. Returns post URI.

    Raises RuntimeError if auth or posting fails.
    """
    password = get_bluesky_password()

    # Authenticate
    auth_data = json.dumps({
        "identifier": "beatthestreakbot.bsky.social",
        "password": password,
    }).encode()
    req = Request("https://bsky.social/xrpc/com.atproto.server.createSession",
                  data=auth_data, headers={"Content-Type": "application/json"})
    session = json.loads(urlopen(req, timeout=15).read())

    # Post
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "repo": session["did"],
        "collection": "app.bsky.feed.post",
        "record": {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": now,
        },
    }
    req = Request(
        "https://bsky.social/xrpc/com.atproto.repo.createRecord",
        data=json.dumps(record).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {session['accessJwt']}"},
    )
    resp = json.loads(urlopen(req, timeout=15).read())
    return resp["uri"]


def _now_et() -> datetime:
    """Return current time in ET. Extracted for testability."""
    return datetime.now(ET)


def should_post_now(game_time_utc: str, already_posted: bool) -> bool:
    """Decide if we should post the pick to Bluesky now.

    Posts if:
    - Not already posted for today
    - Game starts within 3 hours, OR
    - It's the evening run (after 7pm ET)
    """
    if already_posted:
        return False

    now_et = _now_et()
    game_dt = datetime.fromisoformat(game_time_utc).astimezone(ET)

    # Post if game starts within 3 hours (future only)
    time_to_game = game_dt - now_et
    if timedelta(0) <= time_to_game <= timedelta(hours=3):
        return True

    # Post on the final run (after 7pm ET)
    if now_et.hour >= 19:
        return True

    return False

"""Bluesky posting for BTS picks."""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import Request
from zoneinfo import ZoneInfo

from bts.util import retry_urlopen

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
    """Get Bluesky app password from macOS Keychain or environment variable.

    Tries macOS keychain first, then falls back to BTS_BLUESKY_PASSWORD env var.
    Raises RuntimeError if no password is found.
    """
    # Try macOS keychain first
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "claude-cli",
             "-s", "bluesky-bts-app-password", "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass  # Not on macOS

    # Fallback to environment variable
    password = os.environ.get("BTS_BLUESKY_PASSWORD")
    if password:
        return password

    raise RuntimeError(
        "Bluesky app password not found. Set BTS_BLUESKY_PASSWORD or add to macOS keychain."
    )


def post_to_bluesky(text: str) -> str:
    """Post text to Bluesky. Returns post URI.

    Raises RuntimeError if auth or posting fails with a meaningful message.
    """
    password = get_bluesky_password()

    # Authenticate
    auth_data = json.dumps({
        "identifier": "beatthestreakbot.bsky.social",
        "password": password,
    }).encode()
    req = Request("https://bsky.social/xrpc/com.atproto.server.createSession",
                  data=auth_data, headers={"Content-Type": "application/json"})
    try:
        session = json.loads(retry_urlopen(req, timeout=15).read())
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError("Bluesky auth failed — check app password in keychain") from e
        body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"Bluesky auth error (HTTP {e.code}): {body}") from e

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
    try:
        resp = json.loads(retry_urlopen(req, timeout=15).read())
    except HTTPError as e:
        if e.code == 429:
            raise RuntimeError("Bluesky rate limited — try again in a few minutes") from e
        body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"Bluesky post error (HTTP {e.code}): {body}") from e

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

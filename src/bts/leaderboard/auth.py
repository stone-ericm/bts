"""Auth flow for the BTS leaderboard scraper.

Cookies are stored in platform-native keychain:
  - macOS: `security add-generic-password -a claude-cli -s mlb-bts-session-cookies`
  - Linux: `pass insert -m mlb-bts-session-cookies`
The capture is interactive (scripts/capture_bts_cookies.py) and not in scope here.

Per-scrape auth flow:
  1. load_session_cookies() — read JSON cookie list from keychain, return name->value dict
  2. extract_uid(cookies) — pull the Okta-issued uid from the `oktaid` cookie
  3. fetch_xsid(uid, cookies) — POST /api/auth/login {uid, platform: "web"} -> xSid token
  4. Scraper uses cookies + xSid query param on all data calls

xSid expires within hours; mint a fresh one at the start of each scrape run.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys

import httpx

from bts.leaderboard.endpoints import (
    AUTH_LOGIN_URL,
    AUTH_LOGIN_PLATFORM,
    OKTAID_COOKIE_NAME,
    USER_AGENT,
)

log = logging.getLogger(__name__)

KEYCHAIN_ACCOUNT = "claude-cli"
KEYCHAIN_SERVICE = "mlb-bts-session-cookies"


class AuthError(Exception):
    """Raised when session cookies are missing, expired, or rejected."""


def _read_keychain_raw() -> str:
    """Read raw cookie blob from platform keychain. Raises AuthError on failure.

    Mac:   `security find-generic-password ... -w` (may emit hex; we decode)
    Linux: prefer `pass`; fall back to file at $BTS_LEADERBOARD_COOKIE_FILE
           or $HOME/.bts-leaderboard-cookies.json.
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["security", "find-generic-password",
                 "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w"],
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise AuthError(f"could not read cookie store {KEYCHAIN_SERVICE!r}: {e}") from e
        raw = out.decode().strip()
        # `security -w` may emit hex-encoded output when bytes look "binary"
        if re.fullmatch(r"[0-9a-fA-F]+", raw) and len(raw) % 2 == 0:
            try:
                raw = bytes.fromhex(raw).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
        return raw

    if sys.platform.startswith("linux"):
        # Try pass first
        try:
            out = subprocess.check_output(
                ["pass", "show", KEYCHAIN_SERVICE],
                stderr=subprocess.PIPE,
            )
            return out.decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.info(f"`pass` unavailable ({type(e).__name__}); falling back to cookie file")
        # Fall back to file
        import os
        cookie_path = (
            os.environ.get("BTS_LEADERBOARD_COOKIE_FILE")
            or os.path.expanduser("~/.bts-leaderboard-cookies.json")
        )
        if not os.path.exists(cookie_path):
            raise AuthError(
                f"cookie file not found at {cookie_path}. "
                f"Either install `pass` and store at {KEYCHAIN_SERVICE!r}, "
                "or write JSON cookies to that path (chmod 600)."
            )
        with open(cookie_path) as f:
            return f.read().strip()

    raise AuthError(f"unknown platform: {sys.platform}")


def load_session_cookies() -> dict[str, str]:
    """Return name -> value cookie dict from platform keychain. Raises AuthError on failure."""
    raw = _read_keychain_raw()
    try:
        cookies_list = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AuthError(f"keychain payload not valid JSON: {e}") from e
    return {c["name"]: c["value"] for c in cookies_list if "name" in c and "value" in c}


def extract_uid(cookies: dict[str, str]) -> str:
    """Pull the Okta-issued uid from the `oktaid` cookie. Raises AuthError if absent."""
    uid = cookies.get(OKTAID_COOKIE_NAME)
    if not uid:
        raise AuthError(f"missing {OKTAID_COOKIE_NAME!r} cookie; can't authenticate")
    return uid


def fetch_xsid(uid: str, cookies: dict[str, str], timeout: float = 30.0) -> str:
    """POST /api/auth/login -> mint a fresh xSid. Raises AuthError on any failure."""
    response = httpx.post(
        AUTH_LOGIN_URL,
        cookies=cookies,
        json={"uid": uid, "platform": AUTH_LOGIN_PLATFORM},
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise AuthError(f"auth/login returned {response.status_code}: {response.text[:200]}")
    try:
        body = response.json()
    except json.JSONDecodeError as e:
        raise AuthError(f"auth/login response not JSON: {e}") from e
    xsid = body.get("success", {}).get("xSid")
    if not xsid:
        errs = body.get("errors", [])
        raise AuthError(f"xSid missing from auth/login response (errors={errs})")
    return xsid


def is_session_valid(cookies: dict[str, str]) -> bool:
    """Quick check: cookies + uid + auth/login round-trip succeeds."""
    if OKTAID_COOKIE_NAME not in cookies:
        return False
    try:
        uid = extract_uid(cookies)
        fetch_xsid(uid, cookies)
        return True
    except (AuthError, httpx.HTTPError) as e:
        log.warning(f"session probe failed: {e}")
        return False

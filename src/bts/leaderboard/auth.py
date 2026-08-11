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
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

from bts.leaderboard.endpoints import (
    AUTH_LOGIN_URL,
    AUTH_LOGIN_PLATFORM,
    OKTAID_COOKIE_NAME,
    browser_headers,
)

log = logging.getLogger(__name__)

KEYCHAIN_ACCOUNT = "claude-cli"
KEYCHAIN_SERVICE = "mlb-bts-session-cookies"


class AuthError(Exception):
    """Raised when session cookies are missing, expired, or rejected."""


class TransientAuthError(AuthError):
    """auth/login failed in an outage/garbage shape — NOT credential evidence:
    a cookie re-capture is not indicated unless this persists across runs.
    Retryable shapes (5xx, transport errors, EMPTY 200 bodies) retry in-process
    first; rejection-lookalike shapes (non-empty non-JSON 200s, malformed JSON)
    raise this immediately WITHOUT retry (kill-switch: a challenge page must
    not be hammered)."""


class RateLimitedLoginError(AuthError):
    """auth/login returned 429 — the rate-limit kill-switch shape at login
    time. Never retried. Do NOT re-run, do NOT increase cadence, and do NOT
    re-capture cookies (the interactive capture flow adds MORE auth traffic);
    stop and investigate request volume if this persists."""


@dataclass(frozen=True)
class AuthSession:
    xsid: str
    user_id: int | None
    username: str | None


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


def _retry_after_seconds(response, cap: float = 30.0) -> float | None:
    """Server-directed backoff for 5xx: numeric Retry-After, capped so a huge
    header can't stall a cron run. None (= caller's default backoff) when the
    header is absent, HTTP-date-formed, or unparseable."""
    try:
        ra = response.headers.get("Retry-After")
        if ra is None:
            return None
        val = float(str(ra).strip())
        if not math.isfinite(val) or val < 0:
            return None      # hostile/garbage value must never reach time.sleep
        return min(val, cap)
    except (ValueError, TypeError, AttributeError):
        return None


def fetch_login_session(
    uid: str,
    cookies: dict[str, str],
    timeout: float = 30.0,
    attempts: int = 3,
    retry_delay: float = 2.0,
) -> AuthSession:
    """POST /api/auth/login -> mint a fresh xSid and return account identity.

    Response classification (kill-switch philosophy: retry ONLY shapes that
    cannot be a rejection of *us*; never hammer anything rejection-shaped):

      retried in-process (doubling 2s/4s delay; 5xx honors Retry-After <=30s):
        - transport errors
        - 5xx
        - HTTP 200 with an EMPTY body (the observed 2026-08-11 outage shape,
          flapping ~1-in-3 that day)
      TransientAuthError immediately (outage/garbage shaped, but NOT retried —
      a bot-defense challenge page can look exactly like this):
        - HTTP 200 with a non-empty non-JSON body
        - HTTP 200 whose JSON is not an object
        - 1xx / 2xx other than 200
      RateLimitedLoginError immediately: 429
      AuthError immediately (credential-shaped; cookie advice applies):
        - other 4xx, 3xx (redirect-to-login), JSON object without an xSid
    """
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"attempts must be an int >= 1, got {attempts!r}")
    if not (isinstance(retry_delay, (int, float)) and math.isfinite(retry_delay)
            and retry_delay >= 0):
        raise ValueError(f"retry_delay must be a finite number >= 0, got {retry_delay!r}")
    last_transient = None
    last_exc = None
    delay_override = None
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            delay = (delay_override if delay_override is not None
                     else retry_delay * (2 ** (attempt - 2)))
            log.warning(f"auth/login transient failure ({last_transient}); "
                        f"retrying in {delay:.0f}s (attempt {attempt}/{attempts})")
            time.sleep(delay)
        delay_override = None
        try:
            response = httpx.post(
                AUTH_LOGIN_URL,
                cookies=cookies,
                json={"uid": uid, "platform": AUTH_LOGIN_PLATFORM},
                headers={**browser_headers(), "Content-Type": "application/json"},
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            last_transient = f"transport error: {type(e).__name__}: {e}"
            last_exc = e
            continue
        status = response.status_code
        if status >= 500:
            last_transient = f"returned {status}: {response.text[:200]}"
            last_exc = None
            delay_override = _retry_after_seconds(response)
            continue
        if status == 429:
            raise RateLimitedLoginError(
                f"auth/login returned 429 — rate-limited at login; back off: do NOT "
                f"retry or re-capture cookies: {response.text[:200]}")
        if 300 <= status < 500:
            raise AuthError(f"auth/login returned {status}: {response.text[:200]}")
        if status != 200:
            raise TransientAuthError(
                f"auth/login returned unexpected status {status} (not retrying; "
                f"not a cookie problem): {response.text[:200]}")
        try:
            body = response.json()
        except ValueError as e:  # JSONDecodeError / UnicodeDecodeError
            try:
                text = response.text
            except Exception:
                text = None
            if text == "":       # ONLY a zero-length body licenses a retry (r2 #1);
                last_transient = f"response not JSON ({e}); body empty"
                last_exc = e     # whitespace is already a non-empty response.
                continue
            snippet = text[:200] if text is not None else "<undecodable bytes>"
            raise TransientAuthError(
                f"auth/login 200 with a non-empty non-JSON body — NOT retrying "
                f"(could be a challenge/denial page; kill-switch): "
                f"body[:200]={snippet!r}") from e
        if not isinstance(body, dict):
            raise TransientAuthError(
                f"auth/login 200 with non-object JSON body {body!r:.200} — not "
                f"retrying (malformed upstream response, not a cookie problem)")
        success = body.get("success")
        if not isinstance(success, dict):
            success = {}
        xsid = success.get("xSid")
        if not xsid:
            errs = body.get("errors", [])
            detail = (f"errors={errs}" if errs
                      else f"errors empty; body[:200]={response.text[:200]!r}")
            raise AuthError(f"xSid missing from auth/login response ({detail})")
        user = success.get("user")
        if not isinstance(user, dict):     # r2 #2: a non-dict user payload must
            user = {}                      # not AttributeError out of the CLIs
        return AuthSession(
            xsid=xsid,
            user_id=user.get("id"),
            username=user.get("username"),
        )
    raise TransientAuthError(
        f"auth/login transient failure persisted after {attempts} attempts "
        f"(MLB-side flap/outage, not a cookie problem): {last_transient}") from last_exc


def fetch_xsid(uid: str, cookies: dict[str, str], timeout: float = 30.0) -> str:
    """POST /api/auth/login -> mint a fresh xSid. Raises AuthError on any failure."""
    return fetch_login_session(uid, cookies, timeout=timeout).xsid


def is_session_valid(cookies: dict[str, str]) -> bool:
    """Quick check: cookies + uid + auth/login round-trip succeeds.

    NOTE: returns False for transient outages too (TransientAuthError is an
    AuthError) — "couldn't determine" is conflated with "invalid". No current
    production caller; revisit the contract before adding one."""
    if OKTAID_COOKIE_NAME not in cookies:
        return False
    try:
        uid = extract_uid(cookies)
        fetch_xsid(uid, cookies)
        return True
    except (AuthError, httpx.HTTPError) as e:
        log.warning(f"session probe failed: {e}")
        return False

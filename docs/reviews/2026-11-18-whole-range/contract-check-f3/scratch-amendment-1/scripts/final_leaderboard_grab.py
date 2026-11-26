#!/usr/bin/env python3
"""Bounded end-of-season leaderboard grab (season wrap W0.6) — ONE authenticated pass.

Owner-authorized footprint (Eric, 2026-09-14/22): a single pass on the real contest
account after the regular season ends, capped by a hard request budget, with the
existing 403/429 kill-switch (any rate-limit anywhere aborts everything; no retry;
no second attempt without a fresh owner decision). Design + Codex reviews:
`.codex-review/season-wrap/w06-grab-design*.md` (2 design rounds, then code rounds)
and the plan §W0.6.

What it does, in order (network only after step 1 has written plan.json + status.json):
  0. Validate the configuration BEFORE touching credentials or the filesystem
     (profiles ≤ 300 total, board limit ≤ 300, ≥1 page, explicit final-round date).
  1. Preflight: load cookies ONCE (branch + sha256 of the exact bytes recorded, never
     the bytes), copy the frozen early-cohort manifest into the run root, create the
     run root EXCLUSIVELY under data/leaderboard/, write plan.json + status.json.
  2. Login (ONE attempt) through a recording transport; the archived login body has
     the session token redacted; the token lives only in memory and is scrubbed from
     every serialized surface.
  3. Four static JSON lookups (paced, archived, shape-validated).
  4. Full-depth walk of the SEASON-BEST board (limit 300) with validated terminal
     conditions; NO page ceiling by default (owner 2026-09-22: "no request ceiling …
     carefully and courteously try to grab it all") — the walk ends when the board
     does; every page body archived; dedupe by userId; walk exhaustion and population
     coverage reported SEPARATELY; any truncation exposed.
  5. Three continuity tabs (active_streak, all_time top-100; `yesterday` bound to the
     explicit FINAL round of the season).
  6. Profiles for cohort A (top of the final board) + B (frozen early cohort not in A,
     fetched BY ID), shuffled, paced, archived, id-keyed outputs, fail-closed envelopes;
     identity.json rewritten after every profile so an abort preserves results.
  7. status.json rewritten atomically after every request intent and outcome; a
     failed terminal write is exit 4 and is reported outside the failed file.

Never touches the daily corpus (leaderboard_snapshots/, user_picks/, season_stats/,
scrape_status.json): everything lives under data/leaderboard/final_grab_<YYYYMMDD>/.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bts.leaderboard import auth as bts_auth  # noqa: E402
from bts.leaderboard.endpoints import (  # noqa: E402
    LEADERBOARD_ROUND_URL_TEMPLATE,
    LEADERBOARD_URL_TEMPLATE,
    RANKS_TYPE_BY_TAB,
    ROUNDS_URL,
    USER_PROFILE_URL_TEMPLATE,
    browser_headers,
)
from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats  # noqa: E402
from bts.leaderboard.ratelimit import next_gap  # noqa: E402
from bts.leaderboard.scraper import (  # noqa: E402
    DEFAULT_JITTER_S,
    DEFAULT_MIN_INTERVAL_S,
    LeaderboardEnvelopeError,
    ProfileEnvelopeError,
    StaticLookups,
    parse_leaderboard_response,
    parse_rounds_lookup,
    parse_user_profile_response,
    validate_profile_envelope,
)
from bts.leaderboard.storage import (  # noqa: E402
    append_user_picks,
    write_leaderboard_snapshot,
    write_season_stats,
)

SCHEMA_VERSION = "bts_final_leaderboard_grab_v2"
PARSER_SCHEMA = {
    "leaderboard_row": sorted(LeaderboardRow.model_fields),
    "pick_row": sorted(PickRow.model_fields),
    "season_stats": sorted(SeasonStats.model_fields),
    "streak_semantics": "streak = tab ranking field; season_best_streak/active_streak explicit (C-01)",
}
STATIC_JSON_BASE = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json"
RATE_LIMIT_STATUSES = frozenset({403, 429})
MAX_PROFILES_TOTAL = 300
MAX_BOARD_LIMIT = 300
KEYCHAIN_SERVICE = bts_auth.KEYCHAIN_SERVICE
DEFAULT_COOKIE_FILE = Path(os.path.expanduser("~/.bts-leaderboard-cookies.json"))

# An uncapped walk still terminates on: end of board, ANY previously seen page, or this
# many consecutive full pages adding no new users (reporting-only, not a request ceiling).
NO_PROGRESS_PAGES = 3

EXIT_COMPLETE = 0
EXIT_PARTIAL = 2
EXIT_ABORTED_RATE_LIMITED = 3
EXIT_ABORTED_WRITE_FAILURE = 4
EXIT_ABORTED_OTHER = 5
EXIT_CANCELLED = 130  # SIGINT convention: operator Ctrl-C; persisted state says so too

TERMINAL_OK_STATES = ("complete", "complete_with_errors", "complete_with_population_gap")


# ----------------------------------------------------------------------------- errors
class GrabAbort(Exception):
    def __init__(self, terminal_state: str, exit_code: int, message: str):
        super().__init__(message)
        self.terminal_state = terminal_state
        self.exit_code = exit_code


class WriteFailure(GrabAbort):
    def __init__(self, message: str):
        super().__init__("aborted_write_failure", EXIT_ABORTED_WRITE_FAILURE, message)


class RateLimitedAbort(GrabAbort):
    def __init__(self, message: str):
        super().__init__("aborted_rate_limited", EXIT_ABORTED_RATE_LIMITED, message)


class BudgetAbort(GrabAbort):
    def __init__(self, message: str):
        super().__init__("aborted_budget", EXIT_ABORTED_OTHER, message)


class BodyReadError(Exception):
    """The response STATUS was observed but reading the body failed. The status must
    survive so a 403/429 with a truncated body still trips the kill-switch."""

    def __init__(self, status: int, cause: Exception):
        super().__init__(f"body read failed after HTTP {status}: {type(cause).__name__}: {cause}")
        self.status = int(status)
        self.cause = cause


# ----------------------------------------------------------------------------- transport
class HttpxTransport:
    """Live transport. Status is observed BEFORE the body is consumed (streaming), no
    redirects are followed, timeouts apply to every request."""

    def get(self, url: str, *, cookies: dict[str, str], timeout: float = 30.0) -> tuple[int, bytes]:
        with httpx.Client(cookies=cookies, timeout=timeout, headers=browser_headers(), follow_redirects=False) as client:
            with client.stream("GET", url) as r:
                status = r.status_code
                try:
                    return status, r.read()
                except httpx.HTTPError as exc:
                    raise BodyReadError(status, exc) from exc

    def post(self, url, *, cookies, json, headers, timeout):  # httpx.post keyword signature
        with httpx.Client(cookies=cookies, timeout=timeout, headers=headers, follow_redirects=False) as client:
            with client.stream("POST", url, json=json) as r:
                status = r.status_code
                try:
                    r.read()  # decodes any Content-Encoding exactly once
                except httpx.HTTPError as exc:
                    raise BodyReadError(status, exc) from exc
                return r  # already-read response; never rebuild (a rebuild re-applies Content-Encoding)


# ----------------------------------------------------------------------------- credentials
def _pass_show(entry: str) -> bytes | None:
    try:
        return subprocess.check_output(["pass", "show", entry], stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _keychain_show(entry: str) -> bytes | None:
    try:
        out = subprocess.check_output(["security", "find-generic-password", "-a", "claude-cli", "-s", entry, "-w"],
                                      stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    text = out.decode().strip()
    if text and all(c in "0123456789abcdefABCDEF" for c in text) and len(text) % 2 == 0:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    return text.encode()


def load_cookies_with_provenance(*, cookie_file: Path | None = None, platform: str | None = None,
                                 entry: str = KEYCHAIN_SERVICE) -> tuple[dict[str, str], dict[str, Any]]:
    """Read the credential blob EXACTLY ONCE, report which branch fired, and hash the
    bytes that were parsed. Never returns or logs the bytes themselves."""
    platform = platform or sys.platform
    raw: bytes | None = None
    source: str | None = None
    location: str | None = None
    if platform == "darwin":
        raw = _keychain_show(entry)
        source, location = "keychain", f"security:{entry}"
    elif platform.startswith("linux"):
        raw = _pass_show(entry)
        if raw is not None:
            source, location = "pass", f"pass:{entry}"
        else:
            path = Path(os.environ.get("BTS_LEADERBOARD_COOKIE_FILE") or cookie_file or DEFAULT_COOKIE_FILE)
            if not path.exists():
                raise bts_auth.AuthError(f"cookie file not found at {path} and `pass` unavailable")
            raw = path.read_bytes()
            source, location = "file", str(path)
    if raw is None:
        raise bts_auth.AuthError(f"no credential source on platform {platform!r}")
    try:
        cookies_list = json.loads(raw.decode().strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise bts_auth.AuthError(f"credential payload from {source} is not a JSON cookie list: {exc}") from exc
    cookies = {c["name"]: c["value"] for c in cookies_list if isinstance(c, dict) and "name" in c and "value" in c}
    return cookies, {"source": source, "location": location, "sha256": hashlib.sha256(raw).hexdigest(),
                     "bytes": len(raw), "n_cookies": len(cookies)}


# ----------------------------------------------------------------------------- config
@dataclass
class GrabConfig:
    date: str
    season: int
    leaderboard_dir: Path
    run_root: Path
    early_cohort_path: Path
    board_limit: int = 300
    # None = NO CEILING (owner, 2026-09-22: "no request ceiling … carefully and courteously
    # try to grab it all"): the walk runs until the board ends (short page / nextPage=false),
    # a repeated page, or an error. Pacing and the 403/429 kill-switch are unchanged.
    max_board_pages: int | None = None
    cohort_a: int = 150
    cohort_b: int = 150
    transport: Any = field(default_factory=HttpxTransport)
    cookies_loader: Callable[[], tuple[dict[str, str], dict[str, Any]]] = load_cookies_with_provenance
    sleeper: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random())
    rng_seed: int | None = None
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    code_sha: str | None = None
    request_budget: int | None = None
    final_round_date: date | None = None  # REQUIRED (no date-minus-one default)
    include_tabs: bool = True
    min_gap_s: float = DEFAULT_MIN_INTERVAL_S
    jitter_s: float = DEFAULT_JITTER_S
    timeout_s: float = 30.0
    uv_lock_path: Path | None = None

    def planned_max_requests(self) -> int | None:
        """None when the board walk is uncapped (then only profiles/tabs/static/login are bounded)."""
        if self.max_board_pages is None:
            return None
        return 1 + 4 + self.max_board_pages + (3 if self.include_tabs else 0) + self.cohort_a + self.cohort_b

    def planned_non_board_requests(self) -> int:
        return 1 + 4 + (3 if self.include_tabs else 0) + self.cohort_a + self.cohort_b

    def budget(self) -> int | None:
        """Total request cap; None = uncapped (the board class is the only unbounded one)."""
        return self.request_budget if self.request_budget is not None else self.planned_max_requests()

    def validation_problems(self) -> list[str]:
        problems: list[str] = []
        try:
            date.fromisoformat(self.date)
        except ValueError:
            problems.append(f"date {self.date!r} is not YYYY-MM-DD")
        if not isinstance(self.cohort_a, int) or not isinstance(self.cohort_b, int) or self.cohort_a < 0 or self.cohort_b < 0:
            problems.append("cohort sizes must be non-negative integers")
        elif self.cohort_a + self.cohort_b > MAX_PROFILES_TOTAL:
            problems.append(f"cohort_a + cohort_b = {self.cohort_a + self.cohort_b} exceeds the {MAX_PROFILES_TOTAL}-profile cap")
        if not isinstance(self.board_limit, int) or not (1 <= self.board_limit <= MAX_BOARD_LIMIT):
            problems.append(f"board_limit must be 1..{MAX_BOARD_LIMIT}")
        if self.max_board_pages is not None and (not isinstance(self.max_board_pages, int) or self.max_board_pages < 1):
            problems.append("max_board_pages must be None (no ceiling) or >= 1")
        if self.request_budget is not None and (not isinstance(self.request_budget, int) or self.request_budget < 1):
            problems.append("request_budget must be >= 1")
        if self.final_round_date is None:
            problems.append("final_round_date is required (no date-minus-one default)")
        if self.min_gap_s < 0 or self.jitter_s < 0 or self.timeout_s <= 0:
            problems.append("pacing/timeout values must be positive")
        return problems


# ----------------------------------------------------------------------------- io
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n").encode())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _redact_url(url: str) -> str:
    if "xSid=" in url:
        head, _, tail = url.partition("xSid=")
        rest = tail.split("&", 1)
        return head + "xSid=<redacted>" + ("&" + rest[1] if len(rest) > 1 else "")
    return url


class Scrubber:
    """Replaces every known secret (session token, cookie values, uid) in any string
    about to be serialized. Secrets are registered as they become known."""

    def __init__(self) -> None:
        self.secrets: list[str] = []

    def add(self, *values: str | None) -> None:
        for v in values:
            if v and len(v) >= 6 and v not in self.secrets:
                self.secrets.append(v)

    def text(self, s: str) -> str:
        for v in self.secrets:
            if v in s:
                s = s.replace(v, f"<redacted:sha256={_sha256(v.encode())[:16]}>")
        return s

    def bytes(self, data: bytes) -> bytes:
        """Byte-safe replacement of every known secret in utf-8, latin-1 and JSON-escaped forms."""
        for v in self.secrets:
            marker = f"<redacted:sha256={_sha256(v.encode())[:16]}>".encode()
            forms = {v.encode("utf-8"), json.dumps(v)[1:-1].encode(), "".join(f"\\u{ord(c):04x}" for c in v).encode()}
            try:
                forms.add(v.encode("latin-1"))
            except UnicodeEncodeError:
                pass
            for form in forms:
                if form and form in data:
                    data = data.replace(form, marker)
        return data

    def obj(self, o: Any) -> Any:
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, dict):
            # keys are scrubbed too: a credential used as an object key must not survive
            return {(self.text(k) if isinstance(k, str) else k): self.obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self.obj(v) for v in o]
        return o


def redact_login_body(body: bytes, scrub: Scrubber) -> bytes:
    """Archive-safe login response: the xSid value is replaced by a hash marker."""
    try:
        obj = json.loads(body)
    except ValueError:
        # Not JSON (e.g. a 403 "forbidden" page): keep the evidence verbatim, scrubbed of
        # any known secret; only undecodable bytes are replaced by a hashed placeholder.
        try:
            return scrub.text(body.decode()).encode()
        except UnicodeDecodeError:
            return f"<undecodable login body; sha256={_sha256(body)}; bytes={len(body)}>".encode()
    success = obj.get("success") if isinstance(obj, dict) else None
    if isinstance(success, dict) and isinstance(success.get("xSid"), str):
        token = success["xSid"]
        scrub.add(token)
        success["xSid"] = f"<redacted:sha256={_sha256(token.encode())[:16]}>"
    return json.dumps(scrub.obj(obj), sort_keys=True).encode()


# ----------------------------------------------------------------------------- ledger
class Ledger:
    """Request accounting: intent persisted BEFORE send, outcome after; budget enforced;
    raw body archived before parsing; rate-limit status classified even when the body
    read fails; any write failure stops sending."""

    def __init__(self, cfg: GrabConfig, status: dict[str, Any], scrub: Scrubber):
        self.cfg = cfg
        self.status = status
        self.scrub = scrub
        self.issued = 0
        self.status_path = cfg.run_root / "status.json"

    def flush(self) -> None:
        try:
            _atomic_write_json(self.status_path, self.scrub.obj(self.status))
        except OSError as exc:
            raise WriteFailure(f"could not write status.json: {exc}") from exc

    def _pace(self) -> None:
        if self.issued > 0:
            self.cfg.sleeper(next_gap(self.cfg.min_gap_s, self.cfg.jitter_s, self.cfg.rng))

    def scrub_bytes(self, body: bytes) -> bytes:
        """Default archive transform: remove every known secret from a body.
        JSON bodies are parsed and scrubbed as decoded strings (so a token written as
        \\uXXXX escapes is caught); other bodies are scrubbed byte-wise in every
        encoding we know how to produce, decodable or not."""
        try:
            obj = json.loads(body)
        except ValueError:
            obj = None
        if isinstance(obj, (dict, list)):
            scrubbed = self.scrub.obj(obj)
            if scrubbed != obj:
                # re-serialize the scrubbed structure, then ALSO byte-scrub the output so no
                # serialization path can bypass credential removal
                return self.scrub.bytes(json.dumps(scrubbed, ensure_ascii=False, sort_keys=True).encode())
        return self.scrub.bytes(body)

    def request(self, cls: str, name: str, url: str, raw_path: Path,
                send: Callable[[], tuple[int, bytes]],
                archive_transform: Callable[[bytes], bytes] | None = None) -> tuple[int, bytes, dict[str, Any]]:
        budget = self.cfg.budget()
        if budget is not None and self.issued + 1 > budget:
            raise BudgetAbort(f"request budget {budget} reached before {cls}:{name}")
        self._pace()
        entry: dict[str, Any] = {
            "seq": self.issued + 1, "class": cls, "name": name, "url": _redact_url(url),
            "outcome": "intent", "sent_at_utc": self.cfg.now().isoformat(),
            "received_at_utc": None, "duration_s": None, "http_status": None,
            "bytes": None, "sha256": None, "archived_sha256": None,
            "raw_path": str(raw_path.relative_to(self.cfg.run_root)), "raw_redacted": None,
        }
        self.status["requests"].append(entry)
        self.issued += 1
        self.flush()
        t0 = time.monotonic()
        try:
            http_status, body = send()
        except BodyReadError as exc:
            entry.update({"outcome": "error", "http_status": exc.status, "received_at_utc": self.cfg.now().isoformat(),
                          "duration_s": round(time.monotonic() - t0, 3), "body_read_error": self.scrub.text(str(exc.cause))})
            if exc.status in RATE_LIMIT_STATUSES:
                entry["outcome"] = "aborted"
                self.flush()
                raise RateLimitedAbort(f"HTTP {exc.status} on {cls}:{name} (body unreadable) — kill-switch") from exc
            self.flush()
            return exc.status, b"", entry
        except Exception as exc:  # transport-level failure with no status: recorded, no retry
            entry.update({"outcome": "error", "received_at_utc": self.cfg.now().isoformat(),
                          "duration_s": round(time.monotonic() - t0, 3), "error": self.scrub.text(f"{type(exc).__name__}: {exc}")})
            self.flush()
            return 0, b"", entry
        body = bytes(body or b"")
        entry.update({"received_at_utc": self.cfg.now().isoformat(), "duration_s": round(time.monotonic() - t0, 3),
                      "http_status": int(http_status), "bytes": len(body), "sha256": _sha256(body)})
        archived = (archive_transform or self.scrub_bytes)(body)
        entry["archived_sha256"] = _sha256(archived)
        entry["raw_redacted"] = archived != body
        try:
            _atomic_write_bytes(raw_path, gzip.compress(archived))
        except OSError as exc:
            entry["outcome"] = "error"
            entry["error"] = f"raw archive write failed: {exc}"
            raise WriteFailure(f"could not archive raw body for {cls}:{name}: {exc}") from exc
        if int(http_status) in RATE_LIMIT_STATUSES:
            entry["outcome"] = "aborted"
            self.flush()
            raise RateLimitedAbort(f"HTTP {http_status} on {cls}:{name} — kill-switch")
        entry["outcome"] = "success" if int(http_status) == 200 else "error"
        self.flush()
        return int(http_status), body, entry


# ----------------------------------------------------------------------------- cohort
def allocate_cohort(board_rows: list[dict[str, Any]], early_users: list[dict[str, Any]],
                    *, cohort_a: int, cohort_b: int) -> dict[str, Any]:
    """A = first `cohort_a` distinct user_ids by final all_season rank (rank, then id).
    B = first `cohort_b` of the frozen early manifest (its own order) NOT in A — fetched by
    id regardless of final-board presence. E_in_A / E_unfetched / B_shortfall recorded.
    Never backfills B from A's tail (keeps the cohort definition outcome-independent)."""
    seen: set[int] = set()
    ordered: list[int] = []
    for r in sorted(board_rows, key=lambda r: (int(r["rank"]), int(r["user_id"]))):
        uid = int(r["user_id"])
        if uid not in seen:
            seen.add(uid)
            ordered.append(uid)
    A = ordered[:cohort_a]
    a_set = set(A)
    early_ids: list[int] = []
    for u in early_users:
        uid = int(u["user_id"])
        if uid not in early_ids:
            early_ids.append(uid)
    B = [uid for uid in early_ids if uid not in a_set][:cohort_b]
    b_set = set(B)
    return {
        "A": A,
        "B": B,
        "E_in_A": [uid for uid in early_ids if uid in a_set],
        "E_unfetched": [uid for uid in early_ids if uid not in a_set and uid not in b_set],
        "B_shortfall": max(0, cohort_b - len(B)),
        "rule": ("A = first cohort_a distinct user_ids by final all_season (rank, user_id); "
                 "B = first cohort_b of the frozen early manifest order not in A, fetched by id; "
                 "no backfill of B from A's tail; E_unfetched = budget omission (not an abort, not missing history)"),
    }


# ----------------------------------------------------------------------------- pieces
def _load_early(cfg: GrabConfig) -> tuple[list[dict[str, Any]], str]:
    data = json.loads(cfg.early_cohort_path.read_text())
    users = data.get("users")
    if not isinstance(users, list):
        raise GrabAbort("refused_early_cohort_invalid", EXIT_ABORTED_OTHER, "early cohort manifest has no users list")
    if not users and cfg.cohort_b > 0:
        raise GrabAbort("refused_early_cohort_invalid", EXIT_ABORTED_OTHER,
                        f"early cohort manifest is empty but cohort_b={cfg.cohort_b} profiles were requested")
    for u in users:
        if not isinstance(u, dict) or u.get("user_id") is None:
            raise GrabAbort("refused_early_cohort_invalid", EXIT_ABORTED_OTHER, "early cohort manifest row without user_id")
    return users, _file_sha256(cfg.early_cohort_path)


def _valid_timestamp(value: Any) -> str | None:
    """Server timestamp must be a non-empty ISO-8601 string; anything else is invalid metadata."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _validate_page_meta(success: dict[str, Any]) -> dict[str, Any]:
    nxt = success.get("nextPage")
    next_page = nxt if isinstance(nxt, bool) else None
    cnt = success.get("allParticipantsCount")
    participants = int(cnt) if isinstance(cnt, int) and not isinstance(cnt, bool) and cnt >= 0 else None
    raw_ts = success.get("updatedAt")
    return {"next_page": next_page, "all_participants_count": participants,
            "updated_at": _valid_timestamp(raw_ts), "updated_at_raw": None if _valid_timestamp(raw_ts) else repr(raw_ts)[:40]}


def _validate_static(name: str, body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or body.get("errors"):
        raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"static {name} is an error/invalid object")
    key = "rounds" if name == "rounds" else name
    items = body.get(key)
    if not isinstance(items, list):
        raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"static {name} lacks a {key!r} list")
    if name == "rounds":
        if not items:
            raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, "rounds.json has an empty rounds list")
        for r in items:
            if not isinstance(r, dict) or r.get("id") is None or not r.get("date"):
                raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, "rounds.json entry lacks id/date")
    else:
        for it in items:
            if not isinstance(it, dict) or it.get("id") is None:
                raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"{name}.json entry lacks id")
    return body


def _round_for_date(rounds: dict[int, date], target: date) -> int | None:
    hits = [rid for rid, d in rounds.items() if d == target]
    return min(hits) if hits else None


def _population(board: dict[str, Any], participants_seen: list[int]) -> dict[str, Any]:
    """Population coverage, kept separate from walk exhaustion. `census` demands: an
    exhausted, conflict-free walk AND valid participant count + server timestamp on
    EVERY page, all equal. Anything less is an explicit unknown, never a census."""
    listed = board["unique_user_ids"]
    pages = board["per_page"]
    if not participants_seen:
        return {"status": "unknown_no_participant_count", "reported": None, "listed_unique": listed, "gap": None}
    if min(participants_seen) != max(participants_seen):
        return {"status": "unknown_participant_count_drift", "reported": {"min": min(participants_seen), "max": max(participants_seen)},
                "listed_unique": listed, "gap": None}
    reported = participants_seen[0]
    if not board["walk_exhausted"]:
        return {"status": "walk_not_exhausted", "reported": reported, "listed_unique": listed, "gap": reported - listed}
    if board["duplicate_conflicts"]:
        return {"status": "unknown_duplicate_conflicts", "reported": reported, "listed_unique": listed, "gap": None,
                "conflicts": len(board["duplicate_conflicts"])}
    if any(pg["all_participants_count"] is None or pg["updated_at"] is None for pg in pages):
        return {"status": "unknown_incomplete_participant_metadata", "reported": reported, "listed_unique": listed, "gap": None,
                "pages_missing_metadata": [pg["page"] for pg in pages if pg["all_participants_count"] is None or pg["updated_at"] is None]}
    if len({pg["updated_at"] for pg in pages}) > 1:
        return {"status": "unknown_server_version_drift", "reported": reported, "listed_unique": listed, "gap": None,
                "updated_at_values": sorted({str(pg["updated_at"]) for pg in pages})}
    if listed == reported:
        return {"status": "census", "reported": reported, "listed_unique": listed, "gap": 0}
    if listed < reported:
        return {"status": "gap_vs_reported_participants", "reported": reported, "listed_unique": listed, "gap": reported - listed}
    return {"status": "listed_exceeds_reported", "reported": reported, "listed_unique": listed, "gap": reported - listed}


# ----------------------------------------------------------------------------- main
def run_grab(cfg: GrabConfig) -> tuple[int, dict[str, Any]]:
    started = cfg.now()
    scrub = Scrubber()
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "date": cfg.date, "season": cfg.season,
        "started_at_utc": started.isoformat(), "finished_at_utc": None,
        "terminal_state": "running", "exit_code": None, "problems": [],
        "requests": [], "planned_unattempted": [], "requests_accounted": None,
        "board": None, "tabs": {}, "cohort": None, "profiles": [], "artifacts": {},
    }

    # ---- 0. configuration (before credentials or filesystem) ----
    problems = cfg.validation_problems()
    if problems:
        status.update(terminal_state="refused_config", exit_code=EXIT_ABORTED_OTHER, problems=problems)
        return EXIT_ABORTED_OTHER, status

    # ---- refusals (zero requests, zero writes into the run root) ----
    run_root = cfg.run_root
    if run_root.is_symlink():
        status.update(terminal_state="refused_run_root_symlink", exit_code=EXIT_ABORTED_OTHER, problems=[f"{run_root} is a symlink"])
        return EXIT_ABORTED_OTHER, status
    if run_root.exists():
        status.update(terminal_state="refused_run_root_exists", exit_code=EXIT_ABORTED_OTHER,
                      problems=[f"{run_root} already exists; a second attempt needs a fresh owner decision"])
        return EXIT_ABORTED_OTHER, status
    lb = cfg.leaderboard_dir.resolve()
    if not run_root.resolve().is_relative_to(lb) or run_root.resolve() == lb:
        status.update(terminal_state="refused_run_root_outside", exit_code=EXIT_ABORTED_OTHER,
                      problems=[f"{run_root} must be a fresh child of {lb}"])
        return EXIT_ABORTED_OTHER, status
    for protected in ("leaderboard_snapshots", "user_picks", "season_stats"):
        if run_root.resolve().is_relative_to((lb / protected).resolve()):
            status.update(terminal_state="refused_run_root_outside", exit_code=EXIT_ABORTED_OTHER,
                          problems=[f"{run_root} is inside the daily corpus dir {protected}"])
            return EXIT_ABORTED_OTHER, status

    ledger: Ledger | None = None
    exit_code = EXIT_ABORTED_OTHER
    planned_profiles: list[int] = []
    planned_tabs: list[str] = ["active_streak", "all_time", "yesterday"] if cfg.include_tabs else []
    identity: dict[str, Any] = {}
    try:
        # ---- 1. preflight (no network) ----
        early_users, early_sha = _load_early(cfg)
        cookies, provenance = cfg.cookies_loader()
        scrub.add(*cookies.values())
        uid = bts_auth.extract_uid(cookies)
        scrub.add(uid)
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            for sub in ("raw/login", "raw/static", "raw/board", "raw/tabs", "raw/profiles",
                        "leaderboard_snapshots", "user_picks", "season_stats", "inputs"):
                (run_root / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cfg.early_cohort_path, run_root / "inputs" / "early_cohort.json")
        except OSError as exc:
            raise WriteFailure(f"could not create run root: {exc}") from exc
        uv_lock = cfg.uv_lock_path or (REPO_ROOT / "uv.lock")
        plan = {
            "schema_version": SCHEMA_VERSION, "planned_at_utc": started.isoformat(), "date": cfg.date,
            "season": cfg.season, "code_sha": cfg.code_sha, "parser_schema": PARSER_SCHEMA,
            "uv_lock_sha256": _file_sha256(uv_lock) if uv_lock.exists() else None,
            "credential_provenance": {k: v for k, v in provenance.items() if k in ("source", "location", "sha256", "bytes", "n_cookies")},
            "request_budget": cfg.budget(), "planned_max_requests": cfg.planned_max_requests(),
            "planned_non_board_requests": cfg.planned_non_board_requests(),
            "board_ceiling": cfg.max_board_pages,
            "owner_ceiling_decision": {
                "owner_quotes_verbatim": [
                    "no request ceiling. when it comes time, let's carefully and curtiously try to grab it all",
                    "i mean i want to grab all the picks all the public profiles (i assume 95000+)",
                ],
                "interpretation_for_this_run": ("board walk runs to exhaustion (no page ceiling); pacing + kill-switch unchanged; "
                                                "this one-pass run fetches the D5 profile allocation only — the full-field profile "
                                                "campaign is a separate, resumable job (W0.6b)")
                                               if cfg.max_board_pages is None else f"capped at {cfg.max_board_pages} pages",
            },
            "classes": {"login": 1, "static": 4, "board_pages_max": cfg.max_board_pages,
                        "tabs": len(planned_tabs), "profiles_max": cfg.cohort_a + cfg.cohort_b},
            "board_limit": cfg.board_limit, "max_board_pages": cfg.max_board_pages,
            "pacing": {"min_gap_s": cfg.min_gap_s, "jitter_s": cfg.jitter_s, "timeout_s": cfg.timeout_s},
            "kill_switch": "any HTTP 403/429 anywhere (status observed before body) aborts the whole operation; no retry; no second attempt",
            "cohort_rule": {"A": cfg.cohort_a, "B": cfg.cohort_b, "cap": MAX_PROFILES_TOTAL,
                            "text": "A = top of final all_season board; B = frozen early cohort not in A, by id"},
            "early_cohort_path": str(cfg.early_cohort_path), "early_cohort_sha256": early_sha,
            "early_cohort_copy": "inputs/early_cohort.json", "early_cohort_n": len(early_users),
            "final_round_date": cfg.final_round_date.isoformat() if cfg.final_round_date else None,
            "rng_seed": cfg.rng_seed, "run_root": str(run_root),
            "isolation": "never writes to the daily corpus (leaderboard_snapshots/, user_picks/, season_stats/, scrape_status.json)",
        }
        try:
            _atomic_write_json(run_root / "plan.json", scrub.obj(plan))
        except OSError as exc:
            raise WriteFailure(f"could not write plan.json: {exc}") from exc
        ledger = Ledger(cfg, status, scrub)
        ledger.flush()

        # ---- 2. login (one attempt) through the recording transport; token redacted in the archive ----
        def recording_post(url, *, cookies, json, headers, timeout):
            holder: dict[str, Any] = {}

            def send() -> tuple[int, bytes]:
                resp = cfg.transport.post(url, cookies=cookies, json=json, headers=headers, timeout=timeout)
                holder["resp"] = resp
                return resp.status_code, resp.content

            http_status, body, entry = ledger.request("login", "auth_login", url, run_root / "raw/login/001_login.json.gz",
                                                      send, archive_transform=lambda b: redact_login_body(b, scrub))
            if "resp" not in holder:
                raise httpx.TransportError(entry.get("error") or entry.get("body_read_error") or "transport error")
            return holder["resp"]

        try:
            session = bts_auth.fetch_login_session(uid, cookies, timeout=cfg.timeout_s, attempts=1, post=recording_post)
        except bts_auth.RateLimitedLoginError as exc:
            raise RateLimitedAbort(f"login rate-limited: {scrub.text(str(exc))}") from exc
        except bts_auth.TransientAuthError as exc:
            raise GrabAbort("aborted_login_transient", EXIT_ABORTED_OTHER, f"login transient failure: {scrub.text(str(exc))}") from exc
        except bts_auth.AuthError as exc:
            # 403 / 3xx / other 4xx are rejection-shaped: same global abort as a rate limit.
            raise RateLimitedAbort(f"login rejected: {scrub.text(str(exc))}") from exc
        xsid = session.xsid
        scrub.add(xsid)

        def get(url: str) -> tuple[int, bytes]:
            return cfg.transport.get(url, cookies=cookies, timeout=cfg.timeout_s)

        # ---- 3. static lookups (shape-validated) ----
        statics: dict[str, Any] = {}
        for i, (name, url) in enumerate([("rounds", ROUNDS_URL), ("players", f"{STATIC_JSON_BASE}/players.json"),
                                          ("units", f"{STATIC_JSON_BASE}/units.json"), ("squads", f"{STATIC_JSON_BASE}/squads.json")], start=1):
            http_status, body, _ = ledger.request("static", name, url, run_root / f"raw/static/{i:03d}_{name}.json.gz", lambda u=url: get(u))
            if http_status != 200:
                raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"static {name} returned {http_status}")
            try:
                statics[name] = _validate_static(name, json.loads(body))
            except ValueError as exc:
                raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"static {name} not JSON: {exc}") from exc
        lookups = StaticLookups(
            rounds=parse_rounds_lookup(statics["rounds"]),
            players={int(p["id"]): p for p in statics["players"]["players"]},
            units={int(u["id"]): u for u in statics["units"]["units"]},
            squads={int(s["id"]): s for s in statics["squads"]["squads"]},
        )

        # ---- 4. full-depth season-best walk ----
        board: dict[str, Any] = {
            "tab": "all_season", "limit": cfg.board_limit, "ceiling": cfg.max_board_pages,
            "pages": 0, "raw_rows": 0, "unique_user_ids": 0, "pages_without_new_users": 0, "warnings": [],
            "duplicate_rows": 0, "duplicate_conflicts": [], "per_page": [], "termination_reason": None,
            "walk_exhausted": False, "walk_complete": False, "population_complete": False, "population": None,
            "last_rank_reached": None, "all_participants_count": None, "updated_at_values": [],
        }
        status["board"] = board
        rows_by_id: dict[int, LeaderboardRow] = {}
        seen_page_signatures: set[str] = set()  # ANY previously seen page (not just the last) ends the walk
        zero_progress_streak = 0
        participants_seen: list[int] = []
        page = 0
        while True:
            page += 1
            if cfg.max_board_pages is not None and page > cfg.max_board_pages:
                break
            url = LEADERBOARD_URL_TEMPLATE.format(season=cfg.season, page=page, limit=cfg.board_limit,
                                                  ranks_type=RANKS_TYPE_BY_TAB["all_season"], xsid=xsid)
            http_status, body, entry = ledger.request("board", f"page_{page:03d}", url,
                                                      run_root / f"raw/board/page_{page:03d}.json.gz", lambda u=url: get(u))
            board["pages"] = page
            if http_status != 200:
                board["termination_reason"] = "error"
                status["problems"].append(f"board page {page}: HTTP {http_status}")
                break
            try:
                parsed_body = json.loads(body)
                rows = parse_leaderboard_response(parsed_body, tab="all_season", captured_at=cfg.now().replace(tzinfo=None))
            except (ValueError, LeaderboardEnvelopeError) as exc:
                board["termination_reason"] = "error"
                status["problems"].append(f"board page {page}: {type(exc).__name__}: {str(exc)[:160]}")
                break
            success = parsed_body["success"]
            meta = _validate_page_meta(success)
            raw_count = len(success.get("ranks") or [])
            if meta["all_participants_count"] is not None:
                participants_seen.append(meta["all_participants_count"])
            if meta["updated_at"] is not None and meta["updated_at"] not in board["updated_at_values"]:
                board["updated_at_values"].append(meta["updated_at"])
            page_ids = {r.user_id for r in rows}
            new_rows = 0
            for r in rows:
                assert r.user_id is not None
                if r.user_id in rows_by_id:
                    board["duplicate_rows"] += 1
                    prev = rows_by_id[r.user_id]
                    if (prev.rank, prev.streak) != (r.rank, r.streak):
                        board["duplicate_conflicts"].append({"user_id": r.user_id, "first": [prev.rank, prev.streak],
                                                             "later": [r.rank, r.streak], "page": page})
                    continue
                rows_by_id[r.user_id] = r
                new_rows += 1
            board["raw_rows"] += raw_count
            ranks = [r.rank for r in rows]
            board["per_page"].append({"page": page, "raw_rows": raw_count, "new_unique": new_rows,
                                      "rank_min": min(ranks) if ranks else None, "rank_max": max(ranks) if ranks else None,
                                      "next_page": meta["next_page"], "all_participants_count": meta["all_participants_count"],
                                      "updated_at": meta["updated_at"], "updated_at_invalid": meta["updated_at_raw"],
                                      "received_at_utc": entry["received_at_utc"]})
            if ranks:
                board["last_rank_reached"] = max(board["last_rank_reached"] or 0, max(ranks))
            # live totals (visible in status.json after every request, not only at loop exit)
            board["unique_user_ids"] = len(rows_by_id)
            if participants_seen:
                board["all_participants_count"] = {"min": min(participants_seen), "max": max(participants_seen)}
                if len(rows_by_id) > 1.5 * max(participants_seen) and "listed_exceeds_1.5x_reported" not in board["warnings"]:
                    board["warnings"].append("listed_exceeds_1.5x_reported")
            # cycle detection: a page whose userId set was seen on ANY earlier page (A/B/A/B loops)
            signature = _sha256(",".join(str(u) for u in sorted(page_ids)).encode()) if page_ids else f"empty-{page}"
            if signature in seen_page_signatures:
                board["termination_reason"] = "repeated_page"
                break
            seen_page_signatures.add(signature)
            # a valid terminal page is honored FIRST (even after zero-progress pages)
            if raw_count < cfg.board_limit:
                if meta["next_page"] is False:
                    board["termination_reason"] = "short_page"
                    board["walk_exhausted"] = True
                elif meta["next_page"] is True:
                    board["termination_reason"] = "short_page_contradicted_by_next_page"
                else:
                    board["termination_reason"] = "short_page_pagination_unknown"
                break
            if meta["next_page"] is False:
                board["termination_reason"] = "next_page_false"
                board["walk_exhausted"] = True
                break
            # sustained zero progress: FULL CONTINUING pages that add no new users are not progress
            if new_rows == 0:
                zero_progress_streak += 1
                board["pages_without_new_users"] += 1
            else:
                zero_progress_streak = 0
            if zero_progress_streak >= NO_PROGRESS_PAGES:
                board["termination_reason"] = "no_progress"
                break
            if cfg.max_board_pages is not None and page == cfg.max_board_pages:
                board["termination_reason"] = "ceiling"
                break
        board["unique_user_ids"] = len(rows_by_id)
        if participants_seen:
            board["all_participants_count"] = {"min": min(participants_seen), "max": max(participants_seen)}
        board["walk_complete"] = bool(board["walk_exhausted"] and not board["duplicate_conflicts"])
        board["population"] = _population(board, participants_seen)
        board["population_complete"] = board["population"]["status"] == "census"
        if not board["walk_complete"]:
            status["problems"].append(f"board walk incomplete: {board['termination_reason']}"
                                      + (f"; {len(board['duplicate_conflicts'])} duplicate conflicts" if board["duplicate_conflicts"] else ""))
        board_rows = sorted(rows_by_id.values(), key=lambda r: (r.rank, r.user_id))
        snap_path = run_root / "leaderboard_snapshots" / f"{cfg.date}_all_season_full.parquet"
        try:
            write_leaderboard_snapshot(snap_path, board_rows)
        except OSError as exc:
            raise WriteFailure(f"could not write board parquet: {exc}") from exc
        status["artifacts"]["leaderboard_snapshot"] = {"path": str(snap_path.relative_to(run_root)), "sha256": _file_sha256(snap_path), "rows": len(board_rows)}
        ledger.flush()

        # ---- 5. continuity tabs ----
        tab_rows: list[LeaderboardRow] = []
        tabs_done: list[str] = []
        if cfg.include_tabs:
            final_round = _round_for_date(lookups.rounds, cfg.final_round_date)  # type: ignore[arg-type]
            for i, tab in enumerate(planned_tabs, start=1):
                if tab == "yesterday":
                    if final_round is None:
                        status["tabs"]["yesterday"] = {"status": "skipped_no_final_round",
                                                       "note": f"no round dated {cfg.final_round_date.isoformat()} in the archived rounds.json"}  # type: ignore[union-attr]
                        status["problems"].append("tab yesterday skipped: final round not found")
                        continue
                    url = LEADERBOARD_ROUND_URL_TEMPLATE.format(round_id=final_round, page=1, limit=100,
                                                                ranks_type=RANKS_TYPE_BY_TAB[tab], xsid=xsid)
                else:
                    url = LEADERBOARD_URL_TEMPLATE.format(season=cfg.season, page=1, limit=100,
                                                          ranks_type=RANKS_TYPE_BY_TAB[tab], xsid=xsid)
                http_status, body, _ = ledger.request("tab", tab, url, run_root / f"raw/tabs/{i:03d}_{tab}.json.gz", lambda u=url: get(u))
                tabs_done.append(tab)
                info: dict[str, Any] = {"http_status": http_status, "status": "success" if http_status == 200 else "http_error"}
                if tab == "yesterday":
                    info["round_id"] = final_round
                if http_status == 200:
                    try:
                        rows = parse_leaderboard_response(json.loads(body), tab=tab, captured_at=cfg.now().replace(tzinfo=None))  # type: ignore[arg-type]
                        tab_rows.extend(rows)
                        info["rows"] = len(rows)
                    except (ValueError, LeaderboardEnvelopeError) as exc:
                        info.update(status="envelope_error", error=f"{type(exc).__name__}: {str(exc)[:160]}")
                        status["problems"].append(f"tab {tab}: {info['error']}")
                else:
                    status["problems"].append(f"tab {tab}: HTTP {http_status}")
                status["tabs"][tab] = info
            if tab_rows:
                tabs_path = run_root / "leaderboard_snapshots" / f"{cfg.date}_tabs_top100.parquet"
                try:
                    write_leaderboard_snapshot(tabs_path, tab_rows)
                except OSError as exc:
                    raise WriteFailure(f"could not write tabs parquet: {exc}") from exc
                status["artifacts"]["tabs_snapshot"] = {"path": str(tabs_path.relative_to(run_root)), "sha256": _file_sha256(tabs_path), "rows": len(tab_rows)}
            ledger.flush()

        # ---- 6. cohort + profiles ----
        cohort = allocate_cohort([{"user_id": r.user_id, "rank": r.rank} for r in board_rows], early_users,
                                 cohort_a=cfg.cohort_a, cohort_b=cfg.cohort_b)
        order = list(cohort["A"]) + list(cohort["B"])
        cfg.rng.shuffle(order)
        planned_profiles = list(order)
        cohort["request_order"] = order
        cohort["rng_seed"] = cfg.rng_seed
        cohort["early_cohort_sha256"] = early_sha
        cohort["E_n"] = len({int(u["user_id"]) for u in early_users})
        status["cohort"] = {k: (len(v) if isinstance(v, list) and k in ("A", "B", "E_in_A", "E_unfetched", "request_order") else v)
                            for k, v in cohort.items()}
        try:
            _atomic_write_json(run_root / "cohort.json", cohort)
        except OSError as exc:
            raise WriteFailure(f"could not write cohort.json: {exc}") from exc
        status["artifacts"]["cohort"] = {"path": "cohort.json", "sha256": _file_sha256(run_root / "cohort.json")}
        username_by_id = {r.user_id: r.username for r in board_rows}
        for r in tab_rows:
            username_by_id.setdefault(r.user_id, r.username)
        early_name = {int(u["user_id"]): (u.get("usernames_2026_05_01") or ["unknown"])[0] for u in early_users}
        a_set = set(cohort["A"])
        stats_rows: list[SeasonStats] = []

        def persist_identity() -> None:
            try:
                _atomic_write_json(run_root / "identity.json", scrub.obj(identity))
            except OSError as exc:
                raise WriteFailure(f"could not write identity.json: {exc}") from exc

        for user_id in order:
            url = USER_PROFILE_URL_TEMPLATE.format(user_id=user_id, xsid=xsid)
            raw_path = run_root / f"raw/profiles/{user_id}.json.gz"
            username = username_by_id.get(user_id) or early_name.get(user_id) or "unknown"
            # The identity record exists BEFORE the request goes out, so an abort mid-request
            # (kill-switch, write failure) leaves a visible in_flight -> aborted entry.
            rec: dict[str, Any] = {"user_id": user_id, "cohort": "A" if user_id in a_set else "B", "username": username,
                                   "http_status": None, "status": "in_flight", "n_picks": 0, "first_pick_date": None,
                                   "last_pick_date": None, "n_rounds": 0, "skipped_unknown_round_predictions": 0,
                                   "unresolved_round_predictions": 0, "raw_path": str(raw_path.relative_to(run_root)),
                                   "raw_sha256": None, "parsed_path": None, "parsed_sha256": None}
            identity[str(user_id)] = rec
            status["profiles"].append(rec)
            persist_identity()
            http_status, body, entry = ledger.request("profile", str(user_id), url, raw_path, lambda u=url: get(u))
            rec.update({"http_status": http_status, "raw_sha256": entry["sha256"]})
            if http_status != 200:
                rec["status"] = "http_error"
                persist_identity()
                continue
            try:
                parsed_body = json.loads(body)
                env = validate_profile_envelope(parsed_body)
            except (ValueError, ProfileEnvelopeError) as exc:
                rec["status"] = "envelope_error"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                persist_identity()
                continue
            rec.update({"raw_predictions": env.n_predictions, "raw_round_predictions": env.n_round_predictions,
                        "null_field_counts": env.null_field_counts, "no_history": env.no_history,
                        "unresolved_round_predictions": env.unresolved_round_predictions,
                        "unresolved_reasons": env.unresolved_reasons})
            if env.unresolved_round_predictions:
                # A pick cannot be formed from a slot without unitId/playerId; the parser would
                # coerce them to 0 and fabricate rows. Fail closed: raw archive retains the data.
                rec["status"] = "success_with_unresolved"
                persist_identity()
                continue
            try:
                picks, stats = parse_user_profile_response(parsed_body, captured_at=cfg.now().replace(tzinfo=None),
                                                           user_id_unused=user_id, lookups=lookups, username=username)
            except Exception as exc:  # noqa: BLE001 — parse failure is recorded, never fatal
                rec["status"] = "parse_error"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                persist_identity()
                continue
            rec.update({"n_picks": len(picks), "n_rounds": len({p.round_id for p in picks}),
                        "skipped_unknown_round_predictions": stats.skipped_unknown_round_predictions})
            if picks:
                dates = sorted(p.pick_date for p in picks)
                rec["first_pick_date"], rec["last_pick_date"] = dates[0].isoformat(), dates[-1].isoformat()
                parsed_path = run_root / "user_picks" / f"{user_id}.parquet"
                try:
                    append_user_picks(parsed_path, picks)
                except OSError as exc:
                    raise WriteFailure(f"could not write picks for {user_id}: {exc}") from exc
                rec["parsed_path"] = str(parsed_path.relative_to(run_root))
                rec["parsed_sha256"] = _file_sha256(parsed_path)
                rec["status"] = "success_partial_lookup" if stats.skipped_unknown_round_predictions else "success"
                if stats.skipped_unknown_round_predictions:
                    status["problems"].append(f"profile {user_id}: {stats.skipped_unknown_round_predictions} slot(s) skipped for unknown rounds")
            else:
                rec["status"] = "success_no_history" if env.no_history else "success_partial_lookup"
                if not env.no_history:
                    status["problems"].append(f"profile {user_id}: all {env.n_round_predictions} slot(s) skipped for unknown rounds")
            stats_rows.append(stats)
            persist_identity()
        if stats_rows:
            stats_path = run_root / "season_stats" / f"{cfg.date}.parquet"
            try:
                write_season_stats(stats_path, stats_rows)
            except OSError as exc:
                raise WriteFailure(f"could not write season stats: {exc}") from exc
            status["artifacts"]["season_stats"] = {"path": str(stats_path.relative_to(run_root)), "sha256": _file_sha256(stats_path), "rows": len(stats_rows)}
        persist_identity()
        status["artifacts"]["identity"] = {"path": "identity.json", "sha256": _file_sha256(run_root / "identity.json")}

        # ---- terminal classification ----
        profile_problems = [p for p in status["profiles"] if p["status"] in
                            ("http_error", "envelope_error", "parse_error", "success_with_unresolved", "success_partial_lookup")]
        tab_problems = [t for t, info in status["tabs"].items() if info.get("status") != "success"]
        tabs_missing = [t for t in planned_tabs if t not in status["tabs"]]
        clean_ops = board["walk_complete"] and not profile_problems and not tab_problems and not tabs_missing
        if clean_ops and board["population_complete"]:
            status["terminal_state"], exit_code = "complete", EXIT_COMPLETE
        elif clean_ops:
            status["terminal_state"], exit_code = "complete_with_population_gap", EXIT_PARTIAL
            status["problems"].append(f"population coverage: {board['population']['status']}")
        else:
            status["terminal_state"], exit_code = "complete_with_errors", EXIT_PARTIAL
            if profile_problems:
                status["problems"].append(f"{len(profile_problems)} profile(s) not clean")
    except GrabAbort as abort:
        status["terminal_state"] = abort.terminal_state
        status["problems"].append(str(abort))
        exit_code = abort.exit_code
    except KeyboardInterrupt:
        # Operator cancellation: recorded as such (never left as "running"); the in-flight
        # request becomes "aborted" in the accounting below; process exit 130.
        status["terminal_state"] = "cancelled_by_operator"
        status["problems"].append("cancelled by operator (SIGINT); evidence preserved; no rerun without a fresh owner decision")
        exit_code = EXIT_CANCELLED
    except Exception as exc:  # noqa: BLE001 — anything unexpected is a recorded abort, never a silent success
        status["terminal_state"] = "aborted_other"
        status["problems"].append(f"unexpected {type(exc).__name__}: {exc}")
        exit_code = EXIT_ABORTED_OTHER
    finally:
        # ---- accounting: every request terminal; every planned-but-unsent request listed ----
        for entry in status["requests"]:
            if entry.get("outcome") == "intent":
                entry["outcome"] = "aborted"
        for rec in status["profiles"]:
            if rec.get("status") == "in_flight":
                rec["status"] = "aborted"
        if identity and run_root.exists():
            try:
                _atomic_write_json(run_root / "identity.json", scrub.obj(identity))
            except OSError as exc:
                status["problems"].append(f"identity.json final write failed: {exc}")
        sent_by_class: dict[str, set[str]] = {}
        for e in status["requests"]:
            sent_by_class.setdefault(e["class"], set()).add(e["name"])
        unattempted: list[dict[str, Any]] = []
        if "login" not in sent_by_class:
            unattempted.append({"class": "login", "outcome": "unattempted"})
        for name in ("rounds", "players", "units", "squads"):
            if name not in sent_by_class.get("static", set()):
                unattempted.append({"class": "static", "name": name, "outcome": "unattempted"})
        b = status.get("board")
        if b is None:
            unattempted.append({"class": "board", "outcome": "unattempted", "note": "walk never started"})
        elif not b.get("walk_exhausted"):
            unattempted.append({"class": "board", "outcome": "unattempted", "note": f"walk ended by {b.get('termination_reason')}; remaining pages unknown"})
        for tab in planned_tabs:
            if tab in sent_by_class.get("tab", set()):
                continue
            info = status["tabs"].get(tab)
            unattempted.append({"class": "tab", "tab": tab, "outcome": info["status"] if isinstance(info, dict) and "status" in info else "unattempted"})
        sent_profiles = {int(n) for n in sent_by_class.get("profile", set())}
        if planned_profiles:
            for uid_ in planned_profiles:
                if uid_ not in sent_profiles:
                    unattempted.append({"class": "profile", "user_id": uid_, "outcome": "unattempted"})
        elif status.get("cohort") is None and status["terminal_state"] not in TERMINAL_OK_STATES:
            unattempted.append({"class": "profile", "outcome": "unattempted", "note": "cohort never allocated"})
        status["planned_unattempted"] = unattempted
        status["requests_accounted"] = all(e.get("outcome") in ("success", "error", "aborted") for e in status["requests"])
        status["finished_at_utc"] = cfg.now().isoformat()
        status["exit_code"] = exit_code
        if ledger is not None:
            try:
                ledger.flush()
            except WriteFailure as exc:
                status["terminal_state"] = "aborted_write_failure"
                status["exit_code"] = exit_code = EXIT_ABORTED_WRITE_FAILURE
                status["problems"].append(f"terminal status.json write failed: {exc}")
                marker = run_root.parent / f"{run_root.name}.STATUS_WRITE_FAILED"
                try:
                    marker.write_text(json.dumps(scrub.obj({"run_root": str(run_root), "at_utc": cfg.now().isoformat(),
                                                            "intended_terminal_state": status["terminal_state"],
                                                            "problems": status["problems"]}), default=str))
                except OSError:
                    pass
                print(scrub.text(f"STATUS WRITE FAILED for {run_root}: {exc}"), file=sys.stderr)
    # Every surface that leaves this function (CLI summary, tests, logs) is credential-safe.
    return exit_code, scrub.obj(status)


# ----------------------------------------------------------------------------- cli
def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default=date.today().isoformat(), help="grab date YYYY-MM-DD (run root name)")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--leaderboard-dir", type=Path, default=Path("data/leaderboard"))
    p.add_argument("--early-cohort", type=Path, default=Path("docs/audit/2026-09-22-early-cohort-2026-05-01.json"))
    p.add_argument("--board-limit", type=int, default=300)
    p.add_argument("--max-board-pages", type=int, default=None,
                   help="optional ceiling on board pages; default NONE = walk the whole board (owner decision 2026-09-22)")
    p.add_argument("--cohort-a", type=int, default=150)
    p.add_argument("--cohort-b", type=int, default=150)
    p.add_argument("--final-round-date", required=True, help="YYYY-MM-DD of the season's final round (REQUIRED; e.g. 2026-09-27)")
    p.add_argument("--rng-seed", type=int, default=None)
    p.add_argument("--request-budget", type=int, default=None, help="hard cap; default = planned maximum")
    p.add_argument("--no-tabs", action="store_true")
    p.add_argument("--i-have-owner-authorization", action="store_true",
                   help="required: this is ONE authenticated pass on the real contest account")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.i_have_owner_authorization:
        print("refusing: pass --i-have-owner-authorization (one bounded pass, owner-approved)", file=sys.stderr)
        return EXIT_ABORTED_OTHER
    seed = args.rng_seed if args.rng_seed is not None else random.SystemRandom().randrange(1 << 31)
    cfg = GrabConfig(
        date=args.date, season=args.season, leaderboard_dir=args.leaderboard_dir,
        run_root=args.leaderboard_dir / f"final_grab_{args.date.replace('-', '')}",
        early_cohort_path=args.early_cohort, board_limit=args.board_limit, max_board_pages=args.max_board_pages,
        cohort_a=args.cohort_a, cohort_b=args.cohort_b, rng=random.Random(seed), rng_seed=seed,
        code_sha=_git_sha(), request_budget=args.request_budget,
        final_round_date=date.fromisoformat(args.final_round_date),
        include_tabs=not args.no_tabs,
    )
    code, status = run_grab(cfg)
    summary = {k: status.get(k) for k in ("terminal_state", "exit_code", "requests_accounted", "problems")}
    summary["requests_issued"] = len(status.get("requests", []))
    summary["planned_unattempted"] = len(status.get("planned_unattempted", []))
    b = status.get("board") or {}
    summary["board"] = {k: b.get(k) for k in ("pages", "unique_user_ids", "termination_reason", "walk_exhausted", "walk_complete", "population")}
    summary["profiles"] = {"n": len(status.get("profiles", [])), "clean": sum(1 for p in status.get("profiles", []) if p["status"] in ("success", "success_no_history"))}
    print(json.dumps(summary, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())

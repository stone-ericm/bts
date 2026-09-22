#!/usr/bin/env python3
"""Bounded end-of-season leaderboard grab (season wrap W0.6) — ONE authenticated pass.

Owner-authorized footprint (Eric, 2026-09-14/22): a single pass on the real contest
account after the regular season ends, capped by a hard request budget, with the
existing 403/429 kill-switch (any rate-limit anywhere aborts everything; no retry;
no second attempt without a fresh owner decision). Design + Codex reviews:
`.codex-review/season-wrap/w06-grab-design*.md` (2 rounds) and the plan §W0.6.

What it does, in order (network only after step 1 has written plan.json + status.json):
  1. Preflight: load cookies (provenance hashed, never stored), read the frozen early
     cohort manifest, create the run root EXCLUSIVELY under data/leaderboard/, write
     plan.json + status.json.
  2. Login (ONE attempt) through a recording transport that archives the raw body
     before bts.leaderboard.auth parses it.
  3. Four static JSON lookups (paced, archived).
  4. Full-depth walk of the SEASON-BEST board (limit 300) with validated terminal
     conditions; every page body archived; dedupe by userId; truncation exposed.
  5. Three continuity tabs (active_streak, all_time top-100; `yesterday` bound to the
     FINAL round of the season, never "today − 1").
  6. Profiles for cohort A (top of the final board) + B (frozen early cohort not in A,
     fetched BY ID), shuffled, paced, archived, id-keyed outputs.
  7. status.json rewritten atomically after every request intent and outcome.

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
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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

SCHEMA_VERSION = "bts_final_leaderboard_grab_v1"
PARSER_SCHEMA = {
    "leaderboard_row": sorted(LeaderboardRow.model_fields),
    "pick_row": sorted(PickRow.model_fields),
    "season_stats": sorted(SeasonStats.model_fields),
    "streak_semantics": "streak = tab ranking field; season_best_streak/active_streak explicit (C-01)",
}
STATIC_JSON_BASE = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json"
RATE_LIMIT_STATUSES = frozenset({403, 429})

EXIT_COMPLETE = 0
EXIT_PARTIAL = 2
EXIT_ABORTED_RATE_LIMITED = 3
EXIT_ABORTED_WRITE_FAILURE = 4
EXIT_ABORTED_OTHER = 5


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


# ----------------------------------------------------------------------------- config
class HttpxTransport:
    """Live transport: GET returns (status, bytes) without raising on any status."""

    def get(self, url: str, *, cookies: dict[str, str], timeout: float = 30.0) -> tuple[int, bytes]:
        r = httpx.get(url, cookies=cookies, timeout=timeout, headers=browser_headers(), follow_redirects=False)
        return r.status_code, r.content

    def post(self, url, *, cookies, json, headers, timeout):  # httpx.post signature
        return httpx.post(url, cookies=cookies, json=json, headers=headers, timeout=timeout, follow_redirects=False)


def _default_cookies_loader() -> tuple[dict[str, str], dict[str, Any]]:
    """Load the session cookies via bts.leaderboard.auth and describe their provenance
    (which loader branch fired + sha256 of the loaded bytes). Never returns the bytes."""
    raw = bts_auth._read_keychain_raw()
    cookies = bts_auth.load_session_cookies()
    source = "pass" if sys.platform.startswith("linux") else ("keychain" if sys.platform == "darwin" else "unknown")
    return cookies, {"source": source, "sha256": hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest(),
                     "n_cookies": len(cookies)}


@dataclass
class GrabConfig:
    date: str
    season: int
    leaderboard_dir: Path
    run_root: Path
    early_cohort_path: Path
    board_limit: int = 300
    max_board_pages: int = 340
    cohort_a: int = 150
    cohort_b: int = 150
    transport: Any = field(default_factory=HttpxTransport)
    cookies_loader: Callable[[], tuple[dict[str, str], dict[str, Any]]] = _default_cookies_loader
    sleeper: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random())
    rng_seed: int | None = None
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    code_sha: str | None = None
    request_budget: int | None = None
    final_round_date: date | None = None
    include_tabs: bool = True
    min_gap_s: float = DEFAULT_MIN_INTERVAL_S
    jitter_s: float = DEFAULT_JITTER_S
    timeout_s: float = 30.0

    def planned_max_requests(self) -> int:
        return 1 + 4 + self.max_board_pages + (3 if self.include_tabs else 0) + self.cohort_a + self.cohort_b

    def budget(self) -> int:
        return self.request_budget if self.request_budget is not None else self.planned_max_requests()


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


def _redact(url: str) -> str:
    if "xSid=" in url:
        head, _, tail = url.partition("xSid=")
        rest = tail.split("&", 1)
        return head + "xSid=<redacted>" + ("&" + rest[1] if len(rest) > 1 else "")
    return url


# ----------------------------------------------------------------------------- ledger
class Ledger:
    """Request accounting: intent persisted BEFORE send, outcome after; budget enforced;
    raw body archived before parsing; any write failure stops sending."""

    def __init__(self, cfg: GrabConfig, status: dict[str, Any]):
        self.cfg = cfg
        self.status = status
        self.issued = 0
        self.status_path = cfg.run_root / "status.json"

    def flush(self) -> None:
        try:
            _atomic_write_json(self.status_path, self.status)
        except OSError as exc:
            raise WriteFailure(f"could not write status.json: {exc}") from exc

    def _pace(self) -> None:
        if self.issued > 0:
            self.cfg.sleeper(next_gap(self.cfg.min_gap_s, self.cfg.jitter_s, self.cfg.rng))

    def request(self, cls: str, name: str, url: str, raw_path: Path,
                send: Callable[[], tuple[int, bytes]]) -> tuple[int, bytes, dict[str, Any]]:
        if self.issued + 1 > self.cfg.budget():
            raise BudgetAbort(f"request budget {self.cfg.budget()} reached before {cls}:{name}")
        self._pace()
        entry: dict[str, Any] = {
            "seq": self.issued + 1, "class": cls, "name": name, "url": _redact(url),
            "outcome": "intent", "sent_at_utc": self.cfg.now().isoformat(),
            "received_at_utc": None, "duration_s": None, "http_status": None,
            "bytes": None, "sha256": None, "raw_path": str(raw_path.relative_to(self.cfg.run_root)),
        }
        self.status["requests"].append(entry)
        self.issued += 1
        self.flush()
        t0 = time.monotonic()
        try:
            http_status, body = send()
        except Exception as exc:  # transport-level failure: recorded, no retry
            entry.update({"outcome": "error", "received_at_utc": self.cfg.now().isoformat(),
                          "duration_s": round(time.monotonic() - t0, 3), "error": f"{type(exc).__name__}: {exc}"})
            self.flush()
            return 0, b"", entry
        body = bytes(body or b"")
        entry.update({"received_at_utc": self.cfg.now().isoformat(), "duration_s": round(time.monotonic() - t0, 3),
                      "http_status": int(http_status), "bytes": len(body), "sha256": _sha256(body)})
        try:
            _atomic_write_bytes(raw_path, gzip.compress(body))
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
                 "no backfill of B from A's tail"),
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


def _validate_page_meta(success: dict[str, Any]) -> dict[str, Any]:
    nxt = success.get("nextPage")
    next_page = nxt if isinstance(nxt, bool) else None
    cnt = success.get("allParticipantsCount")
    participants = int(cnt) if isinstance(cnt, int) and not isinstance(cnt, bool) and cnt >= 0 else None
    return {"next_page": next_page, "all_participants_count": participants, "updated_at": success.get("updatedAt")}


def _round_for_date(rounds: dict[int, date], target: date) -> int | None:
    hits = [rid for rid, d in rounds.items() if d == target]
    return min(hits) if hits else None


# ----------------------------------------------------------------------------- main
def run_grab(cfg: GrabConfig) -> tuple[int, dict[str, Any]]:
    started = cfg.now()
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "date": cfg.date, "season": cfg.season,
        "started_at_utc": started.isoformat(), "finished_at_utc": None,
        "terminal_state": "running", "exit_code": None, "problems": [],
        "requests": [], "planned_unattempted": [], "requests_accounted": None,
        "board": None, "tabs": {}, "cohort": None, "profiles": [], "artifacts": {},
    }

    # ---- refusals (zero requests, zero writes into the run root) ----
    run_root = cfg.run_root
    if run_root.is_symlink():
        status.update(terminal_state="refused_run_root_symlink", exit_code=EXIT_ABORTED_OTHER,
                      problems=[f"{run_root} is a symlink"])
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
    try:
        # ---- preflight (no network) ----
        early_users, early_sha = _load_early(cfg)
        cookies, provenance = cfg.cookies_loader()
        uid = bts_auth.extract_uid(cookies)
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            for sub in ("raw/login", "raw/static", "raw/board", "raw/tabs", "raw/profiles",
                        "leaderboard_snapshots", "user_picks", "season_stats"):
                (run_root / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WriteFailure(f"could not create run root: {exc}") from exc
        final_round_date = cfg.final_round_date or (date.fromisoformat(cfg.date) - timedelta(days=1))
        plan = {
            "schema_version": SCHEMA_VERSION, "planned_at_utc": started.isoformat(), "date": cfg.date,
            "season": cfg.season, "code_sha": cfg.code_sha, "parser_schema": PARSER_SCHEMA,
            "credential_provenance": provenance, "request_budget": cfg.budget(),
            "planned_max_requests": cfg.planned_max_requests(),
            "classes": {"login": 1, "static": 4, "board_pages_max": cfg.max_board_pages,
                        "tabs": 3 if cfg.include_tabs else 0, "profiles_max": cfg.cohort_a + cfg.cohort_b},
            "board_limit": cfg.board_limit, "max_board_pages": cfg.max_board_pages,
            "pacing": {"min_gap_s": cfg.min_gap_s, "jitter_s": cfg.jitter_s},
            "kill_switch": "any HTTP 403/429 anywhere aborts the whole operation; no retry; no second attempt",
            "cohort_rule": {"A": cfg.cohort_a, "B": cfg.cohort_b,
                            "text": "A = top of final all_season board; B = frozen early cohort not in A, by id"},
            "early_cohort_path": str(cfg.early_cohort_path), "early_cohort_sha256": early_sha,
            "early_cohort_n": len(early_users), "final_round_date": final_round_date.isoformat(),
            "rng_seed": cfg.rng_seed, "run_root": str(run_root),
            "isolation": "never writes to the daily corpus (leaderboard_snapshots/, user_picks/, season_stats/, scrape_status.json)",
        }
        _atomic_write_json(run_root / "plan.json", plan)
        ledger = Ledger(cfg, status)
        ledger.flush()

        # ---- 2. login (one attempt) through the recording transport ----
        login_entry: dict[str, Any] = {}

        def recording_post(url, *, cookies, json, headers, timeout):
            holder: dict[str, Any] = {}

            def send() -> tuple[int, bytes]:
                resp = cfg.transport.post(url, cookies=cookies, json=json, headers=headers, timeout=timeout)
                holder["resp"] = resp
                return resp.status_code, resp.content

            http_status, body, entry = ledger.request("login", "auth_login", url, run_root / "raw/login/001_login.json.gz", send)
            login_entry.update(entry)
            if "resp" not in holder:
                raise httpx.TransportError(entry.get("error", "transport error"))
            return holder["resp"]

        try:
            session = bts_auth.fetch_login_session(uid, cookies, timeout=cfg.timeout_s, attempts=1, post=recording_post)
        except bts_auth.RateLimitedLoginError as exc:
            raise RateLimitedAbort(f"login rate-limited: {exc}") from exc
        except bts_auth.TransientAuthError as exc:
            raise GrabAbort("aborted_login_transient", EXIT_ABORTED_OTHER, f"login transient failure: {exc}") from exc
        except bts_auth.AuthError as exc:
            # 403 / 3xx / other 4xx are rejection-shaped: same global abort as a rate limit.
            raise RateLimitedAbort(f"login rejected: {exc}") from exc
        xsid = session.xsid

        def get(url: str) -> tuple[int, bytes]:
            return cfg.transport.get(url, cookies=cookies)

        # ---- 3. static lookups ----
        statics: dict[str, Any] = {}
        for i, (name, url) in enumerate([("rounds", ROUNDS_URL), ("players", f"{STATIC_JSON_BASE}/players.json"),
                                          ("units", f"{STATIC_JSON_BASE}/units.json"), ("squads", f"{STATIC_JSON_BASE}/squads.json")], start=1):
            http_status, body, _ = ledger.request("static", name, url, run_root / f"raw/static/{i:03d}_{name}.json.gz", lambda: get(url))
            if http_status != 200:
                raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"static {name} returned {http_status}")
            try:
                statics[name] = json.loads(body)
            except ValueError as exc:
                raise GrabAbort("aborted_static_lookup", EXIT_ABORTED_OTHER, f"static {name} not JSON: {exc}") from exc
        lookups = StaticLookups(
            rounds=parse_rounds_lookup(statics["rounds"]),
            players={int(p["id"]): p for p in statics["players"].get("players", [])},
            units={int(u["id"]): u for u in statics["units"].get("units", [])},
            squads={int(s["id"]): s for s in statics["squads"].get("squads", [])},
        )

        # ---- 4. full-depth season-best walk ----
        board: dict[str, Any] = {
            "tab": "all_season", "limit": cfg.board_limit, "pages": 0, "raw_rows": 0, "unique_user_ids": 0,
            "duplicate_rows": 0, "duplicate_conflicts": [], "per_page": [], "termination_reason": None,
            "walk_exhausted": False, "board_complete": False, "last_rank_reached": None,
            "all_participants_count": None, "coverage_of_reported_participants": None, "updated_at_values": [],
        }
        status["board"] = board
        rows_by_id: dict[int, LeaderboardRow] = {}
        prev_ids: set[int] | None = None
        participants_seen: list[int] = []
        for page in range(1, cfg.max_board_pages + 1):
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
                                      "next_page": meta["next_page"], "all_participants_count": meta["all_participants_count"]})
            if ranks:
                board["last_rank_reached"] = max(board["last_rank_reached"] or 0, max(ranks))
            if prev_ids is not None and page_ids and page_ids == prev_ids:
                board["termination_reason"] = "repeated_page"
                break
            prev_ids = page_ids
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
            if page == cfg.max_board_pages:
                board["termination_reason"] = "ceiling"
        board["unique_user_ids"] = len(rows_by_id)
        if participants_seen:
            board["all_participants_count"] = {"min": min(participants_seen), "max": max(participants_seen)}
            if max(participants_seen) > 0:
                board["coverage_of_reported_participants"] = round(len(rows_by_id) / max(participants_seen), 6)
        board["board_complete"] = bool(board["walk_exhausted"] and not board["duplicate_conflicts"])
        if not board["board_complete"]:
            status["problems"].append(f"board walk incomplete: {board['termination_reason']}")
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
        if cfg.include_tabs:
            final_round = _round_for_date(lookups.rounds, final_round_date)
            tab_specs: list[tuple[str, str | None]] = [("active_streak", None), ("all_time", None), ("yesterday", None)]
            for i, (tab, _) in enumerate(tab_specs, start=1):
                if tab == "yesterday":
                    if final_round is None:
                        status["tabs"]["yesterday"] = {"skipped": f"no round dated {final_round_date.isoformat()} in rounds.json"}
                        continue
                    url = LEADERBOARD_ROUND_URL_TEMPLATE.format(round_id=final_round, page=1, limit=100,
                                                                ranks_type=RANKS_TYPE_BY_TAB[tab], xsid=xsid)
                else:
                    url = LEADERBOARD_URL_TEMPLATE.format(season=cfg.season, page=1, limit=100,
                                                          ranks_type=RANKS_TYPE_BY_TAB[tab], xsid=xsid)
                http_status, body, _ = ledger.request("tab", tab, url, run_root / f"raw/tabs/{i:03d}_{tab}.json.gz", lambda u=url: get(u))
                info: dict[str, Any] = {"http_status": http_status}
                if tab == "yesterday":
                    info["round_id"] = final_round
                if http_status == 200:
                    try:
                        rows = parse_leaderboard_response(json.loads(body), tab=tab, captured_at=cfg.now().replace(tzinfo=None))  # type: ignore[arg-type]
                        tab_rows.extend(rows)
                        info["rows"] = len(rows)
                    except (ValueError, LeaderboardEnvelopeError) as exc:
                        info["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
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
        cohort["request_order"] = order
        cohort["rng_seed"] = cfg.rng_seed
        cohort["early_cohort_sha256"] = early_sha
        cohort["E_n"] = len({int(u["user_id"]) for u in early_users})
        status["cohort"] = {k: (len(v) if isinstance(v, list) and k in ("A", "B", "E_in_A", "E_unfetched", "request_order") else v)
                            for k, v in cohort.items()}
        _atomic_write_json(run_root / "cohort.json", cohort)
        status["artifacts"]["cohort"] = {"path": "cohort.json", "sha256": _file_sha256(run_root / "cohort.json")}
        username_by_id = {r.user_id: r.username for r in board_rows}
        for r in tab_rows:
            username_by_id.setdefault(r.user_id, r.username)
        early_name = {int(u["user_id"]): (u.get("usernames_2026_05_01") or ["unknown"])[0] for u in early_users}
        identity: dict[str, Any] = {}
        stats_rows: list[SeasonStats] = []
        for i, user_id in enumerate(order, start=1):
            url = USER_PROFILE_URL_TEMPLATE.format(user_id=user_id, xsid=xsid)
            raw_path = run_root / f"raw/profiles/{user_id}.json.gz"
            http_status, body, entry = ledger.request("profile", str(user_id), url, raw_path, lambda u=url: get(u))
            source = "A" if user_id in set(cohort["A"]) else "B"
            username = username_by_id.get(user_id) or early_name.get(user_id) or "unknown"
            rec: dict[str, Any] = {"user_id": user_id, "cohort": source, "username": username, "http_status": http_status,
                                   "status": None, "n_picks": 0, "first_pick_date": None, "last_pick_date": None,
                                   "n_rounds": 0, "skipped_unknown_round_predictions": 0, "raw_path": entry["raw_path"], "parsed_path": None}
            if http_status != 200:
                rec["status"] = "http_error"
                status["profiles"].append(rec)
                identity[str(user_id)] = rec
                continue
            try:
                parsed_body = json.loads(body)
                env = validate_profile_envelope(parsed_body)
            except (ValueError, ProfileEnvelopeError) as exc:
                rec["status"] = "envelope_error"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                status["profiles"].append(rec)
                identity[str(user_id)] = rec
                continue
            try:
                picks, stats = parse_user_profile_response(parsed_body, captured_at=cfg.now().replace(tzinfo=None),
                                                           user_id_unused=user_id, lookups=lookups, username=username)
            except Exception as exc:  # noqa: BLE001 — parse failure is recorded, never fatal
                rec["status"] = "parse_error"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                status["profiles"].append(rec)
                identity[str(user_id)] = rec
                continue
            rec.update({"n_picks": len(picks), "n_rounds": len({p.round_id for p in picks}),
                        "skipped_unknown_round_predictions": stats.skipped_unknown_round_predictions,
                        "raw_predictions": env.n_predictions, "raw_round_predictions": env.n_round_predictions,
                        "null_field_counts": env.null_field_counts, "no_history": env.no_history})
            if picks:
                dates = sorted(p.pick_date for p in picks)
                rec["first_pick_date"], rec["last_pick_date"] = dates[0].isoformat(), dates[-1].isoformat()
                parsed_path = run_root / "user_picks" / f"{user_id}.parquet"
                try:
                    append_user_picks(parsed_path, picks)
                except OSError as exc:
                    raise WriteFailure(f"could not write picks for {user_id}: {exc}") from exc
                rec["parsed_path"] = str(parsed_path.relative_to(run_root))
                rec["status"] = "success"
            else:
                rec["status"] = "success_no_history"
            stats_rows.append(stats)
            status["profiles"].append(rec)
            identity[str(user_id)] = rec
            if i % 10 == 0:
                ledger.flush()
        if stats_rows:
            stats_path = run_root / "season_stats" / f"{cfg.date}.parquet"
            try:
                write_season_stats(stats_path, stats_rows)
            except OSError as exc:
                raise WriteFailure(f"could not write season stats: {exc}") from exc
            status["artifacts"]["season_stats"] = {"path": str(stats_path.relative_to(run_root)), "sha256": _file_sha256(stats_path), "rows": len(stats_rows)}
        _atomic_write_json(run_root / "identity.json", identity)
        status["artifacts"]["identity"] = {"path": "identity.json", "sha256": _file_sha256(run_root / "identity.json")}

        profile_problems = [p for p in status["profiles"] if p["status"] in ("http_error", "envelope_error", "parse_error")]
        if profile_problems:
            status["problems"].append(f"{len(profile_problems)} profile(s) failed")
        clean = board["board_complete"] and not profile_problems and not any(
            "error" in t or t.get("http_status") not in (None, 200) for t in status["tabs"].values() if isinstance(t, dict) and "skipped" not in t)
        status["terminal_state"] = "complete" if clean else "complete_with_errors"
        exit_code = EXIT_COMPLETE if clean else EXIT_PARTIAL
    except GrabAbort as abort:
        status["terminal_state"] = abort.terminal_state
        status["problems"].append(str(abort))
        exit_code = abort.exit_code
    except Exception as exc:  # noqa: BLE001 — anything unexpected is a recorded abort, never a silent success
        status["terminal_state"] = "aborted_other"
        status["problems"].append(f"unexpected {type(exc).__name__}: {exc}")
        exit_code = EXIT_ABORTED_OTHER
    finally:
        # accounting: every request has a terminal outcome; unattempted classes listed
        for entry in status["requests"]:
            if entry.get("outcome") == "intent":
                entry["outcome"] = "aborted"
        status["requests_accounted"] = all(e.get("outcome") in ("success", "error", "aborted") for e in status["requests"])
        issued = {}
        for e in status["requests"]:
            issued[e["class"]] = issued.get(e["class"], 0) + 1
        planned = {"login": 1, "static": 4, "board": None, "tab": 3 if cfg.include_tabs else 0, "profile": None}
        if status["terminal_state"] not in ("complete", "complete_with_errors"):
            for cls, n in planned.items():
                done = issued.get(cls, 0)
                if n is None:
                    if done == 0 or (cls == "profile" and status.get("cohort") is None):
                        status["planned_unattempted"].append({"class": cls, "outcome": "unattempted", "note": "not reached"})
                elif done < n:
                    status["planned_unattempted"].append({"class": cls, "outcome": "unattempted", "remaining": n - done})
        status["finished_at_utc"] = cfg.now().isoformat()
        status["exit_code"] = exit_code
        if ledger is not None:
            try:
                ledger.flush()
            except WriteFailure:
                status["problems"].append("final status.json write failed")
    return exit_code, status


# ----------------------------------------------------------------------------- cli
def _git_sha() -> str | None:
    try:
        import subprocess
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
    p.add_argument("--max-board-pages", type=int, default=340)
    p.add_argument("--cohort-a", type=int, default=150)
    p.add_argument("--cohort-b", type=int, default=150)
    p.add_argument("--final-round-date", default=None, help="YYYY-MM-DD of the season's final round (default: date − 1)")
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
        final_round_date=date.fromisoformat(args.final_round_date) if args.final_round_date else None,
        include_tabs=not args.no_tabs,
    )
    code, status = run_grab(cfg)
    summary = {k: status.get(k) for k in ("terminal_state", "exit_code", "requests_accounted", "problems")}
    summary["requests_issued"] = len(status.get("requests", []))
    summary["board"] = {k: (status.get("board") or {}).get(k) for k in ("pages", "unique_user_ids", "termination_reason", "walk_exhausted", "board_complete", "all_participants_count")}
    summary["profiles"] = {"n": len(status.get("profiles", [])), "success": sum(1 for p in status.get("profiles", []) if p["status"].startswith("success"))}
    print(json.dumps(summary, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())

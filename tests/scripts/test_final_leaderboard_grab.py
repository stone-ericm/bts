"""Offline rehearsal for the bounded end-of-season leaderboard grab (season wrap W0.6).

Everything here runs against a scripted fake transport: no network, no cookies.
The contract under test is `.codex-review/season-wrap/w06-grab-design-v2.md` plus
the Codex round-2 findings (fail-closed envelopes, login transport boundary,
terminal-page validation, id-keyed outputs, budget cap, intent-before-send).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

from scripts.final_leaderboard_grab import (
    EXIT_ABORTED_RATE_LIMITED,
    EXIT_ABORTED_WRITE_FAILURE,
    EXIT_COMPLETE,
    EXIT_PARTIAL,
    GrabConfig,
    allocate_cohort,
    run_grab,
)

DATE = "2026-09-28"
FINAL_ROUND_ID = 1010  # round dated 2026-09-27 in the fake rounds.json
BASE = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game"


# ----------------------------------------------------------------------------- fakes
def _rank_row(uid: int, rank: int, best: int, active: int = 0, username: str | None = None):
    return {"userId": uid, "rank": rank, "username": username or f"user{uid}", "streak": best,
            "activeStreak": active, "userType": "USER"}


def _board_page(rows, *, next_page, participants=94986, updated="2026-09-28T08:00:00-04:00"):
    return {"success": {"ranks": rows, "nextPage": next_page, "allParticipantsCount": participants,
                        "friendsParticipantsCount": 0, "updatedAt": updated, "user": None}}


def _profile(best=5, active=1, preds=None):
    if preds is None:
        preds = [{"roundId": 1000, "streak": 1, "result": "hit", "streakIncrease": 1,
                  "roundPredictions": [{"number": 1, "unitId": 5, "playerId": 9, "result": "hit",
                                        "atBats": 4, "hits": 2}]}]
    return {"success": {"seasonBestStreak": best, "activeStreak": active, "accuracy": 55,
                        "favouriteBatter": None, "predictions": preds}}


def _statics():
    rounds = [{"id": 1000 + i, "date": f"2026-09-{18 + i:02d}T08:00:00-04:00"} for i in range(10)]
    return {
        f"{BASE}/json/rounds.json": {"rounds": rounds},
        f"{BASE}/json/players.json": {"players": [{"id": 9, "squadId": 1, "feedId": 660271, "name": "P"}]},
        f"{BASE}/json/units.json": {"units": [{"id": 5, "homeSquadId": 1, "awaySquadId": 2}]},
        f"{BASE}/json/squads.json": {"squads": [{"id": 1, "abbreviation": "NYM"}, {"id": 2, "abbreviation": "ATL"}]},
    }


class FakeTransport:
    """Scripted transport. `board_pages` = list of (status, body_dict|bytes) per page; `profiles`
    maps user_id -> (status, body); `login` = (status, body_bytes)."""

    def __init__(self, *, board_pages, profiles, login=(200, b'{"success": {"user": {"id": 50311, "username": "stonehengee"}, "xSid": "x_1"}}'),
                 statics=None, tabs=None, on_request=None):
        self.board_pages = list(board_pages)
        self.profiles = dict(profiles)
        self.login = login
        self.statics = statics or _statics()
        self.tabs = tabs or {}
        self.calls: list[dict] = []
        self.on_request = on_request

    def _record(self, kind, url, status, body):
        self.calls.append({"kind": kind, "url": url, "status": status, "body": body})
        if self.on_request:
            self.on_request(self.calls[-1])

    def post(self, url, *, cookies, json, headers, timeout):
        status, body = self.login
        self._record("login", url, status, body)
        return httpx.Response(status, content=body, request=httpx.Request("POST", url))

    def get(self, url, *, cookies, timeout=None):
        for k, v in self.statics.items():
            if url.startswith(k):
                body = _b(v)
                self._record("static", url, 200, body)
                return 200, body
        if "/api/rank/user" in url:
            uid = int(url.split("/api/rank/user/")[1].split("?")[0].split("/")[0])
            status, body = self.profiles.get(uid, (404, b'{"errors":[{"message":"no such user"}]}'))
            body = _b(body)
            self._record("profile", url, status, body)
            return status, body
        if "/api/rank/leaderboard" in url and "ranksType=SEASON_BEST_STREAK" in url and "page=" in url:
            page = int(url.split("page=")[1].split("&")[0])
            limit = int(url.split("limit=")[1].split("&")[0])
            if limit == 300 and self.board_pages:
                status, body = self.board_pages[min(page, len(self.board_pages)) - 1]
                body = _b(body)
                self._record("board", url, status, body)
                return status, body
        if "/api/rank/leaderboard" in url:
            key = "round" if "/round/" in url else url.split("ranksType=")[1].split("&")[0]
            status, body = self.tabs.get(key, (200, _board_page([_rank_row(1, 1, 3, 3)], next_page=False, participants=100)))
            body = _b(body)
            self._record("tab:" + key, url, status, body)
            return status, body
        raise AssertionError(f"unexpected url {url}")


def _b(v):
    return v if isinstance(v, (bytes, bytearray)) else json.dumps(v).encode()


def _early_manifest(path: Path, ids: list[int]):
    users = [{"order": i + 1, "user_id": uid, "usernames_2026_05_01": [f"early{uid}"],
              "ranks_2026_05_01": {"all_season": i + 1}} for i, uid in enumerate(ids)]
    path.write_text(json.dumps({"schema_version": "bts_early_cohort_manifest_v1", "n_users": len(users), "users": users,
                                "source_fixture_sha256": {"x": "0" * 64}}))
    return path


def _config(tmp_path: Path, transport, *, max_board_pages=340, cohort_a=2, cohort_b=2, early_ids=(11, 12, 13),
            cookies_loader=None, sleeper=None, now=None, board_limit=300):
    daily = tmp_path / "data" / "leaderboard"
    (daily / "user_picks").mkdir(parents=True)
    (daily / "user_picks" / "someone.parquet").write_bytes(b"keep-me")
    (daily / "scrape_status.json").write_text("{}")
    early = _early_manifest(tmp_path / "early.json", list(early_ids))
    return GrabConfig(
        date=DATE,
        season=2026,
        leaderboard_dir=daily,
        run_root=daily / f"final_grab_{DATE.replace('-', '')}",
        early_cohort_path=early,
        board_limit=board_limit,
        max_board_pages=max_board_pages,
        cohort_a=cohort_a,
        cohort_b=cohort_b,
        transport=transport,
        cookies_loader=cookies_loader or (lambda: ({"oktaid": "abc"}, {"source": "test", "sha256": "0" * 64})),
        sleeper=sleeper or (lambda s: None),
        rng=random.Random(7),
        now=now or (lambda: datetime(2026, 9, 28, 13, 0, tzinfo=timezone.utc)),
        code_sha="deadbeef",
        final_round_date=date(2026, 9, 27),
    )


def _status(cfg: GrabConfig) -> dict:
    return json.loads((cfg.run_root / "status.json").read_text())


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "final_grab_" not in str(p)}


def _two_page_board():
    p1 = [_rank_row(100 + i, i + 1, 40 - (i // 30)) for i in range(300)]
    p2 = [_rank_row(400 + i, 301 + i, 10) for i in range(120)]
    return [(200, _board_page(p1, next_page=True)), (200, _board_page(p2, next_page=False))]


# ----------------------------------------------------------------------------- tests
def test_plan_and_status_are_written_before_any_request(tmp_path):
    seen = {}

    def on_request(call):
        if "first_seen" not in seen:
            cfg_root = tmp_path / "data" / "leaderboard" / f"final_grab_{DATE.replace('-', '')}"
            seen["first_seen"] = {"plan": (cfg_root / "plan.json").exists(),
                                  "status": (cfg_root / "status.json").exists(),
                                  "intent_logged": "login" in json.loads((cfg_root / "status.json").read_text())["requests"][0]["class"]
                                  if (cfg_root / "status.json").exists() else False}

    t = FakeTransport(board_pages=_two_page_board(), profiles={}, on_request=on_request)
    cfg = _config(tmp_path, t)
    run_grab(cfg)
    assert seen["first_seen"] == {"plan": True, "status": True, "intent_logged": True}


def test_existing_run_root_means_zero_requests(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={})
    cfg = _config(tmp_path, t)
    cfg.run_root.mkdir(parents=True)
    code, status = run_grab(cfg)
    assert code != EXIT_COMPLETE and status["terminal_state"] == "refused_run_root_exists"
    assert t.calls == []


def test_full_walk_short_last_page_is_exhausted_and_archived(tmp_path):
    profiles = {uid: (200, _profile()) for uid in (100, 101, 11, 12)}
    t = FakeTransport(board_pages=_two_page_board(), profiles=profiles)
    cfg = _config(tmp_path, t, early_ids=(11, 12, 100))
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL, status  # walk exhausted, but 420 listed vs 94,986 reported = population gap
    assert status["terminal_state"] == "complete_with_population_gap"
    board = status["board"]
    assert board["walk_exhausted"] is True and board["termination_reason"] == "short_page"
    assert board["pages"] == 2 and board["unique_user_ids"] == 420 and board["raw_rows"] == 420
    assert board["all_participants_count"] == {"min": 94986, "max": 94986}
    assert board["population"] == {"status": "gap_vs_reported_participants", "reported": 94986, "listed_unique": 420, "gap": 94566}
    assert board["walk_complete"] is True and board["population_complete"] is False
    raw_pages = sorted((cfg.run_root / "raw" / "board").glob("*.json.gz"))
    assert len(raw_pages) == 2
    assert json.loads(gzip.decompress(raw_pages[0].read_bytes()))["success"]["nextPage"] is True
    snap = pq.read_table(cfg.run_root / "leaderboard_snapshots" / f"{DATE}_all_season_full.parquet").to_pandas()
    assert len(snap) == 420 and snap.season_best_streak.notna().all() and snap.tab.eq("all_season").all()
    for req in status["requests"]:
        assert req["outcome"] in ("success", "error", "aborted") and "sha256" in req and "sent_at_utc" in req
    assert all(p["updated_at"] for p in board["per_page"])


def test_short_page_with_next_page_true_is_a_contradiction_not_exhaustion(tmp_path):
    pages = [(200, _board_page([_rank_row(1, 1, 5)], next_page=True))]
    t = FakeTransport(board_pages=pages, profiles={1: (200, _profile())})
    cfg = _config(tmp_path, t, early_ids=(1,))
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL
    assert status["board"]["walk_exhausted"] is False
    assert status["board"]["termination_reason"] == "short_page_contradicted_by_next_page"
    assert status["board"]["walk_complete"] is False


def test_missing_pagination_metadata_is_not_exhaustion(tmp_path):
    body = {"success": {"ranks": [_rank_row(1, 1, 5)], "allParticipantsCount": 10}}  # no nextPage key
    t = FakeTransport(board_pages=[(200, body)], profiles={1: (200, _profile())})
    cfg = _config(tmp_path, t, early_ids=(1,))
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["board"]["walk_exhausted"] is False
    assert status["board"]["termination_reason"] == "short_page_pagination_unknown"


def test_ceiling_is_truncation_with_nonzero_exit_even_when_accounted(tmp_path):
    pages = [(200, _board_page([_rank_row(1000 * p + i, 300 * (p - 1) + i + 1, 30) for i in range(300)], next_page=True))
             for p in range(1, 4)]
    t = FakeTransport(board_pages=pages, profiles={1000: (200, _profile())})
    cfg = _config(tmp_path, t, max_board_pages=2, early_ids=(1000,), cohort_a=1, cohort_b=1)
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL
    assert status["board"]["termination_reason"] == "ceiling" and status["board"]["walk_complete"] is False
    assert status["board"]["pages"] == 2 and status["board"]["last_rank_reached"] == 600
    assert status["requests_accounted"] is True


def test_repeated_page_stops_the_walk_as_incomplete(tmp_path):
    rows = [_rank_row(100 + i, i + 1, 30) for i in range(300)]
    t = FakeTransport(board_pages=[(200, _board_page(rows, next_page=True)), (200, _board_page(rows, next_page=True))],
                      profiles={100: (200, _profile())})
    cfg = _config(tmp_path, t, early_ids=(100,), cohort_a=1, cohort_b=1)
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["board"]["termination_reason"] == "repeated_page"


def test_rate_limit_on_a_board_page_aborts_everything_and_archives_the_error_body(tmp_path):
    pages = [_two_page_board()[0], (429, b"slow down")]
    t = FakeTransport(board_pages=pages, profiles={100: (200, _profile())})
    cfg = _config(tmp_path, t)
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_RATE_LIMITED and status["terminal_state"] == "aborted_rate_limited"
    kinds = [c["kind"] for c in t.calls]
    assert kinds[-1] == "board" and "profile" not in kinds
    err_bodies = [p for p in (cfg.run_root / "raw" / "board").glob("*.json.gz")]
    assert len(err_bodies) == 2 and gzip.decompress(err_bodies[-1].read_bytes()) == b"slow down"
    assert status["requests"][-1]["outcome"] == "aborted" and status["requests"][-1]["http_status"] == 429
    assert any(r["outcome"] == "unattempted" for r in status["planned_unattempted"]) or status["planned_unattempted"]


def test_login_403_archives_the_body_makes_one_post_and_sends_nothing_else(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={}, login=(403, b"forbidden"))
    cfg = _config(tmp_path, t)
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_RATE_LIMITED
    assert [c["kind"] for c in t.calls] == ["login"]
    raw = list((cfg.run_root / "raw" / "login").glob("*.gz"))
    assert len(raw) == 1 and gzip.decompress(raw[0].read_bytes()) == b"forbidden"


def test_login_429_is_the_same_abort(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={}, login=(429, b"rl"))
    cfg = _config(tmp_path, t)
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_RATE_LIMITED and len(t.calls) == 1


def test_envelope_error_on_a_page_terminates_with_error_and_partial_exit(tmp_path):
    pages = [_two_page_board()[0], (200, {"errors": [{"message": "boom"}]})]
    t = FakeTransport(board_pages=pages, profiles={100: (200, _profile())})
    cfg = _config(tmp_path, t, early_ids=(100,), cohort_a=1, cohort_b=1)
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["board"]["termination_reason"] == "error"
    assert status["board"]["walk_complete"] is False


def test_cohort_allocation_rules():
    board = [{"user_id": 100 + i, "rank": i + 1} for i in range(10)]
    early = [{"user_id": u} for u in (105, 106, 500, 501, 502)]
    c = allocate_cohort(board, early, cohort_a=4, cohort_b=2)
    assert c["A"] == [100, 101, 102, 103]
    assert c["B"] == [105, 106]            # early ids not in A, in early order, fetched BY ID even if absent
    assert c["E_in_A"] == []                # 105/106 are ranks 6-7, not in A
    assert c["E_unfetched"] == [500, 501, 502]
    c2 = allocate_cohort(board, [{"user_id": 100}, {"user_id": 101}], cohort_a=4, cohort_b=2)
    assert c2["B"] == [] and c2["B_shortfall"] == 2 and c2["E_in_A"] == [100, 101]


def test_profiles_are_id_keyed_with_identity_map_and_duplicate_usernames_kept_apart(tmp_path):
    rows = [_rank_row(1135, 1, 20, username="jordan"), _rank_row(2002, 2, 19, username="jordan"),
            _rank_row(3, 3, 18, username="a/b"), _rank_row(4, 4, 17, username="a?b")]
    pages = [(200, _board_page(rows, next_page=False, participants=4))]
    profiles = {1135: (200, _profile(best=20)), 2002: (200, _profile(best=19)), 3: (200, _profile(best=18)), 4: (200, _profile(best=17))}
    t = FakeTransport(board_pages=pages, profiles=profiles)
    cfg = _config(tmp_path, t, cohort_a=4, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    assert code == EXIT_COMPLETE, status
    picks_dir = cfg.run_root / "user_picks"
    assert sorted(p.name for p in picks_dir.glob("*.parquet")) == ["1135.parquet", "2002.parquet", "3.parquet", "4.parquet"]
    identity = json.loads((cfg.run_root / "identity.json").read_text())
    assert identity["1135"]["username"] == "jordan" and identity["2002"]["username"] == "jordan"
    stats = pq.read_table(cfg.run_root / "season_stats" / f"{DATE}.parquet").to_pandas()
    assert sorted(stats.user_id.tolist()) == [3, 4, 1135, 2002]
    assert stats[stats.user_id == 2002].best_streak.iloc[0] == 19
    raw = sorted(p.name for p in (cfg.run_root / "raw" / "profiles").glob("*.gz"))
    assert raw == ["1135.json.gz", "2002.json.gz", "3.json.gz", "4.json.gz"]


def test_profile_error_continues_but_profile_rate_limit_aborts(tmp_path):
    rows = [_rank_row(1, 1, 9), _rank_row(2, 2, 8), _rank_row(3, 3, 7)]
    pages = [(200, _board_page(rows, next_page=False, participants=3))]
    profiles = {1: (500, b"oops"), 2: (200, {"success": {"seasonBestStreak": 8}}), 3: (200, _profile())}
    t = FakeTransport(board_pages=pages, profiles=profiles)
    cfg = _config(tmp_path, t, cohort_a=3, cohort_b=0, early_ids=(), sleeper=lambda s: None)
    cfg.rng = random.Random(0)
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["terminal_state"] == "complete_with_errors"
    outcomes = {p["user_id"]: p for p in status["profiles"]}
    assert outcomes[1]["status"] == "http_error" and outcomes[2]["status"] == "envelope_error" and outcomes[3]["status"] == "success"
    assert not (cfg.run_root / "user_picks" / "2.parquet").exists()  # fail closed: no zero-pick user

    profiles_rl = {1: (200, _profile()), 2: (429, b"rl"), 3: (200, _profile())}
    t2 = FakeTransport(board_pages=pages, profiles=profiles_rl)
    cfg2 = _config(tmp_path / "second", t2, cohort_a=3, cohort_b=0, early_ids=())
    code2, status2 = run_grab(cfg2)
    assert code2 == EXIT_ABORTED_RATE_LIMITED
    profile_calls = [c for c in t2.calls if c["kind"] == "profile"]
    assert profile_calls[-1]["status"] == 429  # the 429 is the LAST request sent, whatever the shuffle


def test_budget_cap_is_enforced(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={uid: (200, _profile()) for uid in range(100, 110)})
    cfg = _config(tmp_path, t, cohort_a=10, cohort_b=0, early_ids=())
    cfg.request_budget = 9  # 1 login + 4 static + 2 board + 3 tabs = 10 > 9
    code, status = run_grab(cfg)
    assert status["terminal_state"] == "aborted_budget" and len(t.calls) <= 9


def test_write_failure_stops_sending(tmp_path, monkeypatch):
    t = FakeTransport(board_pages=_two_page_board(), profiles={})
    cfg = _config(tmp_path, t)
    calls_at_failure = {}
    import scripts.final_leaderboard_grab as mod
    real = mod._atomic_write_bytes
    state = {"n": 0}

    def flaky(path, data):
        state["n"] += 1
        if state["n"] == 6:  # a raw board body write fails
            calls_at_failure["n_calls"] = len(t.calls)
            raise OSError("disk full")
        return real(path, data)

    monkeypatch.setattr(mod, "_atomic_write_bytes", flaky)
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_WRITE_FAILURE and status["terminal_state"] == "aborted_write_failure"
    assert len(t.calls) == calls_at_failure["n_calls"]


def test_daily_corpus_is_untouched(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={100: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    before = _tree_hashes(cfg.leaderboard_dir)
    run_grab(cfg)
    assert _tree_hashes(cfg.leaderboard_dir) == before
    assert (cfg.leaderboard_dir / "scrape_status.json").read_text() == "{}"


def test_yesterday_tab_is_bound_to_the_final_round(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={})
    cfg = _config(tmp_path, t, cohort_a=0, cohort_b=0, early_ids=())
    cfg.final_round_date = date(2026, 9, 27)
    code, status = run_grab(cfg)
    round_calls = [c["url"] for c in t.calls if c["kind"] == "tab:round"]
    assert len(round_calls) == 1 and f"/round/{1009}?" in round_calls[0]  # 1000 + 9 -> 2026-09-27
    assert status["tabs"]["yesterday"]["round_id"] == 1009


def test_run_root_symlink_escape_is_refused(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={})
    cfg = _config(tmp_path, t)
    outside = tmp_path / "outside"
    outside.mkdir()
    cfg.run_root.parent.mkdir(parents=True, exist_ok=True)
    cfg.run_root.symlink_to(outside, target_is_directory=True)
    code, status = run_grab(cfg)
    assert status["terminal_state"] in ("refused_run_root_exists", "refused_run_root_symlink") and t.calls == []


def test_provenance_fields_present(tmp_path):
    t = FakeTransport(board_pages=_two_page_board(), profiles={100: (200, _profile()), 11: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=1, early_ids=(11,))
    code, status = run_grab(cfg)
    plan = json.loads((cfg.run_root / "plan.json").read_text())
    for k in ("code_sha", "credential_provenance", "request_budget", "max_board_pages", "board_limit", "cohort_rule",
              "early_cohort_sha256", "parser_schema", "planned_at_utc"):
        assert k in plan, k
    assert plan["credential_provenance"]["sha256"] == "0" * 64 and "cookie" not in json.dumps(plan).lower().replace("cookies_loader", "")
    cohort = json.loads((cfg.run_root / "cohort.json").read_text())
    assert cohort["A"] == [100] and cohort["B"] == [11] and "request_order" in cohort and "rng_seed" in cohort
    assert status["artifacts"]["leaderboard_snapshot"]["sha256"]
    assert all(a["sha256"] for a in status["artifacts"].values() if isinstance(a, dict))


# ----------------------------------------------------------------------------- Codex code-review round 1
from scripts.final_leaderboard_grab import BodyReadError, EXIT_ABORTED_OTHER, load_cookies_with_provenance  # noqa: E402


def _census_board(n=3):
    return [(200, _board_page([_rank_row(i, i, 30 - i) for i in range(1, n + 1)], next_page=False, participants=n))]


def test_rate_limit_status_with_truncated_body_still_aborts(tmp_path):
    class T(FakeTransport):
        def get(self, url, *, cookies, timeout=None):
            if "/api/rank/user/2/" in url:
                self._record("profile", url, 403, b"")
                raise BodyReadError(403, RuntimeError("ReadError: connection closed"))
            return super().get(url, cookies=cookies)

    t = T(board_pages=_census_board(3), profiles={1: (200, _profile()), 2: (403, b""), 3: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=3, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_RATE_LIMITED
    profile_calls = [c for c in t.calls if c["kind"] == "profile"]
    assert profile_calls[-1]["status"] == 403
    last = [r for r in status["requests"] if r["class"] == "profile"][-1]
    assert last["http_status"] == 403 and last["outcome"] == "aborted" and "body_read_error" in last


def test_census_walk_is_complete_and_exit_zero(tmp_path):
    t = FakeTransport(board_pages=_census_board(3), profiles={1: (200, _profile()), 2: (200, _profile()), 3: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=3, cohort_b=0, early_ids=())
    cfg.final_round_date = date(2026, 9, 27)
    code, status = run_grab(cfg)
    assert (code, status["terminal_state"]) == (EXIT_COMPLETE, "complete"), status["problems"]
    assert status["board"]["population"]["status"] == "census" and status["board"]["population_complete"] is True


def test_missing_or_drifting_participant_metadata_makes_population_unknown(tmp_path):
    rows = [_rank_row(i, i, 30 - i) for i in range(1, 4)]
    body = {"success": {"ranks": rows, "nextPage": False}}  # no allParticipantsCount
    t = FakeTransport(board_pages=[(200, body)], profiles={1: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["board"]["population"]["status"] == "unknown_no_participant_count"
    p1 = [_rank_row(100 + i, i + 1, 40) for i in range(300)]
    p2 = [_rank_row(400 + i, 301 + i, 10) for i in range(10)]
    t2 = FakeTransport(board_pages=[(200, _board_page(p1, next_page=True, participants=310)),
                                    (200, _board_page(p2, next_page=False, participants=311))], profiles={100: (200, _profile())})
    cfg2 = _config(tmp_path / "b", t2, cohort_a=1, cohort_b=0, early_ids=())
    code2, status2 = run_grab(cfg2)
    assert code2 == EXIT_PARTIAL and status2["board"]["population"]["status"] == "unknown_participant_count_drift"


def test_error_only_rounds_json_aborts_static_stage(tmp_path):
    statics = _statics(); statics[f"{BASE}/json/rounds.json"] = {"errors": [{"message": "nope"}]}
    t = FakeTransport(board_pages=_census_board(), profiles={}, statics=statics)
    cfg = _config(tmp_path, t)
    code, status = run_grab(cfg)
    assert status["terminal_state"] == "aborted_static_lookup" and code == EXIT_ABORTED_OTHER
    assert not any(c["kind"] == "board" for c in t.calls)


def test_empty_rounds_list_aborts_static_stage(tmp_path):
    statics = _statics(); statics[f"{BASE}/json/rounds.json"] = {"rounds": []}
    t = FakeTransport(board_pages=_census_board(), profiles={}, statics=statics)
    cfg = _config(tmp_path, t)
    code, status = run_grab(cfg)
    assert status["terminal_state"] == "aborted_static_lookup"


def test_profile_prediction_with_only_round_id_is_an_envelope_error(tmp_path):
    body = {"success": {"seasonBestStreak": 1, "activeStreak": 0, "accuracy": 50, "predictions": [{"roundId": 1000}]}}
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, body)})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    assert status["profiles"][0]["status"] == "envelope_error" and code == EXIT_PARTIAL
    assert not (cfg.run_root / "user_picks" / "1.parquet").exists()


def test_profile_slot_missing_required_keys_is_an_envelope_error(tmp_path):
    body = _profile(preds=[{"roundId": 1000, "streak": 1, "roundPredictions": [{}]}])
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, body)})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    assert status["profiles"][0]["status"] == "envelope_error"


def test_profile_with_unknown_round_is_partial_not_clean(tmp_path):
    body = _profile(preds=[{"roundId": 1000, "streak": 1, "result": "hit", "roundPredictions": [{"number": 1, "unitId": 5, "playerId": 9, "result": "hit", "atBats": 4, "hits": 2}]},
                           {"roundId": 4242, "streak": 2, "result": "hit", "roundPredictions": [{"number": 1, "unitId": 5, "playerId": 9, "result": "hit", "atBats": 4, "hits": 1}]}])
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, body)})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    cfg.final_round_date = date(2026, 9, 27)
    code, status = run_grab(cfg)
    assert status["profiles"][0]["status"] == "success_partial_lookup" and status["profiles"][0]["skipped_unknown_round_predictions"] == 1
    assert code == EXIT_PARTIAL and status["terminal_state"] == "complete_with_errors"


def test_unresolved_slot_values_are_counted_and_not_fabricated_as_picks(tmp_path):
    body = _profile(preds=[{"roundId": 1000, "streak": None, "result": None,
                            "roundPredictions": [{"number": 1, "unitId": None, "playerId": None, "result": None, "atBats": None, "hits": None}]}])
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, body)})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    rec = status["profiles"][0]
    assert rec["status"] == "success_with_unresolved" and rec["unresolved_round_predictions"] == 1
    assert not (cfg.run_root / "user_picks" / "1.parquet").exists()


def test_login_raw_archive_is_redacted_and_xsid_never_serialized(tmp_path):
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, _profile())},
                      login=(200, b'{"success": {"user": {"id": 50311, "username": "stonehengee"}, "xSid": "SECRET_XSID_VALUE"}}'))
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    run_grab(cfg)
    raw = gzip.decompress(next((cfg.run_root / "raw" / "login").glob("*.gz")).read_bytes())
    assert b"SECRET_XSID_VALUE" not in raw and b"<redacted" in raw
    blob = b"".join(p.read_bytes() for p in cfg.run_root.rglob("*") if p.is_file() and not p.suffix == ".parquet" and not p.name.endswith(".gz"))
    assert b"SECRET_XSID_VALUE" not in blob
    for gz in cfg.run_root.rglob("*.gz"):
        assert b"SECRET_XSID_VALUE" not in gzip.decompress(gz.read_bytes())


def test_final_status_write_failure_is_exit_4_and_reported_outside_the_file(tmp_path, monkeypatch):
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    cfg.final_round_date = date(2026, 9, 27)
    import scripts.final_leaderboard_grab as mod
    real = mod._atomic_write_json
    state = {"armed": False}

    def flaky(path, obj):
        if state["armed"] and path.name == "status.json" and obj.get("terminal_state") in ("complete", "complete_with_errors", "complete_with_population_gap"):
            raise OSError("disk full at the end")
        return real(path, obj)

    monkeypatch.setattr(mod, "_atomic_write_json", flaky)
    state["armed"] = True
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_WRITE_FAILURE and status["terminal_state"] == "aborted_write_failure"
    assert (cfg.run_root.parent / f"{cfg.run_root.name}.STATUS_WRITE_FAILED").exists()


def test_abort_accounting_lists_every_remaining_planned_request(tmp_path):
    rows = [_rank_row(i, i, 30 - i) for i in range(1, 4)]
    t = FakeTransport(board_pages=[(200, _board_page(rows, next_page=False, participants=3))],
                      profiles={1: (429, b"rl"), 2: (200, _profile()), 3: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=3, cohort_b=0, early_ids=())
    cfg.rng = random.Random(1)
    code, status = run_grab(cfg)
    assert code == EXIT_ABORTED_RATE_LIMITED
    sent = {int(r["name"]) for r in status["requests"] if r["class"] == "profile"}
    unattempted = {u["user_id"] for u in status["planned_unattempted"] if u["class"] == "profile"}
    assert sent | unattempted == {1, 2, 3} and not (sent & unattempted)
    assert status["requests_accounted"] is True
    identity = json.loads((cfg.run_root / "identity.json").read_text())
    assert set(identity) == {str(u) for u in sent}  # incremental identity preserved through the abort


def test_missing_final_round_makes_the_run_partial_not_complete(tmp_path):
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    cfg.final_round_date = date(2026, 12, 25)  # not in rounds.json
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["tabs"]["yesterday"]["status"] == "skipped_no_final_round"
    assert any(u["class"] == "tab" for u in status["planned_unattempted"])


def test_config_validation_refuses_before_credentials_or_filesystem(tmp_path):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return {"oktaid": "abc"}, {"source": "test", "sha256": "0" * 64}

    for kwargs in ({"cohort_a": 301, "cohort_b": 0}, {"cohort_a": -1, "cohort_b": 0}, {"board_limit": 301}, {"max_board_pages": 0}):
        t = FakeTransport(board_pages=_census_board(), profiles={})
        cfg = _config(tmp_path / str(sorted(kwargs.items())), t, early_ids=(), cookies_loader=loader, **{k: v for k, v in kwargs.items() if k in ("cohort_a", "cohort_b", "board_limit", "max_board_pages")})
        code, status = run_grab(cfg)
        assert status["terminal_state"] == "refused_config", kwargs
        assert not cfg.run_root.exists() and t.calls == []
    assert calls["n"] == 0


def test_cookie_loader_reads_once_and_reports_the_real_branch(tmp_path, monkeypatch):
    blob = json.dumps([{"name": "oktaid", "value": "u1"}, {"name": "other", "value": "v"}]).encode()
    monkeypatch.setattr("scripts.final_leaderboard_grab._pass_show", lambda entry: None)  # pass unavailable
    cookie_file = tmp_path / "cookies.json"; cookie_file.write_bytes(blob)
    cookies, prov = load_cookies_with_provenance(cookie_file=cookie_file, platform="linux")
    assert cookies == {"oktaid": "u1", "other": "v"}
    assert prov["source"] == "file" and prov["sha256"] == hashlib.sha256(blob).hexdigest() and prov["bytes"] == len(blob)
    monkeypatch.setattr("scripts.final_leaderboard_grab._pass_show", lambda entry: blob + b"\n")
    cookies2, prov2 = load_cookies_with_provenance(cookie_file=cookie_file, platform="linux")
    assert prov2["source"] == "pass" and prov2["sha256"] == hashlib.sha256(blob + b"\n").hexdigest()


def test_uv_lock_and_manifest_hashes_are_in_the_plan(tmp_path):
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    (tmp_path / "uv.lock").write_text("lock")
    cfg.uv_lock_path = tmp_path / "uv.lock"
    run_grab(cfg)
    plan = json.loads((cfg.run_root / "plan.json").read_text())
    assert plan["uv_lock_sha256"] == hashlib.sha256(b"lock").hexdigest()
    assert (cfg.run_root / "inputs" / "early_cohort.json").exists()
    identity = json.loads((cfg.run_root / "identity.json").read_text())
    assert identity["1"]["parsed_sha256"]


# ----------------------------------------------------------------------------- Codex code-review round 2
def test_gzip_login_through_the_real_transport_does_not_double_decode():
    import gzip as _gz
    from scripts.final_leaderboard_grab import HttpxTransport
    from bts.leaderboard.auth import fetch_login_session
    payload = _gz.compress(b'{"success": {"user": {"id": 50311, "username": "s"}, "xSid": "x_gz"}}')

    def handler(request):
        return httpx.Response(200, content=payload, headers={"Content-Encoding": "gzip", "Content-Type": "application/json"})

    real = HttpxTransport()
    import scripts.final_leaderboard_grab as mod
    orig_client = httpx.Client
    monkey = lambda **kw: orig_client(transport=httpx.MockTransport(handler), **kw)  # noqa: E731
    mod.httpx.Client = monkey
    try:
        session = fetch_login_session("uid", {"c": "v"}, attempts=1, post=real.post)
        assert session.xsid == "x_gz"
        rej = HttpxTransport()

        def handler403(request):
            return httpx.Response(403, content=_gz.compress(b"denied"), headers={"Content-Encoding": "gzip"})
        mod.httpx.Client = lambda **kw: orig_client(transport=httpx.MockTransport(handler403), **kw)
        with pytest.raises(Exception) as ei:
            fetch_login_session("uid", {"c": "v"}, attempts=1, post=rej.post)
        assert "403" in str(ei.value)
        status, body = HttpxTransport().get("https://x/y", cookies={})
        assert (status, body) == (403, b"denied")
    finally:
        mod.httpx.Client = orig_client


def test_census_requires_valid_consistent_metadata_on_every_page(tmp_path):
    p1 = [_rank_row(100 + i, i + 1, 40) for i in range(300)]
    p2 = [_rank_row(400, 301, 10)]
    # page 1 lacks allParticipantsCount -> population unknown even though the walk exhausts
    t = FakeTransport(board_pages=[(200, {"success": {"ranks": p1, "nextPage": True, "updatedAt": "u1"}}),
                                   (200, _board_page(p2, next_page=False, participants=301, updated="u1"))],
                      profiles={100: (200, _profile())})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    assert code == EXIT_PARTIAL and status["board"]["population"]["status"] == "unknown_incomplete_participant_metadata"
    assert status["board"]["population_complete"] is False
    # server version drift (updatedAt differs) -> unknown
    t2 = FakeTransport(board_pages=[(200, _board_page(p1, next_page=True, participants=301, updated="u1")),
                                    (200, _board_page(p2, next_page=False, participants=301, updated="u2"))],
                       profiles={100: (200, _profile())})
    cfg2 = _config(tmp_path / "b", t2, cohort_a=1, cohort_b=0, early_ids=())
    code2, status2 = run_grab(cfg2)
    assert code2 == EXIT_PARTIAL and status2["board"]["population"]["status"] == "unknown_server_version_drift"
    # duplicate conflict across pages -> walk not complete, population not complete
    p2c = [_rank_row(100, 301, 39)]  # id 100 again with a different rank/streak
    t3 = FakeTransport(board_pages=[(200, _board_page(p1, next_page=True, participants=300, updated="u1")),
                                    (200, _board_page(p2c, next_page=False, participants=300, updated="u1"))],
                       profiles={100: (200, _profile())})
    cfg3 = _config(tmp_path / "c", t3, cohort_a=1, cohort_b=0, early_ids=())
    code3, status3 = run_grab(cfg3)
    assert code3 == EXIT_PARTIAL and status3["board"]["walk_complete"] is False and status3["board"]["population_complete"] is False
    assert status3["board"]["population"]["status"] == "unknown_duplicate_conflicts"


@pytest.mark.parametrize("slot", [
    {"number": 1, "unitId": 5, "playerId": 9, "result": None, "atBats": 4, "hits": 1},      # pending result
    {"number": None, "unitId": 5, "playerId": 9, "result": "hit", "atBats": 4, "hits": 1},   # null slot number
    {"number": 3, "unitId": 5, "playerId": 9, "result": "hit", "atBats": 4, "hits": 1},      # invalid slot number
    {"number": 1, "unitId": 5, "playerId": 9, "result": "hit", "atBats": None, "hits": None},  # settled result, missing measurements
])
def test_unresolved_or_invalid_slots_never_become_settled_picks(tmp_path, slot):
    body = _profile(preds=[{"roundId": 1000, "streak": 1, "result": slot["result"], "roundPredictions": [slot]}])
    t = FakeTransport(board_pages=_census_board(1), profiles={1: (200, body)})
    cfg = _config(tmp_path, t, cohort_a=1, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    rec = status["profiles"][0]
    assert rec["status"] == "success_with_unresolved" and rec["unresolved_round_predictions"] == 1, rec
    assert rec["unresolved_reasons"]
    assert not (cfg.run_root / "user_picks" / "1.parquet").exists()
    assert code == EXIT_PARTIAL


def test_echoed_token_is_scrubbed_from_every_archive_the_summary_and_the_marker(tmp_path, monkeypatch):
    token = b"SECRET_XSID_VALUE"
    rows = [_rank_row(1, 1, 9), _rank_row(2, 2, 8)]
    t = FakeTransport(board_pages=[(200, _board_page(rows, next_page=False, participants=2))],
                      profiles={1: (500, b"server error while handling xSid=SECRET_XSID_VALUE"),
                                2: (200, {"errors": [{"message": "bad token SECRET_XSID_VALUE"}]})},
                      login=(200, b'{"success": {"user": {"id": 1, "username": "s"}, "xSid": "SECRET_XSID_VALUE"}}'))
    cfg = _config(tmp_path, t, cohort_a=2, cohort_b=0, early_ids=())
    code, status = run_grab(cfg)
    for gz in cfg.run_root.rglob("*.gz"):
        assert token not in gzip.decompress(gz.read_bytes()), gz
    for f in cfg.run_root.rglob("*.json"):
        assert token not in f.read_bytes(), f
    assert token.decode() not in json.dumps(status, default=str)  # returned object is safe for CLI printing
    reqs = {r["name"]: r for r in status["requests"] if r["class"] == "profile"}
    assert reqs["1"]["archived_sha256"] != reqs["1"]["sha256"] and reqs["1"]["raw_redacted"] is True
    # a clean body archives verbatim: digests agree
    clean = next(r for r in status["requests"] if r["class"] == "board")
    assert clean["archived_sha256"] == clean["sha256"] and clean["raw_redacted"] is False

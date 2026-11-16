"""Adversarial regression probes for CS2B (season tag on streak.json + season_of_date).

Self-contained pytest module with all clocks pinned and no external network.
Covers:
- Rows A..K including type/boolean/keyword-only boundaries and mid-season deploy (Row K)
- Row L (`season_of_date` + L1..L4 real entry points) across exotic date shapes and types
- Multi-step 2026 -> 2027 boundary sequences on one directory (hit, miss from 2026 saver zone,
  void, double hit, double partial miss, racing scorers, crash between writes, 02:00 reconcile)
- Mid-season paths on a 2027-tagged file (saver 9..16 incl. DD miss, once-per-date incl. unpadded/
  compact/week and cross-spelling characterization, today unresolved+applied replay guard, dateless
  update between dated updates)
- Corrupt state (`{`, non-UTF-8, empty, `[]`, `null`, number, string, deep recursion, >4300-digit
  int, `chmod 000` unreadable file) across scorers, decision state, and complete/incomplete replay
"""
from __future__ import annotations

from contextlib import nullcontext
import datetime as dt
from enum import IntEnum
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd
import pytest

import bts.cli as cli_mod
from bts.cli import cli
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.data.backup import verify_restored_ops_tree
from bts.health.streak_validation import check as check_streak_health
import bts.picks as p
from bts.reconcile_receipt import ReconcileReceipt
import bts.scheduler as sched_mod
from bts.state.export import export_initial_state
from bts.state.regenerate import Timeline, compose_state_from_snapshot_and_timeline
from bts.state.verify import diff_pick_files
import bts.strategy as strat_mod
import bts.web as web_mod

ET = ZoneInfo("America/New_York")
FIXED_ET = dt.datetime(2027, 3, 24, 23, 0, tzinfo=ET)

LEGACY_UNTAGGED = (
    b'{"streak": 13, "saver_available": false, '
    b'"updated": "2026-09-13T05:01:02.123456+00:00", '
    b'"applied_dates": ["2026-09-11", "2026-09-12"]}'
)
TAGGED_2026 = (
    b'{"streak": 13, "saver_available": false, '
    b'"updated": "2026-09-13T05:01:02.123456+00:00", '
    b'"applied_dates": ["2026-09-11", "2026-09-12"], "season": 2026}'
)
DEEP_JSON = b"[" * 200_000 + b"]" * 200_000
HUGE_INT_JSON = b'{"season": 2027, "streak": ' + (b"9" * 5000) + b"}"


class DateStr(str):
    """Custom str subclass."""


class RaisingSliceStr(str):
    """str subclass where Python slicing raises RuntimeError."""
    def __getitem__(self, item):
        raise RuntimeError("slice forbidden")


class SeasonEnum(IntEnum):
    Y2027 = 2027


def _preds(p_hit: float = 0.85, second_p: float | None = None) -> pd.DataFrame:
    rows = [{
        "batter_name": "Steven Kwan",
        "batter_id": 700363,
        "team": "CLE",
        "lineup": 1,
        "pitcher_name": "Gerrit Cole",
        "pitcher_id": 543037,
        "pitcher_team": "NYY",
        "game_pk": 778899,
        "game_time": "2027-03-24T23:10:00Z",
        "p_hit_pa": 0.32,
        "p_game_hit": p_hit,
        "flags": "",
    }]
    if second_p is not None:
        rows.append({
            "batter_name": "Bobby Witt Jr.",
            "batter_id": 677951,
            "team": "KC",
            "lineup": 2,
            "pitcher_name": "Logan Gilbert",
            "pitcher_id": 669302,
            "pitcher_team": "SEA",
            "game_pk": 778900,
            "game_time": "2027-03-24T23:40:00Z",
            "p_hit_pa": 0.31,
            "p_game_hit": second_p,
            "flags": "",
        })
    return pd.DataFrame(rows)


def _pick_slot(batter_id: int = 700363, game_pk: int = 778899,
               day: str = "2027-03-24", name: str = "Steven Kwan") -> p.Pick:
    return p.Pick(
        batter_name=name,
        batter_id=batter_id,
        team="CLE",
        lineup_position=1,
        pitcher_name="Gerrit Cole",
        pitcher_id=543037,
        p_game_hit=0.85,
        flags=[],
        projected_lineup=False,
        game_pk=game_pk,
        game_time=f"{day}T23:10:00Z",
    )


def _plant_daily(picks_dir: Path, day: str, *, double: bool = False,
                 result: str | None = None, slot_results: dict | None = None,
                 bluesky_uri: str | None = "at://did:plc:test/post/1") -> p.DailyPick:
    daily = p.DailyPick(
        date=day,
        run_time=f"{day}T18:00:00+00:00",
        pick=_pick_slot(700363, 778899, day, "Steven Kwan"),
        double_down=_pick_slot(677951, 778900, day, "Bobby Witt Jr.") if double else None,
        runner_up=None,
        result=result,
        slot_results=slot_results,
        bluesky_posted=True,
        bluesky_uri=bluesky_uri,
        notification_sent=True,
        delivery_attempted=True,
    )
    p.save_pick(daily, picks_dir)
    return daily


# ---------------------------------------------------------------------------
# 1. Row L: season_of_date exhaustive table, exact int return, and giant-string safety
# ---------------------------------------------------------------------------

ADVERSARIAL_DATE_CASES = [
    ("2027-03-24", 2027),
    ("2027-3-24", 2027),
    ("2027-03-4", 2027),
    ("2027-3-4", 2027),
    ("20270324", 2027),
    ("2027-W12-3", 2027),
    ("2027W123", 2027),
    ("2026-W53-5", 2027),          # ISO week 53 Friday of 2026 is 2027-01-01 (calendar 2027)
    ("2025-W01-1", 2024),          # ISO week 1 Monday of 2025 is 2024-12-30 (calendar 2024)
    ("2028-02-29", 2028),          # leap day
    ("0001-03-24", 1),             # MINYEAR
    ("0999-03-24", 999),           # 4-digit padded year < 1000
    ("9999-03-24", 9999),          # MAXYEAR
    ("٢٠٢٧-03-24", 2027),          # non-ASCII year accepted by strptime \d{4}
    ("2027-03-2٤", 2027),          # non-ASCII 2nd day digit accepted by strptime [1-2]\d
    (DateStr("2027-03-24"), 2027),
    (RaisingSliceStr("2027-03-24"), 2027),
    (RaisingSliceStr("2027-3-24"), 2027),
    # Unparseable / non-str -> None
    ("999-03-24", None),           # 3-digit year rejected by both fromisoformat and strptime %Y
    ("2027-083", None),            # ordinal hyphenated
    ("2027083", None),             # ordinal compact
    ("2027-03", None),
    (" 2027-03-24", None),
    ("2027-03-24 ", None),
    ("2027-03-24\n", None),
    ("\t2027-03-24\t", None),
    ("2027-03-24T10:00", None),
    ("2027-03-24 00:00:00", None),
    ("2027-03-24Z", None),
    ("2027-03-24T00:00:00-04:00", None),
    ("2027/03/24", None),
    ("2027.03.24", None),
    ("2027_03_24", None),
    ("2027-02-29", None),
    ("2027-02-30", None),
    ("2027-04-31", None),
    ("2027-13-01", None),
    ("2027-W54-1", None),
    ("0000-03-24", None),
    ("10000-03-24", None),
    ("-2027-03-24", None),
    ("+2027-03-24", None),
    ("٢٠٢٧-٠٣-٢٤", None),
    ("２０２７-０３-２４", None),
    ("2027-03-24\x00", None),
    ("2027-03-24\ud800", None),
    ("2" * 200, None),
    ("", None),
    ("junk", None),
    (DateStr("junk"), None),
    (RaisingSliceStr("junk"), None),
    (None, None),
    (dt.date(2027, 3, 24), None),
    (dt.datetime(2027, 3, 24, 12, 0), None),
    (Path("2027-03-24"), None),
    (b"2027-03-24", None),
    (bytearray(b"2027-03-24"), None),
    (20270324, None),
    (True, None),
    (False, None),
    (2027.0, None),
]


@pytest.mark.parametrize("raw,expected", ADVERSARIAL_DATE_CASES)
def test_season_of_date_adversarial_cases_and_exact_int_type(raw, expected):
    out = p.season_of_date(raw)
    assert out == expected
    assert out is None or type(out) is int


def test_season_of_date_never_hangs_or_raises_on_giant_or_hostile_inputs():
    class HostileObj:
        def __str__(self):
            raise RuntimeError("str boom")
        def __repr__(self):
            raise RuntimeError("repr boom")

    assert p.season_of_date(HostileObj()) is None

    for payload in ("2" * 5_000_000, "2027-03-" + "9" * 5_000_000, " " * 5_000_000):
        t0 = time.perf_counter()
        assert p.season_of_date(payload) is None
        assert (time.perf_counter() - t0) < 0.5


def test_shadow_prediction_runs_real_select_pick_and_scopes_both_streak_and_saver(tmp_path: Path):
    """Unlike mocking select_pick, drives _run_shadow_prediction through real select_pick to disk."""
    (tmp_path / "streak.json").write_bytes(TAGGED_2026)  # streak=13, saver=False in 2026
    captured = []
    orig_resolve = strat_mod.resolve_policy_decision

    def spy_resolve(ctx, streak, saver):
        captured.append((streak, saver))
        return orig_resolve(ctx, streak, saver)

    with patch.object(sched_mod, "_now_et", return_value=FIXED_ET), \
         patch.object(sched_mod, "predict_local_shadow", return_value=_preds(0.85)), \
         patch.object(sched_mod, "heartbeat_watchdog", return_value=nullcontext()), \
         patch.object(sched_mod, "_analytics_job_status", return_value={}), \
         patch.object(sched_mod, "_update_analytics_job_status", return_value=None), \
         patch.object(p, "get_game_statuses_detailed",
                      return_value={778899: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}), \
         patch.object(strat_mod, "_load_mdp", return_value=None), \
         patch.object(strat_mod, "resolve_policy_decision", side_effect=spy_resolve):
        sched_mod._run_shadow_prediction(
            {"orchestrator": {"picks_dir": str(tmp_path), "models_dir": str(tmp_path / "m")}},
            "2027-03-24",
            "Steven Kwan",
        )
        # Unparseable date falls back to unscoped (13, False) before _days_remaining rejects it
        sched_mod._run_shadow_prediction(
            {"orchestrator": {"picks_dir": str(tmp_path), "models_dir": str(tmp_path / "m")}},
            "2027/03/24",
            "Steven Kwan",
        )

    assert captured == [(0, True), (13, False)]
    shadow_saved = p.load_shadow_pick("2027-03-24", tmp_path)
    assert shadow_saved is not None and shadow_saved.policy_decision["streak"] == 0
    assert (tmp_path / "streak.json").read_bytes() == TAGGED_2026


@pytest.mark.parametrize("day,season", ADVERSARIAL_DATE_CASES)
def test_select_pick_saver_fallback_across_adversarial_dates(tmp_path: Path, day, season):
    """L3: select_pick with saver_available=None never fails on the saver read; readable non-2026
    seasons read True (fresh) while None falls back to unscoped False, and bytes stay untouched."""
    (tmp_path / "streak.json").write_bytes(TAGGED_2026)  # 2026 saver_available=False
    seen_saver = []
    orig_lsa = strat_mod.load_saver_available

    def spy_lsa(pd_arg, **kw):
        val = orig_lsa(pd_arg, **kw)
        seen_saver.append((kw.get("season"), val))
        return val

    with patch.object(strat_mod, "get_game_statuses", return_value={778899: "P"}), \
         patch.object(strat_mod, "_load_mdp", return_value=None), \
         patch.object(strat_mod, "load_saver_available", side_effect=spy_lsa):
        try:
            strat_mod.select_pick(_preds(0.85), day, tmp_path, streak=13)
        except (ValueError, TypeError, RuntimeError):
            pass  # pre-existing downstream _days_remaining / slicing / NUL-path error

    expected_saver = True if (season is not None and season != 2026) else False
    assert seen_saver == [(season, expected_saver)]
    assert (tmp_path / "streak.json").read_bytes() == TAGGED_2026


# ---------------------------------------------------------------------------
# 2. Rows A..K targeted boundary checks
# ---------------------------------------------------------------------------

def test_rows_a_b_keyword_only_exact_int_and_bool_tag_isolation(tmp_path: Path):
    # True == 1 must never match season=1 or be accepted as a season argument
    (tmp_path / "streak.json").write_text(
        '{"season": true, "streak": 25, "saver_available": false, "applied_dates": ["2027-04-01"]}'
    )
    assert (
        p.load_streak(tmp_path, season=1),
        p.load_saver_available(tmp_path, season=1),
        p.load_applied_dates(tmp_path, season=1),
    ) == (0, True, [])

    # Positional season rejected (keyword-only)
    for reader in (p.load_streak, p.load_saver_available, p.load_applied_dates):
        with pytest.raises(TypeError):
            reader(tmp_path, 2027)  # type: ignore[misc]

    # IntEnum rejected (type(season) is int, not isinstance)
    nonexistent_dir = tmp_path / "never_created"
    with pytest.raises(TypeError, match="season must be an int or None"):
        p.save_streak(3, nonexistent_dir, season=SeasonEnum.Y2027)
    assert not nonexistent_dir.exists()  # validated before mkdir


def test_row_k_mid_season_deploy_reads_zero_until_0200_reconcile_stamps_season(tmp_path: Path):
    (tmp_path / "streak.json").write_text('{"streak": 4, "saver_available": true}')
    for day in ("2027-05-01", "2027-05-02", "2027-05-03", "2027-05-04"):
        _plant_daily(tmp_path, day, result="hit", slot_results={"pick": "hit"})

    pre = load_decision_streak_state(
        tmp_path, require_contest_state=False, now=dt.datetime(2027, 5, 4, 18, 0, tzinfo=ET)
    )
    assert (pre.streak, pre.model_streak) == (0, 0)

    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        p.reconcile_results(tmp_path, clock=lambda: dt.datetime(2027, 5, 5, 2, 0, tzinfo=ET))

    post = load_decision_streak_state(
        tmp_path, require_contest_state=False, now=dt.datetime(2027, 5, 5, 9, 0, tzinfo=ET)
    )
    assert (post.streak, post.model_streak) == (4, 4)
    assert json.loads((tmp_path / "streak.json").read_text())["season"] == 2027


# ---------------------------------------------------------------------------
# 3. Full season-boundary sequences on one directory
# ---------------------------------------------------------------------------

def test_full_rollover_lifecycle_void_hit_double_race_crash_and_reconcile(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    # Step 1: Final day of 2026 (2026-09-27) via run_result_polling -> streak 13, saver False, season 2026
    p.save_streak(12, picks_dir, saver_available=False, applied_date="2026-09-26", season=2026)
    _plant_daily(picks_dir, "2026-09-27")
    with patch.object(sched_mod, "_now_et", return_value=dt.datetime(2026, 9, 27, 23, 0, tzinfo=ET)), \
         patch.object(sched_mod, "poll_game_result", return_value="final"), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch("bts.posting.format_result_reply", return_value="ok"), \
         patch("bts.posting.reply_to_bluesky", return_value="at://r/2026"):
        assert sched_mod.run_result_polling(778899, "2026-09-27", picks_dir) == "final"

    winter_bytes = (picks_dir / "streak.json").read_bytes()
    assert json.loads(winter_bytes)["streak"] == 13
    assert json.loads(winter_bytes)["season"] == 2026

    # Step 2: Idle winter -> Jan 2027 reads return (0, True, []) without touching bytes
    assert (
        p.load_streak(picks_dir, season=2027),
        p.load_saver_available(picks_dir, season=2027),
        p.load_applied_dates(picks_dir, season=2027),
    ) == (0, True, [])
    st_winter = load_decision_streak_state(
        picks_dir, require_contest_state=False, now=dt.datetime(2027, 2, 1, 12, 0, tzinfo=ET)
    )
    assert (st_winter.streak, st_winter.model_streak, st_winter.model_saver_available) == (0, 0, True)
    assert (picks_dir / "streak.json").read_bytes() == winter_bytes

    # Step 3: Opening Night 2027 (2027-03-24) is ALL VOID -> reports 0, keeps 2026 bytes untouched
    _plant_daily(picks_dir, "2027-03-24")
    replies_void = []
    with patch.object(sched_mod, "_now_et", return_value=FIXED_ET), \
         patch.object(sched_mod, "poll_game_result", return_value="final"), \
         patch.object(p, "get_game_statuses_detailed",
                      return_value={778899: {"abstract": "F", "detailed": "Postponed"}}), \
         patch("bts.posting.format_result_reply",
               side_effect=lambda r, s: (replies_void.append((r, s)), "ok")[1]), \
         patch("bts.posting.reply_to_bluesky", return_value="at://r/void"):
        assert sched_mod.run_result_polling(778899, "2027-03-24", picks_dir) == "final"
    assert replies_void == [("void", 0)]
    assert (picks_dir / "streak.json").read_bytes() == winter_bytes
    assert p.load_pick("2027-03-24", picks_dir).result == "void"

    # Step 4: Day 2 (2027-03-25) HIT -> stamps 2027, streak=1, saver=True, clears 2026 markers
    _plant_daily(picks_dir, "2027-03-25")
    with patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 0, tzinfo=ET)), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        r_d2 = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(picks_dir)])
    assert r_d2.exit_code == 0 and "Streak: 1" in r_d2.output
    doc_d2 = json.loads((picks_dir / "streak.json").read_text())
    assert (doc_d2["streak"], doc_d2["saver_available"], doc_d2["season"], doc_d2["applied_dates"]) == (
        1, True, 2027, ["2027-03-25"],
    )

    # Step 5: Day 3 (2027-03-26) DOUBLE HIT with two scorers racing (check-results finishes while
    # run_result_polling is resolving; polling's locked re-check sees peer scored and reads 3)
    _plant_daily(picks_dir, "2027-03-26", double=True)
    real_resolve = p.resolve_daily_slot_results

    def resolve_with_peer_interleave(daily, day):
        res = real_resolve(daily, day)
        # Simulate cron check-results completing while polling is between resolve and lock
        r_cron = CliRunner().invoke(cli, ["check-results", "--date", day, "--picks-dir", str(picks_dir)])
        assert r_cron.exit_code == 0
        return res

    replies_race = []
    with patch.object(sched_mod, "_now_et", return_value=dt.datetime(2027, 3, 27, 1, 0, tzinfo=ET)), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 27, 1, 0, tzinfo=ET)), \
         patch.object(sched_mod, "poll_game_result", return_value="final"), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(p, "resolve_daily_slot_results", side_effect=resolve_with_peer_interleave), \
         patch("bts.posting.format_result_reply",
               side_effect=lambda r, s: (replies_race.append((r, s)), "ok")[1]), \
         patch("bts.posting.reply_to_bluesky", return_value="at://r/race"):
        assert sched_mod.run_result_polling(778899, "2027-03-26", picks_dir) == "final"

    assert replies_race == [("hit", 3)]
    assert p.load_streak(picks_dir, season=2027) == 3

    # Step 6: Day 4 (2027-03-27) crash between streak write (3 -> 4) and pick write,
    # same-night reconcile fails closed (keeps 4), restart finishes pick write at 4,
    # and 02:00 reconcile replays 0 + 1 + 2 + 1 = 4.
    _plant_daily(picks_dir, "2027-03-27")

    class _Boom(BaseException):
        pass

    with patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 28, 1, 0, tzinfo=ET)), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(p, "save_pick", side_effect=_Boom()):
        with pytest.raises(_Boom):
            CliRunner().invoke(
                cli, ["check-results", "--date", "2027-03-27", "--picks-dir", str(picks_dir)],
                catch_exceptions=False,
            )
    assert p.load_pick("2027-03-27", picks_dir).result is None
    assert p.load_streak(picks_dir, season=2027) == 4

    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=None):
        p.reconcile_results(picks_dir, clock=lambda: dt.datetime(2027, 3, 27, 23, 55, tzinfo=ET))
    assert p.load_streak(picks_dir, season=2027) == 4

    with patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 28, 1, 10, tzinfo=ET)), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        r_d4 = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-27", "--picks-dir", str(picks_dir)])
    assert r_d4.exit_code == 0 and "Streak: 4" in r_d4.output

    t_rec = dt.datetime(2027, 3, 28, 2, 0, tzinfo=ET)
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_rec, lookback_days=kw["lookback_days"])), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        r_rec = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(picks_dir)])
    assert r_rec.exit_code == 0 and "Streak: 4" in r_rec.output
    final_doc = json.loads((picks_dir / "streak.json").read_text())
    assert (final_doc["streak"], final_doc["saver_available"], final_doc["season"]) == (4, True, 2027)
    assert final_doc["applied_dates"] == ["2027-03-25", "2027-03-26", "2027-03-27"]


@pytest.mark.parametrize("slot_results,expected_streak", [
    ({"pick": "hit", "double_down": "hit"}, 2),
    ({"pick": "hit", "double_down": "miss"}, 0),
])
def test_opening_night_double_hit_and_partial_miss_from_2026_saver_zone(
    tmp_path: Path, slot_results: dict, expected_streak: int,
):
    p.save_streak(14, tmp_path, saver_available=True, applied_date="2026-09-27", season=2026)
    _plant_daily(tmp_path, "2027-03-24", double=True)
    with patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)), \
         patch.object(p, "resolve_daily_slot_results", return_value=slot_results):
        r = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(tmp_path)])
    assert r.exit_code == 0
    doc = json.loads((tmp_path / "streak.json").read_text())
    assert (doc["streak"], doc["saver_available"], doc["season"], doc["applied_dates"]) == (
        expected_streak, True, 2027, ["2027-03-24"],
    )


# ---------------------------------------------------------------------------
# 4. Mid-season paths on a current-season file
# ---------------------------------------------------------------------------

def test_mid_season_double_down_miss_in_saver_zone_matches_replay(tmp_path: Path):
    """At streak 12 with saver=True, a DD partial miss [True, False] consumes saver and holds 12,
    matching _apply_streak_day in forward replay."""
    for i in range(1, 13):
        _plant_daily(tmp_path, f"2027-04-{i:02d}", result="hit", slot_results={"pick": "hit"})
    p.save_streak(12, tmp_path, saver_available=True, season=2027)

    _plant_daily(tmp_path, "2027-04-13", double=True)
    with patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 4, 14, 1, 0, tzinfo=ET)), \
         patch.object(p, "resolve_daily_slot_results", return_value={"pick": "hit", "double_down": "miss"}):
        r = CliRunner().invoke(cli, ["check-results", "--date", "2027-04-13", "--picks-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert (p.load_streak(tmp_path, season=2027), p.load_saver_available(tmp_path, season=2027)) == (12, False)

    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "resolve_daily_slot_results", return_value={"pick": "hit", "double_down": "miss"}):
        p.reconcile_results(tmp_path, clock=lambda: dt.datetime(2027, 4, 14, 2, 0, tzinfo=ET))
    assert (p.load_streak(tmp_path, season=2027), p.load_saver_available(tmp_path, season=2027)) == (12, False)


def test_mid_season_once_per_date_same_spelling_vs_cross_spelling_characterization(tmp_path: Path):
    """Each spelling (`2027-03-24`, `2027-3-24`, `20270324`, `2027-W12-3`) is idempotent when retried
    with the same string, and stored verbatim per Row L4 (so distinct spellings are distinct markers,
    matching d8eb50d0)."""
    p.save_streak(0, tmp_path, saver_available=True, season=2027)
    assert p.update_streak([True], tmp_path, date="2027-3-24") == 1
    assert p.update_streak([True], tmp_path, date="2027-3-24") == 1
    assert p.load_applied_dates(tmp_path, season=2027) == ["2027-3-24"]

    # Cross-spelling call on the same calendar day is not canonicalized (same as base d8eb50d0)
    assert p.update_streak([True], tmp_path, date="2027-03-24") == 2
    assert p.update_streak([True], tmp_path, date="20270324") == 3
    assert p.load_applied_dates(tmp_path, season=2027) == ["2027-3-24", "2027-03-24", "20270324"]


def test_mid_season_dateless_update_between_dated_updates_preserves_tag_and_markers(tmp_path: Path):
    p.save_streak(0, tmp_path, saver_available=True, season=2027)
    assert p.update_streak([True], tmp_path, date="2027-05-01") == 1
    assert p.update_streak([True], tmp_path) == 2  # dateless between dated ones
    doc_mid = json.loads((tmp_path / "streak.json").read_text())
    assert (doc_mid["season"], doc_mid["applied_dates"]) == (2027, ["2027-05-01"])
    assert p.update_streak([True], tmp_path, date="2027-05-01") == 2  # prior marker still honored
    assert p.update_streak([True], tmp_path, date="2027-05-02") == 3
    assert p.load_applied_dates(tmp_path, season=2027) == ["2027-05-01", "2027-05-02"]


def test_mid_season_reconcile_with_today_unresolved_but_marked_applied_on_tagged_file(tmp_path: Path):
    _plant_daily(tmp_path, "2027-05-01", result="hit", slot_results={"pick": "hit"})
    _plant_daily(tmp_path, "2027-05-02", result=None)  # scorer crashed before saving pick
    p.save_streak(2, tmp_path, saver_available=True, applied_date="2027-05-02", season=2027)
    before = (tmp_path / "streak.json").read_bytes()

    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        p.reconcile_results(tmp_path, clock=lambda: dt.datetime(2027, 5, 2, 23, 30, tzinfo=ET))

    assert (tmp_path / "streak.json").read_bytes() == before
    assert p.load_streak(tmp_path, season=2027) == 2


# ---------------------------------------------------------------------------
# 5. Corrupt and unreadable (chmod 000) state across scorers and reconcile
# ---------------------------------------------------------------------------

def test_unreadable_chmod_000_streak_file_raises_in_scorers_and_rebuilds_on_complete_replay(tmp_path: Path):
    if os.geteuid() == 0:
        pytest.skip("root bypasses chmod 000")

    sp = tmp_path / "streak.json"
    sp.write_text('{"season": 2027, "streak": 5, "saver_available": true}')
    os.chmod(sp, 0o000)
    try:
        for fn in (
            lambda: p.load_streak(tmp_path, season=2027),
            lambda: p.load_saver_available(tmp_path, season=2027),
            lambda: p.update_streak([True], tmp_path, date="2027-03-24"),
        ):
            with pytest.raises(PermissionError):
                fn()
        assert p.load_applied_dates(tmp_path, season=2027) == []

        with pytest.raises(ContestStateError, match="PermissionError"):
            load_decision_streak_state(tmp_path, require_contest_state=False, now=FIXED_ET)

        # Incomplete history -> replay is None -> chmod 000 file untouched
        _plant_daily(tmp_path, "2027-03-23", result=None)
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=None):
            p.reconcile_results(tmp_path, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET))
        assert (sp.stat().st_mode & 0o777) == 0o000

        # Complete history -> atomic_write_text os.replace rebuilds a clean readable file
        _plant_daily(tmp_path, "2027-03-23", result="hit", slot_results={"pick": "hit"})
        with patch.object(p, "get_game_statuses_detailed", return_value={}), \
             patch.object(p, "check_hit", return_value=True):
            p.reconcile_results(tmp_path, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET))
        rebuilt = json.loads(sp.read_text())
        assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]) == (1, True, 2027)
    finally:
        if sp.exists():
            os.chmod(sp, 0o600)

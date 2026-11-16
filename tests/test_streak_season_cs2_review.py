# Adopted from the Carbon review of change set 2, round 1 (marker CS2RG-172d8223); two tests that
# pinned the corrupt-file overwrite were changed to the amended contract (marked in their docstrings).
# Source: docs/reviews/2026-11-14-season-rollover/build/cs2-review/carbon-r1/test_cs2_adversarial.py
"""Adversarial regression probes for CS2 (season tag on streak.json).

Designed to be adopted directly into `tests/` once B1 (`RecursionError` in
`picks._load_streak_season`) is fixed. Exercises:
- Rows 1 to 12 with real files on disk and real CLI / scheduler / strategy / contest_state calls
- Full multi-step 2026 -> 2027 rollover sequences on one directory (hit, miss from 2026 saver
  zone, void, double, racing scorers under scoring_lock, crash/restart between streak write and
  pick write, and 02:00 reconcile)
- Mid-season invariants on 2027-tagged state (saver boundaries 9/10/12/15/16, second miss after
  spent saver, once-per-date rule, pp1 B1 on tagged and tagless files, dateless updates/saves)
- Year/date/type boundaries (2026-12-31 -> 2027-01-01, UTC/ET New Year reconcile summary,
  malformed/non-string dates, non-int season tags)
- B1 blocker: deeply nested non-object JSON (`DEEP`) in `streak.json` is corrupt: the seasonal
  readers raise as the dateless ones do (the amended contract), and `reconcile_results` still
  repairs `streak.json`
"""
from __future__ import annotations

from contextlib import nullcontext
import datetime as dt
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
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
from bts.state.verify import diff_pick_files
import bts.strategy as strat_mod
import bts.web as web_mod
from tests.test_strategy import _predictions

ET = ZoneInfo("America/New_York")
LEGACY_2026 = (
    b'{"streak": 13, "saver_available": false, '
    b'"updated": "2026-09-13T05:01:02.123456+00:00", '
    b'"applied_dates": ["2026-09-11", "2026-09-12"]}'
)
DEEP = b"[" * 200_000 + b"]" * 200_000


def _pick(batter_id: int = 700363, game_pk: int = 778899,
          day: str = "2027-03-24", name: str = "Steven Kwan") -> p.Pick:
    return p.Pick(
        batter_name=name,
        batter_id=batter_id,
        team="CLE",
        lineup_position=1,
        pitcher_name="Gerrit Cole",
        pitcher_id=543037,
        p_game_hit=0.82,
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
        pick=_pick(700363, 778899, day, "Steven Kwan"),
        double_down=_pick(666126, 778900, day, "Bobby Witt Jr.") if double else None,
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
# B1 BLOCKER PROBE (Rows 1, 2, 5, 6, 7, 8, 9): RecursionError on DEEP JSON
# ---------------------------------------------------------------------------

def test_b1_deeply_nested_streak_json_isolated_by_season_readers_and_reconcile(tmp_path: Path):
    """Changed by the lead after round 1 (the contract's amendment: corrupt state is an error, not
    "another season"). A deeply nested body is corrupt: the season-scoped streak and saver reads
    raise as the dateless ones do, the marker read returns [] as always, nothing is written, and
    reconcile_results still repairs the file (as d8eb50d0 did; the reviewer's B1)."""
    streak_path = tmp_path / "streak.json"
    streak_path.write_bytes(DEEP)

    for loader in (p.load_streak, p.load_saver_available):
        with pytest.raises(RecursionError):
            loader(tmp_path, season=2027)
        with pytest.raises(RecursionError):
            loader(tmp_path)
    assert p.load_applied_dates(tmp_path, season=2027) == []
    assert streak_path.read_bytes() == DEEP

    _plant_daily(tmp_path, "2027-03-24", result="hit", slot_results={"pick": "hit"})
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        corrections = p.reconcile_results(
            tmp_path, clock=lambda: dt.datetime(2027, 3, 25, 2, 0, tzinfo=ET),
        )
    assert corrections == []
    repaired = json.loads(streak_path.read_text(encoding="utf-8"))
    assert (repaired["streak"], repaired["saver_available"], repaired["season"]) == (1, True, 2027)


# ---------------------------------------------------------------------------
# Item 2: Multi-step sequences across 2026 -> 2027 on one directory
# ---------------------------------------------------------------------------

def test_sequence_2026_to_2027_hit_racing_scorers_double_crash_and_reconcile(tmp_path: Path):
    """End-to-end 2026 -> idle winter -> Opening Night 2027 hit -> second scorer ->
    Day 2 double crash between streak write and pick write -> restart -> 02:00 reconcile."""
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    # 1. Final day of 2026 (at streak 12, saver already spent -> hit makes 13, season=2026)
    p.save_streak(12, picks_dir, saver_available=False, applied_date="2026-09-26", season=2026)
    _plant_daily(picks_dir, "2026-09-27")
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2026, 9, 28, 1, 0, tzinfo=ET)):
        r_2026 = CliRunner().invoke(cli, ["check-results", "--date", "2026-09-27", "--picks-dir", str(picks_dir)])
    assert r_2026.exit_code == 0 and "Streak: 13" in r_2026.output
    winter_bytes = (picks_dir / "streak.json").read_bytes()
    winter_doc = json.loads(winter_bytes)
    assert (winter_doc["streak"], winter_doc["saver_available"], winter_doc["season"]) == (13, False, 2026)

    # 2. Idle winter -> Opening Night 2027 pre-game readers (Rows 8, 9, 10, 11)
    seen_shadow: list[int] = []
    with patch.object(sched_mod, "predict_local_shadow", return_value=object()), \
         patch.object(sched_mod, "heartbeat_watchdog", return_value=nullcontext()), \
         patch.object(sched_mod, "_analytics_job_status", return_value={}), \
         patch.object(sched_mod, "_update_analytics_job_status", return_value=None), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(sched_mod, "select_pick",
                      side_effect=lambda *a, **kw: (seen_shadow.append(kw["streak"]), SimpleNamespace(pick_result=None))[1]):
        sched_mod._run_shadow_prediction({"orchestrator": {"picks_dir": str(picks_dir)}}, "2027-03-24", "Prod")
    assert seen_shadow == [0]

    preds = _predictions([{"p_game_hit": 0.85}])
    with patch.object(strat_mod, "get_game_statuses", return_value={778899: "P"}), \
         patch.object(strat_mod, "_load_mdp", return_value=None), \
         patch.object(strat_mod, "resolve_policy_decision", wraps=strat_mod.resolve_policy_decision) as spy_decide:
        strat_mod.select_pick(preds, "2027-03-24", picks_dir, streak=0)
    assert spy_decide.call_args.args[2] is True  # 2026 spent saver does not carry into 2027

    st_pre = load_decision_streak_state(
        picks_dir, require_contest_state=False, now=dt.datetime(2027, 3, 24, 18, 0, tzinfo=ET),
    )
    assert (st_pre.streak, st_pre.model_streak, st_pre.model_saver_available) == (0, 0, True)
    assert (picks_dir / "streak.json").read_bytes() == winter_bytes  # read-only before first 2027 write

    # 3. Opening Night 2027 HIT via scheduler.run_result_polling (mid-game early reply + final)
    _plant_daily(picks_dir, "2027-03-24")
    replies: list[tuple[str, int]] = []
    polls = iter(["live", "final"])
    with patch.object(sched_mod, "_now_et", return_value=dt.datetime(2027, 3, 24, 21, 30, tzinfo=ET)), \
         patch.object(sched_mod, "poll_game_result", side_effect=lambda *a: next(polls)), \
         patch.object(sched_mod, "_poll_interval_sleep", return_value=None), \
         patch.object(sched_mod, "_check_hits_midgame", return_value=[True]), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch("bts.posting.format_result_reply", side_effect=lambda res, s: (replies.append((res, s)), f"{res}:{s}")[1]), \
         patch("bts.posting.reply_to_bluesky", return_value="at://reply/1"):
        assert sched_mod.run_result_polling(778899, "2027-03-24", picks_dir) == "final"
    assert replies == [("hit", 1)]
    d_night1 = json.loads((picks_dir / "streak.json").read_text())
    assert (d_night1["streak"], d_night1["saver_available"], d_night1["season"], d_night1["applied_dates"]) == (
        1, True, 2027, ["2027-03-24"],
    )
    # Note: mid-game hit sets daily.result="hit" without slot_results; 01:00 cron sees already resolved
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
        r_second = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(picks_dir)])
    assert r_second.exit_code == 0 and "Already resolved" in r_second.output
    assert p.load_streak(picks_dir, season=2027) == 1

    # 4. Day 2 (2027-03-25) DOUBLE hit: crash between streak write and pick write, then restart
    _plant_daily(picks_dir, "2027-03-25", double=True)

    class _Crash(BaseException):
        pass

    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 0, tzinfo=ET)), \
         patch.object(p, "save_pick", side_effect=_Crash("died after streak write")):
        with pytest.raises(_Crash):
            CliRunner().invoke(
                cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(picks_dir)],
                catch_exceptions=False,
            )
    assert p.load_pick("2027-03-25", picks_dir).result is None
    assert p.load_streak(picks_dir, season=2027) == 3

    # Same-night reconcile before restart must respect pp1 B1 and NOT roll 3 back to 1
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=None):
        p.reconcile_results(picks_dir, clock=lambda: dt.datetime(2027, 3, 25, 23, 50, tzinfo=ET))
    assert p.load_streak(picks_dir, season=2027) == 3

    # Restarted scorer finishes pick write without double-counting
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 5, tzinfo=ET)):
        r_retry = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(picks_dir)])
    assert r_retry.exit_code == 0 and "Streak: 3" in r_retry.output
    assert p.load_pick("2027-03-25", picks_dir).result == "hit"

    # 5. 02:00 reconcile via CLI confirms forward replay = 3 (1 + 2), saver True, season 2027
    t_rec = dt.datetime(2027, 3, 26, 2, 0, tzinfo=ET)
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_rec, lookback_days=kw["lookback_days"])), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        r_rec = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(picks_dir)])
    assert r_rec.exit_code == 0 and "Streak: 3" in r_rec.output
    final_doc = json.loads((picks_dir / "streak.json").read_text())
    assert (final_doc["streak"], final_doc["saver_available"], final_doc["season"], final_doc["applied_dates"]) == (
        3, True, 2027, ["2027-03-24", "2027-03-25"],
    )


def test_sequence_opening_night_miss_does_not_trigger_2026_saver_zone(tmp_path: Path):
    """At d8eb50d0, ending 2026 at streak=12, saver_available=True caused an Opening Night
    2027 miss to hold streak 12 and burn the saver! After CS2 it must record (0, True, 2027)."""
    p.save_streak(12, tmp_path, saver_available=True, applied_date="2026-09-27", season=2026)
    _plant_daily(tmp_path, "2027-03-24")
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=False), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
        res = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(tmp_path)])
    assert res.exit_code == 0 and "Streak reset to 0" in res.output
    doc = json.loads((tmp_path / "streak.json").read_text())
    assert (doc["streak"], doc["saver_available"], doc["season"], doc["applied_dates"]) == (
        0, True, 2027, ["2027-03-24"],
    )


def test_sequence_opening_night_void_keeps_legacy_file_untouched_then_day2_stamps(tmp_path: Path):
    (tmp_path / "streak.json").write_bytes(LEGACY_2026)
    _plant_daily(tmp_path, "2027-03-24")
    with patch.object(p, "get_game_statuses_detailed",
                      return_value={778899: {"abstract": "F", "detailed": "Postponed"}}), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
        r_void = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(tmp_path)])
    assert r_void.exit_code == 0 and "Streak unchanged: 0" in r_void.output
    assert (tmp_path / "streak.json").read_bytes() == LEGACY_2026

    _plant_daily(tmp_path, "2027-03-25")
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 26, 1, 0, tzinfo=ET)):
        r_hit = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-25", "--picks-dir", str(tmp_path)])
    assert r_hit.exit_code == 0 and "Streak: 1" in r_hit.output
    doc = json.loads((tmp_path / "streak.json").read_text())
    assert (doc["streak"], doc["saver_available"], doc["season"], doc["applied_dates"]) == (
        1, True, 2027, ["2027-03-25"],
    )


# ---------------------------------------------------------------------------
# Item 3: Mid-season paths on current-season tagged file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prior_streak,expected_streak,expected_saver", [
    (9, 0, True),
    (10, 10, False),
    (12, 12, False),
    (15, 15, False),
    (16, 0, True),
])
def test_mid_season_saver_exact_boundaries_on_tagged_file(
    tmp_path: Path, prior_streak: int, expected_streak: int, expected_saver: bool,
):
    p.save_streak(prior_streak, tmp_path, saver_available=True, season=2027)
    assert p.update_streak([False], tmp_path, date="2027-05-10") == expected_streak
    assert p.load_streak(tmp_path, season=2027) == expected_streak
    assert p.load_saver_available(tmp_path, season=2027) is expected_saver
    assert json.loads((tmp_path / "streak.json").read_text())["season"] == 2027


def test_mid_season_second_miss_after_spent_saver_resets_to_zero(tmp_path: Path):
    p.save_streak(12, tmp_path, saver_available=True, season=2027)
    assert p.update_streak([False], tmp_path, date="2027-05-10") == 12
    assert p.update_streak([False], tmp_path, date="2027-05-10") == 12  # same date idempotent
    assert p.update_streak([False], tmp_path, date="2027-05-11") == 0   # next date resets
    assert (p.load_streak(tmp_path, season=2027), p.load_saver_available(tmp_path, season=2027)) == (0, False)


# ---------------------------------------------------------------------------
# Item 4: Corrupt streak.json during scoring (characterizes CS2 behavior)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [b"{", b"\xff", b"[]"])
def test_corrupt_streak_json_stops_the_scorer_and_is_left_intact(tmp_path: Path, payload: bytes):
    """Changed by the lead after round 1. The reviewer's probe pinned what the code did at 3b4196be:
    a hit OVERWROTE the corrupt file with streak 1 and health went quiet. Both reviewers judged that
    unsafe and the contract was amended: the scorer fails, as on main, on a hit and on a void day,
    the file's bytes are preserved, the pick stays unresolved (retryable), and health still sees it."""
    d_void = tmp_path / "void"
    d_void.mkdir()
    _plant_daily(d_void, "2027-03-24")
    (d_void / "streak.json").write_bytes(payload)
    with patch.object(p, "get_game_statuses_detailed",
                      return_value={778899: {"abstract": "F", "detailed": "Postponed"}}), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
        rv = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d_void)])
    assert rv.exit_code != 0
    assert (d_void / "streak.json").read_bytes() == payload

    d_hit = tmp_path / "hit"
    d_hit.mkdir()
    _plant_daily(d_hit, "2027-03-24")
    (d_hit / "streak.json").write_bytes(payload)
    with patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True), \
         patch.object(cli_mod, "_now_et", return_value=dt.datetime(2027, 3, 25, 1, 0, tzinfo=ET)):
        rh = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d_hit)])
    assert rh.exit_code != 0
    assert (d_hit / "streak.json").read_bytes() == payload
    assert json.loads((d_hit / "2027-03-24.json").read_text()).get("result") not in ("hit", "miss", "void")
    if payload == b"{":
        assert [a.level for a in check_streak_health(d_hit)] == ["CRITICAL"]
    else:   # not UTF-8 / not an object: the check itself raises, which the health runner reports as CRITICAL
        with pytest.raises(Exception):
            check_streak_health(d_hit)


# ---------------------------------------------------------------------------
# Item 5 & Row 12: Year/date boundaries and untouched backup/web/export paths
# ---------------------------------------------------------------------------

def test_contest_dates_dec31_to_jan01_boundary(tmp_path: Path):
    p.save_streak(11, tmp_path, saver_available=False, season=2026)
    assert p.update_streak([True], tmp_path, date="2026-12-31") == 12
    assert json.loads((tmp_path / "streak.json").read_text())["season"] == 2026
    assert p.update_streak([True], tmp_path, date="2027-01-01") == 1
    doc = json.loads((tmp_path / "streak.json").read_text())
    assert (doc["streak"], doc["saver_available"], doc["season"], doc["applied_dates"]) == (
        1, True, 2027, ["2027-01-01"],
    )


def test_reconcile_summary_end_to_end_at_utc_et_new_year_boundary(tmp_path: Path):
    """Unlike mocking reconcile_results, runs the real reconcile_results + cli summary at
    2027-01-01T01:00Z (2026-12-31 20:00 ET) and 2027-01-01T05:05Z (2027-01-01 00:05 ET)."""
    _plant_daily(tmp_path, "2026-12-30", result="hit", slot_results={"pick": "hit"})
    p.save_streak(1, tmp_path, saver_available=True, season=2026)

    t_2026_et = dt.datetime(2027, 1, 1, 1, 0, tzinfo=dt.timezone.utc)  # Dec 31 20:00 ET
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_2026_et, lookback_days=kw["lookback_days"])), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        r1 = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(tmp_path)])
    assert r1.exit_code == 0 and "Streak: 1" in r1.output

    t_2027_et = dt.datetime(2027, 1, 1, 5, 5, tzinfo=dt.timezone.utc)  # Jan 1 00:05 ET
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_2027_et, lookback_days=kw["lookback_days"])), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "check_hit", return_value=True):
        r2 = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(tmp_path)])
    assert r2.exit_code == 0 and "Streak: 0" in r2.output
    assert json.loads((tmp_path / "streak.json").read_text())["season"] == 2027


def test_row12_backup_verify_and_export_accept_season_tagged_streak(tmp_path: Path):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    p.save_streak(8, picks_dir, saver_available=False, applied_date="2027-04-10", season=2027)
    _plant_daily(picks_dir, "2027-04-10", result="hit", slot_results={"pick": "hit"})
    (picks_dir / "account_state").mkdir()
    (picks_dir / "account_state" / "saver_state.json").write_text(json.dumps({"state": "used", "season": 2027}))
    (picks_dir / "account_state" / "contest_ledger.jsonl").write_text('{"date":"2027-04-10"}\n')
    (picks_dir / "decisions" / "2027-04-10").mkdir(parents=True)
    (picks_dir / "decisions" / "2027-04-10" / "decision.json").write_text(
        json.dumps({"date": "2027-04-10", "action": "single"})
    )

    assert check_streak_health(picks_dir) == []
    assert verify_restored_ops_tree(tmp_path) == []
    exported = export_initial_state(picks_dir, tmp_path / "state.json")
    assert (exported["streak_at_cutoff"], exported["saver_available"]) == (8, False)

    regen_dir = tmp_path / "regen"
    regen_dir.mkdir()
    _plant_daily(regen_dir, "2027-04-10", result="hit", slot_results={"pick": "hit"})
    (regen_dir / "streak.json").write_text(json.dumps({"streak": 8, "saver_available": False}))
    assert diff_pick_files(picks_dir, regen_dir).is_clean
    with patch.object(web_mod, "PICKS_DIR", picks_dir):
        assert web_mod.load_streak() == 8

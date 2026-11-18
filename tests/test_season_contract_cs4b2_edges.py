"""Interrupted prefixes, filesystem boundaries and live caller controls."""
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

import bts.cli as cli
import bts.contest_state as cs
import bts.season_contract as sc
import bts.util as util
from tests.test_season_contract_cs4b2 import automatic, dated, live
from tests.test_season_contract_cs4b1 import (
    ID, NOW, ET, contract, fetch, files, offline, pin, put, run, tmp_path,
)


@pytest.mark.parametrize("point", ["before_switch", "after_switch", "after_manual", "after_saver", "printing"])
def test_interrupted_prefixes_and_retry(tmp_path, monkeypatch, point):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.manual.json", dated())
    put(tmp_path, "saver_state.json", dict(state="used", season=2027))
    protected = [put(tmp_path, n, {"evidence": n}) for n in
                 ["contest_streak.json", "contest_ledger.jsonl"]]
    status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
    status.parent.mkdir()
    status.write_text("fetch evidence")
    protected.append(status)
    before = {p: p.read_bytes() for p in protected}
    writer, mover = util.atomic_write_text, os.rename
    events = []
    class Interrupted(BaseException):
        pass
    def write(path, text):
        events.append("switch")
        if point == "before_switch":
            raise Interrupted()
        writer(path, text)
        if point == "after_switch":
            raise Interrupted()
    def move(src, dst):
        kind = "manual" if src.name == "contest_streak.manual.json" else "saver"
        events.append(kind)
        mover(src, dst)
        if point == "after_" + kind:
            raise Interrupted()
    with monkeypatch.context() as patcher:
        patcher.setattr(util, "atomic_write_text", write)
        patcher.setattr(os, "rename", move)
        if point == "printing":
            patcher.setattr("click.echo", Mock(side_effect=Interrupted()))
        with pytest.raises(Interrupted):
            run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "restore")
    assert events == {"before_switch": ["switch"], "after_switch": ["switch"],
                      "after_manual": ["switch", "manual"],
                      "after_saver": ["switch", "manual", "saver"],
                      "printing": ["switch", "manual", "saver"]}[point]
    assert sc.switch_state(tmp_path, NOW).state == ("automatic" if point == "before_switch" else "revoked")
    assert (tmp_path / "account_state/contest_streak.manual.json").exists() == (point in {"before_switch", "after_switch"})
    assert (tmp_path / "account_state/saver_state.json").exists() == (point in {"before_switch", "after_switch", "after_manual"})
    for p, content in before.items():
        assert p.read_bytes() == content
    retry = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "retry")
    assert retry.exit_code == 0, retry.output
    for p, content in before.items():
        assert p.read_bytes() == content


@pytest.mark.parametrize("source", ["manual", "saver"])
@pytest.mark.parametrize("target", [False, True])
def test_lstat_errors_are_not_absence(tmp_path, monkeypatch, source, target):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.manual.json", dated())
    put(tmp_path, "saver_state.json", dict(state="used", season=2027))
    original = Path.lstat
    filename = "contest_streak.manual" if source == "manual" else "saver_state"
    def stat(path, *a, **kw):
        if path.name.startswith(filename) and (".revoked-" in path.name) == target:
            raise PermissionError("injected lstat")
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "lstat", stat)
    result = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "x")
    assert result.exit_code == 1
    assert "injected lstat" in result.output
    assert ("saver flag was not attempted" if source == "manual" else "manual file outcome") in result.output


def test_saver_failure_reports_absent_manual(tmp_path, monkeypatch):
    put(tmp_path, "saver_state.json", dict(state="used", season=2027))
    monkeypatch.setattr(os, "rename", Mock(side_effect=PermissionError("folder is read-only")))
    result = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "x")
    assert result.exit_code == 1
    assert "manual file outcome:" in result.output and "manual.json: absent" in result.output
    assert "saver flag was not set aside" in result.output


def test_checked_rename_named_race_limit(tmp_path, monkeypatch):
    """R31 expressly accepts replacement if another writer wins this narrow race."""
    source = put(tmp_path, "contest_streak.manual.json", dated())
    content = source.read_bytes()
    original = os.rename
    def racing(src, dst):
        dst.write_text("created after the target lstat")
        original(src, dst)
    monkeypatch.setattr(os, "rename", racing)
    assert run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "x").exit_code == 0
    archive = source.with_name("contest_streak.manual.revoked-20270715T160000000000Z.json")
    assert archive.read_bytes() == content


def test_manual_filename_wins_over_claimed_schema_and_source(tmp_path):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.manual.json", automatic(source="misleading automatic"))
    with pytest.raises(cs.ContestStateError, match="legacy"):
        live(tmp_path)
    assert cs.load_decision_streak_state(tmp_path, now=NOW).streak == 12


def test_start_of_season_then_new_year(tmp_path, monkeypatch):
    put(tmp_path, "contest_streak.json", automatic(schema_version="bts_contest_streak_auto_v1", source_date="2026-09-30"))
    put(tmp_path, "saver_state.json", dict(state="active", season=2026))
    with pytest.raises(cs.ContestStateError, match="absent"):
        live(tmp_path)
    assert run(tmp_path, "set-contest-streak", "--date", "2027-07-15", "--streak", "0",
               "--best-streak", "0", "--saver", "not_earned").exit_code == 0
    assert live(tmp_path).streak == 0
    tomorrow = NOW + dt.timedelta(days=1)
    pin(monkeypatch, tomorrow)
    with pytest.raises(cs.ContestStateError, match="absent"):
        live(tmp_path, tomorrow)
    assert run(tmp_path, "season-contract", "enable", "--season", "2027").exit_code == 0
    with pytest.raises(cs.ContestStateError, match="schema_version"):
        live(tmp_path, tomorrow)
    assert fetch(tmp_path).exit_code == 0
    assert live(tmp_path, tomorrow).streak == 7
    january = NOW.replace(year=2028, month=1, day=1)
    pin(monkeypatch, january)
    with pytest.raises(cs.ContestStateError, match="other_season"):
        live(tmp_path, january)
    assert run(tmp_path, "season-contract", "enable", "--season", "2028").exit_code == 0
    with pytest.raises(cs.ContestStateError, match="season"):
        live(tmp_path, january)
    assert fetch(tmp_path).exit_code == 0
    assert live(tmp_path, january).streak == 7


def daily():
    from bts.picks import DailyPick, Pick
    return DailyPick(date="2027-07-15", run_time=NOW.isoformat(), pick=Pick(
        batter_name="Offline", batter_id=1, team="NYM", lineup_position=1, pitcher_name="P",
        pitcher_id=2, p_game_hit=.8, flags=[], projected_lineup=False, game_pk=999,
        game_time="2027-07-15T23:05:00Z"), double_down=None, runner_up=None)


@pytest.mark.parametrize("mode", ["refused", "after_read", "delivered", "private", "crash"])
def test_delivery_snapshot_and_guards(tmp_path, monkeypatch, mode, stamp_saved_pick):
    import bts.scheduler as sch
    from bts.picks import save_pick
    config = dict(orchestrator=dict(picks_dir=str(tmp_path)),
                  scheduler=dict(pick_delivery="private" if mode == "private" else "dm", contest_state_required=True),
                  bluesky=dict(dm_recipient="offline.test"))
    d = daily()
    if mode == "delivered":
        d.notification_sent, d.notification_channel = True, "bluesky_dm"
        d.notification_id, d.delivered_at = "captured", NOW.isoformat()
    elif mode == "crash":
        d.delivery_attempted = True
    state = sch.SchedulerState(date=d.date, schedule_fetched_at="t", games=[], confirmed_game_pks=[],
            runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None)
    monkeypatch.setattr(sch, "_now_et", lambda: NOW)
    monkeypatch.setattr(sch, "_void_committed_games", lambda *a: {})
    monkeypatch.setattr(sch, "_trigger_live_forward_capture_on_lock", lambda *a: None)
    alerts, sends, reads = [], [], []
    monkeypatch.setattr(sch, "_alert_contest_state_failure", lambda *a: alerts.append(str(a[-1])))
    monkeypatch.setattr("bts.dm.send_dm", lambda *a: sends.append(a) or "captured")
    put(tmp_path, "contest_streak.json", automatic())
    put(tmp_path, "season_contract.json", contract())
    original = cs.load_decision_streak_state
    if mode == "after_read":
        stamp_saved_pick(d, original(tmp_path, now=NOW, require_contest_state=True))

    def reading(*a, **kw):
        reads.append(kw)
        if mode != "after_read":
            raise cs.ContestStateError("injected new refusal")
        answer = original(*a, now=NOW, **kw)
        put(tmp_path, "season_contract.json", contract(status="revoked", revoked_at=NOW.isoformat()))
        return answer
    monkeypatch.setattr(cs, "load_decision_streak_state", reading)
    save_pick(d, tmp_path)
    result = sch._deliver_and_lock_pick(d, config, tmp_path, state, d.date, mode)
    assert result == (mode in {"after_read", "delivered", "private"})
    assert len(reads) == (1 if mode in {"refused", "after_read"} else 0)
    if mode == "refused":
        assert alerts == ["injected new refusal"] and not sends and not d.delivery_attempted
    if mode == "after_read":
        assert sends and d.notification_sent


def test_captured_skip_and_grading_ignore_refusal(tmp_path, monkeypatch):
    import bts.scheduler as sch
    import bts.picks as picks
    from bts.daily_decision import write_decision
    monkeypatch.setattr(cs, "load_decision_streak_state", Mock(side_effect=AssertionError("grading read decision")))
    d = daily()
    d.notification_sent, d.notification_channel = True, "bluesky_dm"
    d.notification_id, d.delivered_at = "captured", NOW.isoformat()
    picks.save_pick(d, tmp_path)
    write_decision(d.date, tmp_path, action="single", source="mdp", streak=7, saver_available=False,
                   state_source="contest", state_status="fresh", delivery_status="delivered", scoreable=True)
    monkeypatch.setattr(cli, "_now_et", lambda: NOW)
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *a: {"primary": "hit"})
    result = run(tmp_path, "check-results", "--date", d.date)
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "streak.json").read_text())["streak"] == 1


def test_real_selection_fallbacks_orchestrator_and_clean_outcome(tmp_path, monkeypatch):
    """Retain the reviewed real caller probes as GREEN error-handling controls."""
    import subprocess
    import sys
    # The historical probes install an irreversible network audit hook. Keep
    # them in a child so later tests keep their own transport mocking semantics.
    result = subprocess.run([
        sys.executable, "-c",
        "from pathlib import Path; import sys; "
        "from tests.test_season_contract_cs4b2_edges import _caller_control; "
        "_caller_control(Path(sys.argv[1]), Path(sys.argv[2]))",
        str(Path(__file__).resolve().parents[1]), str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _caller_control(root, directory):
    import sys
    from unittest.mock import patch
    review = root / "docs/reviews/2026-11-14-season-rollover/build/cs4b2-check"
    sys.path.insert(0, str(review))
    import probe as p
    import edges as e
    p.ROOT, p.REPO, p.OUT, e.OUT = directory, root, {}, {}
    # Scheduler late-delivery attention uses its own bound transport.
    with patch("bts.health.alert.send_dm", lambda *a: None), patch.object(p.dt, "datetime", p.Clock):
        p.current_callers()
        e.scheduler_fallbacks()
        e.grading_and_outcome()
    out = p.OUT
    assert out["run_and_pick_injected"]["error"] == "ContestStateError"
    assert all(v["error"] == "ContestStateError" for v in out["later_orchestrator_refusals"].values())
    assert out["prior_callers_reexecuted"]["orchestrate_initial_refusal"] is False
    assert out["prior_callers_reexecuted"]["scheduler_alerts"]
    assert out["eod_skip_under_injection"]["action"] == "skip"
    run = out["bts_run_under_required_only_injection"]
    assert run["exit"] == 1, run
    assert run["output"].splitlines()[-1] == "bts run: no pick: injected NEW automatic refusal"
    assert out["extra_caller_posts"] == []
    for label in ["in_loop", "final"]:
        case = e.OUT["fallback_" + label]
        assert case["alerts"] and case["calls"] and not case["sends"]
        assert not case["state"]["pick_locked"]
    assert e.OUT["polling_streak"]["streak"] == 1
    assert e.OUT["clean_day_outcome"]["exit"] == 0

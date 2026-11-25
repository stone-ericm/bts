"""Contract E operator messages and their offline entry points."""

import ast
import html
import inspect
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from bts import contest_state, orchestrator, picks, scheduler, strategy, web
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_morning_state_cs6b1 import warning
from tests.test_scheduler_tail_objective import _sel, _state, _tail_skip_decision
from tests.test_stale_delivery_cs4c import envelope
from tests.test_stale_delivery_cs4c_carbon import _make_pick


@pytest.mark.parametrize(("fields", "sentence"), [
    ({"reason": "Stopped!"},
     "Candidate used in the decision: A (T) 75.0%. Reason: Stopped!"),
    ({"bar": 0.7501},
     "Candidate used in the decision: A (T) 75.0%, below the streak-8 bar (75.0%) using unrounded values."),
    ({"bar": 0.75},
     "Candidate used in the decision: A (T) 75.0%, at or above the streak-8 bar (75.0%)."),
    ({}, "Candidate used in the decision: A (T) 75.0%. No probability comparison is available for this check."),
    ({"basis": "no_decision", "bar": 0.8},
     "No candidate was chosen. Top model prediction (eligibility unverified): A (T) 75.0%."),
    ({"reason": "<stop & check>", "basis": "no_decision", "best_team": "<T&>"},
     "Candidate used in the decision: A (<T&>) 75.0%. Reason: <stop & check>."),
    ({"reason": "Stopped!", "bar": 0.8},
     "Candidate used in the decision: A (T) 75.0%. Reason: Stopped!"),
    ({"bar": 0.8, "candidate_verified": False},
     "Top model prediction (eligibility unverified): A (T) 75.0%, below the streak-8 bar (80.0%) using unrounded values."),
    ({"reason": " \r\n\t "},
     "Candidate used in the decision: A (T) 75.0%. No probability comparison is available for this check."),
    ({"reason": "  Stop\n here!  "},
     "Candidate used in the decision: A (T) 75.0%. Reason: Stop  here!"),
])
def test_skip_surfaces(fields, sentence):
    summary = dict(best_batter="A", best_team="T", best_p=0.75, streak=8,
                   candidate_verified=True) | fields
    body = sentence
    if fields.get("basis") != "no_decision":
        body += " Streak used at this check: 8."
    assert scheduler.format_skip_dm("2027-07-15", summary) == "BTS 2027-07-15: No pick at this check. " + body
    assert web.render_skip_banner(summary) == (
        '<div class="skip-banner"><strong>LAST RECORDED NO-PICK SUMMARY</strong><span>'
        + html.escape(body) + "</span></div>"
    )


@pytest.mark.parametrize("bad", [None, "0.99", {}, [], True, float("nan"), float("inf"), 10**400])
def test_malformed_persisted_fields(bad):
    summary = dict(best_batter="<A>\r\nB", best_team=[], best_p=bad, bar=bad,
                   streak="8", reason=[], candidate_verified=1)
    body = ("Top model prediction (eligibility unverified): <A> B (?) ?. "
            "No probability comparison is available for this check.")
    assert scheduler.format_skip_dm("day", summary) == "BTS day: No pick at this check. " + body
    assert "<span>" + html.escape(body) + "</span>" in web.render_skip_banner(summary)
    assert web.render_skip_banner(["not an object"]) == ""


@pytest.mark.parametrize("bad", [1e308, 1.5e308, -0.1, 1.0000001, 2])
@pytest.mark.parametrize("field", ["best_p", "bar"])
def test_out_of_range_skip_numbers_are_missing(bad, field):
    summary = dict(best_batter="A", best_team="T", best_p=0.75, bar=0.8)
    summary[field] = bad
    pct = "?" if field == "best_p" else "75.0%"
    assert scheduler.format_skip_summary(summary) == (
        f"Top model prediction (eligibility unverified): A (T) {pct}. "
        "No probability comparison is available for this check."
    )


@pytest.mark.parametrize("edge", [0, 0.0, 1, 1.0])
def test_skip_number_edges_are_valid(edge):
    summary = dict(best_batter="A", best_team="T", best_p=edge, bar=edge)
    assert scheduler.format_skip_summary(summary) == (
        f"Top model prediction (eligibility unverified): A (T) {edge:.1%}, "
        f"at or above the bar ({edge:.1%})."
    )


@pytest.mark.parametrize(("reason", "bar", "ending"), [
    ("Stopped!", 0.9, ". Reason: Stopped!"),
    (None, 0.7, ", at or above the streak-0 bar (70.0%)."),
    (None, None, ". No probability comparison is available for this check."),
], ids=["K1", "K3", "K4"])
def test_run_single_check_skip_journal(reason, bar, ending, tmp_path, monkeypatch, capsys):
    day = datetime(2027, 7, 15).date()
    frame = walk._predictions(day, 0.75)
    selection = _sel(replace(_tail_skip_decision(), reason=reason, pick_bar=bar))
    selection.primary_candidate.update(batter_name="Kept", team="T", p_game_hit=0.75)
    monkeypatch.setattr(orchestrator, "run_and_pick", lambda *a, **kw: (frame, selection, "test"))
    monkeypatch.setattr(scheduler, "count_new_confirmations", lambda *a, **kw: 0)
    monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext())
    scheduler.run_single_check(
        date=day.isoformat(), all_game_pks=[70000], confirmed_sides=set(),
        config=walk._config(tmp_path), early_lock_gap=0.03,
    )
    assert [line for line in capsys.readouterr().err.splitlines() if line.startswith("  SKIP —")] == [
        "  SKIP — Candidate used in the decision: Kept (T) 75.0%"
        + ending + " Streak used at this check: 0."
    ]


def test_candidate_with_raising_class_lookup_preserves_base_summary():
    class UnreadableCandidate:
        @property
        def __class__(self):
            raise RuntimeError("candidate class is unreadable")

    predictions = pd.DataFrame([dict(batter_name="Top", team="T", p_game_hit=0.75)])
    decision = _tail_skip_decision()
    expected = scheduler.build_skip_summary(predictions, 8, decision=decision)
    summary = scheduler.build_skip_summary(
        predictions, 8, decision=decision, candidate=UnreadableCandidate(),
    )
    assert summary == expected
    assert summary["candidate_verified"] is False


def test_candidate_override():
    predictions = pd.DataFrame([dict(batter_name="Removed", team="X", p_game_hit=0.999)])
    summary = scheduler.build_skip_summary(
        predictions, 0, decision=_tail_skip_decision(),
        candidate=dict(batter_name="Kept", team=pd.NA, p_game_hit=0.75),
    )
    assert (summary["best_batter"], summary["best_team"], summary["best_p"]) == ("Kept", "?", 0.75)
    assert summary["basis"] == "decision" and summary["candidate_verified"] is True


@pytest.mark.parametrize(("date", "state", "phrase"), [
    (datetime(2027, 7, 15).date(), walk.State("dated", 56, 56, "used", 0.96),
     "No probability comparison is available for this check."),
    (datetime(2027, 8, 30).date(), walk.State("dated", 56, 56, "used", 0.99), "at or above the"),
    (datetime(2027, 7, 15).date(), walk.State("dated", 12, 12, "used", 0.72), "below the"),
])
def test_shipped_skip_comparisons(date, state, phrase):
    row = walk.walk_one(date, state)
    assert row["action"] == "skip"
    assert phrase in row["skip_dm"]
    assert walk.check([row]) == []


@pytest.mark.parametrize("route", ["delivery", "morning"])
def test_stale_warning_maximum_detail(route, tmp_path, monkeypatch, capsys):
    from tests.test_morning_state_cs6b1 import DAY

    now = datetime.fromisoformat(DAY + "T10:00:00-04:00")
    stamp = "0001-01-01T00:00:00." + "0" * 102 + "+00:00"
    state = contest_state.DecisionStreakState(
        10, False, False, "contest", "fresh", None, None, observation_kind="automatic",
        contest_recorded_at=stamp,
    )
    record = contest_state.automatic_reading_record(state, now)
    detail = f"last accepted fetch {stamp}, about {record['age_hours']:.2f} hours ago"
    assert len(stamp) == 128 and len(detail) == 177
    monkeypatch.setattr(contest_state, "load_decision_streak_state", lambda *a, **kw: state)
    monkeypatch.setattr(scheduler, "_now_et", lambda: now)
    sent = []
    monkeypatch.setattr(alert, "send_dm", lambda recipient, body: sent.append(body))
    config = {"bluesky": {"dm_recipient": "offline.test"}}
    picks_dir = tmp_path / "picks"
    if route == "delivery":
        scheduler._warn_stale_contest_reading(record, config, picks_dir)
        expected, length = envelope(day=DAY, detail=detail), 881
        assert "  STALE CONTEST READING — " + expected.split("- [contest_state] ")[1] in capsys.readouterr().err
    else:
        scheduler._morning_state_check(config, picks_dir, DAY, now, now + timedelta(hours=1))
        expected, length = warning(detail), 895
    assert sent == [expected]
    assert len(sent[0]) == length < 900


@pytest.mark.parametrize("kind", ["late", "postponed"])
def test_refusal_message_and_record(kind, tmp_path, monkeypatch):
    daily = _make_pick()
    now = datetime.fromisoformat(daily.run_time)
    state = _state(date=daily.date)
    sent = []
    monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", lambda alerts, *a, **kw: sent.extend(alerts))
    monkeypatch.setattr(scheduler, "_now_et", lambda: now)
    picks.save_pick(daily, tmp_path)
    if kind == "late":
        cutoff = now
        scheduler._refuse_late_delivery(daily, {}, tmp_path, state, daily.date, "test", cutoff, now)
        prefix = f"LATE DELIVERY REFUSED: the {cutoff:%H:%M} ET cutoff for Steven Kwan had been reached. "
        source = "late_delivery"
    else:
        scheduler._refuse_void_game_delivery(daily, {}, tmp_path, state, daily.date, "test", {101: "Postponed"})
        prefix = f"DELIVERY REFUSED for {daily.date}: game 101 is Postponed. "
        source = "postponed_pick"
    assert [(item.source, item.message) for item in sent] == [(source, prefix + (
        "This pick was withheld. A later check may produce another recommendation. "
        "Use a later official pick message only while the entry window is open. "
        "A manual entry can differ from a later recommendation. "
        "The entry check may report a mismatch if it verifies the entry in time."
    ))]
    saved = scheduler.load_state(daily.date, tmp_path)
    assert saved.delivery_refusals == state.delivery_refusals and not saved.pick_locked
    if kind == "late":
        assert saved.delivery_refusals[0]["late_min"] == 0.0
    assert picks.load_pick(daily.date, tmp_path) is None
    assert (tmp_path / daily.date / saved.delivery_refusals[0]["archive"]).is_file()


@pytest.fixture
def base_scheduler(monkeypatch):
    return _load_base_module("scheduler", monkeypatch)


def _load_base_module(name, monkeypatch):
    filename = f"63669c42:src/bts/{name}.py"
    source = subprocess.run(
        ["git", "show", filename],
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout
    base = ModuleType(f"bts._wr_e_base_{name}")
    monkeypatch.setitem(sys.modules, base.__name__, base)
    exec(compile(source, filename, "exec"), base.__dict__)
    return base


def test_base_summary_gates(base_scheduler):
    class RaisingPredictions:
        @property
        def columns(self):
            raise ValueError("unusable")

    class RaisingCandidate(dict):
        def get(self, key, default=None):
            if key == "team":
                raise ValueError("unusable")
            return super().get(key, default)

    valid = dict(batter_name="Kept", team="T", p_game_hit=0.75)
    good = pd.DataFrame([dict(batter_name="Removed", team="X", p_game_hit=0.999)])
    unusable = [None, RaisingPredictions(), pd.DataFrame(), good.iloc[:0],
                good.drop(columns=["batter_name"]), good.drop(columns=["p_game_hit"]),
                pd.concat([good, good])]
    unusable.extend(good.assign(p_game_hit=invalid) for invalid in (float("nan"), float("inf"), -float("inf")))
    for frame in [*unusable, good]:
        for decision in (None, _tail_skip_decision()):
            old = base_scheduler.build_skip_summary(frame, 0, 0.8, decision)
            for candidate in (None, {}, dict(valid, p_game_hit=True), dict(valid, p_game_hit="0.75"),
                              RaisingCandidate(valid), valid):
                current = scheduler.build_skip_summary(frame, 0, 0.8, decision, candidate)
                if old is None:
                    assert current is None
                else:
                    assert current.pop("basis") == ("decision" if decision else "no_decision")
                    verified = decision is not None and candidate is valid
                    assert current.pop("candidate_verified") is verified
                    assert current == (old | dict(best_batter="Kept", best_team="T", best_p=0.75)
                                       if verified else old)


def test_state_compatibility(base_scheduler, tmp_path):
    day = "2027-07-15"
    old_summary = dict(best_batter="A", best_team="T", best_p=0.75, bar=0.8, streak=10)
    old_state = base_scheduler.SchedulerState(**vars(_state(date=day, skip_summary=old_summary)))
    file = base_scheduler.save_state(old_state, tmp_path)
    loaded = scheduler.load_state(day, tmp_path)
    assert loaded.skip_summary == old_summary
    assert "below the streak-10 bar (80.0%)" in web.render_skip_banner(loaded.skip_summary)
    loaded.skip_summary |= dict(basis="decision", candidate_verified=True)
    scheduler.save_state(loaded, tmp_path)
    assert base_scheduler.load_state(day, tmp_path).skip_summary == loaded.skip_summary
    file.write_text("{")
    assert scheduler.load_state(day, tmp_path) is None
    assert list(file.parent.glob(file.name + ".corrupt-*"))
    file.mkdir()
    with pytest.raises(IsADirectoryError):
        scheduler.load_state(day, tmp_path)


@pytest.mark.parametrize("status_case", ["removed", "none_eligible", "status_failure", "sel_none"])
def test_real_lineup_skip_return_against_base(status_case, base_scheduler, tmp_path, monkeypatch, capsys):
    now = datetime(2027, 7, 15, 10, tzinfo=walk.ET)
    day = now.date()
    frame = walk._predictions(day, 0.76)
    frame.loc[0, "p_game_hit"] = 0.999
    config = walk._config(tmp_path)
    config["orchestrator"]["heartbeat_path"] = str(tmp_path / "heartbeat")
    saved = walk.State("dated", 10, 10, "used", 0.75)
    statuses = {
        70000: {"abstract": "F", "detailed": "Final"},
        70001: {"abstract": "P", "detailed": "Scheduled"},
        70002: {"abstract": "P", "detailed": "Scheduled"},
    }
    if status_case == "none_eligible":
        statuses = dict.fromkeys(statuses, {"abstract": "F", "detailed": "Final"})

    def read_statuses(_):
        if status_case == "status_failure":
            raise OSError("offline")
        return statuses

    monkeypatch.setattr(orchestrator, "run_cascade", lambda *a: (frame.copy(), "test"))
    monkeypatch.setattr(picks, "get_game_statuses_detailed", read_statuses)
    if status_case == "sel_none":
        monkeypatch.setattr(orchestrator, "run_and_pick", lambda *a, **kw: (frame.copy(), None, "test"))
    for module in (base_scheduler, scheduler):
        monkeypatch.setattr(module, "count_new_confirmations", lambda *a, **kw: 0)
        monkeypatch.setattr(module, "heartbeat_watchdog", lambda *a, **kw: nullcontext())
    with walk.offline(now):
        assert walk.install_state(tmp_path, day, saved) == []
        args = dict(date=day.isoformat(), all_game_pks=list(statuses), confirmed_sides=set(),
                    config=config, early_lock_gap=0.03)
        old = base_scheduler.run_single_check(**args)
        capsys.readouterr()
        current = scheduler.run_single_check(**args)
        journal = capsys.readouterr().err
    old_summary = old.pop("skip_summary")
    summary = current.pop("skip_summary")
    assert current == old
    assert {k: v for k, v in summary.items() if k not in
            ("best_batter", "best_team", "best_p", "basis", "candidate_verified")} == {
                k: v for k, v in old_summary.items() if not k.startswith("best_")
            }
    if status_case == "removed":
        assert old_summary["best_batter"] == "Batter 0"
        assert summary["best_batter"] == "Batter 1" and summary["candidate_verified"] is True
        assert journal.splitlines().count(
            "  SKIP — Candidate used in the decision: Batter 1 (T1) 75.0%, below the streak-10 bar "
            "(79.6%) using unrounded values. Streak used at this check: 10."
        ) == 1
    else:
        assert summary["basis"] == "no_decision" and summary["candidate_verified"] is False
        assert summary["streak"] == 10
        assert ("  SKIP — No candidate was chosen. Top model prediction (eligibility unverified): "
                "Batter 0 (T0) 99.9%.") in journal.splitlines()


def test_operator_text_scan():
    def literals(source):
        return [node.value for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]

    source = Path(scheduler.__file__).read_text()
    for phrase in ("Use the current ET date", "below the ~80% bar", "below the pick bar"):
        assert all(phrase not in word for word in literals(source))
    for phrase in ("below the ~80% bar", "below the pick bar"):
        assert all(phrase not in word for word in literals(Path(web.__file__).read_text()))
    for refusal in (scheduler._refuse_late_delivery, scheduler._refuse_void_game_delivery):
        assert all("manually if you want" not in word for word in literals(inspect.getsource(refusal)))
    assert 'Deliver manually if you want to keep the streak alive' in inspect.getsource(scheduler._alert_missed_pick)


@pytest.mark.parametrize("outcome", ["no_id", "ordinary_error", "interrupt", "exit"])
def test_skip_notice_restart_and_retry(outcome, base_scheduler, tmp_path):
    config = {"scheduler": {"pick_delivery": "dm"}, "bluesky": {"dm_recipient": "offline.test"}}
    summary = dict(best_batter="A", best_team="T", best_p=0.75, streak=10)
    now = "2027-07-15T10:00:00-04:00"
    failures = {"ordinary_error": ValueError, "interrupt": KeyboardInterrupt, "exit": SystemExit}
    for module in (base_scheduler, scheduler):
        state = module.SchedulerState(**vars(_state(date=now[:10], skip_summary=summary)))
        module.save_state(state, tmp_path)
        sent = []

        def send(recipient, body):
            sent.append(recipient)
            if len(sent) == 1 and outcome in failures:
                raise failures[outcome]("planted send failure")

        def notify(current):
            return module.maybe_notify_skip(current, current.skip_summary, config, now_iso=now, send=send)

        if outcome in ("interrupt", "exit"):
            with pytest.raises(failures[outcome]):
                notify(state)
        else:
            assert notify(state) is (outcome == "no_id")
        assert state.skip_notified_at == (now if outcome == "no_id" else None)
        before_save = module.load_state(state.date, tmp_path)
        assert before_save.skip_notified_at is None and notify(before_save)
        module.save_state(before_save, tmp_path)
        restarted = module.SchedulerState(**vars(_state(date=state.date)))
        assert module.carry_forward_skip_state(restarted, module.load_state(state.date, tmp_path)) is restarted
        assert restarted.skip_summary == summary and restarted.skip_notified_at == now
        assert not notify(restarted)
        assert sent == ["offline.test", "offline.test"]


def test_shipped_bars_equal_base(monkeypatch):
    base = _load_base_module("strategy", monkeypatch)
    artifact = strategy._load_mdp()
    assert artifact["policy_table"] is not None and artifact["tail"] is not None
    end = walk.LAST_GAME_DAY + timedelta(days=1)
    regimes = set()
    comparisons = 0
    for days in [*range(181), 181]:
        day = (end - timedelta(days=days)).isoformat()
        for streak in range(57):
            regime, _ = strategy._regime(artifact, streak, day)
            regimes.add(regime)
            bests = [(streak, "trusted")]
            if regime == strategy.OBJECTIVE_TAIL:
                bests.extend([(56, "trusted"), (56, "untrusted")])
            for saver in (False, True):
                for best, trust in bests:
                    args = (streak, day, saver, artifact, best, trust)
                    assert strategy.effective_pick_bar(*args) == base.effective_pick_bar(*args)
                    comparisons += 1
    assert len(regimes) == 2
    assert comparisons > 20_748

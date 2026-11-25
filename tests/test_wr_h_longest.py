from contextlib import nullcontext
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from bts import daily_decision, orchestrator, picks, scheduler, strategy
from bts.simulate import mdp, tail_policy
if __name__ != "__main__":
    from scripts.audit.season_walk import walk


# Keep the base runnable so these tests report behavioral failures before H exists.
BOUNDARY = getattr(tail_policy, "LONGEST_STREAK_FROM", 57)
OBJECTIVE = "longest_streak"
FINALE_END = date(2028, 9, 28)


@pytest.fixture
def artifact():
    return strategy._load_mdp()


@pytest.mark.parametrize("days,allow,partner,expected", [
    (range(4, 272), True, True, "single"),
    (range(1, 4), True, True, "double"),
    (range(1, 4), False, True, "single"),
    (range(1, 4), True, False, "single"),
])
def test_h1_h2_h3_resolver_grid(artifact, days, allow, partner, expected):
    for raw, streak, saver, p in product(
            days, (BOUNDARY, BOUNDARY + 1, BOUNDARY + 3, BOUNDARY + 43),
            (False, True), (.01, .5, .8, .99)):
        day = str(FINALE_END - timedelta(days=raw))
        ctx = strategy.DecisionContext(
            p, .95 if partner else None, partner, day, allow, artifact, BOUNDARY, "trusted")
        decision = strategy.resolve_policy_decision(ctx, streak, saver)
        assert (decision.action, decision.policy_action, decision.objective) == (
            expected, "double" if raw <= 3 else "single", OBJECTIVE)
        assert (decision.source, decision.streak, decision.days_effective) == (
            "mdp", streak, min(raw, 180))
        assert (decision.best_supplied, decision.best_status, decision.tail_sha256) == (
            BOUNDARY, "trusted", artifact["tail"].sha256)
        assert decision.effective_best is decision.guard is decision.degraded_reason is None
        assert decision.reason is decision.pick_bar is None


@pytest.mark.parametrize("mode", ["shipped", "missing", "raises"])
def test_h6_display_before_artifacts(monkeypatch, mode):
    if mode == "missing":
        monkeypatch.setattr(strategy, "_load_mdp", lambda: None)
    if mode == "raises":
        def fail():
            raise TypeError("broken loader")
        monkeypatch.setattr(strategy, "_load_mdp", fail)
    assert strategy.effective_pick_bar(BOUNDARY, "2027-07-15", False) == 0.0


def test_h7_finalized_decision(tmp_path):
    day = "2027-07-15"
    rec = daily_decision.write_decision(
        day, tmp_path, objective=OBJECTIVE, action="skip", source="mdp",
        delivery_status="not_applicable", scoreable=False)
    assert rec is not None
    assert daily_decision.load_decision(day, tmp_path) == rec
    assert daily_decision.decision_objective(rec) == OBJECTIVE
    assert not daily_decision.is_reach57_mdp_skip(rec)


def test_finalized_longest_record_is_excluded_by_reach57_readers(tmp_path):
    from bts.health import mdp_policy_alignment
    from bts.skip_policy_shadow import record_skip_from_decision
    from scripts.audit import boundary_shadow_census as census

    day = date(2027, 7, 15)
    text = str(day)
    now = datetime.combine(day, time(12), walk.ET)
    pick = picks.pick_from_row(walk._predictions(day, .96).iloc[0])
    daily = picks.DailyPick(text, now.isoformat(), pick, None, None)
    picks.save_pick(daily, tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    ledger.touch()
    primary = {"batter_id": pick.batter_id, "game_pk": pick.game_pk, "p_game_hit": .96}
    rec = daily_decision.write_decision(
        text, tmp_path, objective=OBJECTIVE, action="single", source="mdp", primary=primary,
        delivery_status="delivered", scoreable=True, now=now)
    assert rec is not None
    assert mdp_policy_alignment.compute_metrics(tmp_path, mdp.DEFAULT_POLICY_PATH, today=day).primary.n == 0
    report = census.run_census(picks_dir=tmp_path, ledger_path=ledger, policy_path=mdp.DEFAULT_POLICY_PATH,
                               saver_active=False, start=text, end=text, as_of=text,
                               output=tmp_path / "census.json")
    assert report["boundaries"]["primary"]["n"] == 0
    assert report["rows"][0]["state_source"] == "excluded_tail_objective"
    assert report["rows"][0]["diffs"] == {}
    path = daily_decision.decision_path(text, tmp_path)
    path.write_text(json.dumps(rec | {"action": "skip", "scoreable": False}))
    assert record_skip_from_decision(text, tmp_path, now=now) is None


@pytest.fixture
def live_case(monkeypatch, tmp_path):
    def prepare(day, kind="dated"):
        frame = walk._predictions(day, .96).head(2)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"}
                    for pk in frame["game_pk"]}
        cfg = walk._config(tmp_path)
        cfg["orchestrator"]["heartbeat_path"] = str(tmp_path / ".heartbeat")
        monkeypatch.setattr(orchestrator, "run_cascade", lambda *args: (frame, "test"))
        monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *args: statuses)
        monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *args, **kwargs: nullcontext())
        monkeypatch.setattr(scheduler, "count_new_confirmations", lambda *args: 6)
        monkeypatch.setattr(scheduler, "_lock_decision_from_predictions",
                            lambda *args, **kwargs: scheduler.LockDecision(True, None, True))
        assert not walk.install_state(tmp_path, day, walk.State(kind, BOUNDARY, BOUNDARY, "used", .96))
        return cfg, list(statuses)
    return prepare


@pytest.mark.parametrize("kind", ["dated", "automatic"])
@pytest.mark.parametrize("entry", ["orchestrator", "scheduler"])
@pytest.mark.parametrize("text,expected", [
    ("2027-07-15", "single"), ("2027-09-24", "double"),
    ("2027-09-25", "double"), ("2027-09-26", "double"),
])
def test_h5_live_selection_save_and_commit(tmp_path, live_case, artifact, capsys, kind, entry, text, expected):
    day = date.fromisoformat(text)
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        cfg, games = live_case(day, kind)
        if entry == "orchestrator":
            _, selection, _ = orchestrator.run_and_pick(cfg, text)
        else:
            result = scheduler.run_single_check(text, games, set(), cfg, .01)
            selection = result["selection"]
        assert selection.action == expected
        daily = selection.pick_result.daily
        assert picks._valid_chosen_on(daily.chosen_on)
        sha = artifact["tail"].sha256
        assert daily.policy_decision["objective"] == OBJECTIVE
        assert daily.policy_decision["tail_sha256"] == sha
        if entry == "scheduler":
            assert "Policy: objective=longest_streak" in capsys.readouterr().err
            saved = picks.load_pick(text, tmp_path)
            assert saved.policy_decision == daily.policy_decision
            scheduler._commit_decision_for_pick(
                tmp_path, text, saved, selection=selection, delivery_status="delivered",
                state=scheduler_state(text))
            record = daily_decision.load_decision(text, tmp_path)
            assert (record["objective"], record["tail_policy_sha256"]) == (OBJECTIVE, sha)


def scheduler_state(day):
    return scheduler.SchedulerState(day, None, [], [], [], False, None, None, None)


@pytest.mark.parametrize("kind", ["dated", "automatic"])
def test_accepted_january_state_uses_raw_days_and_fixed_horizon(tmp_path, live_case, kind):
    day = date(2028, 1, 1)
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        cfg, _ = live_case(day, kind)
        _, sel, _ = orchestrator.run_and_pick(cfg, str(day))
        assert strategy._days_remaining(str(day)) == 271
    assert (sel.action, sel.decision.objective, sel.decision.days_effective) == (
        "single", OBJECTIVE, 180)


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, SystemExit])
def test_rule_error_or_interrupt(monkeypatch, artifact, error, caplog):
    def fail(*args):
        raise error("test interruption")

    monkeypatch.setattr(strategy, "longest_streak_action", fail)
    ctx = strategy.DecisionContext(.01, None, False, "2027-09-26", True, artifact)
    if error is ValueError:
        outcome = strategy.resolve_policy_decision(ctx, BOUNDARY, False)
        assert outcome.action == "single"
        assert outcome.tail_sha256 is None
        assert outcome.degraded_reason == "longest-streak rule unavailable (test interruption); forced single"
        assert outcome.degraded_reason in caplog.text
    else:
        with pytest.raises(error, match="test interruption"):
            strategy.resolve_policy_decision(ctx, BOUNDARY, False)


def test_legacy_display_handles_loader_exception(monkeypatch):
    def fail():
        raise ValueError("test failure")

    monkeypatch.setattr(strategy, "_load_mdp", fail)
    assert strategy.effective_pick_bar(BOUNDARY - 1, "2027-07-15", False) is None


@pytest.mark.parametrize("p_hit,p_both", [(0., 0.), (1., 0.), (1., 1.), (.5, .25), (.75, .50)])
def test_positive_day_rule_selects_a_play_at_exact_ties(p_hit, p_both):
    actions = {tail_policy.longest_streak_action(days, p_hit, p_both)
               for days in range(1, tail_policy.LONGEST_STREAK_HORIZON + 1)}
    assert actions <= {"single", "double"}
    if p_both == 0.0 or (p_hit, p_both) == (.5, .25):
        assert actions == {"single"}


@pytest.mark.parametrize("clamp", ["disabled", "same_game"])
def test_clamped_double_persists_both_actions(tmp_path, clamp):
    day = date(2027, 9, 26)
    frame = walk._predictions(day, .96)
    if clamp == "same_game":
        frame["game_pk"] = frame.iloc[0]["game_pk"]
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        sel = strategy.select_pick(frame, str(day), tmp_path, streak=BOUNDARY, saver_available=False,
                                   allow_double=clamp != "disabled", game_statuses_detailed=statuses)
        picks.save_pick(sel.pick_result.daily, tmp_path)
    daily = picks.load_pick(str(day), tmp_path)
    assert daily.double_down is None
    assert (daily.policy_decision["policy_action"], daily.policy_decision["action"]) == ("double", "single")


@pytest.mark.parametrize("mode", [
    "tail-missing", "tail-corrupt", "tail-truncated", "tail-directory", "tail-mismatch",
    "tail-two-bin", "base-missing", "base-corrupt", "base-negative", "all-missing",
    "tail-malformed-manifest",
])
def test_h4_bad_artifacts_force_single(tmp_path, monkeypatch, artifact, mode):
    import hashlib
    import json
    import shutil
    from dataclasses import replace

    import numpy as np

    base_path, tail_path = tmp_path / "base.npz", tmp_path / "tail.npz"
    shutil.copyfile(mdp.DEFAULT_POLICY_PATH, base_path)
    shutil.copyfile(tail_policy.DEFAULT_TAIL_POLICY_PATH, tail_path)
    path = base_path if mode.startswith("base") else tail_path
    if mode == "all-missing":
        base_path.unlink()
        tail_path.unlink()
    elif mode.endswith("missing"):
        path.unlink()
    elif mode.endswith("corrupt"):
        path.write_bytes(b"damaged")
    elif mode.endswith("truncated"):
        path.write_bytes(path.read_bytes()[:70])
    elif mode.endswith("directory"):
        path.unlink()
        path.mkdir()
    elif mode.endswith("mismatch"):
        with base_path.open("ab") as output:
            output.write(b"trailer")
    elif mode.endswith("negative"):
        rewrite_npz(base_path, season_length=np.array(-1))
        sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
        tail_policy.save_tail_policy(replace(artifact["tail"], base_policy_sha256=sha), tail_path)
    elif mode.endswith("two-bin"):
        tail_policy.save_tail_policy(two_bin_tail(artifact), tail_path)
    elif mode.endswith("manifest"):
        manifest = artifact["tail"].manifest | {"bin_counts": [None]}
        rewrite_npz(tail_path, manifest=np.array(json.dumps(manifest)))
    monkeypatch.setattr(mdp, "DEFAULT_POLICY_PATH", base_path)
    monkeypatch.setattr(tail_policy, "DEFAULT_TAIL_POLICY_PATH", tail_path)
    monkeypatch.setattr(strategy, "_mdp_cache", None)
    day = date(2027, 9, 26)
    frame = walk._predictions(day, .99)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        sel = strategy.select_pick(frame, str(day), tmp_path, streak=BOUNDARY,
                                   saver_available=False, game_statuses_detailed=statuses)
    assert sel.action == "single"
    decision = sel.decision
    assert (decision.objective, decision.policy_action, decision.source) == (OBJECTIVE, "single", "mdp")
    assert decision.tail_sha256 is None and decision.days_effective == 1
    assert decision.degraded_reason.startswith("longest-streak rule unavailable (")
    assert decision.degraded_reason.endswith("); forced single")


@pytest.fixture
def rebound_policy(tmp_path, monkeypatch, artifact):
    def prepare(**changes):
        base_path, tail_path = tmp_path / "base.npz", tmp_path / "tail.npz"
        with np.load(mdp.DEFAULT_POLICY_PATH) as arrays:
            base = dict(arrays)
        np.savez_compressed(base_path, **(base | changes))
        sha = sha256(base_path.read_bytes()).hexdigest()
        tail_policy.save_tail_policy(replace(artifact["tail"], base_policy_sha256=sha), tail_path)
        monkeypatch.setattr(mdp, "DEFAULT_POLICY_PATH", base_path)
        monkeypatch.setattr(tail_policy, "DEFAULT_TAIL_POLICY_PATH", tail_path)
        monkeypatch.setattr(strategy, "_mdp_cache", None)
        policy = strategy._load_mdp()
        assert policy["tail"].base_policy_sha256 == policy["base_sha256"] == sha
        return policy
    return prepare


@pytest.mark.parametrize("malformation", [
    "scalar", "1d", "3d", "saver-axis", "bin-axis", "day-axis", "streak-axis",
    "float", "action-3", "action-minus-1", "boundaries", "bool-horizon",
])
def test_r1_malformed_paired_base_forces_single(artifact, rebound_policy, malformation):
    table = artifact["policy_table"]
    if malformation in {"action-3", "action-minus-1"}:
        broken = table.copy()
        broken[0, 0, 0, 0] = 3 if malformation == "action-3" else -1
        policy = rebound_policy(policy_table=broken)
    elif malformation == "boundaries":
        policy = rebound_policy(boundaries=np.array(.5))
    elif malformation == "bool-horizon":
        policy = rebound_policy(season_length=np.array(True), policy_table=table[:, :2])
        # The loader normalizes the horizon to int. Exercise the decoded-context
        # contract by supplying the boolean to the resolver directly.
        policy = policy | {"season_length": True}
    else:
        broken = {
            "scalar": np.array(0, dtype=table.dtype),
            "1d": table[0, 0, 0],
            "3d": table[0],
            "saver-axis": np.concatenate((table, table[:, :, :1]), axis=2),
            "bin-axis": table[:, :, :, :-1],
            "day-axis": table[:, :-1],
            "streak-axis": table[:BOUNDARY - 1],
            "float": table.astype(float),
        }[malformation]
        policy = rebound_policy(policy_table=broken)
    ctx = strategy.DecisionContext(.84, .83, True, "2027-09-26", True, policy,
                                   BOUNDARY, "trusted")
    decision = strategy.resolve_policy_decision(ctx, BOUNDARY, False)
    assert (decision.action, decision.policy_action, decision.objective) == ("single", "single", OBJECTIVE)
    assert decision.tail_sha256 is None
    assert decision.degraded_reason.startswith("longest-streak rule unavailable (")
    assert decision.degraded_reason.endswith("); forced single")


@pytest.mark.parametrize("horizon", [1, 180])
def test_r1_healthy_paired_base_can_double(artifact, rebound_policy, horizon):
    policy = rebound_policy(season_length=np.array(horizon),
                            policy_table=artifact["policy_table"][:, :horizon + 1])
    ctx = strategy.DecisionContext(.84, .83, True, "2027-09-26", True, policy,
                                   BOUNDARY, "trusted")
    decision = strategy.resolve_policy_decision(ctx, BOUNDARY, False)
    assert (decision.action, decision.tail_sha256, decision.degraded_reason) == (
        "double", policy["tail"].sha256, None)


@pytest.mark.parametrize("base_changes", [
    pytest.param({"policy_table": np.array(0, dtype=np.int8)}, id="scalar"),
    pytest.param({"boundaries": np.array([".76", ".78", ".80", ".82"])}, id="string-edges"),
])
def test_scheduler_delivers_single_for_malformed_pair(
    tmp_path, monkeypatch, live_case, rebound_policy, base_changes,
):
    rebound_policy(**base_changes)
    monkeypatch.setattr(strategy, "_mdp_cache", None)
    day = date(2027, 9, 26)
    text = str(day)
    send = Mock(return_value="local-id")
    monkeypatch.setattr("bts.dm.send_dm", send)
    monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", Mock())
    monkeypatch.setattr(scheduler, "_void_committed_games", lambda *args: {})
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        cfg, games = live_case(day)
        cfg["bluesky"] = {"dm_recipient": "synthetic"}
        result = scheduler.run_single_check(text, games, set(), cfg, .01)
        saved = picks.load_pick(text, tmp_path)
        scheduler._deliver_and_lock_pick(saved, cfg, tmp_path, scheduler_state(text), text,
                                         "final fallback", selection=result["selection"])
    assert send.call_count == 1
    assert saved.double_down is None
    message = send.call_args.args[1]
    assert "Batter 0" in message and "Batter 1" not in message
    record = daily_decision.load_decision(text, tmp_path)
    assert (record["delivery_status"], record["action"], record["objective"]) == (
        "delivered", "single", OBJECTIVE)
    assert record["tail_policy_sha256"] is None
    assert record["degraded_reason"] == saved.policy_decision["degraded_reason"]
    assert record["degraded_reason"].endswith("); forced single")


def finale_decision(policy):
    ctx = strategy.DecisionContext(.84, .83, True, "2027-09-26", True, policy,
                                   BOUNDARY, "trusted")
    return strategy.resolve_policy_decision(ctx, BOUNDARY, False)


def assert_degraded_finale(policy):
    decision = finale_decision(policy)
    assert (decision.action, decision.policy_action, decision.objective, decision.source) == (
        "single", "single", OBJECTIVE, "mdp")
    assert decision.days_effective == 1
    assert decision.tail_sha256 is None
    assert decision.degraded_reason.startswith("longest-streak rule unavailable (")
    assert decision.degraded_reason.endswith("); forced single")


@pytest.mark.parametrize("edges", [
    pytest.param([".76", ".78", ".80", ".82"], id="strings"),
    pytest.param([[.76], [.78], [.80], [.82]], id="nested"),
    pytest.param([.76 + 1j, .78 + 1j, .80 + 1j, .82 + 1j], id="complex"),
    pytest.param([.76, .78, np.nan, .82], id="nan"),
    pytest.param([.76, .78, .80, np.inf], id="infinity"),
    pytest.param([True], id="boolean"),
    pytest.param([.76, .78, .78, .82], id="equal"),
    pytest.param([.76, .80, .78, .82], id="decreasing"),
])
def test_f2_malformed_edges_force_single(artifact, rebound_policy, edges):
    table = artifact["policy_table"][..., :len(edges) + 1]
    policy = rebound_policy(boundaries=np.asarray(edges), policy_table=table)
    assert_degraded_finale(policy)


def test_f2_masked_table_forces_single(rebound_policy):
    policy = rebound_policy()
    # NPZ preserves the underlying values but drops the mask.
    table = np.ma.array(policy["policy_table"], mask=False)
    table[0, 0, 0, 0] = np.ma.masked
    assert_degraded_finale(policy | {"policy_table": table})


@pytest.mark.parametrize("mode", ["shipped", "short", "one-bin", "one-edge"])
def test_f2_healthy_base_can_double(artifact, rebound_policy, mode):
    table = artifact["policy_table"]
    if mode == "shipped":
        policy = artifact
    elif mode == "short":
        policy = rebound_policy(season_length=1, policy_table=table[:, :2])
    else:
        edges = [] if mode == "one-bin" else [.80]
        policy = rebound_policy(boundaries=np.array(edges), policy_table=table[..., :len(edges) + 1])
    decision = finale_decision(policy)
    assert (decision.action, decision.tail_sha256, decision.degraded_reason) == (
        "double", policy["tail"].sha256, None)


@pytest.mark.parametrize("clause", ["integer-horizon", "positive-horizon", "boundary-list", "ndarray"])
def test_f3_decoded_base_clause_forces_single(artifact, rebound_policy, clause):
    if clause == "positive-horizon":
        policy = rebound_policy(season_length=0, policy_table=artifact["policy_table"][:, :1])
    else:
        policy = rebound_policy()
        table = policy["policy_table"]
        changes = {
            "integer-horizon": {"season_length": 180.0},
            "boundary-list": {"boundaries": tuple(policy["boundaries"])},
            "ndarray": {"policy_table": SimpleNamespace(
                ndim=table.ndim, shape=table.shape, dtype=table.dtype, min=table.min, max=table.max)},
        }
        policy = policy | changes[clause]
    assert_degraded_finale(policy)


def test_r2_fresh_selection_uses_fixed_horizon(tmp_path, artifact, rebound_policy):
    policy = rebound_policy(season_length=np.array(1), policy_table=artifact["policy_table"][:, :2])
    selection = fresh_longest_selection(date(2027, 9, 23), tmp_path)
    decision = selection.decision
    assert (decision.action, decision.days_effective, decision.tail_sha256, decision.degraded_reason) == (
        "single", 4, policy["tail"].sha256, None)


def test_r3_fresh_selection_dispatch_precedes_loader(tmp_path, monkeypatch):
    calls = []
    regime = strategy._regime

    def record_regime(*args):
        calls.append("regime")
        return regime(*args)

    def fail_loader():
        calls.append("load")
        raise TypeError("broken loader")

    monkeypatch.setattr(strategy, "_regime", record_regime)
    monkeypatch.setattr(strategy, "_load_mdp", fail_loader)
    selection = fresh_longest_selection(date(2027, 9, 26), tmp_path)
    assert calls[0] == "regime"
    assert "load" in calls
    assert selection.action == "single"
    assert selection.decision.tail_sha256 is None
    assert "broken loader" in selection.decision.degraded_reason


def fresh_longest_selection(day, tmp_path):
    frame = walk._predictions(day, .96)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        return strategy.select_pick(frame, str(day), tmp_path, streak=BOUNDARY,
                                    saver_available=False, game_statuses_detailed=statuses)


def rewrite_npz(path, **changes):
    import numpy as np

    with np.load(path) as arrays:
        data = dict(arrays)
    np.savez_compressed(path, **(data | changes))


def two_bin_tail(artifact):
    from dataclasses import replace

    freq, hit, both = [.5, .5], [.75, .75], [.5, .5]
    policy = tail_policy.solve_emax_season_best(freq, hit, both).policy
    manifest = tail_policy.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2])
    return replace(artifact["tail"], boundaries=[.82], policy_table=policy,
                   bin_freq=freq, bin_p_hit=hit, bin_p_both=both, manifest=manifest)


@pytest.mark.parametrize("operation", ["cached", "restart", "failed", "unconfirmed"])
def test_h8_saved_metadata_and_delivery(tmp_path, live_case, monkeypatch, operation):
    day = date(2027, 9, 26)
    text = str(day)
    state = scheduler_state(text)
    with walk.offline(datetime.combine(day, time(12), walk.ET)):
        cfg, _ = live_case(day, "automatic")
        cfg["bluesky"] = {"dm_recipient": "synthetic"}
        _, selection, _ = orchestrator.run_and_pick(cfg, text)
        if selection.pick_result:
            daily = selection.pick_result.daily
        else:
            # Supply a saved H choice to isolate the base's commit behavior.
            frame = walk._predictions(day, .96)
            daily = picks.DailyPick(text, datetime.now().isoformat(),
                                    picks.pick_from_row(frame.iloc[0]), None, None)
            daily.policy_decision = {"objective": OBJECTIVE, "source": "mdp", "action": "single",
                                     "tail_sha256": strategy._load_mdp()["tail"].sha256}
            from bts.contest_state import load_decision_streak_state
            daily.chosen_on = picks.chosen_on_record(
                load_decision_streak_state(tmp_path, require_contest_state=True), required=True)
        if operation == "unconfirmed":
            daily.delivery_attempted = True
        picks.save_pick(daily, tmp_path)
        saved = picks.load_pick(text, tmp_path)
        assert saved.policy_decision == daily.policy_decision
        action = "double" if saved.double_down else "single"
        if operation == "restart":
            scheduler._write_classification_decision(
                tmp_path, text, action=action, delivered=False, attempted=True,
                primary=scheduler._row_from_daily(saved.pick),
                double_down=scheduler._row_from_daily(saved.double_down),
                state=state, daily=saved)
        else:
            from unittest.mock import Mock

            send = Mock(side_effect=OSError("synthetic rejection") if operation == "failed" else None,
                        return_value="local-id")
            monkeypatch.setattr("bts.dm.send_dm", send)
            monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", Mock())
            monkeypatch.setattr(scheduler, "_alert_delivery_outcome_unknown", Mock())
            monkeypatch.setattr(scheduler, "_void_committed_games", lambda *args: {})
            empty = walk._predictions(day, .96).iloc[:0]
            monkeypatch.setattr(orchestrator, "run_cascade", lambda *args: (empty, "test"))
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg, text, saved, .01)
            assert refresh.daily is saved and not scheduler._refresh_is_genuine_skip(refresh)
            assert getattr(refresh.selection, "decision", None) is None
            scheduler._deliver_and_lock_pick(saved, cfg, tmp_path, state, text, "final fallback",
                                             selection=refresh.selection)
            assert send.call_count == (0 if operation == "unconfirmed" else 1)
        record = daily_decision.load_decision(text, tmp_path)
        if operation == "failed":
            assert record is None
            assert not picks.load_pick(text, tmp_path).delivery_attempted
        else:
            assert record is not None
            assert (record["objective"], record["tail_policy_sha256"]) == (
                OBJECTIVE, saved.policy_decision["tail_sha256"])
            if operation in {"restart", "unconfirmed"}:
                assert record["delivery_status"] == "locked_unconfirmed"


def test_legacy_parity_against_archived_base(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    archive = subprocess.run(["git", "archive", "0a2bec10", "src"], cwd=repo,
                             capture_output=True, check=True, timeout=10).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=archive, check=True, timeout=10)
    record = daily_decision.write_decision(
        "2027-07-15", tmp_path, objective=OBJECTIVE, action="single", source="mdp",
        delivery_status="delivered", scoreable=True)
    results = []
    for src in (tmp_path / "src", repo / "src"):
        env = os.environ | {"PYTHONPATH": os.pathsep.join((str(src), str(repo)))}
        proc = subprocess.run([sys.executable, "-c",
                               "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
                               str(Path(__file__).resolve()), "--parity", str(src), json.dumps(record),
                               "unknown" if src.is_relative_to(tmp_path) else OBJECTIVE],
                              cwd=repo, env=env, capture_output=True, text=True, timeout=27)
        assert proc.returncode == 0, proc.stderr
        imported, result = proc.stdout.splitlines()
        assert Path(imported).resolve() == src / "bts" / "__init__.py"
        result = json.loads(result)
        results.append(result)
        raw, effective = result["raw_days"], result["effective_days"]
        print(f"{imported}: {result['cases']} cases, {result['guards']} guard cells, "
              f"raw days {raw[0]}–{raw[-1]}, effective days {effective[0]}–{effective[-1]}, "
              f"SHA-256 {result['digest']}")
    assert results[0] == results[1]


def parity_fingerprint(src, serialized, expected_objective):
    import csv
    from dataclasses import asdict, replace
    from functools import cache
    from hashlib import sha256
    import logging
    from math import nextafter

    import bts

    print(bts.__file__, flush=True)
    assert Path(bts.__file__).resolve() == Path(src) / "bts" / "__init__.py"
    parsed = daily_decision.parse_decision(serialized)
    assert daily_decision.decision_objective(parsed) == expected_objective
    assert daily_decision.scoreable_commit_from(parsed, None) is True
    logging.disable(logging.CRITICAL)
    strategy._days_remaining = cache(strategy._days_remaining)
    artifact = strategy._load_mdp()
    tail = artifact["tail"]
    two = two_bin_tail(artifact)
    tail_policy.validate_tail_policy(two, expected_base_sha=artifact["base_sha256"])
    modes = (artifact, artifact | {"tail": None, "tail_error": "tail missing"},
             artifact | {"tail": None, "tail_error": "tail corrupt"},
             artifact | {"policy_table": None, "base_sha256": None},
             artifact | {"tail": replace(tail, base_policy_sha256="0" * 64)},
             None, artifact | {"tail": two}, artifact | {"season_length": 90},
             artifact | {"season_length": -1})
    digest = sha256()
    calls, guard_cells = 0, 0
    raw_counts, effective_counts = set(), set()

    def compare(streak, day, saver, p, partner, mode, best, status, allow):
        nonlocal calls
        raw = strategy._days_remaining(day)
        ctx = strategy.DecisionContext(p, partner, partner is not None, day, allow, mode, best, status)
        decision = strategy.resolve_policy_decision(ctx, streak, saver)
        bar = strategy.effective_pick_bar(streak, day, saver, mode, best, status)
        digest.update(json.dumps([streak, day, saver, raw, asdict(decision), bar], sort_keys=True).encode())
        calls += 1
        raw_counts.add(raw)
        effective_counts.add(decision.days_effective)
        return decision

    dates = [str(FINALE_END - timedelta(days=raw)) for raw in range(272)]
    edges = (*artifact["boundaries"], .8, .82)
    probabilities = [.01, .99, *(p for edge in edges for p in
                                (nextafter(edge, 0.), edge, nextafter(edge, 1.)))]
    bests = [(None, None), *product(range(BOUNDARY + 1), ("trusted", "untrusted"))]
    for raw, streak, saver, (quality, p) in product(
            range(272), range(BOUNDARY), (False, True), enumerate(probabilities)):
        index = raw + streak * 3 + saver + quality
        compare(streak, dates[raw], saver, p,
                .86 if index % 3 else None, artifact, *bests[index % len(bests)], bool(index % 2))
    for mode, (best, status), raw, streak in product(
            modes, bests, (1, 22, 90, 180), (0, 25, BOUNDARY - 1)):
        compare(streak, dates[raw], bool(best and best % 2), .82, .90, mode, best, status, True)
    for mode, day, streak, saver, p in product(
            modes, ("2027-07-15", "2027-09-27", "2027-12-31", dates[0]),
            (0, BOUNDARY - 1, BOUNDARY, BOUNDARY + 1, BOUNDARY + 43),
            (False, True), probabilities):
        if streak < BOUNDARY or strategy._days_remaining(day) == 0:
            compare(streak, day, saver, p, .87, mode, BOUNDARY, "trusted", True)
    for streak, cut in ((9, .55), (15, .60), (45, .65)):
        for partner, allow in product((nextafter(cut, 0.), cut, nextafter(cut, 1.), None), (False, True)):
            compare(streak, "2027-07-15", True, 1., partner, None, streak, "trusted", allow)

    guard_path = Path("docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv")
    with guard_path.open() as stream:
        for row in csv.DictReader(stream):
            streak, raw, saver, quality = (int(row[field]) for field in ("streak", "days_left", "saver", "bin"))
            day = str(date(2027, 9, 27) - timedelta(days=raw))
            result = compare(streak, day, bool(saver), artifact["boundaries"][quality - 1],
                             .9, artifact, streak, "trusted", True)
            assert result.guard and result.action == row["guarded_action"], row
            guard_cells += 1
    assert guard_cells == 1511
    print(json.dumps(dict(cases=calls, guards=guard_cells, raw_days=sorted(raw_counts),
                          effective_days=sorted(effective_counts), digest=digest.hexdigest())))


if __name__ == "__main__":
    assert sys.argv[1] == "--parity"
    parity_fingerprint(*sys.argv[2:])

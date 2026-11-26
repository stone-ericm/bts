"""Offline SRD6 probes: real main APIs and explicitly labelled v3 transcriptions."""
from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
from click.testing import CliRunner

from bts.cli import cli
from bts.contest_fetch import build_observation
from bts.daily_decision import load_decision
from bts.health.alert import Alert, dispatch_dm_for_health_alerts
from bts.picks import load_streak
from bts.saver_state import load_saver_state, maybe_auto_earn_saver
from bts.scheduler import (
    SchedulerState, _fallback_min_with_floor, _write_classification_decision,
    compute_wakeup_time, effective_cascade_budget_min, load_state,
    plan_fallback_action, save_state,
)
from bts.strategy import DecisionContext, resolve_policy_decision

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
RESULTS = {}


def blocked(*args, **kwargs):
    raise AssertionError("SRD6 is offline; a network connection was attempted")


def state(day="2027-06-01"):
    return SchedulerState(day, "t", [], [], [], False, None, None, None)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def policy_probe():
    paths = [Path("data/models/mdp_policy.npz"), Path("data/models/mdp_tail_policy.npz")]
    out = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            p = z["policy_table"]
            axis = 2 if p.ndim == 4 else 3
            diff = np.take(p, 0, axis=axis) != np.take(p, 1, axis=axis)
            changed = np.nonzero(np.any(diff, axis=tuple(range(1, diff.ndim))))[0].tolist()
            out[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "shape": list(p.shape), "raw_differing_streaks": changed}
    with np.load(paths[0], allow_pickle=False) as z:
        mdp = {"policy_table": z["policy_table"], "boundaries": z["boundaries"].tolist(),
               "season_length": int(z["season_length"])}
    examples = []
    for s, day, probability in [(38, "2027-08-23", .82), (4, "2027-08-29", .80),
                                 (16, "2027-08-23", .82)]:
        ctx = DecisionContext(probability, probability-.01, True, day, True, mdp, s, "trusted")
        decisions = [resolve_policy_decision(ctx, s, sv) for sv in (False, True)]
        assert decisions[0].action != decisions[1].action
        assert decisions[0].objective == "reach57"
        examples.append({"streak": s, "date": day, "probability": probability,
                         "spent": decisions[0].action, "available": decisions[1].action,
                         "days_remaining": decisions[0].days_effective})
    out["reachable_production_resolver_examples"] = examples
    RESULTS["real_policy_bound"] = out


def interface_probe():
    out = {"plan_signature": str(inspect.signature(plan_fallback_action))}
    try:
        plan_fallback_action(SimpleNamespace(), date="2027-06-01", picks_dir=ROOT,
                             now_utc=datetime.now(timezone.utc))
    except TypeError as exc:
        out["v3_literal_call"] = str(exc)
    else:
        raise AssertionError("v3 call unexpectedly worked")
    plan = plan_fallback_action(now=datetime(2027, 6, 1, 18, tzinfo=ET),
        cutoff=datetime(2027, 6, 1, 19, tzinfo=ET), should_post=False,
        should_post_ungated=False, block_reason="primary_projected", contender_game_pk=None,
        remaining_runs=[], confirmed_sides=set(), budget_min=12, operator_reserve_min=10)
    assert plan.action == "deliver"
    out["real_action"] = plan.action
    out["real_floor_0_12_10"] = _fallback_min_with_floor(0, 12, 10)
    out["configured_budget_2_cold"] = effective_cascade_budget_min(2, [])
    game = {"gamePk": 1, "gameDate": "2027-06-01T23:10:00Z"}
    actual = compute_wakeup_time([game])
    assert actual.hour == 10
    out["evening_slate_actual_wakeup"] = actual.isoformat()
    out["v3_offset_45_plus_15_claim"] = "2027-06-01T18:10:00-04:00"
    RESULTS["real_scheduler_interfaces"] = out


def saver_cli_probe(base):
    picks = base / "saver"
    maybe_auto_earn_saver(picks, best_streak=9, season=2027)
    result = CliRunner().invoke(cli, ["saver-state", "--use", "--season", "2027",
                                    "--picks-dir", str(picks)])
    after_use = load_saver_state(picks, season=2027).state
    assert result.exit_code == 0 and after_use == "not_earned"
    maybe_auto_earn_saver(picks, best_streak=10, season=2027)
    after_fetch = load_saver_state(picks, season=2027).state
    assert after_fetch == "active"
    invalid = CliRunner().invoke(cli, ["saver-state", "--earn", "--season", "2027",
                                     "--picks-dir", str(picks)])
    assert invalid.exit_code == 2
    RESULTS["real_between_fetch_burn_recovery"] = {
        "use_exit": result.exit_code, "use_output": result.output,
        "after_use": after_use, "after_best_10_fetch": after_fetch,
        "earn_exit": invalid.exit_code, "earn_output": invalid.output}


def scoreable_unknown_probe(base):
    picks = base / "commit"
    st = state()
    _write_classification_decision(picks, st.date, action="single", delivered=False,
        attempted=True, primary={"batter_name": "Synthetic", "batter_id": 1,
        "team": "X", "game_pk": 7, "p_game_hit": .82}, double_down=None, state=st)
    decision = load_decision(st.date, picks)
    assert decision["scoreable"] and decision["delivery_status"] == "locked_unconfirmed"
    # Literal v3 sections 3.2.3 and 3.3 exit-0 guard, not an implemented retry.
    v3_exit = 0 if decision["scoreable"] else 6
    RESULTS["real_unconfirmed_commit_vs_v3_guard"] = {
        "main_record": decision, "literal_v3_guard_exit": v3_exit,
        "v3_delivery_failed_or_unconfirmed_required_exit": 6}


def dispatcher_probe(base):
    path = base / "dm.json"
    sent = []
    outcomes = []
    with patch("bts.health.alert.send_dm", lambda recipient, body: sent.append(body)):
        for cmd in ("set-contest-streak", "season-contract enable", "set-contest-streak"):
            key = "contest_state:season_contract_unverified:" + hashlib.sha256(cmd.encode()).hexdigest()[:8]
            outcomes.append(dispatch_dm_for_health_alerts(
                [Alert("CRITICAL", "contest_state", cmd, incident_key=key)],
                "synthetic.invalid", status_path=path, now_et_date=date(2027, 6, 1)))
    assert outcomes == [True, True, False] and len(sent) == 2
    RESULTS["real_action_hash_A_B_A"] = {"sent": outcomes,
        "expected_by_whenever_action_changes": [True, True, True]}


def account_snapshots(base):
    """Read states exposed by v3's specified individual-file order; NOT v3 code."""
    picks = base / "snapshot"
    account = picks / "account_state"
    write(account / "saver_state.json", {"season": 2027, "state": "active",
          "updated_at": "2027-06-01T15:00:00+00:00"})
    manual = {"schema_version": "bts_contest_streak_manual_v3", "season": 2027,
        "effective_date": "2027-06-01", "expires_at": "2027-06-01T23:59:59.999999-04:00",
        "active_streak": 13, "best_streak": 13, "saver_state": "used",
        "recorded_at": "2027-06-01T16:00:00+00:00", "operator_note": "synthetic"}
    with (account / ".state.lock").open("w") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        write(account / "contest_streak.manual.json", manual)
        # Actual current saver reader does not take the writer lock.
        visible = load_saver_state(picks, season=2027)
        assert visible.state == "active"
        with (account / ".state.lock").open("w") as nested:
            try:
                fcntl.flock(nested, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                nested_busy = True
            else:
                nested_busy = False
        assert nested_busy
    RESULTS["v3_manual_first_snapshot"] = {"kind": "specified-order snapshot, not v3 execution",
        "manual_streak": manual["active_streak"], "manual_saver": manual["saver_state"],
        "visible_saver_flag": visible.state, "v3_precedence_policy_saver": visible.is_available,
        "second_open_same_process_lock_busy": nested_busy}
    # Literal Gate 2 returns before inspecting the revoked contract.
    revoked = {"status": "revoked", "last_revoked_at": "2027-06-01T17:00:00+00:00"}
    now = datetime(2027, 6, 1, 18, tzinfo=timezone.utc)
    active_manual = (manual["effective_date"] == now.astimezone(ET).date().isoformat()
        and now <= datetime.fromisoformat(manual["expires_at"]))
    assert active_manual and manual["recorded_at"] < revoked["last_revoked_at"]
    RESULTS["v3_same_date_revoke"] = {"kind": "Gate-2 transcription", "row": "P1",
        "manual_before_revoke": True, "contract": revoked}
    # Re-enable under an already verified contract: contract-last isn't a barrier.
    contract = {"status": "verified", "season": 2027}
    auto = {"season": 2027, "attempt_id": "B", "active_streak": 2}
    fetch_status = {"last_accepted_attempt_id": "A"}
    assert contract["status"] == "verified" and auto["attempt_id"] != fetch_status["last_accepted_attempt_id"]
    RESULTS["v3_reenable_crash_after_step2"] = {"kind": "Gate-3/4 transcription",
        "row": "P2b", "streak": 2, "claim_in_section4": "R2 (or P1)"}


def outside_tx_save(base):
    picks = base / "daemon"
    old = state()
    save_state(old, picks)
    lock_path = picks / old.date / ".selection.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fresh = load_state(old.date, picks)
        fresh.pick_locked = fresh.committed_pick_written = True
        save_state(fresh, picks)
    # Real next-wakeup / startup saves are not named in v3's five transaction sites.
    old.next_wakeup = "2027-06-02T10:00:00-04:00"
    save_state(old, picks)
    final = load_state(old.date, picks)
    assert not final.pick_locked and not final.committed_pick_written
    RESULTS["real_outside_tx_save_overwrites_retry"] = {
        "kind": "serialized real writer calls, not a v3 daemon execution",
        "pick_locked_after_old_next_wakeup_save": final.pick_locked,
        "committed_flag_after_old_next_wakeup_save": final.committed_pick_written}


def main():
    with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked):
        policy_probe()
        interface_probe()
        with TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            saver_cli_probe(base)
            scoreable_unknown_probe(base)
            dispatcher_probe(base)
            account_snapshots(base)
            outside_tx_save(base)
        RESULTS["real_main_writer_schema"] = build_observation(
            {"activeStreak": 7, "seasonBestStreak": 18}, date(2026, 9, 1),
            1, "synthetic", datetime(2026, 9, 2, tzinfo=timezone.utc))["schema_version"]
    (ROOT / "results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()

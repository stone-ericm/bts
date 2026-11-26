"""Run six round-one transitions against full packages and investigate fallback slots."""
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import daily_decision, dm, orchestrator, picks, posting, scheduler
from scripts.audit.boundary_shadow_census import CensusHalt, run_census

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-g-codex"
sys.path.insert(0, str(ARCHIVE))
from commit_paths import DAY, GUARD, NOW, data, setup as archived_setup, state


def setup(stack, root, frame, mode="dm"):
    config, statuses = archived_setup(stack, root, frame, mode)
    games = scheduler.fetch_schedule(DAY)
    for game in games:
        game["teams"] = {side: {"team": {"name": f"{side}-{game['gamePk']}"}}
                         for side in ("home", "away")}
    stack.enter_context(patch.object(scheduler, "fetch_schedule", return_value=games))
    return config, statuses


def choose(config):
    sel = orchestrator.run_and_pick(config, DAY)[1]
    assert sel.decision.action == "single"
    return sel.pick_result.daily, sel


def deliver(root, config, daily, sel=None, current=None):
    return scheduler._deliver_and_lock_pick(daily, config, root, current or state(), DAY,
                                            "review", selection=sel)


def identity(record):
    return record["action"], (record["primary"] or {}).get("batter_id"), record["delivery_status"]


def compare_transitions(side):
    import bts
    import inspect
    source = ROOT / "src" if side == "head" else HERE / "base/src"
    assert Path(bts.__file__).resolve().is_relative_to(source)
    print(side, bts.__file__, flush=True)
    (HERE / f"{side}-writer.py.txt").write_text(inspect.getsource(scheduler._write_commit_decision))

    for scenario in ("concurrent_confirmation", "private_to_dm", "refusal",
                     "orphan", "same_day_rerun", "undecodable"):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            frame = data(.83)
            config, _ = setup(stack, root, frame)
            daily, sel = choose(config)
            path = daily_decision.decision_path(DAY, root)
            send = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="confirmed")))
            expected_batter = 100
            if scenario == "concurrent_confirmation":
                intermediate = []

                def reconcile_during_send(*_):
                    pending = picks.load_pick(DAY, root)
                    assert pending.delivery_attempted
                    assert not deliver(root, config, pending)
                    assert daily_decision.load_decision(DAY, root)["delivery_status"] == "locked_unconfirmed"
                    intermediate.append(path.read_bytes())
                    return "confirmed"

                send.side_effect = reconcile_during_send
                assert deliver(root, config, daily, sel)
                old = intermediate[0]
            elif scenario == "undecodable":
                path.parent.mkdir(exist_ok=True)
                old = b"\xff"
                path.write_bytes(old)
                current = state()
                assert deliver(root, config, daily, sel, current)
                assert current.committed_pick_written
            else:
                config["scheduler"]["pick_delivery"] = "private"
                assert deliver(root, config, daily, sel)
                old = path.read_bytes()
                assert daily_decision.load_decision(DAY, root)["delivery_status"] == "private_locked"
                config["scheduler"]["pick_delivery"] = "dm"
                if scenario == "private_to_dm":
                    assert deliver(root, config, picks.load_pick(DAY, root))
                else:
                    if scenario == "refusal":
                        from bts.health import alert
                        stack.enter_context(patch.object(alert, "dispatch_dm_for_health_alerts"))
                        refused = picks.load_pick(DAY, root)
                        refused.pick.game_time = (NOW - timedelta(minutes=10)).isoformat()
                        picks.save_pick(refused, root)
                        assert not deliver(root, config, refused)
                        assert list(path.parent.glob("refused_delivery_*.json"))
                    else:
                        (root / f"{DAY}.json").rename(root / "old-pick.json")
                    if scenario == "same_day_rerun":
                        (path.parent / "scheduler_state.json").unlink()
                    assert picks.load_pick(DAY, root) is None
                    frame["batter_id"] += 100
                    frame["batter_name"] = [f"Fixture batter {id_}" for id_ in frame.batter_id]
                    scheduler.run_day(DAY, config)
                    expected_batter = 200
            send.assert_called_once()
            assert f"Fixture batter {expected_batter}" in send.call_args.args[1]
            assert picks.pick_was_delivered(picks.load_pick(DAY, root))
            record = daily_decision.load_decision(DAY, root)
            after = identity(record)
            assert after == ("single", expected_batter, "delivered")
            assert record.get("policy_guard") is None and path.read_bytes() != old
            print(side, scenario, after, "rewrote record")


def cached_matrix():
    for mode in ("dm", "public", "private"):
        for variant in ("failed", "warmup", "candidate_without_decision", "drop_second",
                        "replace_second", "malformed_slot", "missing_slot", "unguarded"):
            with TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                frame = data(.83 if variant == "unguarded" else .96)
                config, statuses = setup(stack, root, frame, mode)
                old, _ = choose(config)
                sel_second = picks.pick_from_row(frame.iloc[1])
                if variant in ("drop_second", "replace_second"):
                    old.double_down = sel_second
                file = picks.save_pick(old, root)
                if variant in ("malformed_slot", "missing_slot"):
                    body = json.loads(file.read_bytes())
                    if variant == "malformed_slot":
                        body["double_down"] = {"batter_name": "no required fields"}
                    else:
                        del body["double_down"]
                    file.write_text(json.dumps(body))
                    try:
                        picks.load_pick(DAY, root)
                    except (TypeError, KeyError) as exc:
                        assert daily_decision.load_decision(DAY, root) is None
                        print(mode, variant, "reader rejects", type(exc).__name__)
                        continue
                    raise AssertionError("Expected malformed saved pick to fail reading")
                old = picks.load_pick(DAY, root)
                if variant in ("drop_second", "replace_second"):
                    old.double_down = (None if variant == "drop_second"
                                       else picks.pick_from_row(frame.iloc[2]))
                    picks.save_pick(old, root)
                if variant == "failed":
                    stack.enter_context(patch.object(orchestrator, "run_cascade",
                                                     side_effect=RuntimeError("offline failed refresh")))
                else:
                    for status in statuses.values():
                        status["detailed"] = "Warmup"
                refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, old, .03)
                sel = refresh.selection
                assert (sel is None) == (variant == "failed")
                if sel is not None:
                    assert sel.decision is None and sel.primary_candidate is None
                if variant in ("candidate_without_decision", "drop_second", "replace_second"):
                    # Stress the writer's identity rule even if a future caller sends
                    # ranking data alongside an old selection that made no decision.
                    sel = replace(sel, primary_candidate=scheduler._row_from_daily(sel_second),
                                  double_candidate=scheduler._row_from_daily(sel_second))
                send = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="capture")))
                stack.enter_context(patch.object(posting, "post_to_bluesky", send))
                assert deliver(root, config, refresh.daily, sel)
                current = daily_decision.load_decision(DAY, root)
                assert identity(current) == ("double" if old.double_down else "single", 100,
                                             "private_locked" if mode == "private" else "delivered")
                assert current["double_down"] == scheduler._row_from_daily(old.double_down)
                assert current.get("policy_guard") == (None if variant == "unguarded" else GUARD)
                assert send.call_count == int(mode != "private")
                print(mode, variant, identity(current),
                      "second", (current["double_down"] or {}).get("batter_id"))
                if mode == "dm" and variant == "warmup":
                    assert old.chosen_on["streak"] == 50 and old.chosen_on["saver_available"] is False
                    assert current["streak"] == 0 and current["saver_available"] is None
                    ledger = root / "ledger.jsonl"
                    ledger.write_text(json.dumps({"recorded_at": f"{DAY}T00:00:00Z",
                                                  "active_streak": 50}))
                    try:
                        run_census(picks_dir=root, ledger_path=ledger,
                                   policy_path=ROOT / "data/models/mdp_policy.npz", saver_active=False,
                                   start=DAY, end=DAY, as_of=DAY, output=root / "census.json")
                    except CensusHalt as exc:
                        assert "gate A" in str(exc) and "'recorded': 0, 'ledger_asof': 50" in str(exc)
                        print("Warmup state mismatch:", str(exc))
                    else:
                        raise AssertionError("Expected a census halt with the wrong recorded state")


if __name__ == "__main__":
    compare_transitions(sys.argv[1])
    if sys.argv[1] == "head":
        cached_matrix()

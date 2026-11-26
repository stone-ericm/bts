"""Compare established delivery exits and state selection, on base and HEAD."""

import contextlib
import inspect
import io
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

from offline import forbid_network

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent


def child(revision):
    active_src = REVIEW / "base" / "src" if revision == "base" else ROOT / "src"
    sys.path[:0] = [str(active_src), str(ROOT)]
    forbid_network()
    from click.testing import CliRunner
    from pytest import MonkeyPatch

    from bts import cli, contest_state, dm, orchestrator, picks, posting, scheduler, strategy
    from bts import daily_decision
    from bts.daily_decision import load_decision
    from bts.health import alert
    from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, daily_pick, set_automatic, set_dated
    from tests.test_saved_pick_paths_cs5a import captured_fetch, invoke, prepare_run_day, scheduler_state, world

    assert Path(picks.__file__).resolve().is_relative_to(active_src)
    snapshots = {f"source/{function.__name__}": inspect.getsource(function) for function in (
        strategy.select_pick, scheduler._refresh_pick_at_fallback_decision,
        scheduler.plan_fallback_action, scheduler.run_single_check,
    )}

    def save(key, value):
        # The same strings must not differ because the fixture used a different
        # directory in the second process.
        serialized = json.dumps(value, default=str, sort_keys=True).replace(str(root), "<scratch>")
        snapshots[key] = json.loads(serialized)

    def delivery_facts(config, path):
        daily = picks.load_pick(DAY, path)
        data = asdict(daily) if daily is not None else None
        if data:
            data.pop("chosen_on", None)
        decision = load_decision(DAY, path)
        state = scheduler.load_state(DAY, path)
        return dict(pick=data, locked=state.pick_locked if state else False, decision=decision,
                    dms=[c.args for c in dm.send_dm.call_args_list],
                    posts=[c.args for c in posting.post_to_bluesky.call_args_list],
                    alerts=[c.args for c in alert.send_dm.call_args_list])

    def compare_exits(mode, name):
        config, path, clock, statuses = world.__wrapped__(root, mp)
        mp.setattr(daily_decision, "datetime", contest_state.datetime)
        daily = daily_pick()
        set_dated(path)
        config["scheduler"]["pick_delivery"] = mode
        if name == "delivered":
            daily.notification_sent, daily.notification_id = True, "prior-dm"
        if name == "attempted":
            daily.delivery_attempted = True
        if name == "private":
            config["scheduler"]["pick_delivery"] = "private"
        if name == "optional":
            config["scheduler"]["contest_state_required"] = False
        if name == "late":
            clock[0] = picks.submission_cutoff_et(daily)
        if name == "void":
            statuses[101] = {"abstract": "F", "detailed": "Postponed", "code": "PO"}
        if name == "read-error":
            (path / "account_state" / "contest_streak.manual.json").unlink()
        picks.save_pick(daily, path)
        no_judge = Mock(side_effect=AssertionError("early exit reached saved-pick judge"))
        mp.setattr(picks, "saved_pick_refusal", no_judge, raising=False)
        no_state = Mock(wraps=contest_state.load_decision_streak_state)
        mp.setattr(contest_state, "load_decision_streak_state", no_state)
        trace = io.StringIO()
        state = scheduler_state()
        with contextlib.redirect_stderr(trace):
            result = scheduler._deliver_and_lock_pick(daily, config, path, state, DAY, "probe")
        assert not no_judge.called
        assert no_state.call_count == (name in {"optional", "read-error"})
        facts = delivery_facts(config, path)
        facts.update(result=result, stderr=trace.getvalue())
        save(f"scheduler/{mode}/{name}", facts)

    def commands(command, name):
        config, path, _, statuses = world.__wrapped__(root, mp)
        set_dated(path)
        daily = daily_pick()
        daily.bluesky_posted = name == "already-posted"
        daily.bluesky_uri = "at://old" if daily.bluesky_posted else None
        picks.save_pick(daily, path)
        no_judge = Mock(side_effect=AssertionError("unchanged path reached saved-pick judge"))
        mp.setattr(picks, "saved_pick_refusal", no_judge, raising=False)
        if name == "optional-orchestrate":
            config["scheduler"]["contest_state_required"] = False
        if name in {"optional-orchestrate", "second-read-error"}:
            statuses.update(dict.fromkeys(statuses, {"abstract": "P", "detailed": "Warmup", "code": "PW"}))
        if name == "first-read-error":
            (path / "account_state" / "contest_streak.manual.json").unlink()
        if name == "second-read-error":
            reader = contest_state.load_decision_streak_state
            mp.setattr(contest_state, "load_decision_streak_state", Mock(side_effect=[
                reader(path, now=NOW, require_contest_state=True),
                contest_state.ContestStateError("captured second read error"),
            ]))
        trace = io.StringIO()
        if name == "optional-orchestrate":
            with contextlib.redirect_stderr(trace):
                rc = orchestrator.orchestrate(root / "unused.toml", DAY, config=config)
            result = (rc, trace.getvalue())
        else:
            try:
                execution = invoke(command, config, root)
                # bts run prints a wall-clock start line before model selection.
                result = (execution.exit_code, execution.stderr, execution.stdout.splitlines()[1:] if command == "run" else execution.stdout)
            except contest_state.ContestStateError as exc:
                result = (type(exc).__name__, str(exc))
        assert not no_judge.called
        save(f"command/{command}/{name}", {**delivery_facts(config, path), "result": result})

    def loader(auto, manual, switch, required):
        path = root / "picks"
        if auto != "absent":
            set_automatic(path, streak=5, covered=(switch == "covered"))
            auto_file = path / "account_state" / "contest_streak.json"
            body = json.loads(auto_file.read_text())
            if auto == "invalid":
                body["active_streak"] = 58
            if auto == "undated":
                body.pop("source_date")
                body.pop("recorded_at")
            auto_file.write_text(json.dumps(body))
        if switch in {"mismatch", "revoked", "corrupt"} or (switch == "covered" and auto == "absent"):
            account_file(path, "season_contract.json", {
                "schema_version": "bts_season_contract_v1", "season": 2027,
                "status": "revoked" if switch == "revoked" else "automatic",
                "activation_id": "b" * 32, "switched_at": NOW.isoformat(),
                "revoked_at": NOW.isoformat() if switch == "revoked" else None,
                "note": None,
            } if switch != "corrupt" else [])
        if manual in {"dated", "lapsed"}:
            set_dated(path, now=NOW if manual == "dated" else NOW - timedelta(days=1))
        if manual in {"legacy", "legacy-expiring", "invalid"}:
            account_file(path, "contest_streak.manual.json", {
                "active_streak": "bad" if manual == "invalid" else 31,
                "source": "mlb_bts_profile", "override_expires_at": (NOW + timedelta(hours=1)).isoformat() if manual == "legacy-expiring" else None,
            })
        try:
            snapshot = asdict(contest_state.load_decision_streak_state(path, now=NOW, require_contest_state=required))
            snapshot.pop("observation_kind", None)
            snapshot.pop("dated_identity", None)
        except contest_state.ContestStateError as exc:
            snapshot = [type(exc).__name__, str(exc)]
        save(f"loader/{auto}/{manual}/{switch}/{required}", snapshot)

    def ordinary(which):
        config, path, clock, _ = world.__wrapped__(root, mp)
        set_automatic(path, streak=5)
        frame, _ = orchestrator.run_cascade.return_value
        frame.loc[1, "flags"] = "PROJECTED"
        mp.setattr(scheduler, "check_confirmed_lineups", lambda *_: {})
        first = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
        assert not first["should_post"] and first["selection"].decision is not None
        chosen = picks.load_pick(DAY, path)
        assert chosen and not dm.send_dm.called
        if which == "flag":
            account_file(path, "saver_state.json", {"season": 2027, "state": "active"})
        else:
            captured_fetch(path, mp, 6, reactivate=(which == "reactivated"))
        orchestrator.run_cascade.return_value = (None, None)
        prepare_run_day(mp)
        scheduler.run_day(date=DAY, config=config)
        assert dm.send_dm.call_count == 1 and not alert.send_dm.called
        assert scheduler.load_state(DAY, path).pick_locked
        delivered = load_decision(DAY, path)
        assert delivered["delivery_status"] == "delivered"
        save(f"ordinary/{which}", {"pick_ids": [chosen.pick.batter_id, chosen.double_down.batter_id if chosen.double_down else None],
                                  "text": dm.send_dm.call_args.args[1], "status": delivered["delivery_status"]})

    from itertools import product

    trace = io.StringIO()
    jobs = [(compare_exits, args) for args in product(["dm", "public"],
             ["delivered", "attempted", "private", "optional", "late", "void", "read-error"])]
    jobs += [(commands, args) for args in product(["run", "orchestrate"],
             ["already-posted", "first-read-error", "second-read-error"])]
    jobs += [(commands, ("orchestrate", "optional-orchestrate"))]
    jobs += [(loader, args) for args in product(["absent", "valid", "undated", "invalid"],
             ["absent", "dated", "lapsed", "legacy", "legacy-expiring", "invalid"],
             ["absent", "covered", "mismatch", "revoked", "corrupt"], [False, True])]
    jobs += [(ordinary, (change,)) for change in ["fetch", "flag", "reactivated"]]
    for test, args in jobs:
        with tempfile.TemporaryDirectory(dir=REVIEW) as temp, MonkeyPatch.context() as mp:
            root = Path(temp)
            with contextlib.redirect_stderr(trace):
                test(*args)
    (REVIEW / f"comparison-{revision}.json").write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        for label in ("base", "branch"):
            subprocess.run([sys.executable, __file__, label], check=True)
        previous, current = [json.loads((REVIEW / f"comparison-{label}.json").read_text()) for label in ("base", "branch")]
        different = [key for key in previous if previous[key] != current[key]]
        for key in different:
            print(key, previous[key], current[key])
        assert not different and previous.keys() == current.keys(), different
        source_comparisons = sum(key.startswith("source/") for key in previous)
        print(f"{len(previous) - source_comparisons} base/branch cases and {source_comparisons} source bodies agreed")

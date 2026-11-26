"""Compare archived base and branch using the real delivery and file paths."""

import argparse
import contextlib
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from bts import contest_state, daily_decision, dm, picks, posting, scheduler
from bts.health import alert
from tests.conftest import stamp_saved_pick
from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, daily_pick, set_automatic, set_dated
from tests.test_saved_pick_paths_cs5a import scheduler_state, world
from tests.test_stale_delivery_cs4c import OLD, WARNING_FILE, fresh, seed_automatic


SCRATCH = Path(__file__).resolve().parent
CRITICAL = "health_dm_delivery_status.json"
BRANCH = hasattr(contest_state, "automatic_reading_record")
WORKSPACE = None


class SimulatedExit(BaseException):
    pass


class BrokenDiagnostic(RuntimeError):
    def __str__(self):
        raise KeyboardInterrupt("diagnostic")


class CloseAfterLock(io.StringIO):
    close_on_newline = False

    def write(self, text):
        result = super().write(text)
        if self.close_on_newline and text == "\n":
            self.saved = self.getvalue()
            self.close()
        elif text.startswith("  LOCKED") and ("sent:" in text or "Posted to" in text):
            self.close_on_newline = True
        return result


def seed(directory, reading):
    if reading == "dated":
        set_dated(directory)
    elif reading in {"old", "fresh", "future", "unreadable", "missing-time"}:
        recorded = {
            "old": OLD["recorded_at"], "fresh": (NOW - timedelta(hours=12)).isoformat(),
            "future": (NOW + timedelta(hours=1)).isoformat(),
            "unreadable": "not ISO", "missing-time": None,
        }[reading]
        seed_automatic(directory, recorded)
    elif reading == "uncovered":
        set_automatic(directory, covered=False)
    elif reading == "legacy":
        account_file(directory, "contest_streak.manual.json", {
            "active_streak": 7, "source_date": DAY, "recorded_at": "literal",
        })
    elif reading == "revoked":
        set_automatic(directory)
        path = directory / "account_state" / "season_contract.json"
        data = json.loads(path.read_bytes())
        data.update(status="revoked", revoked_at=NOW.isoformat())
        path.write_text(json.dumps(data))
    elif reading == "corrupt":
        set_automatic(directory)
        (directory / "account_state" / "season_contract.json").write_text("{")


def normalized_file(path):
    raw = path.read_bytes().decode()
    if path.name == f"{DAY}.json" or path.name.startswith("refused_delivery"):
        data = json.loads(raw)
        if "automatic_reading" in data:
            keys = list(data)
            index = keys.index("automatic_reading")
            assert keys[index - 1] == "chosen_on", keys
            field = json.dumps({"automatic_reading": data.pop("automatic_reading")}, indent=2)[2:-2]
            raw = raw.replace(f"{field},\n" if index < len(keys) - 1 else f",\n{field}", "")
            assert raw == json.dumps(data, indent=2)
    return raw


def snapshot(root):
    return {
        str(path.relative_to(root)): normalized_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != WARNING_FILE
        and "account_state" not in path.parts and path.name != "seeded"
    }


@contextlib.contextmanager
def cell_directory():
    root = WORKSPACE / "case"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


def run_cell(reading, mode, required, marker, choice, outcome="id"):
    with cell_directory() as root:
        with pytest.MonkeyPatch.context() as patch:
            config, directory, clock, statuses = world.__wrapped__(root, patch)
            config["scheduler"].update(contest_state_required=required, pick_delivery=mode)
            if outcome == "no-recipient":
                del config["bluesky"]["dm_recipient"]
            csv = contest_state.datetime
            for module in (picks, daily_decision, scheduler):
                patch.setattr(module, "datetime", csv)
            events = []
            snapshots = []
            capture = Mock(side_effect=lambda *args: events.append("capture"))
            patch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", capture)
            state = scheduler_state()
            seed(directory, reading)
            daily = daily_pick()
            if choice in {"saved", "same-reuse", "wrong-object"}:
                try:
                    decision = contest_state.load_decision_streak_state(
                        directory, require_contest_state=True,
                    )
                except contest_state.ContestStateError:
                    pass
                else:
                    stamp_saved_pick.__wrapped__()(daily, decision)
            prior = dict(OLD, stale=False, age_hours=12.0, reason=None)
            if BRANCH:
                daily.automatic_reading = prior
            if marker == "attempted":
                daily.delivery_attempted = True
            elif marker == "delivered":
                daily.notification_sent = True
                daily.notification_id = "prior"
            path = picks.save_pick(daily, directory)
            if choice == "saved":
                daily = picks.load_pick(DAY, directory)
            selection = None
            if choice in {"fresh", "wrong-object", "same-reuse"}:
                selection = fresh(daily_pick() if choice == "wrong-object" else daily)
                if choice == "same-reuse":
                    selection.decision = None
            status = root / "health_state" / CRITICAL
            status.parent.mkdir()
            status.write_bytes(b'{ "seed": true, "status": "sent", "sent_day": "2027-07-15" }\n')
            warn_status = status.with_name(WARNING_FILE)
            if outcome == "directory-status":
                warn_status.mkdir()
            if outcome == "readonly-status":
                warn_status.write_bytes(b'{"status":"seeded"}')
                warn_status.chmod(0o444)
            if outcome == "parent-is-file":
                status.unlink()
                status.parent.rmdir()
                status.parent.write_text("seeded")
            if outcome in {"initial-late", "final-cutoff"}:
                cutoff = picks.submission_cutoff_et(daily)
                times = iter([cutoff] if outcome == "initial-late" else [NOW, cutoff])
                patch.setattr(scheduler, "_now_et", lambda: next(times, cutoff))
            if outcome == "all-void":
                statuses[101] = {"abstract": "F", "detailed": "Postponed", "code": "D"}

            read = Mock(wraps=contest_state.load_decision_streak_state)
            patch.setattr(contest_state, "load_decision_streak_state", read)
            compute = Mock(wraps=getattr(contest_state, "automatic_reading_record", None))
            patch.setattr(contest_state, "automatic_reading_record", compute, raising=False)
            error = RuntimeError("probe\nexception")
            special = {"interrupt": KeyboardInterrupt, "exit": SystemExit,
                       "generator-exit": GeneratorExit, "custom": SimulatedExit,
                       "exception-group": ExceptionGroup("warning", [error]),
                       "base-group": BaseExceptionGroup("warning", [KeyboardInterrupt()])}
            where, _, scenario = outcome.partition(":")
            if where == "calculation":
                compute.side_effect = special.get(scenario, error)
            if where == "malformed":
                age = object() if scenario == "unserializable" else "bad"
                compute.return_value = {"stale": True, "reason": "old", "age_hours": age}
            if where == "dispatcher":
                patch.setattr(alert, "dispatch_dm_for_health_alerts",
                              Mock(side_effect=special.get(scenario, BrokenDiagnostic() if
                                   scenario == "diagnostic" else error)))
            if where == "warning-clock":
                def after_commit(*args, **kwargs):
                    daily_decision_commit(*args, **kwargs)
                    patch.setattr(scheduler, "_now_et", Mock(side_effect=special.get(scenario, error)))

                daily_decision_commit = scheduler._commit_decision_for_pick
                patch.setattr(scheduler, "_commit_decision_for_pick", after_commit)
            if where == "warning-stderr":
                def print_with_interrupt(*args, **kwargs):
                    if args[0].startswith("  STALE CONTEST READING"):
                        raise KeyboardInterrupt("stderr")
                    print(*args, **kwargs)

                patch.setattr(scheduler, "print", print_with_interrupt, raising=False)

            saves = picks.save_pick

            def observe_save(obj, directory):
                events.append("save")
                saved_path = saves(obj, directory)
                value = json.loads(saved_path.read_bytes())
                snapshots.append((value["delivery_attempted"], value["delivered_at"],
                                  value.get("automatic_reading")))
                return saved_path

            patch.setattr(picks, "save_pick", observe_save)

            def send(*args):
                events.append("pick")
                if outcome == "raises":
                    raise error
                if outcome == "transport-crash":
                    raise SimulatedExit
                return {"no-id": None, "empty-id": ""}.get(outcome, "confirmed")

            def health(recipient, text):
                warning = text.startswith("BTS health WARN attention:")
                events.append("warn" if warning else "alert")
                if outcome == "warn-transport-interrupt" and warning:
                    raise KeyboardInterrupt
                return "health"

            dm.send_dm.side_effect = posting.post_to_bluesky.side_effect = send
            alert.send_dm.side_effect = health
            stderr = CloseAfterLock() if outcome == "closed-stderr" else io.StringIO()
            if outcome == "stderr-closed-at-entry":
                stderr.saved = ""
                stderr.close()
            with contextlib.redirect_stderr(stderr):
                try:
                    result = scheduler._deliver_and_lock_pick(
                        daily, config, directory, state, DAY, "review", selection=selection,
                    )
                except BaseException as exc:
                    result = {"raised": type(exc).__name__}
            lines = (stderr.saved if stderr.closed else stderr.getvalue()).splitlines()
            pick = json.loads(path.read_bytes()) if path.exists() else {}
            warning, existing = [], []
            for call in alert.send_dm.call_args_list:
                target = warning if call.args[1].startswith("BTS health WARN attention:") else existing
                target.append(call.args)
            warning_stderr = [
                line for line in lines if line.startswith("  STALE CONTEST READING")
                or "Stale-reading warning failed:" in line or WARNING_FILE in line
            ]
            unchanged = {
                "result": result, "state": asdict(state),
                "pick_dm": [call.args for call in dm.send_dm.call_args_list],
                "public": [call.args for call in posting.post_to_bluesky.call_args_list],
                "health": existing, "capture": [call.args[1] for call in capture.call_args_list],
                "stderr": [line for line in lines if line not in warning_stderr],
                "files": snapshot(root),
            }
            return {
                "existing": unchanged, "reads": read.call_count, "computes": compute.call_count,
                "compute_time": compute.call_args.args[1].isoformat() if compute.called else None,
                "same_snapshot": compute.call_args.args[0].contest_recorded_at if compute.called else None,
                "value": pick.get("automatic_reading"), "saves": snapshots,
                "events": events, "warn": warning, "warn_stderr": warning_stderr,
                "warn_status": json.loads(warn_status.read_bytes()) if warn_status.is_file() else None,
                "criteria_delivered": picks.pick_was_delivered(picks.load_pick(DAY, directory))
                if path.exists() else False,
            }


def state_matrix():
    result = {}
    axes = (("dated", "old", "fresh", "unreadable", "missing-time", "future", "legacy",
             "uncovered", "revoked", "corrupt", "model"),
            (False, True), (-1, 0, 1))
    for reading, required, offset in itertools.product(*axes):
        with cell_directory() as root:
            seed(root, reading)
            try:
                value = asdict(contest_state.load_decision_streak_state(
                    root, now=NOW + timedelta(days=offset), require_contest_state=required,
                ))
            except contest_state.ContestStateError as exc:
                value = {"refusal": str(exc)}
            result[f"{reading}/{required}/{offset}"] = {
                key: val.isoformat() if isinstance(val, date) else val
                for key, val in value.items()
            }
    return result


def collect():
    matrix = {}
    for cell in itertools.product(
        ("dated", "old", "fresh", "future", "unreadable", "uncovered", "model"),
        ("dm", "public", "private"), (True, False), ("none", "attempted", "delivered"),
        ("fresh", "saved", "none", "same-reuse", "wrong-object"),
    ):
        matrix["/".join(map(str, cell))] = run_cell(*cell)
    faults = {}
    for mode, outcome in itertools.product(("dm", "public"), (
        "raises", "no-id", "empty-id", "all-void", "initial-late", "final-cutoff",
        "no-recipient", "transport-crash", "closed-stderr", "stderr-closed-at-entry",
        "directory-status", "readonly-status", "parent-is-file",
        "malformed", "malformed:unserializable", "warn-transport-interrupt",
        "warning-stderr:interrupt", "calculation:error", "calculation:interrupt",
        "calculation:exit", "calculation:generator-exit", "calculation:custom",
        "dispatcher:error", "dispatcher:interrupt", "dispatcher:exit", "dispatcher:generator-exit",
        "dispatcher:custom", "dispatcher:diagnostic", "dispatcher:exception-group",
        "dispatcher:base-group", "warning-clock:error", "warning-clock:interrupt",
    )):
        faults[f"{mode}/{outcome}"] = run_cell("old", mode, True, "none", "fresh", outcome)
    return {"module": contest_state.__file__, "matrix": matrix, "faults": faults, "states": state_matrix()}


def compare(base, branch):
    print(base["module"], branch["module"], sep="\n")
    classifications = {}
    difference_count = 0
    for label, before in base["matrix"].items():
        after = branch["matrix"][label]
        for name in before["existing"]:
            assert before["existing"][name] == after["existing"][name], (label, name)
        reading, mode, required, marker, _ = label.split("/")
        qualifies = (
            mode != "private" and required == "True" and marker == "none"
            and ("pick" in after["events"])
        )
        assert before["reads"] == after["reads"], label
        assert after["computes"] == int(qualifies), label
        if qualifies:
            assert after["compute_time"] == NOW.isoformat(), label
            assert after["reads"] == 1 and len(after["saves"]) == 2, label
            assert after["saves"][0] == [True, None, after["value"]], label
            assert after["saves"][1] == [True, NOW.isoformat(), after["value"]], label
            reason = {"dated": None, "fresh": None, "old": "old", "future": "future",
                      "unreadable": "unreadable"}[reading]
            assert (after["value"]["reason"] if after["value"] else None) == reason, label
            assert len(after["warn"]) == len(after["warn_stderr"]) == int(reason is not None), label
            if reason:
                assert after["events"][-2:] == ["capture", "warn"], label
            classifications[reading] = classifications.get(reading, 0) + 1
        else:
            assert not after["warn"] and not after["warn_stderr"], label
    for label, before in base["states"].items():
        after = branch["states"][label].copy()
        after.pop("contest_recorded_at", None)
        assert before == after, (label, before, after)
    for label, before in base["faults"].items():
        after = branch["faults"][label]
        changes = [key for key in before["existing"]
                   if before["existing"][key] != after["existing"][key]]
        code_changes = bool(changes)
        difference_count += code_changes
        print(f"{label}: base={before['existing']['result']} branch={after['existing']['result']}; "
              f"changes={','.join(changes) or '-'}; pick={after['events'].count('pick')}; "
              f"compute={after['computes']}; delivered={after['criteria_delivered']}; "
              f"capture={after['events'].count('capture')}; warn={len(after['warn'])}; "
              f"diagnostics={len(after['warn_stderr'])}")
    print(f"{len(base['matrix'])} exact existing-output comparisons; stamps {classifications}")
    print(f"{len(base['states'])} loader comparisons; {len(base['faults'])} fault cells; "
          f"{difference_count} altered existing outputs")


def main():
    global WORKSPACE
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["replay", "collect", "compare"])
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.action == "replay":
        root = SCRATCH.parent.parent
        base = SCRATCH / "base"
        results = []
        with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="delivery-") as folder:
            for name, source in (("base", base), ("branch", root)):
                result = SCRATCH / f"delivery-{name}.json"
                env = os.environ | {"PYTHONPATH": os.pathsep.join(map(str, (
                    SCRATCH, source / "src", source, root,
                )))}
                subprocess.run(
                    [sys.executable, __file__, "collect", str(result), "--workspace", folder],
                    env=env, cwd=root, check=True,
                )
                results.append(json.loads(result.read_bytes()))
        compare(*results)
    elif args.action == "collect":
        WORKSPACE = args.workspace
        args.paths[0].write_text(json.dumps(collect(), indent=2))
    else:
        compare(*(json.loads(path.read_bytes()) for path in args.paths))


if __name__ == "__main__":
    main()

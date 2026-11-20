"""Probe opt-in acknowledgements, damaged receipts and interrupt windows."""

from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import stat
import sys

from probe_support import (
    FAILURE_PREFIX, attempt_direct, day_pick, deadline, fixtures, prepare_schedules, raw,
    receipt, shared, status_body, world_at,
)


DAY = "2027-03-25"
SENT_FIELDS = {"message_id", "sent_day", "sent_sources"}


def failure_lines(journal):
    return [line for line in journal.getvalue().splitlines() if line.startswith(FAILURE_PREFIX)]


def own_payload(**fields):
    from bts.health.alert import HEALTH_DM_STATUS_SCHEMA_VERSION

    return {
        "schema_version": HEALTH_DM_STATUS_SCHEMA_VERSION,
        "status": "sent", "message_id": "prior-ack", "sent_day": DAY,
        "sent_sources": ["CRITICAL:late_delivery"],
        **fields,
    }


def route(root, patch, opening):
    return world_at(root, patch, opening, opening.at(DAY, "18:20"))


def acknowledgements(root, patch, opening, label, reply):
    from bts import dm, picks, scheduler
    from bts.health import alert

    with route(root, patch, opening) as (config, directory, _, _, scoped):
        daily = day_pick(DAY)
        daily.delivery_attempted = True
        prepare_schedules(daily)
        path = receipt(directory, DAY)
        legacy = shared(directory)
        legacy.parent.mkdir()
        legacy.write_bytes(b'{"status":"failed","critical_count":1}\n')
        checkpoint = legacy.read_bytes()
        dispatcher = alert.dispatch_dm_for_health_alerts
        returns = []

        def observe(*args, **kwargs):
            value = dispatcher(*args, **kwargs)
            returns.append(value)
            return value

        scoped.setattr(alert, "dispatch_dm_for_health_alerts", observe)
        if label == "no_recipient":
            config["bluesky"].pop("dm_recipient")
            alert.send_dm.return_value = "retry-ack"
        else:
            alert.send_dm.side_effect = [reply, "retry-ack"]
        first = StringIO()
        with redirect_stderr(first):
            assert attempt_direct(config, directory, daily) is False
        assert legacy.read_bytes() == checkpoint
        payload = json.loads(path.read_bytes())
        actual = payload["status"]
        if label == "usable":
            assert actual == "sent" and payload["message_id"] == reply and not failure_lines(first)
        else:
            assert actual == ("skipped_no_recipient" if label == "no_recipient" else "failed")
            assert not SENT_FIELDS & payload.keys()
            assert len(failure_lines(first)) == 1
            if label not in {"no_recipient", "exception"}:
                assert payload["error"] == "send_dm returned no usable message id"
            if label == "exception":
                assert r"bad\nresponse" in failure_lines(first)[0]
        before_second = path.read_bytes()
        config["bluesky"]["dm_recipient"] = "offline.test"
        assert scheduler.run_day(DAY, config) is None
        assert legacy.read_bytes() == checkpoint
        after_second = path.read_bytes()
        assert scheduler.run_day(DAY, config) is None
        assert path.read_bytes() == after_second and legacy.read_bytes() == checkpoint
        expected_returns = ([True, False, False] if label == "usable" else
                            [False, True, False] if label == "no_recipient" else [True, True, False])
        assert returns == expected_returns
        expected_count = 1 if label in {"usable", "no_recipient"} else 2
        assert [call.args[1] for call in alert.send_dm.call_args_list] == [status_body(DAY)] * expected_count
        assert (before_second == after_second) is (label == "usable")
        assert json.loads(after_second)["sent_day"] == DAY
        assert json.loads(after_second)["sent_sources"] == ["CRITICAL:late_delivery"]
        assert not picks.pick_was_delivered(picks.load_pick(DAY, directory))
        dm.send_dm.assert_not_called()
    print(f"reply={label}: first status={actual}, attempts={returns}, body stable across routes.")


def damage(root, patch, opening, injury):
    from bts import dm, scheduler
    from bts.health import alert

    with route(root, patch, opening) as (config, directory, _, _, _):
        daily = day_pick(DAY)
        daily.delivery_attempted = True
        prepare_schedules(daily)
        path = receipt(directory, DAY)
        legacy = shared(directory)
        legacy.parent.mkdir()
        original = json.dumps(own_payload()).encode()
        legacy.write_bytes(original)
        evidence = path
        if injury == "blocked_parent":
            path.parent.write_bytes(b"not a directory")
            evidence = path.parent
        elif injury == "symlink_parent":
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            path.parent.symlink_to(elsewhere, target_is_directory=True)
            evidence = path.parent
        else:
            path.parent.mkdir()
            if injury == "directory":
                path.mkdir()
            elif injury == "fifo":
                os.mkfifo(path)
            elif injury == "unreadable":
                path.write_bytes(original)
                path.chmod(0)
                assert not os.access(path, os.R_OK), "probe must run without elevated file access"
            elif injury == "unwritable_parent":
                path.parent.chmod(0o500)
                assert not os.access(path.parent, os.W_OK), "probe must run without elevated file access"
                evidence = path.parent
            elif injury == "symlink":
                path.symlink_to(legacy)
            elif injury == "hardlink":
                os.link(legacy, path)
            else:
                payload = {
                    "object": b"{ }", "truncated": b"{", "array": b"[]", "null": b"null",
                    "wrong_sent": json.dumps(own_payload(sent_sources={"late_delivery": True})).encode(),
                    "bad_id": json.dumps(own_payload(message_id="  ")).encode(),
                    "wrong_date": json.dumps(own_payload(sent_day="2027-03-26")).encode(),
                    "wrong_schema": json.dumps(own_payload(schema_version=["damaged"])).encode(),
                }[injury]
                path.write_bytes(payload)
        initial_info = evidence.lstat()
        journal = StringIO()
        try:
            with redirect_stderr(journal):
                assert attempt_direct(config, directory, daily) is False
                assert scheduler.run_day(DAY, config) is None
            assert legacy.read_bytes() == original
            assert scheduler.load_state(DAY, directory).pick_locked
            assert scheduler.load_state(DAY, directory).committed_pick_written
            dm.send_dm.assert_not_called()
            failures = failure_lines(journal)
            broken_storage = injury in {"unreadable", "directory", "fifo", "symlink", "hardlink",
                                        "symlink_parent", "blocked_parent", "unwritable_parent"}
            actual_count = alert.send_dm.call_count
            expected_count = 2 if broken_storage else 1
            assert [call.args[1] for call in alert.send_dm.call_args_list] == [status_body(DAY)] * actual_count
            if broken_storage:
                assert len(failures) == 2
                for line in failures:
                    assert len(line.removeprefix(FAILURE_PREFIX)) <= 200
                    assert str(path) in line and "receipt" in line
                after = evidence.lstat()
                assert (after.st_ino, stat.S_IFMT(after.st_mode)) == (
                    initial_info.st_ino, stat.S_IFMT(initial_info.st_mode),
                )
                if injury == "symlink_parent":
                    assert not list(elsewhere.iterdir())
            else:
                assert not failures
            if actual_count != expected_count:
                assert injury == "wrong_schema", (injury, actual_count, expected_count)
                assert actual_count == 0 and path.read_bytes() == payload
                print("CONTRACT MISMATCH: schema_version=['damaged'] suppressed both encounters; "
                      "malformed receipt was treated as acknowledgement.")
            print(f"{injury}: sends={actual_count} (contract {expected_count}), diagnostics={len(failures)}.")
        finally:
            if injury == "unwritable_parent":
                path.parent.chmod(0o700)
            elif injury == "unreadable":
                path.chmod(0o600)
    return actual_count == expected_count


def interrupted(root, patch, opening, entry, exception, write_landed):
    from bts import daily_decision, dm, picks, scheduler
    from bts.health import alert

    with route(root, patch, opening) as (config, directory, _, _, scoped):
        daily = day_pick(DAY)
        daily.delivery_attempted = True
        prepare_schedules(daily)
        picks.save_pick(daily, directory)
        legacy = shared(directory)
        legacy.parent.mkdir()
        legacy.write_bytes(b"[]\n")
        path = receipt(directory, DAY)
        decision = daily_decision.decision_path(DAY, directory)
        replace = os.replace
        accepted = []

        def replace_boundary(source, target, **kwargs):
            if target != path.name or kwargs.get("dst_dir_fd") is None:
                return replace(source, target, **kwargs)
            assert alert.send_dm.call_count == 1
            accepted.append(alert.send_dm.call_args.args[1])
            assert scheduler.load_state(DAY, directory).pick_locked
            assert scheduler.load_state(DAY, directory).committed_pick_written
            assert raw(decision) is not None
            if write_landed:
                replace(source, target, **kwargs)
            raise exception("receipt write boundary")

        with scoped.context() as fault:
            fault.setattr(os, "replace", replace_boundary)
            try:
                if entry == "direct":
                    attempt_direct(config, directory, daily)
                else:
                    scheduler.run_day(DAY, config)
            except exception:
                pass
            else:
                raise AssertionError("interrupt was swallowed")
        original = decision.read_bytes()
        assert path.exists() is write_landed and accepted == [status_body(DAY)]
        assert not list(path.parent.glob(f".{path.name}.*"))
        assert legacy.read_bytes() == b"[]\n"
        for _ in range(2):
            scheduler.run_day(DAY, config)
            assert decision.read_bytes() == original
            assert legacy.read_bytes() == b"[]\n"
        total = alert.send_dm.call_count
        assert total == (1 if write_landed else 2)
        dm.send_dm.assert_not_called()
    print(f"{entry} {exception.__name__} (receipt replaced={write_landed}): "
          f"propagated, total alerts={total}, no recommendation resend.")


def main():
    with deadline(50), fixtures("receipt") as (root, patch, opening):
        for label, reply in [
            ("none", None), ("empty", ""), ("whitespace", " \t "), ("number", 17),
            ("exception", RuntimeError("bad\nresponse")), ("usable", "  original id  "),
            ("no_recipient", None),
        ]:
            acknowledgements(root / label, patch, opening, label, reply)
        for injury in (
            "object", "truncated", "array", "null", "wrong_sent", "bad_id", "wrong_date",
            "directory", "fifo", "unreadable", "symlink", "hardlink", "unwritable_parent",
            "blocked_parent", "symlink_parent", "wrong_schema",
        ):
            damage(root / injury, patch, opening, injury)
        for entry in ("direct", "classification"):
            for exception in (KeyboardInterrupt, SystemExit):
                for write_landed in (False, True):
                    interrupted(root / f"{entry}-{exception.__name__}-{write_landed}",
                                patch, opening, entry, exception, write_landed)


if __name__ == "__main__":
    main()

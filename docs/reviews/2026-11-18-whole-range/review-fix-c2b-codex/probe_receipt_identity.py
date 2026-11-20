"""Challenge the production receipt with two real scheduler entry routes."""

from contextlib import redirect_stderr
from datetime import date, datetime, timedelta
from io import StringIO
import json
from pathlib import Path
import sys
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parent / "replay" / "codex"))
from probe_support import (ET, attempt_direct, day_pick, deadline, fixtures,
                           prepare_schedules, receipt, shared, status_body, world_at)

from bts import dm, picks, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_saved_pick_paths_cs5a import scheduler_state


DAY = "2027-03-25"
OTHER_DAY = "2027-03-26"
NOW = datetime.fromisoformat(f"{DAY}T18:20:00").replace(tzinfo=ET)
KEY = "CRITICAL:late_delivery"
SHAPE = {"schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION, "status": "sent",
         "sent_day": DAY, "sent_sources": [KEY], "message_id": "seed-id"}


def stage_pick(day, directory):
    daily = day_pick(day)
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)
    return daily


def encounter(route, day, config, directory, daily):
    if route == "direct":
        assert attempt_direct(config, directory, daily) is False
    else:
        assert scheduler.run_day(day, config) is None


def wrong_field_cases():
    invalid = {
        "schema_version": ["damaged", [SHAPE["schema_version"]]],
        "status": ["failed", "skipped_no_recipient", "SENT", ["sent"]],
        "sent_day": [OTHER_DAY, DAY + " ", [DAY]],
        "sent_sources": [KEY, [], ["CRITICAL:pick_delivery"], [KEY, KEY],
                         [KEY, "CRITICAL:pick_delivery"], ["CRITICAL:pick_delivery", KEY],
                         [True], [None], {KEY: True}],
        "message_id": ["", "\t \r\n", "\u2003", ["seed-id"]],
    }
    for field, examples in invalid.items():
        yield f"missing {field}", json.dumps({k: v for k, v in SHAPE.items() if k != field})
        for candidate in [None, False, 17, 1.5, {}, *examples]:
            yield f"{field}={candidate!r}", json.dumps(SHAPE | {field: candidate})


def from_cutoff(config, directory, sender):
    other_store = directory.parent / "cutoff_candidates"
    other_store.mkdir()
    daily = day_pick(DAY)
    picks.save_pick(daily, other_store)
    sender.return_value = "cutoff-id"
    scheduler._refuse_late_delivery(daily, config, other_store, scheduler_state(), DAY,
                                    "receipt probe", NOW - timedelta(minutes=1), NOW)
    sent_body = sender.call_args.args[1]
    assert "LATE DELIVERY REFUSED" in sent_body
    old = shared(directory).read_bytes()
    assert json.loads(old)["sent_day"] == DAY
    assert json.loads(old)["sent_sources"] == [KEY]
    assert "message_id" not in json.loads(old)
    sender.reset_mock()
    return old, sent_body.split("[late_delivery] ", 1)[1]


def seed(label, document, config, directory, sender, now):
    path = receipt(directory, DAY)
    if label == "real prior C2 acknowledgement":
        daily = stage_pick(DAY, directory)
        sender.return_value = "seed-id"
        assert attempt_direct(config, directory, daily) is False
        sender.assert_called_once_with("offline.test", status_body(DAY))
        sender.reset_mock()
    elif label == "real other-slate copy":
        other = stage_pick(OTHER_DAY, directory)
        prepare_schedules(other)
        now[0] = NOW + timedelta(days=1)
        assert scheduler.run_day(OTHER_DAY, config) is None
        sender.assert_called_once_with("offline.test", status_body(OTHER_DAY))
        sender.reset_mock()
        now[0] = NOW
        path.write_bytes(receipt(directory, OTHER_DAY).read_bytes())
    elif label in {"real legacy cutoff copy", "cutoff copy with message ID", "other opt-in alert"}:
        legacy, cutoff_text = from_cutoff(config, directory, sender)
        if label == "real legacy cutoff copy":
            path.write_bytes(legacy)
        elif label == "cutoff copy with message ID":
            path.write_text(json.dumps(json.loads(legacy) | {"message_id": "cutoff-id"}))
        else:
            assert alert.dispatch_dm_for_health_alerts(
                [alert.Alert("CRITICAL", "late_delivery", cutoff_text)], "offline.test",
                status_path=path, now_et_date=date.fromisoformat(DAY), require_message_id=True,
            ) is True
            assert "LATE DELIVERY REFUSED" in sender.call_args.args[1]
            assert json.loads(path.read_bytes()) == SHAPE | {"message_id": "cutoff-id"}
            sender.reset_mock()
    else:
        path.write_text(document)
    return path.read_bytes()


def observe(root, patch, opening, route, label, document):
    with world_at(root, patch, opening, NOW) as (config, directory, now, _, scoped):
        clock = Mock(wraps=alert.datetime)
        clock.now.side_effect = lambda tz=None: now[0].astimezone(tz)
        scoped.setattr(alert, "datetime", clock)
        sender = Mock(return_value="fresh-id")
        scoped.setattr(alert, "send_dm", sender)
        path = receipt(directory, DAY)
        path.parent.mkdir(parents=True)
        initial = seed(label, document, config, directory, sender, now)
        shared_before = shared(directory).read_bytes() if shared(directory).exists() else None
        sender.return_value = "fresh-id"
        daily = stage_pick(DAY, directory)
        prepare_schedules(daily)
        stderr = StringIO()
        with redirect_stderr(stderr):
            encounter(route, DAY, config, directory, daily)
            after = path.read_bytes()
            assert scheduler.run_day(DAY, config) is None
        assert scheduler.load_state(DAY, directory).pick_locked
        assert load_decision(DAY, directory)["delivery_status"] == "locked_unconfirmed"
        assert path.read_bytes() == after
        assert (shared(directory).read_bytes() if shared(directory).exists() else None) == shared_before
        assert "Unknown-outcome alert failed: " not in stderr.getvalue()
        dm.send_dm.assert_not_called()
        result = sender.call_count
        if result:
            sender.assert_called_once_with("offline.test", status_body(DAY))
            assert json.loads(after) == SHAPE | {"message_id": "fresh-id"}
        else:
            assert after == initial
        return result


def multi_source_order(root, patch):
    path = root / "receipt.json"
    keys = [KEY, "CRITICAL:pick_delivery"]
    path.write_text(json.dumps(SHAPE | {"sent_sources": list(reversed(keys))}))
    sender = Mock(return_value="fresh-id")
    patch.setattr(alert, "send_dm", sender)
    args = ([alert.Alert("CRITICAL", "pick_delivery", "one"),
             alert.Alert("CRITICAL", "late_delivery", "two")], "offline.test")
    kwargs = dict(status_path=path, now_et_date=date.fromisoformat(DAY), require_message_id=True)
    assert alert.dispatch_dm_for_health_alerts(*args, **kwargs) is True
    assert json.loads(path.read_bytes())["sent_sources"] == keys
    assert alert.dispatch_dm_for_health_alerts(*args, **kwargs) is False
    sender.assert_called_once()
    print("Reordered two-source opt-in receipt: sent once, wrote sorted order, then deduped.")


def main():
    suspicious = {
        "extra contradictory error": json.dumps(SHAPE | {"error": "send_dm returned no usable message id"}),
        "extra unknown key": json.dumps(SHAPE | {"unexpected": [False]}),
        "duplicate failed then sent status": '{"status":"failed",' + json.dumps(SHAPE)[1:],
        "duplicate bad then correct schema": '{"schema_version":null,' + json.dumps(SHAPE)[1:],
        "cutoff copy with message ID": None,
        "other opt-in alert": None,
    }
    controls = dict(wrong_field_cases()) | {
        "malformed JSON": "{", "wrong top-level type": "[]", "real other-slate copy": None,
        "real legacy cutoff copy": None, "real prior C2 acknowledgement": None,
    }
    divergences = []
    with deadline(50), fixtures("receipt-identity") as (root, patch, opening):
        for route in ("direct", "classification"):
            for index, (label, document) in enumerate((controls | suspicious).items()):
                result = observe(root / f"{route}-{index}", patch, opening, route, label, document)
                expected = int(label != "real prior C2 acknowledgement")
                print(f"{route}: {label}: {result} C2 DMs across two encounters (contract {expected})")
                if result != expected:
                    assert label in suspicious, (route, label, result)
                    divergences.append((route, label))
        multi_source_order(root, patch)
    print(f"{len(divergences)} mismatches across {2 * (len(controls) + len(suspicious))} route cases.")


if __name__ == "__main__":
    main()

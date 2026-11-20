"""Run the round-2 cases and additional byte encodings through both C2 routes."""

from contextlib import redirect_stderr
from datetime import date, datetime, timedelta
from hashlib import sha256
from io import StringIO
import json
from unittest.mock import Mock

import probe_receipt_identity as archived
from probe_receipt_identity import DAY, KEY, NOW, OTHER_DAY, encounter, from_cutoff, stage_pick
from probe_support import ET, deadline, fixtures, prepare_schedules, receipt, shared, status_body, world_at

from bts import dm, scheduler
from bts.daily_decision import load_decision
from bts.health import alert, runner


def digest(day):
    return sha256(status_body(day).encode("utf-8")).hexdigest()


SHAPE = archived.SHAPE | {"body_sha256": digest(DAY)}


def encoded(obj):
    return json.dumps(obj, ensure_ascii=True).encode("utf-8")


def assert_locked(day, directory):
    assert scheduler.load_state(day, directory).pick_locked
    assert load_decision(day, directory)["delivery_status"] == "locked_unconfirmed"
    dm.send_dm.assert_not_called()


def acknowledge(day, route, config, directory):
    daily = stage_pick(day, directory)
    prepare_schedules(daily)
    encounter(route, day, config, directory, daily)
    assert_locked(day, directory)
    return receipt(directory, day).read_bytes()


def static_cases():
    archived.SHAPE = SHAPE.copy()
    for label, text in archived.wrong_field_cases():
        yield label, text.encode("utf-8"), True
    valid = encoded(SHAPE)
    yield "missing body hash (round-2 writer shape)", encoded({k: v for k, v in SHAPE.items() if k != "body_sha256"}), True
    yield "extra error", encoded(SHAPE | {"error": "send_dm returned no usable message id"}), True
    yield "extra member", encoded(SHAPE | {"unexpected": [False]}), True
    yield "duplicate status", b'{"status":"failed",' + valid[1:], True
    yield "duplicate key with escape", b'{"statu\\u0073":"failed",' + valid[1:], True
    yield "duplicate schema", b'{"schema_version":null,' + valid[1:], True
    yield "duplicate identical value", b'{"status":"sent",' + valid[1:], True
    yield "nested duplicate", b'{"nested":[{"x":0,"x":1}],' + valid[1:], True
    yield "NaN", valid.replace(b'"seed-id"', b'NaN'), True
    yield "Infinity", valid.replace(b'"seed-id"', b'1e100000'), True
    yield "integer at conversion limit", valid.replace(b'"seed-id"', b'8' * 4300), True
    yield "integer beyond conversion limit", valid.replace(b'"seed-id"', b'8' * 5000), True
    yield "deep array", b'[' * 2000 + b'0' + b']' * 2000, True
    yield "malformed", b'{', True
    yield "not an object", b'[]', True
    yield "UTF-8 BOM", b'\xef\xbb\xbf' + valid, True
    yield "UTF-16 with BOM", valid.decode().encode("utf-16"), True
    yield "invalid UTF-8", valid.replace(b'seed-id', b'\xff'), True
    yield "non-JSON whitespace", b'\xc2\xa0' + valid, True
    yield "NFKC-equivalent status", encoded(SHAPE | {"status": "ｓｅｎｔ"}), True
    yield "NFKC-equivalent day", encoded(SHAPE | {"sent_day": DAY.replace("2", "２")}), True
    yield "NFKC-equivalent source", encoded(SHAPE | {"sent_sources": [KEY.replace(":", "：")]}), True
    for label, value in (("uppercase hash", digest(DAY).upper()), ("padded hash", " " + digest(DAY)),
                         ("other slate hash", digest(OTHER_DAY)), ("null hash", None), ("numeric hash", 12)):
        yield label, encoded(SHAPE | {"body_sha256": value}), True
    yield "escaped keys and values", valid.replace(b'"status"', b'"statu\\u0073"').replace(b'"sent"', b'"\\u0073ent"'), False
    yield "reordered and whitespace", b'\t\r\n' + encoded(dict(reversed(list(SHAPE.items())))) + b' \r\t\n', False
    yield "literal UTF-8 message ID", json.dumps(SHAPE | {"message_id": " é "}, ensure_ascii=False).encode(), False
    yield "NFD message ID", encoded(SHAPE | {"message_id": " e\u0301 "}), False
    yield "escaped NUL and surrogate message ID", encoded(SHAPE | {"message_id": "\x00\ud800"}), False


def writer_bytes(label, route, root, config, directory, now, scoped, sender):
    if label in {"real cutoff", "cutoff plus ID", "foreign opt-in"}:
        legacy, text = from_cutoff(config, directory, sender)
        if label == "real cutoff":
            return legacy
        if label == "cutoff plus ID":
            return encoded(json.loads(legacy) | {"message_id": "cutoff-id"})
        destination = root / "foreign.json"
        assert alert.dispatch_dm_for_health_alerts(
            [alert.Alert("CRITICAL", "late_delivery", text)], "offline.test", status_path=destination,
            now_et_date=date.fromisoformat(DAY), require_message_id=True,
        )
        raw = destination.read_bytes()
        assert json.loads(raw) == SHAPE | {
            "message_id": "cutoff-id", "body_sha256": sha256(sender.call_args.args[1].encode()).hexdigest(),
        }
        assert json.loads(raw)["body_sha256"] != digest(DAY)
        return raw
    if label == "same body legacy runner override":
        target = receipt(directory, DAY)
        message = status_body(DAY).split("[late_delivery] ", 1)[1]
        scoped.setattr(runner, "_safe_run", lambda name, _: [alert.Alert("CRITICAL", "late_delivery", message)]
                       if name == "post_failure" else [])
        scoped.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        runner.run_all_checks(directory, root / "models", "offline.test", today=date.fromisoformat(DAY),
                              thresholds_overrides={"health_dm_delivery_status": target})
        sender.assert_called_once_with("offline.test", status_body(DAY))
        raw = target.read_bytes()
        assert json.loads(raw)["body_sha256"] == digest(DAY)
        assert "message_id" not in json.loads(raw)
        return raw
    other = label == "genuine other-slate copy"
    day = OTHER_DAY if other else DAY
    now[0] = NOW + timedelta(days=int(other))
    source = root / "previous" / "picks"
    other_config = config | {"orchestrator": config["orchestrator"] | {"picks_dir": str(source)}}
    raw = acknowledge(day, route, other_config, source)
    sender.assert_called_once_with("offline.test", status_body(day))
    now[0] = NOW
    return raw


def check(root, patch, opening, route, label, initial, sends):
    with world_at(root, patch, opening, NOW) as (config, directory, now, _, scoped):
        sender = Mock(return_value="seed-id")
        scoped.setattr(alert, "send_dm", sender)
        path = receipt(directory, DAY)
        path.parent.mkdir(parents=True)
        with redirect_stderr(StringIO()) as stderr:
            if initial is None:
                initial = writer_bytes(label, route, root, config, directory, now, scoped, sender)
            path.write_bytes(initial)
            original_shared = shared(directory).read_bytes() if shared(directory).exists() else None
            sender.reset_mock(return_value=True)
            sender.return_value = "fresh-id"
            daily = stage_pick(DAY, directory)
            prepare_schedules(daily)
            encounter(route, DAY, config, directory, daily)
            after = path.read_bytes()
            encounter(route, DAY, config, directory, daily)
        assert_locked(DAY, directory)
        assert path.read_bytes() == after
        assert (shared(directory).read_bytes() if shared(directory).exists() else None) == original_shared
        assert "Unknown-outcome alert failed:" not in stderr.getvalue()
        if sends:
            sender.assert_called_once_with("offline.test", status_body(DAY))
            assert json.loads(after) == SHAPE | {"message_id": "fresh-id"}
        else:
            sender.assert_not_called()
            assert initial == after
        print(f"{route}: {label}: {sender.call_count} C2 sends over two encounters", flush=True)
        if label in {"genuine same-slate copy", "foreign opt-in", "same body legacy runner override"}:
            print(f"{label} bytes: {initial!r}", flush=True)


def rollover(root, patch, opening, route, day):
    start = datetime.fromisoformat(f"{day}T23:58:00").replace(tzinfo=ET)
    with world_at(root, patch, opening, start) as (config, directory, now, _, scoped):
        sender = Mock(return_value="rollover-id")
        scoped.setattr(alert, "send_dm", sender)
        with redirect_stderr(StringIO()):
            initial = acknowledge(day, route, config, directory)
            daily = stage_pick(day, directory)
            prepare_schedules(daily)
            encounter(route, day, config, directory, daily)
            for delta, next_route in ((3, route), (10, "classification" if route == "direct" else "direct")):
                now[0] = start + timedelta(minutes=delta)
                encounter(next_route, day, config, directory, daily)
                assert receipt(directory, day).read_bytes() == initial
        sender.assert_called_once_with("offline.test", status_body(day))
        assert_locked(day, directory)
        assert json.loads(initial)["sent_day"] == day
        assert not receipt(directory, now[0].date().isoformat()).exists()
        print(f"{route}: {start.isoformat()} through {now[0].isoformat()}: same-slate C2 DM sent once over four encounters", flush=True)


def main():
    cases = list(static_cases()) + [
        (label, None, label != "genuine same-slate copy")
        for label in ("real cutoff", "cutoff plus ID", "foreign opt-in", "same body legacy runner override",
                      "genuine other-slate copy", "genuine same-slate copy")
    ]
    with deadline(50), fixtures("receipt-matrix") as (root, patch, opening):
        for route in ("direct", "classification"):
            for index, (label, initial, sends) in enumerate(cases):
                check(root / f"{route}-{index}", patch, opening, route, label, initial, sends)
            for day in (DAY, "2026-12-31"):
                rollover(root / f"rollover-{route}-{day}", patch, opening, route, day)
    print(f"PASS: {2 * len(cases)} receipt route cases and four ET rollover runs")


if __name__ == "__main__":
    main()

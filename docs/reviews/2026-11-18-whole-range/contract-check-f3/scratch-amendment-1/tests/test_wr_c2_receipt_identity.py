"""Receipt identity checks through both C2 lock routes."""

from datetime import date, timedelta
from hashlib import sha256
import json
import os

import pytest

from bts import dm, picks, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, daily_pick
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world
from tests.test_wr_c2_recovery import copy, own, restore_policy_cache


OTHER_DAY = (date.fromisoformat(DAY) + timedelta(days=1)).isoformat()
KEY = "CRITICAL:late_delivery"
BODY_HASH = sha256(copy(DAY).encode("utf-8")).hexdigest()
RECEIPT = {
    "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
    "status": "sent",
    "sent_day": DAY,
    "sent_sources": [KEY],
    "message_id": "seed-id",
    "body_sha256": BODY_HASH,
}


def wrong_field_cases():
    invalid = {
        "schema_version": ["damaged", [RECEIPT["schema_version"]]],
        "status": ["failed", "skipped_no_recipient", "SENT", ["sent"]],
        "sent_day": [OTHER_DAY, DAY + " ", [DAY]],
        "sent_sources": [KEY, [], ["CRITICAL:pick_delivery"], [KEY, KEY],
                         [KEY, "CRITICAL:pick_delivery"], ["CRITICAL:pick_delivery", KEY],
                         [True], [None], {KEY: True}],
        "message_id": ["", "\t \r\n", "\u2003", ["seed-id"]],
    }
    for field, examples in invalid.items():
        yield pytest.param(
            json.dumps({key: value for key, value in RECEIPT.items() if key != field}),
            id=f"missing-{field}",
        )
        for candidate in [None, False, 17, 1.5, {}, *examples]:
            yield pytest.param(json.dumps(RECEIPT | {field: candidate}), id=f"{field}={candidate!r}")


def extra_cases():
    yield pytest.param(json.dumps(RECEIPT | {"error": "send_dm returned no usable message id"}), id="extra-error")
    yield pytest.param(json.dumps(RECEIPT | {"unexpected": False}), id="extra-unknown")
    yield pytest.param('{"status":"failed",' + json.dumps(RECEIPT)[1:], id="duplicate-status")
    yield pytest.param('{"schema_version":null,' + json.dumps(RECEIPT)[1:], id="duplicate-schema")
    yield pytest.param('{"nested":{"key":1,"key":2},' + json.dumps(RECEIPT)[1:], id="nested-duplicate")
    old_receipt = {key: value for key, value in RECEIPT.items() if key != "body_sha256"}
    yield pytest.param(json.dumps(old_receipt), id="old-five-member-receipt")
    for label, value in {
        "null": None,
        "number": 17,
        "uppercase": BODY_HASH.upper(),
        "whitespace": f" {BODY_HASH} ",
        "other-slate": sha256(copy(OTHER_DAY).encode("utf-8")).hexdigest(),
    }.items():
        yield pytest.param(json.dumps(RECEIPT | {"body_sha256": value}), id=f"hash-{label}")


@pytest.fixture
def encounter(world, monkeypatch, route):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    path = own(directory)
    path.parent.mkdir(parents=True)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    def run():
        if route == "direct":
            assert scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "C2b") is False
        else:
            assert scheduler.run_day(DAY, config) is None
        dm.send_dm.assert_not_called()
        assert load_decision(DAY, directory)["delivery_status"] == "locked_unconfirmed"

    return path, run


def assert_replaced(path, run):
    initial = path.read_bytes()
    run()
    replacement = path.read_bytes()
    run()
    alert.send_dm.assert_called_once_with("offline.test", copy(DAY))
    assert replacement != initial
    assert path.read_bytes() == replacement


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("document", [*extra_cases(), *wrong_field_cases()])
def test_rejects_invalid_receipt(encounter, document):
    path, run = encounter
    path.write_text(document, encoding="utf-8")
    assert_replaced(path, run)


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("seed", ["foreign-body", "legacy-cutoff", "legacy-with-id", "other-slate"])
def test_foreign_dispatcher_receipts_do_not_suppress_c2(encounter, seed, tmp_path):
    path, run = encounter
    source = tmp_path / "foreign.json"
    other_slate = seed == "other-slate"
    day = OTHER_DAY if other_slate else DAY
    message = copy(day).split("[late_delivery] ", 1)[1] if other_slate else "LATE DELIVERY REFUSED: fixture"
    assert alert.dispatch_dm_for_health_alerts(
        [alert.Alert("CRITICAL", "late_delivery", message)],
        "offline.test",
        status_path=source,
        now_et_date=date.fromisoformat(day),
        require_message_id=seed in {"foreign-body", "other-slate"},
    )
    if seed == "legacy-with-id":
        path.write_text(json.dumps(json.loads(source.read_bytes()) | {"message_id": "cutoff-id"}))
    else:
        path.write_bytes(source.read_bytes())
    alert.send_dm.reset_mock()
    assert_replaced(path, run)


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("rewrite", ["none", "order", "whitespace"])
def test_genuine_receipt_deduplicates(encounter, rewrite):
    path, run = encounter
    run()
    initial = path.read_bytes()
    if rewrite == "order":
        path.write_text(json.dumps(dict(reversed(list(json.loads(initial).items())))))
    elif rewrite == "whitespace":
        path.write_bytes(b" \n" + initial + b"\t")
    seeded = path.read_bytes()
    run()
    run()
    alert.send_dm.assert_called_once_with("offline.test", copy(DAY))
    assert path.read_bytes() == seeded


@pytest.mark.parametrize("route", ["direct", "classification"])
def test_receipt_hash_matches_exact_transport_body(encounter):
    path, run = encounter
    run()
    body = alert.send_dm.call_args.args[1]
    assert json.loads(path.read_bytes()) == RECEIPT | {
        "message_id": "health-dm",
        "body_sha256": sha256(body.encode("utf-8")).hexdigest(),
    }


def test_reader_rejects_duplicate_members_inside_nested_objects(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"array":[{"key":1,"key":2}]}', encoding="utf-8")
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert alert._read_receipt(directory_fd, path.name) is None
    finally:
        os.close(directory_fd)

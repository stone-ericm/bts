"""Exercise EOD transport, restart, status-path collisions and bad receipts."""

import datetime as dt
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts

from review_support import (
    DAY, DIGEST, MISMATCH, NEXT, SUFFIX, alert, contest_state, eod, files, flag,
    health, observe, receipt, select,
)

DIGEST_ALERT = alert.Alert("CRITICAL", "fixture", "Prior critical alert")
REPEATED = alert.Alert("WARN", "calibration_drift", "Calibration drift in both slates")


def state(path):
    return json.loads(path.read_text())


def transport_cases(root):
    picks = root / "limit" / "picks"
    observe(picks)
    flag(picks, "used")
    # Padding is single ASCII graphemes. The digest header and item marker are ASCII.
    prefix = alert.format_health_dm_body([DIGEST_ALERT], [])
    long_alert = alert.Alert("CRITICAL", DIGEST_ALERT.source,
                             DIGEST_ALERT.message + "x" * (1000 - len(prefix)))
    def capped(_, body):
        assert len(body) <= 1000
        return "accepted"
    base_alerts, control = eod(picks, existing=[long_alert], transport=capped)
    assert len(control[0][1]) == 1000
    base_bytes = receipt(picks).read_bytes()
    receipt(picks).unlink()
    flag(picks, "active")
    raw, sends = eod(picks, existing=[long_alert], transport=capped)
    assert [row for row in raw if row not in select(raw)] == base_alerts
    assert sends[:1] == control and len(sends) == 2
    assert sends[1][1].startswith("BTS health WARN attention:") and "fixture" not in sends[1][1]
    assert receipt(picks).read_bytes() == base_bytes
    assert state(receipt(picks, MISMATCH))["message_id"] == "accepted"

    for mode, wall in (("live", DAY), ("historical", NEXT)):
        for first in ("used", "not_earned"):
            picks = root / mode / first / "p"
            flag(picks, "active")
            second = "used" if first == "not_earned" else "not_earned"
            for saver in (first, second):
                observe(picks, saver=saver)
                raw, sends = eod(picks, wall=wall)
                item, = raw
                assert len(sends) == 1 and bool(select(raw)) == (saver == "used")
                body = sends[0][1]
                if mode == "historical":
                    pointer = health.HISTORICAL_POINTER
                    assert body.count(pointer) == 1 and body.endswith(pointer)
                    assert body.index("1st observed day") < body.index(pointer)
                else:
                    assert item.message in body
                assert not eod(picks, wall=wall)[1]
            assert state(receipt(picks))["sent_day"] == str(DAY.date())
            assert state(receipt(picks, MISMATCH))["sent_day"] == str(DAY.date())

    picks = root / "streak" / "picks"
    flag(picks, "active")
    for offset, label in enumerate(("1st observed day", "2nd consecutive day", "3rd consecutive day")):
        day = DAY + dt.timedelta(days=offset)
        observe(picks, day)
        _, sends = eod(picks, day, wall=(day + dt.timedelta(days=1)).replace(hour=0))
        assert len(sends) == 1 and label in sends[0][1]
        assert not eod(picks, day, wall=(day + dt.timedelta(days=1)).replace(hour=0))[1]
    flag(picks, "not_earned")
    assert "read not_earned" in eod(picks, day)[1][0][1]
    print("1,000-grapheme digest, both incident orders in both modes, processed dates and labels held")


def receipt_cases(root):
    for kind in ("symlink", "hardlink", "directory", "fifo"):
        picks = root / kind / "picks"
        observe(picks)
        actual_flag = flag(picks, "active")
        before = actual_flag.read_bytes()
        path = receipt(picks, MISMATCH)
        path.parent.mkdir()
        if kind == "symlink":
            path.symlink_to(actual_flag)
        elif kind == "hardlink":
            os.link(actual_flag, path)
        elif kind == "directory":
            path.mkdir()
        else:
            os.mkfifo(path)
        inode = path.lstat()
        for attempt in range(2):
            raw, sends = eod(picks, existing=[DIGEST_ALERT])
            assert len(select(raw)) == 1
            assert len(sends) == 2 - attempt and "dated observation" in sends[-1][1]
        assert (path.lstat().st_ino, path.lstat().st_mode) == (inode.st_ino, inode.st_mode)
        assert actual_flag.read_bytes() == before

    for override, name in (("/", MISMATCH), ("custom-digest.json", "custom-digest.json" + SUFFIX)):
        picks = root / name / "picks"
        observe(picks)
        flag(picks, "active")
        custom = override if override == "/" else str(receipt(picks, override))
        _, sends = eod(picks, existing=[DIGEST_ALERT], overrides={"health_dm_delivery_status": custom})
        assert len(sends) == 2 and DIGEST_ALERT.message in sends[0][1]
        assert state(receipt(picks, name))["message_id"] == "ack"

    picks = root / "bad-attention" / "picks"
    observe(picks)
    flag(picks, "active")
    path = receipt(picks, "warn_attention_state.json")
    path.parent.mkdir()
    path.write_text('{"sources":{"saver_state:dated_used_mismatch":{"streak":"bad"}}}')
    try:
        eod(picks, existing=[DIGEST_ALERT])
    except ValueError:
        assert not receipt(picks).exists()
        print("Known limit 8: malformed persisted mismatch streak stopped both dispatches")
    else:
        raise AssertionError("expected inherited malformed-row failure")
    print("Both receipt aliases, directory, FIFO, root fallback, and named override verified")


def send_failures_and_restart(root):
    for failing_dispatch in (0, 1):
        for outcome in ("raise", "blank"):
            picks = root / outcome / str(failing_dispatch) / "picks"
            observe(picks)
            flag(picks, "active")
            seen = []
            def transport(_, body):
                seen.append(body)
                if len(seen) != failing_dispatch + 1:
                    return "ack"
                if outcome == "raise":
                    raise OSError("offline transport fixture")
                return "  "
            _, sends = eod(picks, existing=[DIGEST_ALERT], transport=transport)
            assert len(sends) == 2
            failed = outcome == "raise" or failing_dispatch == 1
            relevant = receipt(picks, DIGEST if failing_dispatch == 0 else MISMATCH)
            assert state(relevant)["status"] == ("failed" if failed else "sent")
            _, retry = eod(picks, existing=[DIGEST_ALERT])
            assert len(retry) == int(failed)
            if failed:
                assert (DIGEST_ALERT.message in retry[0][1]) == (failing_dispatch == 0)
            assert not eod(picks, existing=[DIGEST_ALERT])[1]

    picks = root / "interrupt" / "picks"
    observe(picks)
    flag(picks, "active")
    account = files(picks)
    with patch.object(alert, "_replace_receipt", side_effect=KeyboardInterrupt):
        sent = []
        try:
            eod(picks, existing=[DIGEST_ALERT], transport=lambda _, body: sent.append(body) or "ack")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("interrupt swallowed")
    assert len(sent) == 2 and state(receipt(picks))["status"] == "sent"
    assert not receipt(picks, MISMATCH).exists() and files(picks) == account
    _, retry = eod(picks, existing=[DIGEST_ALERT])
    assert [body for _, body in retry] == sent[1:] and files(picks) == account
    assert not eod(picks, existing=[DIGEST_ALERT])[1]

    picks = root / "first-interrupt" / "picks"
    observe(picks)
    flag(picks, "active")
    attempted = []
    def interrupted(_, body):
        attempted.append(body)
        raise KeyboardInterrupt
    try:
        eod(picks, existing=[DIGEST_ALERT], transport=interrupted)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("first interrupt swallowed")
    assert len(attempted) == 1 and DIGEST_ALERT.message in attempted[0]
    assert len(eod(picks, existing=[DIGEST_ALERT])[1]) == 2
    print("Thrown and blank sends and interrupts on either dispatch; retries used only persisted state")


def collision(root, base):
    picks = root / "picks"
    observe(picks)
    flag(picks, "active")
    digest = receipt(picks, "digest.json")
    shared = digest.with_name(digest.name + SUFFIX)
    overrides = {"health_dm_delivery_status": str(digest), "warn_attention_state": str(shared)}
    _, first = eod(picks, existing=[REPEATED], overrides=overrides)
    first_disk = state(shared)
    if base:
        assert not first and first_disk["sources"]["calibration_drift"]["streak"] == 1
    else:
        assert len(first) == 1 and first_disk["message_id"] == "ack" and "sources" not in first_disk
    tomorrow = DAY + dt.timedelta(days=1)
    observe(picks, tomorrow, saver="active")
    raw, second = eod(picks, tomorrow, existing=[REPEATED], overrides=overrides)
    assert raw == [REPEATED]
    second_disk = state(shared)
    expected_streak = 2 if base else 1
    assert second_disk["sources"]["calibration_drift"]["streak"] == expected_streak
    assert len(second) == int(base)
    if base:
        assert "(2nd consecutive day)" in second[0][1]
    print(f"Collision ({'base' if base else 'candidate'}): first-file keys={list(first_disk)}; "
          f"next-day streak={expected_streak}, digest count={len(second)}")


if __name__ == "__main__":
    print("bts.__file__ =", bts.__file__)
    with TemporaryDirectory(prefix="f2-dm-", dir="/tmp") as directory:
        root = Path(directory)
        if sys.argv[1:] in (["base"], ["candidate"]):
            collision(root, sys.argv[1:] == ["base"])
        else:
            transport_cases(root)
            receipt_cases(root)
            send_failures_and_restart(root)

import datetime as dt
import json
import unicodedata
from unittest.mock import patch

from probe_support import (
    DAY, DECEMBER, INCIDENT, RECIPIENT, SECOND, STATUS, alert, bts, eod, flag,
    health, observed, read_receipt, receipt, scratch,
)

KEY = f"WARN_ATTENTION:{INCIDENT}"
OLD_KEY = "WARN_ATTENTION:saver_state"
CRITICAL = alert.Alert("CRITICAL", "offline-probe", "Existing failure")


def detect_raw(rows):
    return [row for row in rows if row.incident_key == INCIDENT]


def same_day(root):
    days = (DAY, (DAY + dt.timedelta(days=1)).replace(hour=0, minute=1))
    for wall in days:
        for first, second in (("used", "not_earned"), ("not_earned", "used")):
            picks = root / f"order-{wall.isoformat()}-{first}" / "picks"
            flag(picks, "active")
            former = None
            for state in (first, second):
                manual = observed(picks, state=state)
                before = {p.name: p.read_bytes() for p in manual.parent.iterdir()}
                raw, sends = eod(picks, wall=wall)
                assert len(raw) == len(sends) == 1
                assert len(detect_raw(raw)) == int(state == "used")
                name, key = (SECOND, KEY) if state == "used" else (STATUS, OLD_KEY)
                assert read_receipt(picks, name)["sent_sources"] == [key]
                assert read_receipt(picks, name)["sent_day"] == str(DAY.date())
                if former is not None:
                    assert former[0].read_bytes() == former[1]
                former = (receipt(picks, name), receipt(picks, name).read_bytes())
                assert not eod(picks, wall=wall)[1]
                assert {p.name: p.read_bytes() for p in manual.parent.iterdir()} == before
    picks = root / "consecutive" / "picks"
    flag(picks, "active", DECEMBER.year)
    observed(picks, DECEMBER)
    january = (DECEMBER + dt.timedelta(days=1)).replace(hour=0, minute=4)
    for index, when in enumerate((DECEMBER, january), start=1):
        if index == 2:
            flag(picks, "active", when.year)
            observed(picks, when)
        raw, sends = eod(picks, when, wall=january)
        warn, = detect_raw(raw)
        _, body = sends[0]
        assert read_receipt(picks, SECOND)["sent_day"] == str(when.date())
        assert read_receipt(picks, SECOND)["sent_sources"] == [KEY]
        if index == 1:
            assert warn.message.endswith(health.HISTORICAL_POINTER)
            assert body.count(health.HISTORICAL_POINTER) == 1
            assert body.index("(1st observed day)") < body.index(health.HISTORICAL_POINTER)
        else:
            assert body.endswith("(2nd consecutive day)")
    print("Both same-day incident orders on live and historical runs retained separate receipts. "
          "Historical label precedes the pointer. Consecutive days across January 1 each sent.")


def near_limit(root):
    picks = root / "limit" / "picks"
    observed(picks)
    flag(picks, "used")
    existing = alert.Alert("CRITICAL", "close_to_limit", "e\u0301" * 951)
    counts = []

    def constrained(_, body):
        # In this fixture only e + combining acute is multi-code-point.
        length = sum(not unicodedata.combining(character) for character in body)
        counts.append(length)
        if length > 1000:
            raise RuntimeError(f"message too long: {length} graphemes")
        return "sent"

    _, reference = eod(picks, existing=[existing], transport=constrained)
    previous = receipt(picks).read_bytes()
    receipt(picks).unlink()
    flag(picks, "active")
    raw, sends = eod(picks, existing=[existing], transport=constrained)
    assert len(raw) == 2 and len(sends) == 2 and sends[:1] == reference
    assert receipt(picks).read_bytes() == previous
    assert 985 <= counts[0] == counts[1] <= 1000 and counts[1] + counts[2] > 1000
    assert read_receipt(picks, SECOND)["critical_count"] == 0
    assert sends[1][1].startswith("BTS health WARN attention:\n")
    assert all(recipient == RECIPIENT for recipient, _ in sends)
    print(f"Near-limit digest and receipt unchanged. Grapheme counts reference, digest, mismatch: {counts}")


def failures(root):
    for broken in ("digest", "mismatch", "empty-digest", "empty-mismatch"):
        picks = root / broken / "picks"
        observed(picks)
        flag(picks, "active")

        def transport(_, body):
            which = "digest" if "Existing failure" in body else "mismatch"
            if which == broken:
                raise OSError(f"offline {which}")
            return "" if f"empty-{which}" == broken else "sent"

        _, sends = eod(picks, existing=(CRITICAL,), transport=transport)
        assert len(sends) == 2
        assert read_receipt(picks)["status"] == ("failed" if broken == "digest" else "sent")
        assert read_receipt(picks, SECOND)["status"] == ("failed" if broken == "mismatch" else "sent")
        _, retries = eod(picks, existing=(CRITICAL,))
        assert len(retries) == int(not broken.startswith("empty"))
        if retries:
            assert ("Existing failure" in retries[0][1]) == (broken == "digest")

    for name in (STATUS, SECOND):
        picks = root / f"interrupt-{name}" / "picks"
        observed(picks)
        flag(picks, "active")
        before = flag(picks, "active").read_bytes()
        stored = alert._persist_health_dm_delivery_status
        attempted = []

        def interrupt(path, **payload):
            if path.name == name:
                raise KeyboardInterrupt("pre-status")
            stored(path, **payload)

        with patch.object(alert, "_persist_health_dm_delivery_status", interrupt):
            try:
                eod(picks, existing=(CRITICAL,), transport=lambda _, body: attempted.append(body))
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("interrupt did not propagate")
        assert len(attempted) == (1 if name == STATUS else 2)
        assert not receipt(picks, name).exists()
        assert (picks / "account_state/saver_state.json").read_bytes() == before
        assert len(eod(picks, existing=(CRITICAL,))[1]) == (2 if name == STATUS else 1)
        assert not eod(picks, existing=(CRITICAL,))[1]
    print("Thrown transport errors isolated in either dispatch and retried. Empty IDs recorded sent. "
          "Interrupt before either receipt write left the expected partial set; restart retried and then deduped.")


def bad_receipt_paths(root):
    for kind in ("fifo", "directory", "malformed", "linked-account", "empty-basename"):
        picks = root / kind / "picks"
        observed(picks)
        flag_file = flag(picks, "active")
        sibling = receipt(picks, SECOND)
        sibling.parent.mkdir(parents=True)
        overrides = None
        if kind == "fifo":
            import os
            os.mkfifo(sibling)
        elif kind == "directory":
            sibling.mkdir()
        elif kind == "malformed":
            sibling.write_text("{")
        elif kind == "linked-account":
            sibling.symlink_to(flag_file)
        else:
            overrides = {"health_dm_delivery_status": "/"}
        sent = []
        try:
            eod(picks, existing=(CRITICAL,), overrides=overrides,
                transport=lambda _, body: sent.append(body) or "sent")
        except ValueError as error:
            assert kind == "empty-basename" and len(sent) == 1
            flag(picks, "used")
            assert len(eod(picks, existing=(CRITICAL,), overrides=overrides)[1]) == 1
            print(f"status override '/': existing digest sent; mismatch derivation raised {error}; no-mismatch control returned")
            continue
        assert len(sent) == 2 and sent[0].startswith("BTS health CRITICAL")
        assert read_receipt(picks)["status"] == "sent"
        if kind == "linked-account":
            actual = json.loads(flag_file.read_text())
            print(f"mismatch status symlink to saver flag: overwritten with {actual.get('schema_version')}; "
                  f"flag state now {actual.get('state')!r}")
        elif kind == "malformed":
            assert read_receipt(picks, SECOND)["status"] == "sent"
        else:
            assert not sibling.is_file()
    print("FIFO, directory and malformed mismatch receipts preserved the existing digest.")


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    with scratch() as directory:
        same_day(directory)
        near_limit(directory)
        failures(directory)
        bad_receipt_paths(directory)

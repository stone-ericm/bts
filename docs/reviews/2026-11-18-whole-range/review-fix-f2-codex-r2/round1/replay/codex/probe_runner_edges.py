import datetime as dt
import os
import sys

from probe_support import (
    DAY, INCIDENT, SECOND, alert, automatic, bts, contest_state, eod, flag, observed,
    put, read_receipt, receipt, scratch,
)

CRITICAL = alert.Alert("CRITICAL", "offline-probe", "A pre-existing critical alert")


def alias(root, expected_base):
    for kind in ("symlink", "hardlink"):
        picks = root / kind / "picks"
        manual = observed(picks)
        actual_flag = flag(picks, "active")
        original = actual_flag.read_bytes()
        manual_bytes = manual.read_bytes()
        next_day = DAY.replace(hour=12) + dt.timedelta(days=1)
        automatic(picks, next_day, coverage=DAY.date())
        sibling = receipt(picks, SECOND)
        sibling.parent.mkdir(parents=True)
        if kind == "symlink":
            sibling.symlink_to(actual_flag)
        else:
            os.link(actual_flag, sibling)
        raw, sends = eod(picks, existing=(CRITICAL,))
        assert raw[0] == CRITICAL and len(sends) == (1 if expected_base else 2)
        assert read_receipt(picks)["status"] == "sent"
        assert manual.read_bytes() == manual_bytes
        decision = contest_state.load_decision_streak_state(picks, now=next_day, require_contest_state=True)
        preserved = actual_flag.read_bytes() == original
        print(f"{kind}: flag preserved={preserved}, day-two saver_available={decision.saver_available}, "
              f"DMs={len(sends)}")
        assert preserved is expected_base and decision.saver_available is expected_base


def no_basename(root, expected_base):
    picks = root / "no-basename" / "picks"
    observed(picks)
    flag(picks, "active")
    captured = []
    try:
        eod(picks, existing=(CRITICAL,), overrides={"health_dm_delivery_status": "/"},
            transport=lambda _, body: captured.append(body) or "sent")
    except ValueError as error:
        print(f"Root receipt path: first DM sent; mismatch raises {error}")
        assert not expected_base
    else:
        print("Root receipt path: first DM sent and runner returns normally")
        assert expected_base
    assert len(captured) == 1 and CRITICAL.message in captured[0]


def bad_attention(root, expected_base):
    picks = root / "broken-attention" / "picks"
    observed(picks)
    flag(picks, "active")
    put(picks.parent / "health_state", "warn_attention_state.json", {
        "sources": {INCIDENT: {"last_seen": str(DAY.date()), "streak": "bad"}},
    })
    captured = []
    try:
        eod(picks, existing=(CRITICAL,), transport=lambda _, body: captured.append(body) or "sent")
    except ValueError as error:
        assert not expected_base
        print(f"Corrupt attention metadata: runner raises {error} before either transport call")
    else:
        assert expected_base
        print("Corrupt attention metadata: base sends its critical DM")
    assert len(captured) == int(expected_base)


if __name__ == "__main__":
    base = sys.argv[1:] == ["base"]
    print(f"bts.__file__ = {bts.__file__}")
    with scratch() as directory:
        alias(directory, base)
        no_basename(directory, base)
        bad_attention(directory, base)

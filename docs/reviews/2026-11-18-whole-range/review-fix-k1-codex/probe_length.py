"""Exercise the real loader/parser, dispatcher and complete wire body."""

import ctypes as c
import inspect
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts import contest_state as cs, scheduler
from bts.health import alert
from scripts.audit.season_walk.walk import offline
from tests.test_wr_k1_refusal_dm import AT, BOX, DAY, cases, fixtures, refusal


def grapheme_counter():
    """Use the host ICU character break iterator if present."""
    try:
        icu = c.CDLL("/usr/lib/libicucore.A.dylib")
        open_break = icu.ubrk_open
    except (OSError, AttributeError):
        return lambda _: None
    open_break.restype = c.c_void_p
    open_break.argtypes = [c.c_int, c.c_char_p, c.c_void_p, c.c_int32, c.POINTER(c.c_int)]
    icu.ubrk_next.restype = c.c_int32
    icu.ubrk_next.argtypes = [c.c_void_p]
    icu.ubrk_close.argtypes = [c.c_void_p]

    def count(body):
        raw = body.encode("utf-16-le")
        status = c.c_int()
        iterator = open_break(0, b"en", raw, len(raw) // 2, c.byref(status))
        assert iterator and status.value <= 0, status.value
        result = 0
        try:
            while icu.ubrk_next(iterator) != -1:
                result += 1
        finally:
            icu.ubrk_close(iterator)
        return result

    return count


class BadString(cs.ContestStateError):
    def __str__(self):
        raise RuntimeError("string unavailable")


class StrangeName(BadString):
    pass


StrangeName.__name__ = "🏟️" * 900


def main():
    print("bts:", bts.__file__)
    logging.disable(logging.CRITICAL)
    graphemes = grapheme_counter()
    current = "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters
    synthetic = {
        "long first line": cs.ContestStateError("👩🏽‍💻" * 25000 + "\nHIDDEN"),
        "embedded newline": cs.ContestStateError(" \t one \u2028HIDDEN"),
        "empty first line": cs.ContestStateError("\r\nHIDDEN"),
        "only whitespace": cs.ContestStateError(" \t"),
        "surrogates": cs.ContestStateError("broken \udfff \ud800"),
        "str raises": BadString(),
        "str raises with long type name": StrangeName(),
    }
    metrics = {}
    with offline(AT), TemporaryDirectory(prefix="wrk1-length-") as temp:
        root = Path(temp)
        receipt = root / "health_state" / "health_dm_delivery_status.json"
        receipt.parent.mkdir()
        cfg = {"orchestrator": {"picks_dir": str(root / BOX.lstrip("/"))},
               "bluesky": {"dm_recipient": "capture.invalid"}}
        receipt = Path(cfg["orchestrator"]["picks_dir"]).parent / "health_state" / receipt.name
        receipt.parent.mkdir(parents=True)

        def sample(name, error, rows, day=DAY):
            for count in rows:
                if count == "corrupt":
                    receipt.write_bytes(b"\xff{")
                elif count is None:
                    receipt.unlink(missing_ok=True)
                else:
                    receipt.write_text(json.dumps({"status": "failed", "critical_count": count}))
                bodies = []

                def send(_, body):
                    bodies.append(body)
                    body.encode("utf-8")
                    if len(body) >= 1000:
                        raise ValueError("body refused by length")
                    return "accepted"

                try:
                    with patch.object(alert, "send_dm", send):
                        scheduler._alert_contest_state_failure(cfg, error, **({"slate_date": day} if current else {}))
                except Exception as exc:
                    assert not current, (name, exc)
                    print(name, "raised", type(exc).__name__)
                    continue
                body, = bodies
                saved = json.loads(receipt.read_text())
                try:
                    clusters = graphemes(body)
                except UnicodeEncodeError:
                    clusters = None
                row = (len(body), clusters, saved["status"])
                print(name, repr(count) if count != 10**700 else "701 digits", row)
                metrics.setdefault(str(count), []).append(row)
                if current:
                    assert saved["status"] == "sent" and len(body) < 1000
                    assert body.count("[contest_state]") == 1 and "HIDDEN" not in body
                    assert (body.startswith("⚠️") == (type(count) is int and count > 0))
                    if day != DAY:
                        assert "for this slate at this check" in body
                    if type(count) is int and count > 999:
                        assert "(999+ CRITICAL)" in body

        for index, (case, variant) in enumerate(cases):
            if case == "synthetic":
                continue
            picks_dir = root / str(index) / "data" / "picks"
            account = picks_dir / "account_state"
            account.mkdir(parents=True)
            fixtures(case, variant, picks_dir, account)
            sample(f"{case} {variant!r}", refusal(case, variant, picks_dir), (None, 1, 999))

        picks_dir = root / "long-on-disk" / "picks"
        account = picks_dir / "account_state"
        account.mkdir(parents=True)
        (account / "contest_streak.json").write_text(json.dumps({"active_streak": "👩🏽‍💻" * 4000}))
        try:
            cs.load_decision_streak_state(picks_dir, require_contest_state=True, now=AT)
        except cs.ContestStateError as exc:
            sample("huge wrong type from actual JSON", cs.ContestStateError(str(exc).replace(str(picks_dir), BOX)),
                   (None, 1, 10**700))
        else:
            raise AssertionError("expected loader refusal")

        for name, error in synthetic.items():
            sample(name, error, (None, 1, 999, 1000, 10**700, "corrupt"))
        for day in (None, "", "2027-02-29", "\ud800", "2027-07-03\x00", "x" * 100000, 20270703):
            sample(f"bad date {type(day).__name__} {len(day) if isinstance(day, str) else day}", synthetic["long first line"],
                   (None, 10**700, "corrupt"), day=day)

    for key, rows in metrics.items():
        print("MAX", key if len(key) < 10 else "701 digits", max(row[0] for row in rows),
              max((row[1] for row in rows if row[1] is not None), default=None))


if __name__ == "__main__":
    main()

"""Check real CLI behavior for bad optional files, read limits and interruption."""
import datetime as dt
import errno
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import daily_decision, day_outcome

from review_support import (ARCHIVE, CURRENT, DATE, ET, SKIP, SWITCH, command, dated, identify_package,
                            put, setup, snapshot, state, switch)


CAP = 65536
OVERFLOW_STAMPS = ("0001-01-01T00:00:00Z", "9999-12-31T23:59:59-12:00")


def valid(name, day=DATE):
    return switch(day) if name == SWITCH else dated(day)


def encode_at_size(obj, size):
    head = json.dumps(obj | {"ignored_padding": ""}, separators=(",", ":")).encode()
    assert len(head) <= size and head.endswith(b'""}')
    return head[:-2] + b"q" * (size - len(head)) + head[-2:]


def assert_out(result, wanted, *, transport="confirmed"):
    assert state(result) == wanted, result
    problem = wanted != "deliberate_skip"
    assert result["exit"] == int(problem) and len(result["sends"]) == int(problem), result
    assert result["stderr"] == ("day-outcome: DM failed: OSError\n" if transport == "error" and problem else "")


def bad_files(root, base):
    faults = ("absent", "directory", "fifo", "denied", "utf8", "bad_json", "deep_json", "long_number",
              "list", "wrong_schema", "wrong_day", "wrong_season", "timestamp_type", "over_cap", "at_cap")
    for fault in faults:
        for name in (CURRENT, ARCHIVE, SWITCH):
            cfg, picks, account = setup(root / name / fault)
            path = account / name
            raw = json.dumps(valid(name)).encode()
            if fault == "directory":
                path.mkdir()
            elif fault == "fifo":
                os.mkfifo(path)
            elif fault != "absent":
                payloads = {
                    "utf8": b"\xff", "bad_json": b'{"field":', "list": b"[]",
                    "deep_json": b'{"field":' + b"[" * 2500 + b"0" + b"]" * 2500 + b"}",
                    "long_number": b'{"field":' + b"8" * 5100 + b"}",
                    "over_cap": encode_at_size(valid(name), CAP + 1), "at_cap": encode_at_size(valid(name), CAP),
                    "wrong_schema": json.dumps(valid(name) | {"schema_version": "legacy"}).encode(),
                    "wrong_day": json.dumps(valid(name, "2027-04-01")).encode(),
                    "wrong_season": json.dumps(valid(name) | {"season": 2028}).encode(),
                    "timestamp_type": json.dumps(valid(name) | {
                        "revoked_at" if name == SWITCH else "recorded_at": {"timestamp": "wrong"},
                    }).encode(),
                }
                path.write_bytes(payloads.get(fault, raw))

            real_open, real_read = os.open, os.read
            evidence_opens, read_counts = [], []
            target_fds = set()

            def open_probe(target, flags, *args, **kwargs):
                if os.fspath(target) != str(path):
                    return real_open(target, flags, *args, **kwargs)
                assert flags & os.O_NONBLOCK and flags & os.O_ACCMODE == os.O_RDONLY
                evidence_opens.append(flags)
                if fault == "denied":
                    raise PermissionError(errno.EACCES, "injected EACCES", target)
                fd = real_open(target, flags, *args, **kwargs)
                target_fds.add(fd)
                return fd

            def read_probe(fd, count):
                chunk = real_read(fd, count)
                if fd in target_fds:
                    read_counts.append(len(chunk))
                return chunk

            before = snapshot(picks)
            with patch.object(os, "open", open_probe), patch.object(os, "read", read_probe):
                result = command(cfg, picks)
            expected = "skip_state_review" if fault == "at_cap" and not base else "deliberate_skip"
            assert_out(result, expected)
            assert snapshot(picks) == before
            assert len(evidence_opens) <= 1
            if not base and fault == "over_cap":
                assert sum(read_counts) == CAP + 1, read_counts
            if fault == "fifo":
                assert not read_counts
            if fault != "at_cap":
                other = ARCHIVE if name == SWITCH else SWITCH
                put(account / other, valid(other))
                with patch.object(os, "open", open_probe):
                    result = command(cfg, picks)
                assert_out(result, "deliberate_skip" if base else "skip_state_review")
            print(f"{name} {fault}: {expected}, {sum(read_counts)} bytes, other evidence checked={fault != 'at_cap'}")


def timestamp_edges(root, base):
    for i, stamp in enumerate(OVERFLOW_STAMPS):
        try:
            dt.datetime.fromisoformat(stamp).astimezone(ET)
        except OverflowError:
            pass
        else:
            raise AssertionError(stamp)
        for name in (CURRENT, ARCHIVE, SWITCH):
            cfg, picks, account = setup(root / str(i) / name)
            obj = switch(stamp=stamp) if name == SWITCH else dated(stamp=stamp)
            put(account / name, obj)
            result = command(cfg, picks)
            assert result["exit"] in (0, 1), result
            if name == SWITCH or base:
                assert_out(result, "deliberate_skip")
                put(account / ARCHIVE, dated())
                assert_out(command(cfg, picks), "deliberate_skip" if base else "skip_state_review")
            print(json.dumps({"edge": name, "stamp": stamp, **result}))
    cfg, picks, account = setup(root / "out_of_range_archive_masks_valid")
    put(account / ARCHIVE, dated(streak=2))
    put(account / "contest_streak.manual.revoked-extreme.json", dated(streak=30, stamp=OVERFLOW_STAMPS[1]))
    result = command(cfg, picks)
    print(json.dumps({"edge": "overflowing timestamp outranks a valid differing archive", **result}))


def malformed_skips(root):
    changes = ({"streak": True}, {"streak": False}, {"streak": -1}, {"streak": "30"}, {"streak": []},
               {"streak": 30.0}, {"saver_available": 0}, {"saver_available": "false"}, {"saver_available": {}},
               {"saver_available": None}, {"scoreable": True}, {"date": "2027-04-01"}, {"date": [DATE]},
               {"date": None}, {"scoreable": 0}, {"action": {}}, {"schema_version": []})
    for index, change in enumerate(changes):
        cfg, picks, account = setup(root / str(index), SKIP | change)
        for name in (CURRENT, ARCHIVE, SWITCH):
            put(account / name, valid(name))
        result = command(cfg, picks)
        if "action" in change or "schema_version" in change:
            assert result == {"exit": 1, "stdout": "", "stderr": "", "sends": [], "exception": "TypeError"}
        elif change == {"scoreable": 0}:
            assert_out(result, "no_decision")
        else:
            assert_out(result, "deliberate_skip")
        print(json.dumps({"skip_corruption": change, **result}))


def enumeration(root, base):
    cfg, picks, account = setup(root)
    put(account / ARCHIVE, dated(streak=2))
    put(account / "contest_streak.manual.revoked-other.json", dated(streak=2))
    real_scandir = os.scandir

    class PartialListing:
        def __enter__(self):
            self.real = real_scandir(account)
            return self.entries()

        def __exit__(self, *_exc):
            self.real.close()

        def entries(self):
            for entry in self.real:
                yield entry
                raise OSError("injected directory failure after one returned archive")

    def scan(target):
        return PartialListing() if target == account else real_scandir(target)

    with patch.object(os, "scandir", scan):
        assert_out(command(cfg, picks), "deliberate_skip" if base else "skip_state_review")
    print("Archive enumeration failure preserved the already read archive")


def standing(root, base):
    cfg, picks, account = setup(root)
    put(account / CURRENT, dated())
    before = snapshot(root)
    real_open, real_read = os.open, os.read
    fds = []

    def capture(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == str(account / CURRENT):
            fds.append(fd)
        return fd

    def interrupt_after_read(fd, length):
        raw = real_read(fd, min(length, 8))
        if fd in fds:
            raise KeyboardInterrupt
        return raw

    with patch.object(os, "open", capture), patch.object(os, "read", interrupt_after_read):
        interrupted = command(cfg, picks)
    if base:
        assert_out(interrupted, "deliberate_skip")
    else:
        assert interrupted["exit"] == 1 and not interrupted["stdout"] and not interrupted["sends"], interrupted
        assert "Aborted!" in interrupted["stderr"] and len(fds) == 1
        try:
            os.fstat(fds[0])
        except OSError as exc:
            assert exc.errno == errno.EBADF, exc
        else:
            raise AssertionError("interruption left the evidence file open")
    assert snapshot(root) == before
    wanted = "deliberate_skip" if base else "skip_state_review"
    first = command(cfg, picks)
    assert_out(first, wanted)
    assert command(cfg, picks) == first
    assert_out(command(cfg, picks, transport="no-receipt"), wanted, transport="no-receipt")
    failed = command(cfg, picks, transport="error")
    assert_out(failed, wanted, transport="error")
    assert snapshot(root) == before
    print(json.dumps({"interrupt": interrupted, "two_restarts": first, "failed_send": failed}))

    cfg.write_text("[not TOML")
    before = snapshot(root)
    failed = command(cfg, picks)
    assert failed["exit"] == 2 and not failed["stdout"] and len(failed["sends"]) == 1
    assert "THE CHECK COULD NOT RUN (TOMLDecodeError)" in failed["sends"][0][1]
    assert snapshot(root) == before
    print(json.dumps({"invalid_config": failed}))


if __name__ == "__main__":
    identify_package()
    base = "--base" in sys.argv
    with TemporaryDirectory() as temp:
        root = Path(temp)
        bad_files(root / "files", base)
        timestamp_edges(root / "timestamps", base)
        malformed_skips(root / "skips")
        enumeration(root / "enumeration", base)
        standing(root / "standing", base)

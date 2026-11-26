"""Check dated timestamp rejection, local file failures, and repeat runs through the CLI."""
import errno
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "docs/reviews/2026-11-18-whole-range/review-fix-f3-codex"))

import probe_inputs
from review_support import ARCHIVE, CURRENT, DATE, SKIP, SWITCH, command, dated, identify_package
from review_support import put, setup, snapshot, state, switch


def report(config, picks, trial, *reasons):
    before = snapshot(picks)
    outcome = command(config, picks)
    assert snapshot(picks) == before
    assert not outcome["stderr"], outcome
    if not trial or not reasons:
        probe_inputs.assert_out(outcome, "deliberate_skip")
        return outcome
    probe_inputs.assert_out(outcome, "skip_state_review")
    detail = " ".join((f"The policy recorded a skip for {DATE} using streak 30 and policy saver availability "
                      "unavailable.", *reasons,
                      "Check MLB for unsettled entries and verify the current streak, season best and saver."))
    assert json.loads(outcome["stdout"]) == {"date": DATE, "state": "skip_state_review", "ok": False,
                                             "detail": detail, "pick": None}
    assert outcome["sends"] == [("operator.invalid", f"BTS day check {DATE}: SKIP STATE REVIEW. {detail}")]
    return outcome


def timestamps(root, trial):
    archive_reason = f"An archived dated observation for {DATE} records streak 1 and policy saver availability unavailable."
    revoke_reason = f"The season switch records a revocation on {DATE} at 16:00 ET."
    for i, stamp in enumerate(probe_inputs.OVERFLOW_STAMPS):
        for filename in (CURRENT, ARCHIVE):
            config, picks, account = setup(root / str(i) / filename)
            put(account / filename, dated(stamp=stamp))
            print(filename, stamp, state(report(config, picks, trial)))
    config, picks, account = setup(root / "mixed")
    put(account / SWITCH, switch())
    put(account / ARCHIVE, dated())
    put(account / "contest_streak.manual.revoked-overflow.json",
        dated(streak=30, stamp=probe_inputs.OVERFLOW_STAMPS[1]))
    put(account / CURRENT, dated(stamp=probe_inputs.OVERFLOW_STAMPS[0]))
    print("mixed", json.dumps(report(config, picks, trial, archive_reason, revoke_reason)))


def file_failures(root, trial):
    config, picks, account = setup(root)
    put(account / CURRENT, dated(streak=30))
    put(account / ARCHIVE, dated())
    real_open, real_read, real_close = os.open, os.read, os.close
    fd = None
    read_sizes, requests, closed = [], [], []

    def open_current(path, flags, *args, **kwargs):
        nonlocal fd
        result = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == str(account / CURRENT):
            assert fd is None and flags & os.O_NONBLOCK and flags & os.O_ACCMODE == os.O_RDONLY
            fd = result
        return result

    def short_reads(handle, size):
        if handle != fd or closed:
            return real_read(handle, size)
        requests.append(size)
        data = real_read(handle, min(size, 8192))
        read_sizes.append(len(data))
        return data

    def fail_after_partial_read(handle, size):
        if handle == fd and not closed and read_sizes:
            raise OSError(errno.EIO, "test interruption reading the installed observation")
        return short_reads(handle, size)

    def close(handle):
        real_close(handle)
        if handle == fd and not closed:
            closed.append(handle)

    archive_reason = f"An archived dated observation for {DATE} records streak 1 and policy saver availability unavailable."
    with patch.object(os, "open", open_current), patch.object(os, "read", fail_after_partial_read), \
            patch.object(os, "close", close):
        print("EIO after partial read", json.dumps(report(config, picks, trial, archive_reason)))
    assert bool(closed) == trial

    fd = None
    read_sizes.clear()
    requests.clear()
    closed.clear()
    put(account / CURRENT, dated(streak=30) | {"padding": "a" * (1024 * 1024)})
    with patch.object(os, "open", open_current), patch.object(os, "read", short_reads), \
            patch.object(os, "close", close):
        print("short reads and oversized input", json.dumps(report(config, picks, trial, archive_reason)))
    if trial:
        assert sum(read_sizes) == 65537 and requests[-1] == 1 and len(closed) == 1
    else:
        assert not read_sizes
    print("bytes read", sum(read_sizes), "read requests", requests)

    put(account / CURRENT, dated(streak=30))
    real_scandir = os.scandir

    def fail_listing(path):
        if path == account:
            raise PermissionError(errno.EACCES, "test archive directory listing failure")
        return real_scandir(path)

    put(account / SWITCH, switch())
    with patch.object(os, "scandir", fail_listing):
        print("matching installed observation", json.dumps(report(config, picks, trial)))


def legacy_and_restart(root, trial):
    config, picks, account = setup(root, SKIP | {"schema_version": []})
    put(account / CURRENT, dated())
    result = command(config, picks)
    probe_inputs.assert_out(result, "no_decision")
    print("legacy list schema", json.dumps(result))
    probe_inputs.enumeration(root / "enumeration", not trial)
    probe_inputs.standing(root / "standing", not trial)


if __name__ == "__main__":
    identify_package()
    trial = "--base" not in sys.argv
    with TemporaryDirectory() as temp:
        root = Path(temp)
        timestamps(root / "timestamps", trial)
        file_failures(root / "files", trial)
        legacy_and_restart(root / "legacy-and-restart", trial)

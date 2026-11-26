"""Run the contract-check inputs plus file lifecycle and read-failure checks."""

import errno
import json
import os
import socket
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import day_outcome, util
from report_harness import _box, report, snapshot

MIB = 1024 * 1024
NON_OBJECT = "not a JSON object"
BAD_COUNT = "streak is not a non-negative integer"
BIG_INT = "JSON exceeds decoder limits (ValueError)"
TOO_DEEP = "JSON exceeds decoder limits (RecursionError)"
NOT_REGULAR = "not a regular file"
UNREADABLE = "cannot be read (PermissionError)"
MAX_BYTES = "exceeds 1 MiB"
CASES = {
    "absent": (None, None), "object": (b"{}", None), "empty": (b"", "not valid JSON"),
    "syntax": (b"{", "not valid JSON"), "invalid UTF-8": (b"\xff", "cannot be read (UnicodeDecodeError)"),
    "raw array": (b"[]", NON_OBJECT), "raw string": (b'"x"', NON_OBJECT), "raw bool": (b"true", NON_OBJECT),
    "raw int": (b"-1", NON_OBJECT), "raw float": (b"3.0", NON_OBJECT), "raw null": (b"null", NON_OBJECT),
    "raw parseable large": (b"1" + b"0" * 99, NON_OBJECT), "raw too large": (b"1" + b"0" * 4999, BIG_INT),
    "object too large": (b'{"streak":' + b"9" * 5000 + b"}", BIG_INT),
    "huge allowed": (b'{"season":2027,"streak":' + b"1" + b"0" * 99 + b"}", None),
    "max digits allowed": (b'{"season":2027,"streak":' + b"9" * 4300 + b"}", None),
    "wrong applied_dates": (b'{"season":2027,"streak":3,"applied_dates":"2027-03-24"}', None),
    "mixed applied_dates": (b'{"season":2027,"streak":3,"applied_dates":[null,"2027-03-24","2027-03-24"]}', None),
    "malformed saver metadata": (b'{"season":2027,"streak":10,"saver_available":{}}', None),
    "missing count and current tag": (b'{"season":2027}', None),
    "prior tag": (b'{"season":2026,"streak":3}', None), "unknown tag": (b'{"season":"2027","streak":3}', None),
    "prior tag and bad count": (b'{"season":2026,"streak":"3"}', BAD_COUNT),
    "bad count with saver disabled": (b'{"season":2027,"streak":"3","saver_available":false}', BAD_COUNT),
    "bad count with applied date": (b'{"season":2027,"streak":"3","applied_dates":["2027-03-24"]}', BAD_COUNT),
    "NaN in other member": (b'{"streak":3,"other":NaN}', None), "Infinity count": (b'{"streak":1e400}', BAD_COUNT),
    "duplicate streak keys": (b'{"streak":"3","streak":3}', None), "negative zero": (b'{"streak":-0}', None),
    "exactly 1 MiB": (b" " * (MIB - 2) + b"{}", None), "over 1 MiB": (b" " * (MIB - 1) + b"{}", MAX_BYTES),
}
for tag in (b"", b'"season":2027,'):
    for count in (b'"3"', b"true", b"-1", b"3.0", b"null"):
        CASES[f"{tag!r} {count!r}"] = (b"{" + tag + b'"streak":' + count + b"}", BAD_COUNT)
for depth in (2300, 6000, 12000):
    CASES[f"nesting-{depth}"] = (b'{"nested":' + b"[" * depth + b"0" + b"]" * depth + b"}",
                                None if depth < 12000 else TOO_DEEP)


def check(picks, config, reason):
    fixture = picks.parent.parent
    before = snapshot(fixture)
    for games, state in ((0, "no_games"), (1, "no_decision")):
        result, chat = report(picks, config, games)
        assert result.exit_code == int(reason is not None or games > 0), (result.output, result.exception)
        payload = json.loads(result.stdout)
        assert payload["state"] == state and payload["streak_problem"] == reason, payload
        assert payload["ok"] is (result.exit_code == 0)
        assert result.stderr == "" and len(chat.accepted) == result.exit_code, (result.stderr, chat.attempted)
        if chat.accepted:
            head = "NO DECISION" if games else "MODEL STREAK FILE NEEDS REVIEW"
            assert chat.accepted[0].startswith(f"BTS day check 2027-04-02: {head}. "), chat.accepted
        assert snapshot(fixture) == before


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    for idx, (name, (content, reason)) in enumerate(CASES.items()):
        picks, config = _box(root / str(idx))
        if content is not None:
            (picks / "streak.json").write_bytes(content)
        check(picks, config, reason)
        print(f"{name}: {reason}")

    for name, reason in [("dir", NOT_REGULAR), ("fifo", NOT_REGULAR), ("socket", NOT_REGULAR),
                         ("loop", "cannot be read (OSError)"), ("dangling", None), ("mode-zero", UNREADABLE),
                         ("symlink-to-fifo", NOT_REGULAR), ("symlink-to-regular", NON_OBJECT)]:
        picks, config = _box(root / name)
        path = picks / "streak.json"
        sock = None
        if name == "dir":
            path.mkdir()
        elif name == "fifo":
            os.mkfifo(path)
        elif name == "socket":
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(str(path))
        elif name == "loop":
            path.symlink_to(path.name)
        elif name == "dangling":
            path.symlink_to("missing.json")
        elif name == "mode-zero":
            assert os.geteuid() != 0
            path.write_bytes(b"{}")
            path.chmod(0)
        else:
            target = path.with_name("target")
            if name == "symlink-to-fifo":
                os.mkfifo(target)
            else:
                target.write_bytes(b"[]")
            path.symlink_to(target.name)
        original_open = os.open

        def reject_open(candidate, *args, **kwargs):
            assert Path(candidate) != path or reason != NOT_REGULAR, name
            return original_open(candidate, *args, **kwargs)

        try:
            with patch.object(util.os, "open", reject_open):
                check(picks, config, reason)
        finally:
            if sock is not None:
                sock.close()
        print(f"{name}: {reason}")

    picks, config = _box(root / "growth")
    path = picks / "streak.json"
    path.write_bytes(b"{}")
    original_read = os.read
    original_open = os.open
    active = None
    sizes = []

    def track_open(candidate, *args, **kwargs):
        global active
        fd = original_open(candidate, *args, **kwargs)
        if Path(candidate) == path:
            active = fd
        return fd

    def read_and_grow(fd, size):
        data = original_read(fd, size)
        if fd == active:
            sizes.append((size, len(data)))
            if len(sizes) == 1:
                with path.open("ab", buffering=0) as handle:
                    handle.write(b" " * (2 * MIB))
        return data

    with patch.object(util.os, "open", track_open), patch.object(util.os, "read", read_and_grow):
        result, chat = report(picks, config, 1)
    assert result.exit_code == 1 and json.loads(result.stdout)["streak_problem"] == MAX_BYTES
    assert sizes == [(MIB + 1, 2), (MIB - 1, MIB - 1)], sizes
    assert len(chat.accepted) == 1 and path.stat().st_size == 2 * MIB + 2
    print(f"growth after the first real read: {sizes} (requested, read); {MAX_BYTES}")

    picks, config = _box(root / "swap")
    path = picks / "streak.json"
    path.write_bytes(b"{}")
    replacement = path.with_name("replacement")
    os.mkfifo(replacement)
    active = None

    def swap_and_open(candidate, flags, *args, **kwargs):
        if Path(candidate) == path:
            assert flags & os.O_NONBLOCK
            os.replace(replacement, path)
        return track_open(candidate, flags, *args, **kwargs)

    def forbid_special_read(fd, size):
        assert fd != active, "read from the FIFO swapped in after stat"
        return original_read(fd, size)

    with patch.object(util.os, "open", swap_and_open), patch.object(util.os, "read", forbid_special_read):
        result, chat = report(picks, config, 0)
    assert result.exit_code == 1 and json.loads(result.stdout)["streak_problem"] == NOT_REGULAR
    assert len(chat.accepted) == 1
    print("rename FIFO over regular file at open: descriptor caught, no read")

    picks, config = _box(root / "error")
    path = picks / "streak.json"
    path.write_bytes(b"{}")
    for code in (errno.EIO, errno.ENOENT):
        active = None

        def fail_read(fd, size):
            if fd == active:
                raise OSError(code, os.strerror(code))
            return original_read(fd, size)

        with patch.object(util.os, "open", track_open), patch.object(util.os, "read", fail_read):
            result, chat = report(picks, config, 0)
        assert active is not None and path.read_bytes() == b"{}", result.output
        expected = f"cannot be read ({type(OSError(code, '')).__name__})"
        actual = json.loads(result.stdout)["streak_problem"]
        print(f"descriptor opened then os.read raised {errno.errorcode[code]}: exit={result.exit_code}, "
              f"reason={actual!r}, DMs={len(chat.accepted)} (expected {expected!r})")
        if code == errno.EIO:
            assert actual == expected and result.exit_code == 1 and len(chat.accepted) == 1
        with patch.object(util.os, "open", track_open), patch.object(util.os, "read", fail_read):
            assert day_outcome._streak_problem(path) == actual

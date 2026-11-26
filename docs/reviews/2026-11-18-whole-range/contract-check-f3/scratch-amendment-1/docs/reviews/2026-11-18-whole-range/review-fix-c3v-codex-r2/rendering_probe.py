"""Exercise real replay filenames and exact stderr/receipt rendering."""

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import signal
import socket
import tempfile
from unittest.mock import patch

from round1.codex.probe_common import RUN, examine, receipt, refuse_network


REVIEW = Path(__file__).resolve().parent
DAY = "2027-04-16"
TODAY = "2027-04-20"
PENDING = "no final result on the pick"
INVALID = "pick file unreadable or not valid JSON"
APPLIED = "today's result is applied to streak.json but not yet saved on the pick"
LOST = "hit pick could not be loaded"
BEFORE = b'{"streak":9,"season":2027,"applied_dates":["2027-04-20"]}'


@contextmanager
def deadline():
    def expire(*_args):
        raise TimeoutError("rendering probe deadline")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(25)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def setup(scratch, name):
    root = scratch / name / "picks"
    root.mkdir(parents=True)
    (root / "streak.json").write_bytes(BEFORE)
    return root


def expected(evidence, day, rendered_day, reason, rendered_reason):
    want = (f"Season replay unavailable for {rendered_day} ({rendered_reason}); "
            "streak retained, not recalculated.\n")
    assert evidence["stderr"] == want, evidence
    assert evidence["stderr"].isascii() and len(evidence["stderr"].splitlines()) == 1
    assert all(32 <= ord(char) <= 126 for char in evidence["stderr"][:-1])
    assert receipt(evidence)["replay"] == {
        "state": "unavailable", "blocked_date": day, "reason": reason,
    }
    assert evidence["stdout"].endswith("No scoring changes detected. Streak retained: 9\n")
    assert evidence["strict"][1:] == [0, None, BEFORE.hex(), BEFORE.hex()]


def filenames(scratch):
    from bts import picks

    cases = (
        ("plain", DAY, DAY),
        ("lf", DAY + "\n", DAY + r"\n"),
        ("arabic_digit", "2027-04-1٦", r"2027-04-1\u0666"),
        ("arabic_digit_lf", "2027-04-1٦\n", r"2027-04-1\u0666\n"),
        ("supplementary_digit", "2027-04-1𝟨", r"2027-04-1\U0001d7e8"),
        ("mixed_nd", "2027-03-٠६", r"2027-03-\u0660\u096c"),
        ("impossible_date", "2027-02-99", "2027-02-99"),
        ("arabic_month_sorts_after_today", "2027-٠٤-١٠", None),
        ("carriage_return", DAY + "\r", None),
        ("crlf", DAY + "\r\n", None),
        ("repeated_lf", DAY + "\n\n", None),
        ("middle_lf", "2027-04-\n16", None),
        ("tab", DAY + "\t", None),
        ("bell", DAY + "\a", None),
        ("escape", DAY + "\x1b", None),
        ("us", DAY + "\x1f", None),
        ("delete", DAY + "\x7f", None),
        ("nel", DAY + "\x85", None),
        ("line_separator", DAY + "\u2028", None),
        ("paragraph_separator", DAY + "\u2029", None),
        ("nondecimal_number", "2027-04-1²", None),
        ("backslash", DAY + "\\", None),
    )
    results = {}
    for name, day, rendered in cases:
        root = setup(scratch, name)
        (root / f"{day}.json").write_text("{}")
        assert (root / f"{day}.json").exists()
        assert (
            bool(picks._ISO_DATE_RE.match(day)) and day <= TODAY
        ) == (rendered is not None), (name, day)
        run = examine(root, lookback=8)
        if rendered is None:
            assert run["stderr"] == "" and run["strict"][:3] == [["return", (0, True)], 0, None]
            assert receipt(run)["replay"] == {"state": "saved", "streak": 0, "saver_available": True}
        else:
            assert run["strict"][0] == ["return", None]
            expected(run, day, rendered, PENDING, PENDING)
        results[name] = {"stderr": run["stderr"], "replay": receipt(run)["replay"]}

    root = setup(scratch, "invalid_filename_bytes")
    raw = os.fsencode(root) + b"/2027-04-16\xff.json"
    try:
        fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        if exc.errno not in (errno.EILSEQ, errno.EINVAL):
            raise
        results["invalid_filename_bytes"] = {"creation_refused": [exc.errno, exc.strerror]}
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"{}")
        [path] = root.glob("2027-*.json")
        assert path.stem == DAY + "\udcff" and not picks._ISO_DATE_RE.match(path.stem)
        run = examine(root)
        assert run["stderr"] == "" and receipt(run)["replay"]["state"] == "saved"
        results["invalid_filename_bytes"] = {"stem": path.stem, "stderr": run["stderr"]}
    return results


def reasons(scratch):
    from bts import picks

    results = {}
    for name, day, rendered, body, reason in (
        ("invalid", DAY, DAY, "{", INVALID),
        ("invalid-lf", DAY + "\n", DAY + r"\n", "{", INVALID),
        ("applied", TODAY, TODAY, "{}", APPLIED),
        ("pending", DAY, DAY, "{}", PENDING),
        ("lost", DAY, DAY, '{"result":"hit"}', LOST),
        ("lost-lf", DAY + "\n", DAY + r"\n", '{"result":"hit"}', LOST),
    ):
        root = setup(scratch, f"reason-{name}")
        path = root / f"{day}.json"
        path.write_text(body)
        if reason == LOST:
            read = Path.read_text

            def unlink_after_read(candidate, *args, **kwargs):
                content = read(candidate, *args, **kwargs)
                if candidate == path:
                    path.unlink()
                return content

            with patch.object(Path, "read_text", unlink_after_read):
                run = examine(root, direct=False)
        else:
            run = examine(root)
        expected(run, day, rendered, reason, reason)
        results[name] = {"stderr": run["stderr"], "replay": receipt(run)["replay"]}
    return results


def injected_fields(scratch):
    from bts import picks

    # Narrow seam for characters that cannot reach the real replay callback.
    suffix = "\n\r\v\f\x00\x1b\x7f\x85\u2028\u2029\\\udcff"
    escaped = r"\n\r\x0b\x0c\x00\x1b\x7f\x85\u2028\u2029\\\udcff"
    root = setup(scratch, "injected")

    def replay(*_args, receipt, **_kwargs):
        receipt.replayed(None, blocked_date=DAY + suffix, reason=PENDING + suffix)
        return []

    with patch.object(picks, "reconcile_results", replay):
        run = examine(root, direct=False)
    expected(run, DAY + suffix, DAY + escaped, PENDING + suffix, PENDING + escaped)
    return {"stderr": run["stderr"], "replay": receipt(run)["replay"]}


def main():
    from bts import cli

    assert Path(cli.__file__).resolve().parent == REVIEW.parents[1] / "src" / "bts"
    with tempfile.TemporaryDirectory(dir=REVIEW) as tmp, deadline(), \
         patch.object(socket.socket, "connect", refuse_network), \
         patch.object(socket.socket, "connect_ex", refuse_network), \
         patch.object(socket, "getaddrinfo", refuse_network):
        scratch = Path(tmp)
        result = {"filenames": filenames(scratch), "reasons": reasons(scratch),
                  "injected_fields": injected_fields(scratch), "clock": RUN.isoformat()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

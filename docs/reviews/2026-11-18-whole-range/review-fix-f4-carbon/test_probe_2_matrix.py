"""Review probe 2: adversarial file matrix, growing/partial reads, special files, TOCTOU, FD leaks, order, standing rows."""
from __future__ import annotations

import datetime as dt
import json
import os
import signal
import socket
import stat
import tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import contest_state, day_outcome, dm, picks, scheduler, util
from bts.cli import cli
from tests.test_day_outcome import DATE, _bound_receipt, _box, _publish_receipt, _write_decision

ET = ZoneInfo("America/New_York")
MIB = 1024 * 1024


class PinnedClock(dt.datetime):
    wall = dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.wall.astimezone(tz) if tz else cls.wall.replace(tzinfo=None)


@contextmanager
def bounded_alarm(seconds: int = 2):
    def on_alarm(*_):
        raise AssertionError("operation blocked past alarm")

    prev = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def open_fds() -> list[int]:
    fds = []
    for fd in range(256):
        try:
            os.fstat(fd)
            fds.append(fd)
        except OSError:
            pass
    return fds


def snapshot_tree(root: Path) -> dict[str, tuple]:
    snap = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        st = p.lstat()
        if stat.S_ISLNK(st.st_mode):
            snap[rel] = ("symlink", os.readlink(p))
        elif stat.S_ISREG(st.st_mode):
            try:
                body = p.read_bytes()
            except PermissionError:
                body = b"<unreadable>"
            snap[rel] = ("reg", st.st_mode & 0o777, body)
        else:
            snap[rel] = ("special", stat.S_IFMT(st.st_mode))
    return snap


def invoke_day_outcome(
    picks_dir: Path,
    cfg_path: Path,
    games: int | None = 1,
    *,
    date: str | None = DATE,
    wall: dt.datetime = PinnedClock.wall,
    dm_error: Exception | None = None,
    recipient: bool = True,
):
    sent = []
    idempotent_flags = []

    def fake_schedule(requested):
        if games is None:
            raise OSError("schedule unreachable")
        return [{"gamePk": 900}] * games

    def fake_urlopen(req, **opts):
        route = req.full_url.rsplit("/", 1)[-1]
        if route == "chat.bsky.convo.sendMessage":
            idempotent_flags.append(opts.get("idempotent"))
            body = json.loads(req.data)["message"]["text"]
            sent.append(body)
            if len(body) > 1000:
                raise HTTPError(req.full_url, 400, "maxGraphemes=1000", {}, None)
            if dm_error is not None:
                raise dm_error
            return BytesIO(b'{"id":"dm-ok"}')
        if route == "com.atproto.server.createSession":
            return BytesIO(b'{"accessJwt":"tok"}')
        if route.startswith("chat.bsky.convo.getConvoForMembers?"):
            return BytesIO(b'{"convo":{"id":"cid"}}')
        raise AssertionError(req.full_url)

    argv = ["day-outcome", "--config", str(cfg_path), "--picks-dir", str(picks_dir),
            "--expected-username", "stonehengee"]
    if date is not None:
        argv += ["--date", date]
    if recipient:
        argv += ["--dm-recipient", "did:plc:operator"]

    with patch.object(PinnedClock, "wall", wall), \
         patch.object(dt, "datetime", PinnedClock), \
         patch.object(scheduler, "fetch_schedule", fake_schedule), \
         patch.object(dm, "get_dm_password", lambda: "pw"), \
         patch.object(dm, "retry_urlopen", fake_urlopen):
        res = CliRunner().invoke(cli, argv)
    return res, sent, idempotent_flags


MATRIX_CASES = [
    # (label, writer_or_bytes, expected_problem)
    ("absent", None, None),
    ("empty", b"", "not valid JSON"),
    ("whitespace_only", b"   \r\n\t ", "not valid JSON"),
    ("utf8_bom_object", b"\xef\xbb\xbf{}", "not valid JSON"),
    ("syntax_brace", b"{", "not valid JSON"),
    ("trailing_garbage", b"{} trailing", "not valid JSON"),
    ("raw_nul_in_string", b'{"streak": 1, "note": "a\x00b"}', "not valid JSON"),
    ("raw_cr_in_string", b'{"streak": 1, "note": "a\rb"}', "not valid JSON"),
    ("invalid_utf8_ff", b"\xff", "cannot be read (UnicodeDecodeError)"),
    ("invalid_utf8_truncated_seq", b'{"a": "\xe2\x82"}', "cannot be read (UnicodeDecodeError)"),
    ("top_array", b"[]", "not a JSON object"),
    ("top_string", b'"x"', "not a JSON object"),
    ("top_bool_true", b"true", "not a JSON object"),
    ("top_bool_false", b"false", "not a JSON object"),
    ("top_int", b"0", "not a JSON object"),
    ("top_negative", b"-1", "not a JSON object"),
    ("top_float", b"3.0", "not a JSON object"),
    ("top_null", b"null", "not a JSON object"),
    ("top_nan", b"NaN", "not a JSON object"),
    ("top_huge_100_digits", b"1" + b"0" * 99, "not a JSON object"),
    ("top_huge_5000_digits", b"1" + b"0" * 4999, "JSON exceeds decoder limits (ValueError)"),
    ("obj_empty", b"{}", None),
    ("obj_season_only", b'{"season":2027}', None),
    ("obj_negative_zero_int", b'{"streak":-0}', None),
    ("obj_negative_zero_float", b'{"streak":-0.0}', "streak is not a non-negative integer"),
    ("obj_sci_zero_float", b'{"streak":0e0}', "streak is not a non-negative integer"),
    ("obj_bool_true", b'{"streak":true}', "streak is not a non-negative integer"),
    ("obj_bool_false", b'{"streak":false}', "streak is not a non-negative integer"),
    ("obj_neg_one", b'{"streak":-1}', "streak is not a non-negative integer"),
    ("obj_float", b'{"streak":3.0}', "streak is not a non-negative integer"),
    ("obj_string", b'{"streak":"3"}', "streak is not a non-negative integer"),
    ("obj_null", b'{"streak":null}', "streak is not a non-negative integer"),
    ("obj_list", b'{"streak":[1]}', "streak is not a non-negative integer"),
    ("obj_dict", b'{"streak":{}}', "streak is not a non-negative integer"),
    ("obj_nan_streak", b'{"streak":NaN}', "streak is not a non-negative integer"),
    ("obj_inf_streak", b'{"streak":Infinity}', "streak is not a non-negative integer"),
    ("obj_overflow_float_streak", b'{"streak":1e500}', "streak is not a non-negative integer"),
    ("obj_surrogate_streak", b'{"streak":"\\ud800"}', "streak is not a non-negative integer"),
    ("obj_prior_season_good", b'{"season":2026,"streak":5}', None),
    ("obj_prior_season_bad_str", b'{"season":2026,"streak":"3"}', "streak is not a non-negative integer"),
    ("obj_prior_season_bad_bool", b'{"season":2026,"streak":false}', "streak is not a non-negative integer"),
    ("obj_unknown_season_good", b'{"season":"2027","streak":3}', None),
    ("obj_bad_applied_dates_and_saver", b'{"season":2027,"streak":10,"applied_dates":"bad","saver_available":{}}', None),
    ("obj_nan_ignored", b'{"season":2027,"streak":3,"other":NaN}', None),
    ("obj_dup_last_good", b'{"season":2027,"streak":"3","streak":3}', None),
    ("obj_dup_last_bad", b'{"season":2027,"streak":3,"streak":"3"}', "streak is not a non-negative integer"),
    ("obj_100_digits", b'{"season":2027,"streak":' + b"1" + b"0" * 99 + b"}", None),
    ("obj_4300_nines", b'{"season":2027,"streak":' + b"9" * 4300 + b"}", None),
    ("obj_4301_digits", b'{"streak":' + b"1" + b"0" * 4300 + b"}", "JSON exceeds decoder limits (ValueError)"),
    ("obj_5000_digits", b'{"streak":' + b"9" * 5000 + b"}", "JSON exceeds decoder limits (ValueError)"),
    ("obj_neg_5000_digits", b'{"streak":-' + b"9" * 5000 + b"}", "JSON exceeds decoder limits (ValueError)"),
    ("obj_ignored_5000_digits", b'{"streak":3,"other":' + b"9" * 5000 + b"}", "JSON exceeds decoder limits (ValueError)"),
    ("nested_200_ok", b'{"ignored":' + b"[" * 200 + b"0" + b"]" * 200 + b"}", None),
    ("nested_12000_arrays", b"[" * 12000 + b"0" + b"]" * 12000, "JSON exceeds decoder limits (RecursionError)"),
    ("nested_12000_objects", b'{"a":' * 12000 + b"0" + b"}" * 12000, "JSON exceeds decoder limits (RecursionError)"),
    ("exact_1mib", b" " * (MIB - 2) + b"{}", None),
    ("over_1mib_by_1", b" " * (MIB - 1) + b"{}", "exceeds 1 MiB"),
    ("over_1mib_split_utf8", b" " * (MIB - 1) + "\U0001f680".encode("utf-8"), "exceeds 1 MiB"),
]


def test_full_rule_matrix_clean_and_problem_days(tmp_path):
    # Clean day ('entered') and problem day ('no_decision')
    clean_dir, clean_cfg = _box(tmp_path / "clean")
    _write_decision(clean_dir)
    _publish_receipt(clean_dir, _bound_receipt(clean_dir))

    prob_dir, prob_cfg = _box(tmp_path / "prob")

    fds_before = open_fds()
    for label, payload, expected in MATRIX_CASES:
        for kind, (pdir, cfg) in (("clean", (clean_dir, clean_cfg)), ("problem", (prob_dir, prob_cfg))):
            sp = pdir / "streak.json"
            if sp.exists() or sp.is_symlink():
                sp.unlink()
            if payload is not None:
                sp.write_bytes(payload)
            snap_before = snapshot_tree(pdir)
            with bounded_alarm(3):
                res, sent, idem = invoke_day_outcome(pdir, cfg, games=1)
            assert res.exception is None or isinstance(res.exception, SystemExit), (label, kind, res.exception)
            assert res.exit_code != 2, f"{label}/{kind} unexpectedly exited 2: {res.stderr}"
            out = json.loads(res.stdout)
            assert out["streak_problem"] == expected, (label, kind, out)
            if kind == "clean":
                assert out["state"] == "entered"
                if expected is None:
                    assert res.exit_code == 0 and out["ok"] is True and sent == []
                else:
                    assert res.exit_code == 1 and out["ok"] is False and len(sent) == 1
                    assert sent[0].startswith(f"BTS day check {DATE}: MODEL STREAK FILE NEEDS REVIEW. ")
                    assert f"({expected})" in sent[0]
                    assert idem == [False]
            else:
                assert out["state"] == "no_decision"
                assert res.exit_code == 1 and out["ok"] is False and len(sent) == 1
                assert sent[0].startswith(f"BTS day check {DATE}: NO DECISION. ")
                if expected is not None:
                    assert f"({expected})" in sent[0]
            assert snapshot_tree(pdir) == snap_before, f"Disk modified on {label}/{kind}"
    assert open_fds() == fds_before, "File descriptor leak across matrix"


def test_special_files_symlinks_sockets_and_toctou(tmp_path):
    pdir, cfg = _box(tmp_path / "special")
    sp = pdir / "streak.json"
    fds_before = open_fds()

    # 1. FIFO, mode-000 FIFO, directory, symlink->FIFO, symlink->dir, symlink->/dev/null, AF_UNIX socket
    with tempfile.TemporaryDirectory(dir="/tmp") as short_tmp:
        sock_target = Path(short_tmp) / "s.sock"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(sock_target))
        try:
            fifo_target = tmp_path / "ext_fifo"
            os.mkfifo(fifo_target)
            dir_target = tmp_path / "ext_dir"
            dir_target.mkdir()

            special_setups = [
                ("fifo", lambda: os.mkfifo(sp)),
                ("fifo_mode000", lambda: (os.mkfifo(sp), sp.chmod(0))),
                ("directory", lambda: sp.mkdir()),
                ("symlink_to_fifo", lambda: sp.symlink_to(fifo_target)),
                ("symlink_to_dir", lambda: sp.symlink_to(dir_target)),
                ("symlink_to_devnull", lambda: sp.symlink_to("/dev/null")),
                ("symlink_to_unix_socket", lambda: sp.symlink_to(sock_target)),
            ]
            real_open = os.open

            def forbid_streak_open(candidate, *a, **kw):
                assert Path(candidate) != sp, f"Special file {sp} was passed to os.open!"
                return real_open(candidate, *a, **kw)

            for name, creator in special_setups:
                if sp.exists() or sp.is_symlink():
                    if sp.is_dir() and not sp.is_symlink():
                        sp.rmdir()
                    else:
                        sp.chmod(0o600) if not sp.is_symlink() else None
                        sp.unlink()
                creator()
                try:
                    with bounded_alarm(2), patch.object(util.os, "open", forbid_streak_open):
                        res, sent, _ = invoke_day_outcome(pdir, cfg, games=0)
                    assert res.exit_code == 1, (name, res.output)
                    payload = json.loads(res.stdout)
                    assert payload["state"] == "no_games"
                    assert payload["streak_problem"] == "not a regular file", (name, payload)
                    assert len(sent) == 1
                finally:
                    if sp.is_symlink():
                        sp.unlink()
                    elif sp.is_dir():
                        sp.rmdir()
                    elif sp.exists():
                        sp.chmod(0o600)
                        sp.unlink()

            # Direct AF_UNIX socket at picks_dir/streak.json in short /tmp path
            short_picks = Path(short_tmp) / "p"
            short_picks.mkdir()
            direct_sock_path = short_picks / "streak.json"
            sock2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock2.bind(str(direct_sock_path))
            try:
                with bounded_alarm(2):
                    res, _, _ = invoke_day_outcome(short_picks, cfg, games=0)
                assert res.exit_code == 1
                assert json.loads(res.stdout)["streak_problem"] == "not a regular file"
            finally:
                sock2.close()
        finally:
            sock.close()

    # 2. Dangling symlink (allowed), symlink loop (OSError), symlink through regular file (NotADirectoryError)
    sp.symlink_to("does_not_exist.json")
    res, sent, _ = invoke_day_outcome(pdir, cfg, games=0)
    assert res.exit_code == 0 and json.loads(res.stdout)["streak_problem"] is None and sent == []
    sp.unlink()

    sp.symlink_to("streak.json")
    res, sent, _ = invoke_day_outcome(pdir, cfg, games=0)
    assert res.exit_code == 1 and json.loads(res.stdout)["streak_problem"] == "cannot be read (OSError)"
    sp.unlink()

    reg_parent = tmp_path / "regular_parent"
    reg_parent.write_text("not a dir")
    sp.symlink_to(reg_parent / "child.json")
    res, sent, _ = invoke_day_outcome(pdir, cfg, games=0)
    assert res.exit_code == 1 and json.loads(res.stdout)["streak_problem"] == "cannot be read (NotADirectoryError)"
    sp.unlink()

    # 3. TOCTOU: path.stat() sees regular file, swapped to FIFO before os.open -> fstat rejects without os.read
    sp.write_bytes(b"{}")
    swap_fifo = tmp_path / "swap_fifo"
    os.mkfifo(swap_fifo)
    real_open = os.open
    real_read = os.read

    def swap_on_open(candidate, flags, *a, **kw):
        if Path(candidate) == sp:
            return real_open(swap_fifo, flags, *a, **kw)
        return real_open(candidate, flags, *a, **kw)

    def forbid_fifo_read(fd, n):
        assert not stat.S_ISFIFO(os.fstat(fd).st_mode), "os.read called on FIFO descriptor!"
        return real_read(fd, n)

    with bounded_alarm(2), patch.object(util.os, "open", swap_on_open), patch.object(util.os, "read", forbid_fifo_read):
        res, _, _ = invoke_day_outcome(pdir, cfg, games=0)
    assert res.exit_code == 1 and json.loads(res.stdout)["streak_problem"] == "not a regular file"
    assert open_fds() == fds_before, "FD leak in special-file/TOCTOU checks"


def test_growing_file_and_partial_reads_bounded(tmp_path):
    pdir, cfg = _box(tmp_path / "growing")
    sp = pdir / "streak.json"
    sp.write_bytes(b" " * 65536)
    real_read = os.read
    read_calls = 0

    def growing_partial_read(fd, count):
        nonlocal read_calls
        if os.fstat(fd).st_ino == sp.stat().st_ino:
            read_calls += 1
            # Append more data on every read so EOF would never be reached
            with sp.open("ab") as wf:
                wf.write(b" " * 100000)
            return real_read(fd, min(count, 65536))
        return real_read(fd, count)

    with bounded_alarm(3), patch.object(util.os, "read", growing_partial_read):
        res, sent, _ = invoke_day_outcome(pdir, cfg, games=0)
    assert res.exit_code == 1
    assert json.loads(res.stdout)["streak_problem"] == "exceeds 1 MiB"
    assert 1 <= read_calls <= 20
    assert len(sent) == 1

    # Also check 1-byte partial reads at exact max_bytes vs max_bytes + 1
    exact = tmp_path / "exact128.json"
    exact.write_bytes(b" " * 126 + b"{}")
    over = tmp_path / "over128.json"
    over.write_bytes(b" " * 127 + b"{}")
    with patch.object(util.os, "read", lambda fd, n: real_read(fd, min(n, 1))):
        assert util.read_regular_file_text(exact, max_bytes=128) == " " * 126 + "{}"
        with pytest.raises(util.FileReadLimitError):
            util.read_regular_file_text(over, max_bytes=128)


def test_day_classified_before_streak_file_read(tmp_path):
    pdir, cfg = _box(tmp_path / "order")
    _write_decision(pdir, action="skip", source="mdp", scoreable=False, delivery_status=None,
                    streak=30, saver_available=False)
    acc = pdir / "account_state"
    acc.mkdir()
    end = dt.datetime.combine(dt.date.fromisoformat(DATE), dt.time.max, ET)
    (acc / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": contest_state.DATED_MANUAL_SCHEMA,
        "season": 2027,
        "effective_date": DATE,
        "expires_at": end.isoformat(),
        "active_streak": 1,
        "best_streak": 30,
        "saver_state": "used",
        "recorded_at": f"{DATE}T14:00:00Z",
        "source": "manual_cli",
    }))
    sp = pdir / "streak.json"
    sp.write_bytes(b"[]")

    events = []
    orig_classify = day_outcome.classify_day
    orig_skip = day_outcome._skip_review
    orig_read = day_outcome.read_regular_file_text

    def rec_classify(*a, **kw):
        events.append("classify_day")
        return orig_classify(*a, **kw)

    def rec_skip(*a, **kw):
        events.append("_skip_review")
        return orig_skip(*a, **kw)

    def rec_read(path, *a, **kw):
        if Path(path) == sp:
            events.append("read_streak")
        return orig_read(path, *a, **kw)

    with patch.object(day_outcome, "classify_day", rec_classify), \
         patch.object(day_outcome, "_skip_review", rec_skip), \
         patch.object(day_outcome, "read_regular_file_text", rec_read):
        res, sent, _ = invoke_day_outcome(pdir, cfg, games=1)

    assert res.exit_code == 1
    out = json.loads(res.stdout)
    assert out["state"] == "skip_state_review"
    assert out["streak_problem"] == "not a JSON object"
    assert events == ["classify_day", "_skip_review", "read_streak"], events


def test_standing_rows_interrupt_midnight_send_restart_and_exit2(tmp_path):
    # 1. Independent exit-2 failures stay exit 2 even when streak.json is broken
    pdir, cfg = _box(tmp_path / "exit2_cases")
    (pdir / "streak.json").write_bytes(b"[]")
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("not = [valid toml")
    res_toml, sent_toml, _ = invoke_day_outcome(pdir, bad_toml, games=0)
    assert res_toml.exit_code == 2 and res_toml.stdout == ""
    assert "THE CHECK COULD NOT RUN (TOMLDecodeError)" in sent_toml[0]

    # Invalid UTF-8 in decision.json stays exit 2 (base behaviour)
    (pdir / DATE).mkdir(parents=True)
    (pdir / DATE / "decision.json").write_bytes(b"\xff")
    res_dec, sent_dec, _ = invoke_day_outcome(pdir, cfg, games=1)
    assert res_dec.exit_code == 2 and res_dec.stdout == ""
    assert "THE CHECK COULD NOT RUN (UnicodeDecodeError)" in sent_dec[0]

    # 2. Interrupt inside day-outcome mid-read and mid-send writes nothing
    pdir2, cfg2 = _box(tmp_path / "interrupt")
    (pdir2 / "streak.json").write_bytes(b"{")
    snap = snapshot_tree(pdir2)
    with patch.object(day_outcome, "read_regular_file_text", side_effect=KeyboardInterrupt):
        res_int, _, _ = invoke_day_outcome(pdir2, cfg2, games=0)
    assert isinstance(res_int.exception, KeyboardInterrupt)
    assert snapshot_tree(pdir2) == snap

    # 3. Failed / unconfirmed sends (HTTP 500, URLError, TimeoutError): single attempt (idempotent=False), stateless retry
    for err in (HTTPError("https://api.bsky.chat/xrpc/chat.bsky.convo.sendMessage", 500, "err", {}, None),
                URLError("conn reset"),
                TimeoutError("lost response")):
        res_err, sent_err, idem = invoke_day_outcome(pdir2, cfg2, games=0, dm_error=err)
        assert res_err.exit_code == 1
        assert len(sent_err) == 1 and idem == [False]
        assert f"day-outcome: DM failed: {type(err).__name__}\n" == res_err.stderr
        assert snapshot_tree(pdir2) == snap

    # Subsequent restart/re-run still reports identically
    res_ok, sent_ok, _ = invoke_day_outcome(pdir2, cfg2, games=0)
    assert res_ok.exit_code == 1 and len(sent_ok) == 1 and res_ok.stderr == ""
    assert snapshot_tree(pdir2) == snap

    # 4. Midnight ET default --date resolution and New Year / Opening boundary
    pdir3, cfg3 = _box(tmp_path / "midnight")
    (pdir3 / "streak.json").write_bytes(b'{"season":2027,"streak":9}')
    for wall_dt, expected_date in [
        (dt.datetime(2027, 12, 31, 23, 59, 59, tzinfo=ET), "2027-12-31"),
        (dt.datetime(2028, 1, 1, 0, 0, 1, tzinfo=ET), "2028-01-01"),
    ]:
        res_m, sent_m, _ = invoke_day_outcome(pdir3, cfg3, games=0, date=None, wall=wall_dt)
        assert res_m.exit_code == 0 and sent_m == []
        out_m = json.loads(res_m.stdout)
        assert out_m["date"] == expected_date and out_m["streak_problem"] is None


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        test_full_rule_matrix_clean_and_problem_days(base / "t1")
        test_special_files_symlinks_sockets_and_toctou(base / "t2")
        test_growing_file_and_partial_reads_bounded(base / "t3")
        test_day_classified_before_streak_file_read(base / "t4")
        test_standing_rows_interrupt_midnight_send_restart_and_exit2(base / "t5")
    print("PROBE 2 PASSED")

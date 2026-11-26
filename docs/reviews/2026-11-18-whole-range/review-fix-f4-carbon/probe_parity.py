"""Review probe 1 (normalized temp paths): base (ddbc531a) vs F4 (b3d5170b) parity across all day states and read_regular_file_text."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAD_SRC = REPO_ROOT / "src"

WORKER_CODE = r"""
import datetime as dt
import json
import os
import signal
import sys
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

target_src = Path(os.environ["TARGET_SRC"]).resolve()
repo_root = Path(os.environ["REPO_ROOT"]).resolve()
side = os.environ["SIDE_NAME"]

sys.meta_path = [
    finder for finder in sys.meta_path
    if "editable" not in type(finder).__name__.lower() and "bts" not in repr(finder).lower()
]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(target_src))

import bts
from bts import cli as cli_mod, contest_state, dm, saver_state, scheduler, season_contract, util
from click.testing import CliRunner
from tests.test_day_outcome import DATE, PICK, _bound_receipt, _box, _publish_receipt, _write_decision

actual_init = Path(bts.__file__).resolve()
expected_init = (target_src / "bts" / "__init__.py").resolve()
assert actual_init == expected_init, f"Expected {expected_init}, got {actual_init}"

ET = ZoneInfo("America/New_York")
MIB = 1024 * 1024


class PinnedClock(dt.datetime):
    wall = dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.wall.astimezone(tz) if tz else cls.wall.replace(tzinfo=None)


@contextmanager
def alarm_limit(seconds=2):
    def on_alarm(*_):
        raise TimeoutError("blocked in file read")

    prev = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def build_state_fixture(root: Path, case_name: str):
    sub = root / case_name
    if case_name == "entered":
        p, cfg = _box(sub)
        _write_decision(p)
        _publish_receipt(p, _bound_receipt(p))
        games = [{"gamePk": 900}]
    elif case_name == "entered_double":
        p, cfg = _box(sub)
        dd = {**PICK, "batter_id": 2, "batter_name": "Ramirez", "game_pk": 901}
        _write_decision(p, action="double", double_down=dd)
        _publish_receipt(p, _bound_receipt(p))
        games = [{"gamePk": 900}, {"gamePk": 901}]
    elif case_name == "entered_game_unverified":
        p, cfg = _box(sub)
        _write_decision(p)
        rec = _bound_receipt(p, confirmed=False, reason="match")
        rec["qualification"]["slots"] = [{"state": "unit_unverified"}]
        _publish_receipt(p, rec)
        games = [{"gamePk": 900}]
    elif case_name == "deliberate_skip":
        p, cfg = _box(sub)
        _write_decision(p, action="skip", source="mdp", scoreable=False, delivery_status=None)
        games = [{"gamePk": 900}]
    elif case_name == "no_games":
        p, cfg = _box(sub)
        games = []
    elif case_name == "no_games_all_void":
        p, cfg = _box(sub)
        _write_decision(p)
        games = [{"status": {"detailedState": "Postponed"}}, {"status": {"detailedState": "Cancelled"}}]
    elif case_name == "research_commit":
        p, cfg = _box(sub, intent="research", delivery="private")
        _write_decision(p, delivery_status="private_locked")
        games = [{"gamePk": 900}]
    elif case_name == "delivered_under_research":
        p, cfg = _box(sub, intent="research", delivery="private")
        _write_decision(p, delivery_status="delivered")
        games = [{"gamePk": 900}]
    elif case_name == "intent_unknown":
        p, cfg = _box(sub, intent="bogus", delivery="dm")
        games = [{"gamePk": 900}]
    elif case_name == "schedule_unverifiable":
        p, cfg = _box(sub)
        games = None
    elif case_name == "no_decision":
        p, cfg = _box(sub)
        games = [{"gamePk": 900}]
    elif case_name == "skip_not_deliberate":
        p, cfg = _box(sub)
        _write_decision(p, action="skip", source="crash_guard", scoreable=False, delivery_status=None)
        games = [{"gamePk": 900}]
    elif case_name == "commit_not_scoreable":
        p, cfg = _box(sub)
        _write_decision(p, delivery_status="failed", scoreable=False)
        games = [{"gamePk": 900}]
    elif case_name == "contradiction":
        p, cfg = _box(sub)
        _write_decision(p)
        games = []
    elif case_name == "entered_late":
        p, cfg = _box(sub)
        _write_decision(p)
        _publish_receipt(p, _bound_receipt(p, before_cutoff=False))
        games = [{"gamePk": 900}]
    elif case_name == "not_entered":
        p, cfg = _box(sub)
        _write_decision(p)
        _publish_receipt(p, _bound_receipt(p, confirmed=False, reason="no_pick"))
        games = [{"gamePk": 900}]
    elif case_name == "entry_unverifiable":
        p, cfg = _box(sub)
        _write_decision(p)
        games = [{"gamePk": 900}]
    elif case_name == "skip_state_review_dated":
        p, cfg = _box(sub)
        _write_decision(p, action="skip", source="mdp", scoreable=False, delivery_status=None,
                        streak=30, saver_available=False)
        acc = p / "account_state"
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
        games = [{"gamePk": 900}]
    elif case_name == "skip_state_review_revoked":
        p, cfg = _box(sub)
        _write_decision(p, action="skip", source="mdp", scoreable=False, delivery_status=None,
                        streak=30, saver_available=False)
        acc = p / "account_state"
        acc.mkdir()
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "revoked",
            "activation_id": "a" * 32,
            "switched_at": None,
            "revoked_at": f"{DATE}T20:00:00Z",
        }))
        games = [{"gamePk": 900}]
    else:
        raise ValueError(case_name)
    return sub, p, cfg, games


CASES = [
    "entered", "entered_double", "entered_game_unverified", "deliberate_skip",
    "no_games", "no_games_all_void", "research_commit", "delivered_under_research",
    "intent_unknown", "schedule_unverifiable", "no_decision", "skip_not_deliberate",
    "commit_not_scoreable", "contradiction", "entered_late", "not_entered",
    "entry_unverifiable", "skip_state_review_dated", "skip_state_review_revoked",
]

STREAK_GOOD_VARIANTS = {
    "absent": None,
    "dangling": "DANGLING",
    "empty_obj": b"{}",
    "current_zero": b'{"season":2027,"streak":0,"saver_available":true,"applied_dates":[]}',
    "prior_bad_meta": b'{"season":2026,"streak":14,"saver_available":"bad","applied_dates":null}',
    "unknown_tag": b'{"season":"unknown","streak":58,"applied_dates":[1,"2027-04-01"]}',
    "huge_4300": b'{"streak":' + b"9" * 4300 + b"}",
    "nan_and_dup_good": b'{"streak":"bad","streak":7,"other":NaN}',
    "crlf": b'{\r\n  "season": 2027,\r\n  "streak": 2\r\n}\r\n',
    "exact_1mib": b" " * (MIB - 2) + b"{}",
}


def snapshot(root: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = b"SYMLINK:" + os.readlink(p).encode("utf-8")
        elif p.is_file():
            out[rel] = p.read_bytes()
    return out


def invoke_cli(root: Path, p: Path, cfg: Path, games, *, recipient: bool = True, dm_fail: bool = False):
    sent = []

    def fake_schedule(d):
        assert d == DATE
        if games is None:
            raise OSError("schedule down")
        return games

    def fake_urlopen(req, **opts):
        route = req.full_url.rsplit("/", 1)[-1]
        if route == "chat.bsky.convo.sendMessage":
            body = json.loads(req.data)["message"]["text"]
            sent.append(body)
            if dm_fail:
                raise TimeoutError("simulated timeout")
            return BytesIO(b'{"id":"msg-1"}')
        if route == "com.atproto.server.createSession":
            return BytesIO(b'{"accessJwt":"jwt-1"}')
        if route.startswith("chat.bsky.convo.getConvoForMembers?"):
            return BytesIO(b'{"convo":{"id":"convo-1"}}')
        raise AssertionError(req.full_url)

    argv = ["day-outcome", "--date", DATE, "--config", str(cfg), "--picks-dir", str(p),
            "--expected-username", "stonehengee"]
    if recipient:
        argv += ["--dm-recipient", "did:plc:operator"]

    with patch.object(dt, "datetime", PinnedClock), \
         patch.object(scheduler, "fetch_schedule", fake_schedule), \
         patch.object(dm, "get_dm_password", lambda: "pw"), \
         patch.object(dm, "retry_urlopen", fake_urlopen):
        res = CliRunner().invoke(cli_mod.cli, argv)
    return {
        "exit_code": res.exit_code,
        "stdout": res.stdout,
        "stderr": res.stderr.replace(str(root), "<TMP>"),
        "sent": sent,
    }


def collect_all():
    day_records = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for case_name in CASES:
            sub, p, cfg, games = build_state_fixture(root, case_name)
            streak_path = p / "streak.json"
            for vname, content in STREAK_GOOD_VARIANTS.items():
                if streak_path.exists() or streak_path.is_symlink():
                    streak_path.unlink()
                if content == "DANGLING":
                    streak_path.symlink_to("missing_target.json")
                elif content is not None:
                    streak_path.write_bytes(content)

                snap_before = snapshot(sub)
                rec = invoke_cli(root, p, cfg, games, recipient=True, dm_fail=False)
                snap_after = snapshot(sub)
                assert snap_before == snap_after, f"Mutation in {case_name}/{vname}"
                day_records[f"{case_name}::{vname}"] = rec

            streak_path.write_bytes(b"{}")
            day_records[f"{case_name}::no_recipient"] = invoke_cli(root, p, cfg, games, recipient=False)
            day_records[f"{case_name}::dm_fail"] = invoke_cli(root, p, cfg, games, recipient=True, dm_fail=True)

        sub2, p2, cfg2, games2 = build_state_fixture(root / "exit2", "no_games")
        cfg2.unlink()
        (p2 / "streak.json").write_bytes(b"{}")
        day_records["exit2::missing_config"] = invoke_cli(root, p2, cfg2, games2)

        rr_root = root / "rr"
        rr_root.mkdir()
        reader_records = {}

        def run_default_reader(label: str, target: Path):
            try:
                with alarm_limit(2):
                    val = util.read_regular_file_text(target)
                reader_records[label] = {"ok": True, "val_len": len(val), "val_sample": val[:64]}
            except Exception as exc:
                msg = str(exc).replace(str(rr_root), "<RR>")
                reader_records[label] = {
                    "ok": False,
                    "exc_exact_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "msg": msg,
                }

        f_utf8 = rr_root / "utf8.txt"
        f_utf8.write_bytes("hello\nhéllo\n".encode("utf-8"))
        run_default_reader("utf8", f_utf8)

        f_crlf = rr_root / "crlf.txt"
        f_crlf.write_bytes(b"line1\r\nline2\rline3\n")
        run_default_reader("crlf", f_crlf)

        f_large = rr_root / "large.txt"
        f_large.write_bytes(b"x" * (MIB + 500))
        run_default_reader("over_1mib_default", f_large)

        run_default_reader("missing", rr_root / "absent.txt")

        f_dangling = rr_root / "dangling.txt"
        f_dangling.symlink_to("no_target.txt")
        run_default_reader("dangling", f_dangling)

        f_loop = rr_root / "loop.txt"
        f_loop.symlink_to("loop.txt")
        run_default_reader("loop", f_loop)

        f_fifo = rr_root / "fifo.txt"
        os.mkfifo(f_fifo)
        run_default_reader("fifo", f_fifo)

        f_dir = rr_root / "dir.txt"
        f_dir.mkdir()
        run_default_reader("dir", f_dir)

        f_bad_utf8 = rr_root / "bad.txt"
        f_bad_utf8.write_bytes(b"\xff\xfe")
        run_default_reader("bad_utf8", f_bad_utf8)

        if os.geteuid() != 0:
            f_perm = rr_root / "perm.txt"
            f_perm.write_bytes(b"{}")
            f_perm.chmod(0)
            try:
                run_default_reader("perm000", f_perm)
            finally:
                f_perm.chmod(0o600)

        caller_dir = rr_root / "callers"
        (caller_dir / "account_state").mkdir(parents=True)
        now_dt = dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)
        sw_path = caller_dir / "account_state" / "season_contract.json"
        sw_path.write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": "b" * 32,
            "switched_at": "2027-03-25T12:00:00Z",
            "revoked_at": None,
        }))
        sw_ok = season_contract._read_switch_state(caller_dir, now_dt, historical=True)
        sw_path.unlink()
        os.mkfifo(sw_path)
        with alarm_limit(2):
            sw_fifo = season_contract._read_switch_state(caller_dir, now_dt, historical=True)

        sv_path = caller_dir / "account_state" / "saver_state.json"
        sv_path.write_text(json.dumps({"season": 2027, "state": "active", "source": "cli", "updated_at": "2027-04-01T00:00:00Z"}))
        sv_ok = saver_state._read_saver_state(caller_dir, season=2027, historical=True)
        sv_path.unlink()
        os.mkfifo(sv_path)
        with alarm_limit(2):
            sv_fifo = saver_state._read_saver_state(caller_dir, season=2027, historical=True)

        cfs_path = rr_root / "fetch_status.json"
        cfs_path.write_text(json.dumps({"last_alert_at": "2027-04-02T12:00:00+00:00"}))
        cfs_ok = cli_mod._read_contest_fetch_status(cfs_path)
        cfs_path.unlink()
        os.mkfifo(cfs_path)
        with alarm_limit(2):
            cfs_fifo = cli_mod._read_contest_fetch_status(cfs_path)

        caller_records = {
            "switch_ok": sw_ok.state,
            "switch_fifo": sw_fifo.state,
            "saver_ok": (sv_ok.state, sv_ok.season),
            "saver_fifo": (sv_fifo.state, sv_fifo.season),
            "fetch_status_ok": cfs_ok,
            "fetch_status_fifo": cfs_fifo,
        }

    print(json.dumps({
        "side": side,
        "bts_file": str(actual_init),
        "day_records": day_records,
        "reader_records": reader_records,
        "caller_records": caller_records,
    }))


collect_all()
"""


def _run_side(side_name: str, target_src: Path) -> dict:
    env = os.environ.copy()
    env["SIDE_NAME"] = side_name
    env["TARGET_SRC"] = str(target_src)
    env["REPO_ROOT"] = str(REPO_ROOT)
    env["TZ"] = "America/New_York"
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_CODE],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


@contextmanager
def _base_src_dir():
    configured = os.environ.get("BTS_BASE_SRC")
    if configured and Path(configured).is_dir():
        yield Path(configured)
        return
    with TemporaryDirectory() as tmp:
        dest = Path(tmp)
        chk = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", "ddbc531a^{commit}"],
            capture_output=True,
            timeout=10,
        )
        if chk.returncode != 0:
            pytest.skip("Base commit ddbc531a not present in git object store")
        archive = subprocess.Popen(
            ["git", "-C", str(REPO_ROOT), "archive", "ddbc531a", "src"],
            stdout=subprocess.PIPE,
        )
        try:
            subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True, timeout=15)
        finally:
            if archive.stdout:
                archive.stdout.close()
            archive.wait(timeout=10)
        assert archive.returncode == 0
        yield dest / "src"


def test_base_and_f4_parity_all_states_and_default_reader():
    with _base_src_dir() as base_src:
        base_data = _run_side("base_ddbc531a", base_src)
        head_data = _run_side("head_b3d5170b", HEAD_SRC)

    print(f"base bts.__file__ = {base_data['bts_file']}")
    print(f"head bts.__file__ = {head_data['bts_file']}")
    assert base_data["bts_file"] != head_data["bts_file"]

    assert head_data["reader_records"] == base_data["reader_records"]
    assert head_data["reader_records"]["fifo"]["exc_exact_type"] == "builtins.OSError"
    assert head_data["reader_records"]["crlf"]["val_sample"] == "line1\nline2\nline3\n"
    assert head_data["reader_records"]["over_1mib_default"]["ok"] is True
    assert head_data["caller_records"] == base_data["caller_records"]

    assert set(head_data["day_records"]) == set(base_data["day_records"])
    compared = 0
    for key, base_rec in base_data["day_records"].items():
        head_rec = head_data["day_records"][key]
        assert head_rec["exit_code"] == base_rec["exit_code"], key
        assert head_rec["stderr"] == base_rec["stderr"], key
        assert head_rec["sent"] == base_rec["sent"], key
        if key.startswith("exit2::"):
            assert head_rec["stdout"] == base_rec["stdout"] == ""
        else:
            head_json = json.loads(head_rec["stdout"])
            assert "streak_problem" in head_json, key
            assert head_json.pop("streak_problem") is None, key
            assert json.dumps(head_json) + "\n" == base_rec["stdout"], key
        compared += 1
    print(f"Verified {compared} day-outcome runs and {len(head_data['reader_records'])} default-reader cases.")


if __name__ == "__main__":
    test_base_and_f4_parity_all_states_and_default_reader()
    print("PROBE PARITY PASSED")

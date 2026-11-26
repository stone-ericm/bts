"""Probe 03: Independent end-to-end verification of closing-test BL-O4 (base RED vs F5 GREEN), M1-M7, and N1-N8."""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
from contextlib import ExitStack
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner  # noqa: E402
from bts import entry_receipt  # noqa: E402
from bts.cli import cli  # noqa: E402
from tests.test_entry_intent import ENTRY_CMD, LIVE_DM, LIVE_DM_PROBLEM, RESEARCH, _exe  # noqa: E402

ET = ZoneInfo("America/New_York")
BASE_REV = "a4e8278d"
DAY = "2027-03-24"
PREV_DAY = "2027-03-23"
ENTER_CFG = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
RIGHT_U = {"id": 77, "roundId": 1, "feedId": 101}
WRONG_U = {"id": 77, "roundId": 1, "feedId": 102}


def _forbid_network() -> None:
    def _deny(*a, **kw):
        raise AssertionError(f"Unexpected network access: {a}")

    socket.socket.connect = _deny  # type: ignore[assignment]
    socket.socket.connect_ex = _deny  # type: ignore[assignment]
    socket.socket.sendto = _deny  # type: ignore[assignment]
    socket.getaddrinfo = _deny  # type: ignore[assignment]


def make_box(root: Path, script_path: Path, src_path: Path):
    home = root / "home"
    (home / "projects" / "bts").mkdir(parents=True)
    (home / "projects" / "bts" / ".env").write_text("")
    (home / ".local" / "bin").mkdir(parents=True)
    _exe(
        home / ".local" / "bin" / "uv",
        '#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || { echo "uv shim: unexpected $*" >&2; exit 97; }\n'
        f'shift 2\nPYTHONPATH="{src_path}" exec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n',
    )
    shims = root / "shims"
    shims.mkdir()
    crontab = root / "crontab.txt"
    _exe(
        shims / "crontab",
        f'#!/bin/sh\nf="{crontab}"\necho "$1" >> "$f.calls"\ncase "$1" in\n'
        '  -l) [ -f "$f.unreadable" ] && { echo "crontab: cannot read" >&2; exit 1; }\n'
        '      [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
        '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n  *) exit 98 ;;\nesac\n',
    )
    (root / "tmp").mkdir()
    env = {
        "HOME": str(home),
        "PATH": f"{shims}:/usr/bin:/bin",
        "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/probe03",
        "TMPDIR": str(root / "tmp"),
    }

    class Box:
        config = home / ".bts-orchestrator.toml"
        cron = crontab
        shims_dir = shims
        home_dir = home

        @staticmethod
        def run(action: str):
            return subprocess.run(["bash", str(script_path), action], env=env, capture_output=True, text=True, timeout=20)

        @staticmethod
        def lines():
            return crontab.read_text().splitlines() if crontab.exists() else None

    return Box()


def cron_slots(line: str, day: str = DAY) -> list[str]:
    minute_field, hour_field, *_ = line.split()
    step = int(minute_field.removeprefix("*/"))
    hours = range(24) if hour_field == "*" else range(10, 24)
    midnight = datetime.fromisoformat(day).replace(tzinfo=ET)
    cur = midnight.astimezone(timezone.utc)
    stop = (midnight + timedelta(days=1)).astimezone(timezone.utc)
    out = []
    while cur < stop:
        et = cur.astimezone(ET)
        if et.hour in hours:
            out.append(et.strftime("%H:%M"))
        cur += timedelta(minutes=step)
    return out


def check_installer_rows_m1_m4_n1_to_n5(work_dir: Path) -> dict:
    script = REPO / "scripts/cron-setup-hetzner.sh"
    src = REPO / "src"

    # M1: show and install under enter
    b_m1 = make_box(work_dir / "m1", script, src)
    b_m1.config.write_text(ENTER_CFG)
    show_m1 = b_m1.run("show")
    inst_m1 = b_m1.run("install")
    assert show_m1.returncode == 0 and inst_m1.returncode == 0
    wd_show, = [ln for ln in show_m1.stdout.splitlines() if ENTRY_CMD in ln and not ln.startswith("#")]
    wd_inst, = [ln for ln in b_m1.lines() if ENTRY_CMD in ln and not ln.startswith("#")]
    assert wd_show == wd_inst
    assert wd_inst.startswith("*/15 * * * * ")
    assert "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840 " in wd_inst
    assert "--cron" in shlex.split(wd_inst)
    assert wd_inst.endswith("# BTS-HETZNER")

    # M4 + N3 + N4 + N1 + N5: foreign, unmarked legacy, active old marked, commented old marked, empty, unreadable, filter err
    b_m4 = make_box(work_dir / "m4", script, src)
    foreign = "0 9 * * * echo foreign"
    unmarked_legacy = f"*/15 10-23 * * * uv run {ENTRY_CMD}"
    old_marked = f"*/15 10-23 * * * uv run {ENTRY_CMD} # BTS-HETZNER"
    b_m4.cron.write_text(f"{foreign}\n{unmarked_legacy}\n# {unmarked_legacy}\n{old_marked}\n# {old_marked}\n")
    b_m4.config.write_text(ENTER_CFG)
    assert b_m4.run("install").returncode == 0
    lines_enter = b_m4.lines()
    assert lines_enter[:3] == [foreign, unmarked_legacy, f"# {unmarked_legacy}"]
    marked_wd = [ln for ln in lines_enter if ENTRY_CMD in ln and "# BTS-HETZNER" in ln]
    assert len(marked_wd) == 1 and marked_wd[0].startswith("*/15 * * * * ")

    # N4: remove removes every marked line (including new one) and preserves foreign + unmarked legacy
    assert b_m4.run("remove").returncode == 0
    assert b_m4.lines() == [foreign, unmarked_legacy, f"# {unmarked_legacy}"]

    # N1 + N5: research omits watchdog & fetch recipient, keeps all other lines byte-for-byte identical
    b_m4.config.write_text(RESEARCH)
    assert b_m4.run("install").returncode == 0
    lines_res = b_m4.lines()
    assert not [ln for ln in lines_res if ENTRY_CMD in ln and "# BTS-HETZNER" in ln]
    bts_enter = [ln for ln in lines_enter if ln.endswith("# BTS-HETZNER") and ENTRY_CMD not in ln]
    bts_res = [ln for ln in lines_res if ln.endswith("# BTS-HETZNER")]
    assert bts_res == [
        ln.replace(" --dm-recipient stonehengee.bsky.social", "") if "fetch-contest-streak" in ln else ln
        for ln in bts_enter
    ]

    # N3: unreadable crontab & filter failure
    before_bytes = b_m4.cron.read_bytes()
    Path(f"{b_m4.cron}.unreadable").touch()
    shown_unread = b_m4.run("show")
    assert shown_unread.returncode == 0 and "(none)" in shown_unread.stdout and "Would install:" in shown_unread.stdout
    for act in ("install", "remove"):
        assert b_m4.run(act).returncode == 1
        assert b_m4.cron.read_bytes() == before_bytes
    Path(f"{b_m4.cron}.unreadable").unlink()
    _exe(b_m4.shims_dir / "grep", '#!/bin/sh\n[ "$1" = "-v" ] && exit 2\nexec /usr/bin/grep "$@"\n')
    for act in ("install", "remove"):
        assert b_m4.run(act).returncode == 1
        assert b_m4.cron.read_bytes() == before_bytes

    # N2: missing, malformed, disagreeing, and live-gate refusal leave crontab untouched and never call crontab
    for idx, bad_cfg in enumerate([
        None,
        '[scheduler]\npick_delivery = "private"\n',
        '[scheduler\nentry_intent = "enter"\n',
        '[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\n',
        '[scheduler]\nentry_intent = "research"\npick_delivery = "dm"\n',
        LIVE_DM,
    ]):
        b_n2 = make_box(work_dir / f"n2_{idx}", script, src)
        b_n2.cron.write_text(old_marked + "\n")
        if bad_cfg is not None:
            b_n2.config.write_text(bad_cfg)
        for act in ("show", "install"):
            r = b_n2.run(act)
            assert r.returncode == 1
            if bad_cfg == LIVE_DM:
                assert LIVE_DM_PROBLEM in r.stderr
            assert not Path(f"{b_n2.cron}.calls").exists()
            assert b_n2.cron.read_text() == old_marked + "\n"

    return {"installed_watchdog_line": wd_inst, "rows": ["M1", "M4", "N1", "N2", "N3", "N4", "N5"]}


def replay_closing_test_base_vs_f5(work_dir: Path, f5_wd_line: str) -> dict:
    """Replay the real closing-test fixture (06:10 pitch committed at 05:10) on archived base a4e8278d vs F5."""
    fixture_gz = REPO / "docs/reviews/2026-11-18-whole-range/closing-test/codex-blind/evidence/cron-early-entry.json.gz"
    with gzip.open(fixture_gz, "rt") as stream:
        snapshot = next(ev[-1] for ev in json.load(stream) if ev[1] == "snapshot")
    saved_pick = snapshot[f"data/picks/{DAY}.json"]
    saved_dec = snapshot[f"data/picks/{DAY}/decision.json"]

    # Extract base a4e8278d to run its real installer & real CLI in a subprocess
    base_dir = work_dir / "base_src"
    base_dir.mkdir()
    tar_bytes = subprocess.run(
        ["git", "-C", str(REPO), "archive", BASE_REV, "src", "scripts"],
        check=True, capture_output=True, timeout=20,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        tf.extractall(base_dir, filter="data")

    b_box = make_box(work_dir / "base_box", base_dir / "scripts/cron-setup-hetzner.sh", base_dir / "src")
    b_box.config.write_text(ENTER_CFG)
    assert b_box.run("install").returncode == 0
    base_wd_line, = [ln for ln in b_box.lines() if ENTRY_CMD in ln and not ln.startswith("#")]
    assert base_wd_line.startswith("*/15 10-23 * * * ") and "--cron" not in base_wd_line

    # Run full-day closing-test replay in child processes against base src and candidate src
    helper_code = """
import gzip, json, os, shlex, socket, sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
from bts import entry_receipt
from bts.cli import cli

ET = ZoneInfo("America/New_York")
cfg_in = json.loads(Path(sys.argv[1]).read_text())
day = cfg_in["day"]
wd_line = cfg_in["wd_line"]
saved_pick = cfg_in["saved_pick"]
saved_dec = cfg_in["saved_dec"]
present = cfg_in["present"]
root = Path(cfg_in["root"])

for n in ("connect", "connect_ex", "sendto"):
    setattr(socket.socket, n, lambda *a, **k: (_ for _ in ()).throw(AssertionError("net")))
socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(AssertionError("net"))

pdir = root / "picks"
pdir.mkdir(parents=True)
cfg_path = root / "config.toml"
cfg_path.write_text(cfg_in["enter_cfg"])
udir = entry_receipt.units_snapshot_dir(pdir)
udir.mkdir(parents=True)
(udir / f"{day}T030000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [{"id": 77, "roundId": 1, "feedId": saved_pick["pick"]["game_pk"]}]}).encode(), mtime=0))

tokens = shlex.split(wd_line, comments=True)
args = tokens[tokens.index("check-pick-entered"):tokens.index(">>")]
args[args.index("--picks-dir") + 1] = str(pdir)
uname = args[args.index("--expected-username") + 1]
recip = args[args.index("--dm-recipient") + 1]

minute_f, hour_f, *_ = wd_line.split()
step = int(minute_f.removeprefix("*/"))
hours = range(24) if hour_f == "*" else range(10, 24)
mid = datetime.fromisoformat(day).replace(tzinfo=ET)
cur, stop = mid.astimezone(timezone.utc), (mid + timedelta(days=1)).astimezone(timezone.utc)
slot_list = []
while cur < stop:
    et = cur.astimezone(ET)
    if et.hour in hours:
        slot_list.append(et.strftime("%H:%M"))
    cur += timedelta(minutes=step)

runner = CliRunner()
reads, dms_by_slot, outcomes = [], {}, {}
row_item = [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None}]

with patch("bts.leaderboard.auth.load_session_cookies", lambda: {}), \\
     patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"), \\
     patch("bts.leaderboard.auth.fetch_login_session", lambda *a, **k: SimpleNamespace(user_id="u1", username=uname, xsid="x1")), \\
     patch("bts.contest_fetch.fetch_profile", lambda *a, **k: {"predictions": []}), \\
     patch("bts.cli._fetch_rounds", lambda: {1: date.fromisoformat(day)}), \\
     patch("bts.cli._fetch_bts_to_mlb", lambda: {1: saved_pick["pick"]["batter_id"]}), \\
     patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), \\
     patch("os.fsync", lambda fd: None), \\
     patch("time.monotonic", lambda: 1000.0):
    for s in slot_list:
        if s >= "05:10" and not (pdir / f"{day}.json").exists():
            (pdir / f"{day}.json").write_text(json.dumps(saved_pick))
            (pdir / day).mkdir(exist_ok=True)
            (pdir / day / "decision.json").write_text(json.dumps(saved_dec))
        slot_calls, slot_dms = [], []
        with patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: slot_calls.append("pending") or (row_item if present else [])), \\
             patch("bts.dm.send_dm", lambda r, m: slot_dms.append(m)):
            res = runner.invoke(cli, [*args, "--now-et", f"{day}T{s}:00"])
            assert res.exit_code in (0, 1), res.output
        if slot_calls:
            reads.append(s)
        if slot_dms:
            dms_by_slot[s] = slot_dms
        recs = entry_receipt.discover(pdir, day)
        rec_s, = [r for r in recs if r["started_at"].startswith(f"{day}T{s}:00")]
        outcomes[s] = rec_s["outcome"]

morning_dms = []
with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": saved_pick["pick"]["game_pk"]}]), \\
     patch("bts.dm.send_dm", lambda r, m: morning_dms.append(m)):
    m_res = runner.invoke(cli, ["day-outcome", "--date", day, "--config", str(cfg_path), "--picks-dir", str(pdir), "--expected-username", uname, "--dm-recipient", recip])

counts = {}
for v in outcomes.values():
    counts[v] = counts.get(v, 0) + 1

Path(sys.argv[2]).write_text(json.dumps({
    "slot_count": len(slot_list),
    "reads": reads,
    "dms_by_slot": dms_by_slot,
    "outcome_counts": counts,
    "sample_0500_to_0615": {k: outcomes[k] for k in ("05:00", "05:15", "05:30", "05:45", "06:00", "06:15") if k in outcomes},
    "morning_exit": m_res.exit_code,
    "morning_body": json.loads(m_res.stdout),
    "morning_dms": morning_dms,
}))
"""
    results = {}
    for label, src_p, wd_l, present in (
        ("base_entered", base_dir / "src", base_wd_line, True),
        ("f5_entered", REPO / "src", f5_wd_line, True),
        ("f5_missing", REPO / "src", f5_wd_line, False),
    ):
        rdir = work_dir / label
        rdir.mkdir()
        in_p = rdir / "in.json"
        out_p = rdir / "out.json"
        in_p.write_text(json.dumps({
            "day": DAY, "wd_line": wd_l, "saved_pick": saved_pick, "saved_dec": saved_dec,
            "present": present, "root": str(rdir / "state"), "enter_cfg": ENTER_CFG,
        }))
        env = {**os.environ, "PYTHONPATH": str(src_p), "TZ": "America/New_York", "TMPDIR": str(rdir)}
        proc = subprocess.run([sys.executable, "-c", helper_code, str(in_p), str(out_p)], env=env, capture_output=True, text=True, timeout=40)
        assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"
        results[label] = json.loads(out_p.read_text())

    # Base RED: 56 slots, 0 reads, morning exit 1 ENTRY UNVERIFIABLE (outside_window x56)
    assert results["base_entered"]["slot_count"] == 56
    assert results["base_entered"]["reads"] == []
    assert results["base_entered"]["outcome_counts"] == {"outside_window": 56}
    assert results["base_entered"]["morning_exit"] == 1
    assert results["base_entered"]["morning_body"]["state"] == "entry_unverifiable"
    assert "outside_window x56" in results["base_entered"]["morning_dms"][0]

    # F5 GREEN (entered): 96 slots, 05:00 no_pick_file, 05:15 observed/confirmed, 05:30-06:00 already_confirmed, 06:15 outside_window, morning exit 0
    assert results["f5_entered"]["slot_count"] == 96
    assert results["f5_entered"]["reads"] == ["05:15"]
    assert results["f5_entered"]["sample_0500_to_0615"] == {
        "05:00": "no_pick_file",
        "05:15": "observed",
        "05:30": "already_confirmed",
        "05:45": "already_confirmed",
        "06:00": "already_confirmed",
        "06:15": "outside_window",
    }
    assert results["f5_entered"]["outcome_counts"] == {
        "no_pick_file": 21,       # 00:00..05:00 inclusive = 21 slots
        "observed": 1,            # 05:15
        "already_confirmed": 3,   # 05:30, 05:45, 06:00
        "outside_window": 71,     # 06:15..23:45 = 71 slots
    }
    assert results["f5_entered"]["morning_exit"] == 0
    assert results["f5_entered"]["morning_body"]["state"] == "entered"
    assert results["f5_entered"]["morning_dms"] == []

    # F5 M2 missing: 05:15, 05:30, 05:45, 06:00 all read account; DMs at 05:15 (initial), 05:45 (t30), 06:00 (t15); morning NOT ENTERED
    assert results["f5_missing"]["reads"] == ["05:15", "05:30", "05:45", "06:00"]
    assert sorted(results["f5_missing"]["dms_by_slot"].keys()) == ["05:15", "05:45", "06:00"]
    assert results["f5_missing"]["morning_exit"] == 1
    assert results["f5_missing"]["morning_body"]["state"] == "not_entered"

    return results


def check_m3_m5_m6_m7_n6_n7_n8(work_dir: Path, f5_wd_line: str) -> dict:
    """Verify M3, M5, M6, M7, N6, N7, and N8 schedule/interval properties."""
    fixture_gz = REPO / "docs/reviews/2026-11-18-whole-range/closing-test/codex-blind/evidence/cron-early-entry.json.gz"
    with gzip.open(fixture_gz, "rt") as stream:
        snapshot = next(ev[-1] for ev in json.load(stream) if ev[1] == "snapshot")
    saved_pick = snapshot[f"data/picks/{DAY}.json"]
    saved_dec = snapshot[f"data/picks/{DAY}/decision.json"]

    tokens = shlex.split(f5_wd_line, comments=True)
    base_args = tokens[tokens.index("check-pick-entered"):tokens.index(">>")]
    runner = CliRunner()

    def make_env(name: str):
        rdir = work_dir / name
        pdir = rdir / "picks"
        pdir.mkdir(parents=True)
        cfg = rdir / "config.toml"
        cfg.write_text(ENTER_CFG)
        marker = rdir / "health_state" / "pick_entry_check.json"
        udir = entry_receipt.units_snapshot_dir(pdir)
        udir.mkdir(parents=True)
        args = list(base_args)
        args[args.index("--picks-dir") + 1] = str(pdir)
        uname = args[args.index("--expected-username") + 1]
        recip = args[args.index("--dm-recipient") + 1]

        def set_pick(pitch_hm: str, *, day: str = DAY, pitch_day: str | None = None,
                     deliver_hm: str = "04:00", committed: bool = True):
            first = datetime.fromisoformat(f"{pitch_day or day}T{pitch_hm}:00").replace(tzinfo=ET)
            commit = datetime.fromisoformat(f"{day}T{deliver_hm}:00").replace(tzinfo=ET).isoformat()
            pk = {
                **saved_pick, "date": day, "run_time": commit, "delivered_at": commit if committed else None,
                "pick": {**saved_pick["pick"], "game_pk": 101, "game_time": first.isoformat()},
            }
            if not committed:
                pk.update(notification_sent=False, notification_id=None, delivery_attempted=False)
            (pdir / f"{day}.json").write_text(json.dumps(pk))
            if committed:
                (pdir / day).mkdir(exist_ok=True)
                (pdir / day / "decision.json").write_text(json.dumps({**saved_dec, "date": day, "finalized_at": commit}))

        def add_units(u_dict: dict, stamp: str = "030000"):
            (udir / f"{DAY}T{stamp}Z.json.gz").write_bytes(
                gzip.compress(json.dumps({"units": [u_dict]}).encode(), mtime=0)
            )

        def run_slot(hm: str, *, day: str = DAY, present: bool = True):
            calls, dms = [], []
            before_m = marker.read_bytes() if marker.exists() else None
            before_ids = {r["attempt_id"] for r in entry_receipt.discover(pdir, day)}
            row_item = [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None}]
            with ExitStack() as stack:
                stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", lambda: calls.append("cookies") or {}))
                stack.enter_context(patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"))
                stack.enter_context(patch("bts.leaderboard.auth.fetch_login_session", lambda *a, **k: calls.append("session") or SimpleNamespace(user_id="u1", username=uname, xsid="x1")))
                stack.enter_context(patch("bts.contest_fetch.fetch_profile", lambda *a, **k: calls.append("profile") or {"predictions": []}))
                stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: calls.append("pending") or (row_item if present else [])))
                stack.enter_context(patch("bts.cli._fetch_rounds", lambda: calls.append("rounds") or {1: date.fromisoformat(day)}))
                stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", lambda: calls.append("xw") or {1: saved_pick["pick"]["batter_id"]}))
                stack.enter_context(patch("bts.dm.send_dm", lambda r, m: dms.append(m)))
                stack.enter_context(patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"))
                stack.enter_context(patch("os.fsync", lambda fd: None))
                stack.enter_context(patch("time.monotonic", lambda: 1000.0))
                res = runner.invoke(cli, [*args, "--now-et", f"{day}T{hm}:00"])
            assert res.exit_code in (0, 1), res.output
            after_m = marker.read_bytes() if marker.exists() else None
            new_recs = [r for r in entry_receipt.discover(pdir, day) if r["attempt_id"] not in before_ids]
            assert len(new_recs) == 1, len(new_recs)
            latest = new_recs[0]
            return latest["outcome"], calls, dms, before_m != after_m

        def run_morning(day: str = DAY):
            dms = []
            with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}]), \
                 patch("bts.dm.send_dm", lambda r, m: dms.append(m)):
                res = runner.invoke(cli, ["day-outcome", "--date", day, "--config", str(cfg), "--picks-dir", str(pdir), "--expected-username", uname, "--dm-recipient", recip])
            return res.exit_code, json.loads(res.stdout), dms

        return rdir, pdir, marker, set_pick, add_units, run_slot, run_morning

    # M3 missing + M7 confirmed for 10:05
    for present in (False, True):
        rdir, _, _, set_pick, add_units, run_slot, run_morning = make_env(f"m3_{present}")
        add_units(RIGHT_U)
        set_pick("10:05", deliver_hm="08:40")
        res_map = {hm: run_slot(hm, present=present) for hm in ("08:45", "09:00", "09:15", "09:30", "09:45", "10:00")}
        assert res_map["08:45"][0] == "outside_window" and not res_map["08:45"][1]
        assert res_map["10:00"][0] == "outside_window" and not res_map["10:00"][1]
        reads = [hm for hm, v in res_map.items() if "pending" in v[1]]
        if present:
            assert reads == ["09:00"]
            assert [res_map[hm][0] for hm in ("09:15", "09:30", "09:45")] == ["already_confirmed"] * 3
            assert run_morning()[0] == 0
        else:
            assert reads == ["09:00", "09:15", "09:30", "09:45"]
            assert run_morning()[0] == 1
        shutil.rmtree(rdir)

    # M5 (10:50 wrong-game at 10:00 after 09:45 batter match) & M6 (10:50 entry removed before 10:00)
    for wrong_game in (True, False):
        rdir, _, _, set_pick, add_units, run_slot, run_morning = make_env(f"m5_m6_{wrong_game}")
        set_pick("10:50", deliver_hm="09:30")
        if not wrong_game:
            add_units(RIGHT_U)
        trace = {}
        for hm in ("09:30", "09:45", "10:00", "10:15", "10:30", "10:45"):
            if hm == "10:00" and wrong_game:
                add_units(WRONG_U, "140000")
            trace[hm] = run_slot(hm, present=(hm in ("09:30", "09:45") or wrong_game))
        assert trace["09:30"] == ("outside_window", [], [], False)
        assert trace["09:45"] == ("base_slot_available", [], [], False)
        assert "pending" in trace["10:00"][1] and trace["10:00"][3] is True
        assert trace["10:45"] == ("outside_window", [], [], False)
        assert sum(len(v[2]) for v in trace.values()) == 3
        m_code, m_body, m_dms = run_morning()
        assert m_code == 1 and len(m_dms) == 1
        assert m_body["state"] == ("entry_unverifiable" if wrong_game else "not_entered")
        shutil.rmtree(rdir)

    # N6: 00:00, 03:00 (preview on disk), 05:00, 09:45 local-only gates + yesterday pick/marker
    rdir, _, marker, set_pick, add_units, run_slot, _ = make_env("n6")
    assert run_slot("00:00") == ("no_pick_file", [], [], False)
    set_pick("00:35", day=PREV_DAY, pitch_day=DAY, deliver_hm="23:00")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"date": PREV_DAY, "status": "confirmed", "receipt": "c" * 32}))
    assert run_slot("00:15") == ("no_pick_file", [], [], False)
    set_pick("06:10", committed=False, deliver_hm="03:00")
    for hm in ("00:00", "03:00", "05:00", "09:45"):
        assert run_slot(hm) == ("not_committed", [], [], False)
    set_pick("19:10", committed=True, deliver_hm="08:00")
    for hm in ("00:00", "03:00", "05:00", "09:45"):
        assert run_slot(hm) == ("outside_window", [], [], False)
    marker.write_text(json.dumps({"date": DAY, "status": "confirmed", "receipt": "d" * 32}))
    assert run_slot("18:00") == ("already_confirmed", [], [], False)
    shutil.rmtree(rdir)

    # N7: 19:10 evening game has exact same 5 slots (18:00..19:00)
    rdir, _, _, set_pick, add_units, run_slot, run_morning = make_env("n7")
    add_units(RIGHT_U)
    set_pick("19:10", deliver_hm="16:00")
    n7_trace = {hm: run_slot(hm, present=True) for hm in ("17:45", "18:00", "18:15", "18:30", "18:45", "19:00", "19:15")}
    assert n7_trace["17:45"] == ("outside_window", [], [], False)
    assert n7_trace["18:00"][0] == "observed" and "pending" in n7_trace["18:00"][1]
    for hm in ("18:15", "18:30", "18:45", "19:00"):
        assert n7_trace[hm] == ("already_confirmed", [], [], False)
    assert n7_trace["19:15"] == ("outside_window", [], [], False)
    assert run_morning()[0] == 0
    shutil.rmtree(rdir)

    # N8: wrapper inequality 840 + 30 < 900, and 900s elapsed spacing across normal, spring-forward, and fall-back days
    m = re.search(r"/usr/bin/flock -n /home/bts/\.check-pick-entered\.lock /usr/bin/timeout --verbose -k (\d+) (\d+) ", f5_wd_line)
    assert m and m.groups() == ("30", "840")
    kill_grace, term_limit = map(int, m.groups())
    assert term_limit + kill_grace == 870 < 900

    # Verify Debian cron wildcard launch instants across normal day, 2027-03-14 (spring), and 2027-11-07 (fall)
    dst_counts = {}
    for label, d_str, exp_len in (("ordinary", DAY, 96), ("spring_dst", "2027-03-14", 92), ("fall_dst", "2027-11-07", 100)):
        mid = datetime.fromisoformat(d_str).replace(tzinfo=ET)
        cur = mid.astimezone(timezone.utc)
        stop = (mid + timedelta(days=1)).astimezone(timezone.utc)
        utc_instants = []
        et_labels = []
        while cur < stop:
            et = cur.astimezone(ET)
            if et.minute % 15 == 0:
                utc_instants.append(cur)
                et_labels.append(et.strftime("%H:%M%z"))
            cur += timedelta(minutes=15)
        assert len(utc_instants) == exp_len, (label, len(utc_instants))
        gaps = [(b - a).total_seconds() for a, b in zip(utc_instants, utc_instants[1:])]
        assert set(gaps) == {900.0}, (label, set(gaps))
        dst_counts[label] = {"count": len(utc_instants), "min_gap_sec": min(gaps), "max_gap_sec": max(gaps)}

    return {"M3_M5_M6_M7_N6_N7": "PASS", "N8_wrapper_and_dst_gaps": dst_counts}


def main() -> None:
    t0 = time.perf_counter()
    _forbid_network()
    work = OUT_DIR / ".work_probe_03"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        inst = check_installer_rows_m1_m4_n1_to_n5(work / "inst")
        closing = replay_closing_test_base_vs_f5(work / "closing", inst["installed_watchdog_line"])
        rows = check_m3_m5_m6_m7_n6_n7_n8(work / "rows", inst["installed_watchdog_line"])
        elapsed = round(time.perf_counter() - t0, 3)
        report = {
            "status": "PASS",
            "elapsed_sec": elapsed,
            "installer_rows": inst,
            "closing_test_replay": closing,
            "remaining_rows": rows,
        }
        (OUT_DIR / "log_03_m_and_n_rows_end_to_end.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS probe_03 in {elapsed}s: closing-test base=UNVERIFIABLE(x56) -> F5=ENTERED(1/96), M1-M7 & N1-N8 green")
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()

"""Stop the real installed Click entrypoint at filesystem and transport boundaries."""
from contextlib import ExitStack
from datetime import date
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from bts import entry_receipt, receipt_io
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World, XW


STAGES = ("before-fetch", "during-fetch", "dm-before-marker", "failed-dm-before-marker",
          "marker-temp", "receipt-temp", "receipt-renamed", "receipt-sealed", "all-clear")


def child(stage, root):
    def pause(label):
        if stage == label:
            print(f"child {os.getpid()} at {label}", flush=True)
            (root / "ready").write_text(label)
            time.sleep(8)
            raise AssertionError("parent did not send signal")

    def send(_, message):
        if stage == "all-clear":
            pause("all-clear")
        if stage == "failed-dm-before-marker":
            raise RuntimeError("captured send failure")
        (root / "accepted-dm").write_text(message)

    from bts.cli import _atomic_write_json
    real_fdopen = os.fdopen
    real_replace = os.replace

    def atomic_write(path, value):
        pause("dm-before-marker")
        pause("failed-dm-before-marker")
        return _atomic_write_json(path, value)

    class PartialWrite:
        def __init__(self, stream, label):
            self.stream, self.label = stream, label

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:len(data) // 2])
            self.stream.flush()
            pause(self.label)

    def fdopen(fd, *args, **kwargs):
        stream = real_fdopen(fd, *args, **kwargs)
        return PartialWrite(stream, "marker-temp") if stage == "marker-temp" else stream

    def receipt_open(path, mode):
        stream = open(path, mode)
        return PartialWrite(stream, "receipt-temp") if stage == "receipt-temp" else stream

    def replace(src, dest):
        real_replace(src, dest)
        path = Path(dest)
        if "pick_entry_receipts" in path.parts:
            pause("receipt-sealed" if path.name.endswith(".sealed") else "receipt-renamed")

    session = SimpleNamespace(user_id="fixture", username="operator", xsid="fixture")
    leaves = {
        "bts.leaderboard.auth.load_session_cookies": lambda: pause("before-fetch") or {},
        "bts.leaderboard.auth.extract_uid": lambda _: "fixture",
        "bts.leaderboard.auth.fetch_login_session": lambda *_, **__: session,
        "bts.contest_fetch.fetch_profile": lambda *_: {"predictions": []},
        "bts.contest_fetch.fetch_pending_predictions": lambda *_: pause("during-fetch") or MATCH,
        "bts.cli._fetch_rounds": lambda: {1: date.fromisoformat(DAY)},
        "bts.cli._fetch_bts_to_mlb": lambda: XW,
        "bts.dm.send_dm": send,
        "bts.picks._git_head_sha": lambda *_: "fixture-revision",
        "bts.cli._atomic_write_json": atomic_write,
        "os.fdopen": fdopen,
        "os.replace": replace,
    }
    cli = REPO / ".venv/bin/bts"
    sys.argv = [str(cli), "check-pick-entered", "--picks-dir", str(root / "picks"), "--now-et", f"{DAY}T18:30:00",
                "--expected-username", "operator", "--dm-recipient", "captured.invalid"]
    with ExitStack() as stack:
        for address, fn in leaves.items():
            stack.enter_context(patch(address, fn))
        stack.enter_context(patch.object(receipt_io, "open", receipt_open, create=True))
        runpy.run_path(str(cli), run_name="__main__")


def check(stage, root):
    w = World(root)
    w.commit(pitch=PITCH)
    w.capture([RIGHT if stage == "all-clear" else WRONG])
    if stage != "failed-dm-before-marker":
        assert w.check("18:25")["marker"]["escalations"] == ["initial"]
    before = w.marker.read_bytes() if w.marker.exists() else None
    before_receipts = entry_receipt.discover(w.picks, DAY)
    sig = signal.SIGKILL if stage in ("marker-temp", "receipt-renamed") else signal.SIGTERM
    proc = subprocess.Popen([sys.executable, __file__, stage, str(root)], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        end = time.monotonic() + 4
        while not (root / "ready").exists() and proc.poll() is None and time.monotonic() < end:
            time.sleep(0.01)
        assert (root / "ready").read_text() == stage
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=2)
        assert proc.returncode == -sig, (stage, proc.returncode, out, err)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=1)

    marker = w.read_marker()
    after_receipts = entry_receipt.discover(w.picks, DAY)
    marker_temps = sorted(p.name for p in w.marker.parent.glob("*.tmp"))
    receipt_dir = entry_receipt.receipts_root(w.picks) / DAY
    orphan_receipts = sorted(p.name for p in receipt_dir.glob("*.json")
                             if not p.with_name(p.name + ".sealed").exists())
    receipt_temps = sorted(p.name for p in receipt_dir.glob("*.tmp"))
    accepted = (root / "accepted-dm").exists()
    if stage in STAGES[:5]:
        assert (w.marker.read_bytes() if w.marker.exists() else None) == before, (stage, marker)
    else:
        assert marker["status"] == ("confirmed" if stage == "all-clear" else "alerted"), marker
        if stage != "all-clear":
            assert marker["escalations"] == ["initial", "wrong_game"], marker
    assert len(after_receipts) == len(before_receipts) + (stage == "receipt-sealed"), stage
    assert bool(marker_temps) == (stage == "marker-temp"), (stage, marker_temps)
    assert bool(receipt_temps) == (stage == "receipt-temp"), (stage, receipt_temps)
    assert bool(orphan_receipts) == (stage == "receipt-renamed"), (stage, orphan_receipts)
    assert accepted == (stage in ("dm-before-marker", "marker-temp", "receipt-temp", "receipt-renamed", "receipt-sealed"))
    health = [a["level"] for a in w.health()]
    morning = w.morning()["outcome"]["state"]
    result = w.check("18:45", MATCH, dm="failure" if stage == "failed-dm-before-marker" else "success")
    if stage == "all-clear":
        assert result["fetches"] == [] and result["dm"] == []
        assert result["receipts"][-1]["references"]["confirmed_by"] == marker["receipt"]
        assert all(r["attempt_id"] != marker["receipt"] for r in result["receipts"])
    elif stage == "failed-dm-before-marker":
        assert result["marker"]["status"] == "dm_failed" and result["marker"]["escalations"] == []
        retried = w.check("19:00", MATCH)
        assert retried["marker"]["escalations"] == ["wrong_game"] and retried["dm"]
    else:
        assert bool(result["dm"]) == (stage in STAGES[:5]), (stage, result)
    print(json.dumps(dict(stage=stage, pid=proc.pid, sig=sig.name, output=out, stderr=err,
                          marker=marker, marker_temps=marker_temps, receipt_temps=receipt_temps,
                          orphan_receipts=orphan_receipts, accepted_dm=accepted, health=health, morning=morning,
                          next_dm=result["dm"], next_marker=result["marker"]), ensure_ascii=False), flush=True)


def serial_loss(root):
    w = World(root)
    w.commit(pitch=PITCH)
    assert w.check("18:25")["marker"]["escalations"] == ["initial"]
    w.capture([WRONG])
    proc = subprocess.Popen([sys.executable, __file__, "failed-dm-before-marker", str(root)], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        end = time.monotonic() + 4
        while not (root / "ready").exists() and proc.poll() is None and time.monotonic() < end:
            time.sleep(0.01)
        assert (root / "ready").is_file()
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=2)
        assert proc.returncode == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=1)
    w.capture([RIGHT], f"{DAY}T230000Z")
    result = w.check("18:45", MATCH)
    assert result["marker"]["status"] == "confirmed" and result["dm"][0].startswith("✅")
    assert result["receipts"][-1]["qualification"]["slots"][0]["state"] == "unit_unverified"
    print("serial observation loss", proc.pid, out, err, result["marker"], result["dm"], w.morning(), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1], Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            for stage in STAGES:
                check(stage, Path(root) / stage)
            serial_loss(Path(root) / "serial-loss")

"""Exercise sealed early receipts, the final pick read and ordering in the real morning CLI."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from replay_support import DAY, ET, RIGHT, SCRATCH, World, compact, deadline, row, source
import morning_fifo
import state_replay
from bts import day_outcome, entry_receipt, picks


def report(w):
    with deadline():
        result = w.morning()
    print(w.root.name, result)
    return result[0], result[1]["state"]


def fifo(root):
    w = World(root / "R10")
    w.commit_at("10:05", at="09:40")
    early = w.cron("09:45")["receipt"]
    assert early["early_cron_run"] is True and early["verifier"]["reason"] == "no_pick"
    assert report(w) == (1, "not_entered")
    path = w.picks / f"{DAY}.json"
    path.unlink()
    os.mkfifo(path)
    assert report(w) == (1, "entry_unverifiable")
    child = subprocess.run([sys.executable, __file__, str(w.root)], cwd=SCRATCH / "base",
                           env=os.environ | {"PYTHONPATH": "src:."}, text=True, capture_output=True, timeout=4)
    assert child.returncode == 0, (child.stdout, child.stderr)
    print("same sealed input on real base:", child.stdout)


def windows(root):
    w = World(root / "boundaries")
    final = w.commit_at("10:05", at="08:00")
    w.capture([RIGHT], f"{DAY}T135000Z")
    assert w.cron("09:45", pending=[row()])["receipt"]["early_cron_run"] is True
    starts = [datetime.fromisoformat(f"{DAY}T10:00").replace(tzinfo=ET) + timedelta(minutes=15 * i)
              for i in range(56)]
    for pitch, has, literal in (
        ("06:10", False, None), ("10:04:59", False, None), ("10:05", False, None),
        ("10:05:01", True, None), ("10:06", True, None), ("10:20", True, None),
        ("11:15", True, None), ("11:16", True, None), ("19:10", True, None),
        ("23:50", True, None), ("00:31", True, "2027-03-25T00:31-04:00"),
        ("01:00", True, "2027-03-25T01:00-04:00"), ("01:00:01", False, "2027-03-25T01:00:01-04:00"),
        ("10:05", False, f"{DAY}T14:05:00"), ("10:06", True, f"{DAY}T14:06:00Z"),
    ):
        final.pick.game_time = literal or f"{DAY}T{pitch}-04:00"
        picks.save_pick(final, w.picks)
        lead = picks.earliest_pick_game_et(final)
        eligible = any(picks.SUBMISSION_CUTOFF_MIN < (lead - at).total_seconds() / 60 <= 75 for at in starts)
        assert eligible == has and day_outcome._allow_early_evidence(DAY, w.picks) == (not has), pitch
        if lead.date().isoformat() == DAY and "08:50" <= pitch <= "10:55":
            at_early = w.invoke("09:45", auth="login", persist=False)
            assert ("base_slot_available" in at_early["stdout"] or "early check deferred" in at_early["stdout"]) == has
        assert report(w) == ((1, "entry_unverifiable") if has else (0, "entered"))

    final.pick.game_time = f"{DAY}T19:00-04:00"
    final.double_down = w.commit(dd=(1020, 202, f"{DAY}T10:05-04:00"), delivery=None).double_down
    picks.save_pick(final, w.picks)
    assert report(w) == (0, "entered")
    w.edit_pitch(final, "10:06", "double_down")
    assert report(w) == (1, "entry_unverifiable")

    path = w.picks / f"{DAY}.json"
    saved = path.read_bytes()
    for label, content in (
        ("invalid utf8", b'"\xff"'), ("list", b"[]"), ("bad JSON", b"{"),
        ("null", b"null"), ("missing slot", b'{"pick": {}}'), ("oversize", b" " * 65537), ("absent", None),
    ):
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
        assert report(w) == (1, "entry_unverifiable"), label
    path.mkdir()
    assert report(w) == (1, "entry_unverifiable")
    path.rmdir()
    path.write_bytes(saved)
    path_data = json.loads(saved)
    path_data["double_down"]["game_time"] = "not a pitch"
    path.write_text(json.dumps(path_data))
    assert report(w) == (1, "entry_unverifiable")


def chronology(root, version):
    state_replay.slot_gained_then_lost(root / "gained-then-lost", version)
    state_replay.moving_double(root / "moving-double", version)
    w = World(root / "manual-removal", version)
    w.commit_at("10:05", at="09:40")
    w.capture([RIGHT], f"{DAY}T134000Z")
    w.cron("09:45", pending=[row()])
    manual = w.invoke("09:50", manual=True)
    assert manual["sent"] and manual["receipt"]["verifier"]["reason"] == "no_pick"
    assert report(w) == ((1, "not_entered") if version == "base" else (0, "entered"))
    print("manual-removal live receipt:", compact(manual))

    if version == "f5":
        w = World(root / "late", version)
        saved = w.commit_at("10:00:30", at="09:40")
        w.capture([RIGHT], f"{DAY}T134000Z")
        r = w.cron("09:45", pending=[row()], elapsed=650)
        assert r["receipt"]["qualification"]["all_confirmed"] and not r["receipt"]["observation"]["before_cutoff"]
        assert report(w) == (1, "entered_late")
        w.edit_pitch(saved, "10:50")
        assert report(w) == (1, "entry_unverifiable")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode.startswith("/"):
        with patch.object(morning_fifo, "source", source):
            morning_fifo.read(Path(mode), "base")
    else:
        source(mode)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            if mode == "f5":
                fifo(root)
                windows(root)
            chronology(root, mode)

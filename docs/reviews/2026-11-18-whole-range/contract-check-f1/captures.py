"""Probe malformed captures, bounded blocking, interrupts and date boundaries."""
from datetime import date
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import entry_receipt
from harness import RIGHT, WRONG, World, row

HERE = Path(__file__).resolve().parent


def bad_capture(w, raw):
    path = w.units / f"{w.day}T210000Z.json.gz"
    path.write_bytes(raw)
    return path


def archive(obj):
    return gzip.compress(json.dumps(obj).encode(), mtime=0)


def run(root):
    results = {}
    malformed = {
        "plain": b"abc",
        "bad_json": gzip.compress(b"{"),
        "bad_unicode": gzip.compress(b"\xff"),
        "truncated_gzip": archive({"units": [RIGHT]})[:15],
        "top_list": archive([]),
        "top_null": archive(None),
        "units_absent": archive({}),
        "units_null": archive({"units": None}),
        "units_false": archive({"units": False}),
        "units_int": archive({"units": 1}),
        "units_map": archive({"units": {"id": 77}}),
        "units_text": archive({"units": "x"}),
        "units_nonobjects": archive({"units": [False, None, 42, "x", []]}),
        "units_mistyped": archive({"units": [{"id": "77", "feedId": 102, "roundId": 1}]}),
        "units_mix_valid_invalid": archive({"units": [WRONG, None, {"id": "77", "feedId": 101, "roundId": 1}]}),
    }
    for name, raw in malformed.items():
        w = World(root / name)
        w.commit()
        bad_capture(w, raw)
        results[name] = {"run": w.check("18:30", [row()]), "morning": w.morning()}

    for name in ("directory", "unreadable", "large", "symlink", "dangling_symlink"):
        w = World(root / name)
        w.commit()
        path = w.units / f"{w.day}T210000Z.json.gz"
        if name == "directory":
            path.mkdir()
        elif name == "unreadable":
            path = w.capture([WRONG])
            path.chmod(0)
        elif name == "symlink":
            target = w.root / "outside-inventory.json.gz"
            target.write_bytes(archive({"units": [WRONG]}))
            path.symlink_to(target)
        elif name == "dangling_symlink":
            path.symlink_to(w.root / "missing.json.gz")
        else:
            path.write_bytes(archive({"units": [WRONG], "padding": " " * (48 * 1024**2)}))
        results[name] = {"file_bytes": path.stat().st_size if path.exists() else None,
                         "run": w.check("18:30", [row()]), "morning": w.morning()}
        if name == "unreadable":
            path.chmod(0o600)

    for name, fault, times in (("interrupt_once", KeyboardInterrupt, 1), ("exit_once", SystemExit, 1),
                               ("interrupt_persistent", KeyboardInterrupt, 3),
                               ("memory", MemoryError, 3)):
        w = World(root / name)
        w.commit()
        w.capture([WRONG])
        loads, real_load = [], entry_receipt.load_units

        def inject(*args):
            loads.append(True)
            if len(loads) <= times:
                raise fault(17) if fault is SystemExit else fault()
            return real_load(*args)

        results[name] = {"run": w.check("18:30", [row()], more_patches=[("bts.entry_receipt.load_units", inject)]),
                         "loads": len(loads), "morning": w.morning()}

    for name, initial, later in (("wrong_then_right", WRONG, RIGHT), ("right_then_wrong", RIGHT, WRONG),
                                 ("wrong_then_unreadable", WRONG, None)):
        w = World(root / name)
        w.commit()
        loads, real_load = [], entry_receipt.load_units
        path = w.capture([initial])

        def shift(*args):
            loads.append(True)
            result = real_load(*args)
            if len(loads) == 1:
                path.write_bytes(archive({"units": [later]}) if later else b"broken after first read")
            return result

        results[name] = {"run": w.check("18:30", [row()], more_patches=[("bts.entry_receipt.load_units", shift)]),
                         "loads": len(loads), "morning": w.morning()}

    for label, day, next_day, offset in (("midnight", "2027-07-04", "2027-07-05", "-04:00"),
                                        ("year", "2027-12-31", "2028-01-01", "-05:00")):
        first, second = World(root / label, day), World(root / label, next_day)
        first.commit(pitch=f"{next_day}T00:45:00{offset}")
        second.commit(pitch=f"{next_day}T00:50:00{offset}")
        first.capture([WRONG], day.replace("-", "") + "T235900Z")
        second.capture([RIGHT], next_day.replace("-", "") + "T045000Z")
        kwargs = {"rounds": {1: date.fromisoformat(day), 2: date.fromisoformat(next_day)}}
        r1 = first.check("23:50", [row()], **kwargs)
        r2 = first.check("23:59", [row()], elapsed=120, **kwargs)  # response arrives after midnight ET
        r3 = second.check("00:05", [row(), row(rnd=2)], **kwargs)
        results[label] = [r1, r2, r3]

    for pending in ("match", "no_pick"):
        for mode in ("fifo", "link_fifo"):
            w = World(root / f"{mode}-{pending}")
            w.commit()
            path = w.units / f"{w.day}T210000Z.json.gz"
            if mode == "link_fifo":
                target = w.root / "pipe"
                os.mkfifo(target)
                path.symlink_to(target)
            else:
                os.mkfifo(path)
            child = subprocess.run(["nice", "-n", "19", "timeout", "2s", sys.executable, __file__,
                                    "fifo-child", str(w.root), pending], timeout=3, capture_output=True, text=True)
            results[f"{mode}-{pending}"] = {"exit": child.returncode, "marker": w.read_marker(),
                                            "dm_attempted": (w.root / "send-attempt").exists()}
    return results


if len(sys.argv) > 1:
    w = World(Path(sys.argv[2]))
    def record_send(*args):
        (w.root / "send-attempt").touch()
    w.check("18:30", [row()] if sys.argv[3] == "match" else [],
            more_patches=[("bts.dm.send_dm", record_send)])
else:
    with TemporaryDirectory(dir=HERE) as scratch:
        print(json.dumps(run(Path(scratch)), indent=2))

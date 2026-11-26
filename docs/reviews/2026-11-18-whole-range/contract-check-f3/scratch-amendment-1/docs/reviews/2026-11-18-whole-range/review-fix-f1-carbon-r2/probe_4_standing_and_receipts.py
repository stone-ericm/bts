"""Items 6 and 7: Standing hostile-input rows, FIFO/interrupts/year-boundary, receipts, and day-outcome."""
from __future__ import annotations

from datetime import date
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from bts import entry_receipt
from probe_common import (
    DAY,
    HERE,
    PITCH,
    SandboxWorld,
    UNIT_RIGHT_1,
    UNIT_WRONG_1,
    bts_source_path,
    make_row,
)

BASE_SRC = HERE / "base_archive" / "src"


def gz_json(obj) -> bytes:
    return gzip.compress(json.dumps(obj).encode(), mtime=0)


def fifo_worker(root_str: str, mode: str):
    w = SandboxWorld(Path(root_str))
    rows = [make_row(77, 1, 1)] if mode == "match" else []
    w.check("18:30", rows)


def marker_error_worker(root_str: str, raw_hex: str | None, is_dir: bool):
    w = SandboxWorld(Path(root_str))
    w.commit()
    w.capture([UNIT_RIGHT_1])
    w.marker.parent.mkdir(parents=True, exist_ok=True)
    if is_dir:
        w.marker.mkdir()
    elif raw_hex is not None:
        w.marker.write_bytes(bytes.fromhex(raw_hex))
    res = w.check("18:30", [])
    print(json.dumps({"exit": res["exit"], "exception": res["exception"]}))


def run_standing_and_receipts():
    report = {}
    with tempfile.TemporaryDirectory(prefix="f1_standing_") as tmp:
        root = Path(tmp)

        # 1. Malformed & non-regular captures: direct load_units vs CLI (Amendment 2 & Amendment 3)
        capture_cases = [
            # Raising shapes -> match_game_unchecked + 2 stderr lines + no receipt
            ("plain_bytes", b"not gzip at all", "BadGzipFile"),
            ("truncated_gz", gz_json({"units": [UNIT_WRONG_1]})[:12], "EOFError"),
            ("bad_utf8", gzip.compress(b"\xff\xfe"), "UnicodeDecodeError"),
            ("bad_json", gzip.compress(b"{broken"), "JSONDecodeError"),
            ("top_list", gz_json([UNIT_WRONG_1]), "AttributeError"),
            ("top_null", gz_json(None), "AttributeError"),
            ("top_int", gz_json(42), "AttributeError"),
            ("top_str", gz_json("hello"), "AttributeError"),
            ("top_bool", gz_json(True), "AttributeError"),
            ("units_int", gz_json({"units": 99}), "TypeError"),
            ("units_bool_true", gz_json({"units": True}), "TypeError"),
            ("units_float", gz_json({"units": 1.5}), "TypeError"),
            ("is_directory", "DIR", "IsADirectoryError"),
            ("unreadable_perm", "CHMOD000", "PermissionError"),
            # Silent shapes -> unchanged load_units accepts, unit_unverified, base match confirmation, no stderr
            ("empty_obj", gz_json({}), None),
            ("units_null", gz_json({"units": None}), None),
            ("units_false", gz_json({"units": False}), None),
            ("units_empty_str", gz_json({"units": ""}), None),
            ("units_nonempty_str", gz_json({"units": "xyz"}), None),
            ("units_dict_map", gz_json({"units": {"id": 77, "feedId": 102, "roundId": 1}}), None),
            ("units_bad_items", gz_json({"units": [None, False, 0, "x", [], {"id": "77", "feedId": 102, "roundId": 1}, {"id": True, "feedId": 102, "roundId": 1}]}), None),
            ("unit_bool_feed", gz_json({"units": [{"id": 77, "feedId": True, "roundId": 1}]}), None),
            ("unit_str_feed", gz_json({"units": [{"id": 77, "feedId": "102", "roundId": 1}]}), None),
            # Mixed valid wrong_game + malformed items -> still catches wrong_game!
            ("mixed_with_wrong", gz_json({"units": [None, {"id": True, "feedId": 101, "roundId": 1}, UNIT_WRONG_1]}), "WRONG_GAME"),
        ]

        checked_shapes = 0
        for name, spec, expected in capture_cases:
            w = SandboxWorld(root / f"cap_{name}")
            w.commit()
            cap_file = w.units_dir / f"{DAY}T210000Z.json.gz"
            if spec == "DIR":
                cap_file.mkdir()
            elif spec == "CHMOD000":
                cap_file.write_bytes(gz_json({"units": [UNIT_WRONG_1]}))
                cap_file.chmod(0)
            else:
                cap_file.write_bytes(spec)

            try:
                # Direct reader check
                raised_type = None
                try:
                    entry_receipt.load_units(w.picks, 2027)
                except Exception as exc:
                    raised_type = type(exc).__name__

                res = w.check("18:45", [make_row(77, 1, 1)])
                if expected == "WRONG_GAME":
                    assert raised_type is None
                    assert res["exit"] == 1 and res["marker"]["reason"] == "wrong_game"
                elif expected is not None:
                    assert raised_type == expected, (name, raised_type, expected)
                    assert res["exit"] == 0
                    assert res["marker"]["status"] == "confirmed" and res["marker"]["reason"] == "match_game_unchecked"
                    assert res["stderr"] == (
                        f"check-pick-entered: game check unavailable ({expected}); batter match only\n"
                        f"check-pick-entered: entry receipt unavailable ({expected})\n"
                    )
                    assert res["receipts"] == []
                else:
                    assert raised_type is None, (name, raised_type)
                    assert res["exit"] == 0
                    assert res["marker"]["status"] == "confirmed" and res["marker"]["reason"] == "match"
                    assert res["stderr"] == ""
                    assert res["receipts"][-1]["qualification"]["slots"][0]["state"] == "unit_unverified"
            finally:
                if spec == "CHMOD000":
                    cap_file.chmod(0o644)
            checked_shapes += 1
        report["capture_shapes_checked"] = checked_shapes

        # 2. Named pipe (FIFO) at capture path: blocks before marker on match, after marker on no_pick (bounded 2s each)
        for mode in ("no_pick", "match"):
            w_fifo = SandboxWorld(root / f"fifo_{mode}")
            w_fifo.commit()
            fifo_path = w_fifo.units_dir / f"{DAY}T210000Z.json.gz"
            os.mkfifo(fifo_path)
            timed_out = False
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--fifo-worker", str(w_fifo.root), mode],
                    env={**os.environ, "PYTHONPATH": str(HERE.parents[1] / "src") + ":" + str(HERE), "TZ": "America/New_York"},
                    capture_output=True,
                    timeout=2.0,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
            assert timed_out, f"FIFO did not block as expected in {mode}"
            m_after = w_fifo.read_marker()
            if mode == "match":
                assert m_after is None
            else:
                assert m_after is not None and m_after["status"] == "alerted" and m_after["reason"] == "no_pick"
        report["fifo_both_paths"] = "verified"

        # 3. Interrupts during new read (publish succeeds vs publish also fails) and between DM and marker write
        for exc_cls, code_arg in ((KeyboardInterrupt, ()), (SystemExit, (17,))):
            # Case A: interrupt once in qualify_run, publish succeeds
            w = SandboxWorld(root / f"int_once_{exc_cls.__name__}")
            w.commit()
            w.capture([UNIT_WRONG_1])
            orig_lu = entry_receipt.load_units
            calls = []

            def raise_first(*a):
                calls.append(1)
                if len(calls) == 1:
                    raise exc_cls(*code_arg)
                return orig_lu(*a)

            ri = w.check("18:45", [make_row(77, 1, 1)], extra_patches=[("bts.entry_receipt.load_units", raise_first)])
            assert ri["marker"] is None and ri["attempted_dm"] == [] and len(ri["receipts"]) == 1
            rec = ri["receipts"][0]
            if exc_cls is KeyboardInterrupt:
                assert rec["detail"] == "raised KeyboardInterrupt" and rec["exit_code"] is None
            else:
                assert rec["exit_code"] == 17 and rec["detail"] is None

            # Case B: interrupt in qualify_run AND failure in publish -> no marker, no DM, no sealed receipt
            w2 = SandboxWorld(root / f"int_both_{exc_cls.__name__}")
            w2.commit()
            w2.capture([UNIT_WRONG_1])
            calls2 = []

            def raise_then_corrupt(*a):
                calls2.append(1)
                if len(calls2) == 1:
                    raise exc_cls(*code_arg)
                raise RuntimeError("publish read also failed")

            ri2 = w2.check("18:45", [make_row(77, 1, 1)], extra_patches=[("bts.entry_receipt.load_units", raise_then_corrupt)])
            assert ri2["marker"] is None and ri2["attempted_dm"] == [] and ri2["receipts"] == []

        # DM raises AFTER transport accepted it -> repeats on next run (same as base)
        w_dup = SandboxWorld(root / "dm_ack_lost")
        w_dup.commit()
        w_dup.capture([UNIT_WRONG_1])
        delivered_wire = []

        def send_then_drop_ack(rec_h, msg):
            delivered_wire.append(msg)
            raise ConnectionResetError("connection closed after POST accepted")

        r_drop = w_dup.check("18:45", [make_row(77, 1, 1)], extra_patches=[("bts.dm.send_dm", send_then_drop_ack)])
        assert r_drop["marker"]["status"] == "dm_failed" and len(delivered_wire) == 1
        r_rep = w_dup.check("18:46", [make_row(77, 1, 1)])
        assert r_rep["dm"] == delivered_wire and r_rep["marker"]["escalations"] == ["wrong_game"]
        report["interrupts_and_lost_ack"] = "verified"

        # 4. Inherited marker errors (A5.4): compare base and candidate on [], null, invalid UTF-8, bad escalations, directory
        marker_faults = [
            ("json_list", b"[]", False),
            ("json_null", b"null", False),
            ("bad_utf8", b"\xff", False),
            ("bad_escalations", json.dumps({"date": DAY, "status": "alerted", "escalations": 5}).encode(), False),
            ("marker_is_dir", None, True),
        ]
        for f_name, raw_b, is_dir in marker_faults:
            pair = {}
            for side_lbl, s_dir in (("base", BASE_SRC), ("cand", HERE.parents[1] / "src")):
                sub_root = root / f"mf_{f_name}_{side_lbl}"
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--marker-worker",
                        str(sub_root),
                        raw_b.hex() if raw_b is not None else "NONE",
                        "1" if is_dir else "0",
                    ],
                    env={**os.environ, "PYTHONPATH": str(s_dir) + ":" + str(HERE), "TZ": "America/New_York"},
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=True,
                )
                pair[side_lbl] = json.loads(proc.stdout)
            assert pair["base"] == pair["cand"], (f_name, pair)
        report["inherited_marker_faults_parity"] = "verified"

        # 5. Midnight ET & Dec 31 UTC-filename year boundary (A6.6)
        nye_root = root / "nye_utc_gap"
        w_dec31 = SandboxWorld(nye_root, "2027-12-31")
        w_jan01 = SandboxWorld(nye_root, "2028-01-01")
        w_dec31.commit(pitch="2028-01-01T00:30:00-05:00")
        w_jan01.commit(pitch="2028-01-01T00:30:00-05:00")
        # Capture taken at 2027-12-31 19:15 ET == 2028-01-01T001500Z UTC (2028 filename prefix!)
        w_dec31.capture(
            [{"id": 77, "roundId": 1, "feedId": 102}, {"id": 88, "roundId": 2, "feedId": 102}],
            "20280101T001500Z",
        )
        rounds_nye = {1: date(2027, 12, 31), 2: date(2028, 1, 1)}
        # On Dec 31 ET (season 2027), 2028*.json.gz is NOT matched -> unit_unverified -> confirms (limit A6.6!)
        r_nye = w_dec31.check("23:50", [make_row(77, 1, 1, rnd=1)], rounds=rounds_nye)
        assert r_nye["exit"] == 0 and r_nye["marker"]["status"] == "confirmed"
        assert r_nye["receipts"][-1]["observation"]["sources"]["units"] is None
        # On Jan 1 ET (season 2028), 2028*.json.gz IS matched -> detects wrong_game!
        r_nyd = w_jan01.check("00:05", [make_row(88, 1, 1, rnd=2)], rounds=rounds_nye)
        assert r_nyd["exit"] == 1 and r_nyd["marker"]["reason"] == "wrong_game"
        report["dec31_utc_year_boundary_a6_6"] = "verified"

        # 6. Receipt schema & separate-read confirmation-with-unverifiable-morning + all day-outcome end states (Item 7)
        # Separate read: qualify_run sees unbound capture -> confirms; before publish() a wrong-game capture lands -> morning entry_unverifiable!
        w_sep = SandboxWorld(root / "sep_confirm_unverifiable")
        w_sep.commit()
        w_sep.capture([{"id": 77, "roundId": 1, "feedId": None}], f"{DAY}T200000Z")
        orig_lu2 = entry_receipt.load_units
        sep_calls = []

        def swap_to_wrong_for_publish(*a):
            sep_calls.append(1)
            out = orig_lu2(*a)
            if len(sep_calls) == 1:
                w_sep.capture([UNIT_WRONG_1], f"{DAY}T200000Z")  # overwrite before publish()!
            return out

        r_sep = w_sep.check("18:45", [make_row(77, 1, 1)], extra_patches=[("bts.entry_receipt.load_units", swap_to_wrong_for_publish)])
        assert r_sep["exit"] == 0 and r_sep["marker"]["status"] == "confirmed"
        assert r_sep["receipts"][-1]["qualification"]["slots"][0]["state"] == "wrong_game"
        assert w_sep.morning()["stdout"]["state"] == "entry_unverifiable"

        # Late game proof -> morning entered_late
        w_late = SandboxWorld(root / "morning_late")
        w_late.commit()
        w_late.capture([UNIT_RIGHT_1])
        r_late = w_late.check("19:34", [make_row(77, 1, 1)], elapsed=61)  # responses_done after 19:35 cutoff!
        assert r_late["exit"] == 0 and r_late["receipts"][-1]["observation"]["before_cutoff"] is False
        assert w_late.morning()["stdout"]["state"] == "entered_late"

        # Deliberate skip and no_games in day-outcome
        w_skip = SandboxWorld(root / "morning_skip")
        w_skip.commit(delivery="not_applicable")
        assert w_skip.morning()["stdout"]["state"] == "deliberate_skip"
        w_nogames = SandboxWorld(root / "morning_nogames")
        assert w_nogames.morning(games=[])["stdout"]["state"] == "no_games"
        report["receipt_and_day_outcome_states"] = "verified"

    (HERE / "log_4_standing_and_receipts.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if "--fifo-worker" in sys.argv:
        idx = sys.argv.index("--fifo-worker")
        fifo_worker(sys.argv[idx + 1], sys.argv[idx + 2])
    elif "--marker-worker" in sys.argv:
        idx = sys.argv.index("--marker-worker")
        h = sys.argv[idx + 2]
        marker_error_worker(sys.argv[idx + 1], None if h == "NONE" else h, sys.argv[idx + 3] == "1")
    else:
        run_standing_and_receipts()

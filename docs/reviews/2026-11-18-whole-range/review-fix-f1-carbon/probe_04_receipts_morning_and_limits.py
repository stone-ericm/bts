"""Probe 04 (Brief Item 7):
- Receipt field-by-field comparison against base (1e69d0f0) across wrong-game, follow-up, and capture-shift runs
- bts day-outcome across every end state (entered, entered_game_unverified, entered_late, not_entered, entry_unverifiable)
- Direct empirical tests of known limits A6.4 (rollback to base) and A6.5 (bts day-outcome priority & M9/mismatch/non-unverified cases)
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_SRC = HERE / "base_pkg" / "src"
HEAD_SRC = REPO_ROOT / "src"


def run_side_worker(mode: str, work_dir: Path, out_file: Path):
    from bts import entry_receipt
    from review_harness import (
        DEFAULT_DAY,
        UNIT_RIGHT_1,
        UNIT_WRONG_1,
        ReviewWorld,
        make_row,
    )

    if mode == "receipt_compare":
        res = {}
        for dm_mode in ("success", "failure"):
            w = ReviewWorld(work_dir / f"rc_{dm_mode}")
            w.write_pick_and_decision()
            w.write_capture([UNIT_WRONG_1])
            r1 = w.run_watchdog("18:30", [make_row()], dm_mode=dm_mode)
            r2 = w.run_watchdog("18:45", [make_row()], dm_mode="success")
            res[dm_mode] = {
                "run1_receipt": r1["receipts"][0],
                "run2_receipts": r2["receipts"],
                "run1_exit": r1["exit_code"],
                "run2_exit": r2["exit_code"],
            }
        out_file.write_text(json.dumps(res))
        return

    if mode == "m9_compare":
        res = {}
        for label, pre_no_pick, dm_mode in [
            ("m9_direct", False, "success"),
            ("m9_after_no_pick", True, "success"),
            ("m9_dm_failed", False, "failure"),
        ]:
            w = ReviewWorld(work_dir / label)
            w.write_pick_and_decision()
            w.write_capture([UNIT_WRONG_1], "20270324T200000Z")
            if pre_no_pick:
                w.run_watchdog("18:26", [], dm_mode=dm_mode)
            r1 = w.run_watchdog("18:30", [make_row()], dm_mode=dm_mode)
            # Conflicting capture arrives before 19:05 -> unit_unverified
            w.write_capture([UNIT_RIGHT_1], "20270324T210000Z")
            r2 = w.run_watchdog("19:05", [make_row()], dm_mode=dm_mode)
            morn = w.run_day_outcome()
            hlth = w.run_health_pick_entry()
            res[label] = {
                "r1_exit": r1["exit_code"],
                "r1_marker": r1["marker"],
                "r1_dm": r1["dm_sent"],
                "r2_exit": r2["exit_code"],
                "r2_marker": r2["marker"],
                "r2_dm": r2["dm_sent"],
                "morning_exit": morn["exit_code"],
                "morning_state": morn["outcome"]["state"],
                "morning_dm_count": len(morn["dm_sent"]),
                "health": hlth,
            }
        out_file.write_text(json.dumps(res))
        return

    if mode == "rollback_step":
        # Run a single watchdog check on an existing ReviewWorld directory using BASE_SRC
        w = ReviewWorld(work_dir)
        r1 = w.run_watchdog("18:50", [make_row()])
        r2 = w.run_watchdog("19:00", [make_row()])
        morn = w.run_day_outcome()
        out_file.write_text(json.dumps({
            "r1_exit": r1["exit_code"],
            "r1_marker": r1["marker"],
            "r1_dm": r1["dm_sent"],
            "r2_stdout": r2["stdout"],
            "r2_fetches": r2["fetches"],
            "morning_state": morn["outcome"]["state"],
        }))
        return


def call_worker(src_dir: Path, mode: str, work_dir: Path, out_file: Path):
    env = dict(os.environ, PYTHONPATH=f"{src_dir}:{HERE}", TZ="America/New_York")
    subprocess.run(
        [sys.executable, __file__, "--worker", mode, str(work_dir), str(out_file)],
        env=env,
        timeout=45,
        check=True,
    )


def main():
    sys.path.insert(0, str(HEAD_SRC))
    sys.path.insert(0, str(HERE))
    from bts import entry_receipt
    from review_harness import (
        DEFAULT_DAY,
        UNIT_RIGHT_1,
        UNIT_RIGHT_2,
        UNIT_WRONG_1,
        UNIT_WRONG_2,
        ReviewWorld,
        make_row,
        strip_volatile,
    )

    summary = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. Receipt field-by-field comparison against base
        b_rc = root / "b_rc.json"
        h_rc = root / "h_rc.json"
        call_worker(BASE_SRC, "receipt_compare", root / "b_rc_w", b_rc)
        call_worker(HEAD_SRC, "receipt_compare", root / "h_rc_w", h_rc)
        b_rc_data = json.loads(b_rc.read_text())
        h_rc_data = json.loads(h_rc.read_text())

        receipt_diffs = {}
        for dm_mode, expected_status in (("success", "alerted"), ("failure", "dm_failed")):
            br1 = strip_volatile(b_rc_data[dm_mode]["run1_receipt"], root / "b_rc_w" / f"rc_{dm_mode}" / "picks")
            hr1 = strip_volatile(h_rc_data[dm_mode]["run1_receipt"], root / "h_rc_w" / f"rc_{dm_mode}" / "picks")
            # Normalize picks_dir sha differences (picks JSON doesn't embed path, decision.json doesn't embed path)
            diff_keys = sorted(k for k in (br1.keys() | hr1.keys()) if br1.get(k) != hr1.get(k))
            assert diff_keys == ["exit_code", "marker_status"], (dm_mode, diff_keys, br1, hr1)
            assert (br1["marker_status"], br1["exit_code"]) == ("confirmed", 0)
            assert (hr1["marker_status"], hr1["exit_code"]) == (expected_status, 1)
            assert hr1["verifier"] == {"ok": True, "reason": "match", "required_mlb_ids": [1010], "game_qualified": False}
            receipt_diffs[dm_mode] = diff_keys
        summary["receipt_run1_diff_keys_vs_base"] = receipt_diffs

        # 2. Separate capture reads between watchdog and receipt publish (A5.2 / limit 3)
        shift_outcomes = {}
        for name, first_units, second_payload in [
            ("wrong_then_right", [UNIT_WRONG_1], gzip.compress(json.dumps({"units": [UNIT_RIGHT_1]}).encode(), mtime=0)),
            ("right_then_wrong", [UNIT_RIGHT_1], gzip.compress(json.dumps({"units": [UNIT_WRONG_1]}).encode(), mtime=0)),
            ("wrong_then_corrupt", [UNIT_WRONG_1], b"corrupt"),
        ]:
            w = ReviewWorld(root / f"shift_{name}")
            w.write_pick_and_decision()
            cpath = w.write_capture(first_units)
            real_lu = entry_receipt.load_units
            seen_calls = []

            def swap_after_first(*a, _real=real_lu, _cp=cpath, _sp=second_payload):
                seen_calls.append(1)
                out = _real(*a)
                if len(seen_calls) == 1:
                    _cp.write_bytes(_sp)
                return out

            r = w.run_watchdog("18:30", [make_row()], extra_patches=[("bts.entry_receipt.load_units", swap_after_first)])
            morn = w.run_day_outcome()
            hlth = w.run_health_pick_entry()
            shift_outcomes[name] = {
                "watchdog_exit": r["exit_code"],
                "marker_status": r["marker"]["status"],
                "sealed_receipts": len(r["receipts"]),
                "morning_state": morn["outcome"]["state"],
                "health_levels": [h["level"] for h in hlth],
            }
        assert shift_outcomes["wrong_then_right"] == {
            "watchdog_exit": 1, "marker_status": "alerted", "sealed_receipts": 1,
            "morning_state": "entered", "health_levels": ["WARN"],
        }
        assert shift_outcomes["right_then_wrong"] == {
            "watchdog_exit": 0, "marker_status": "confirmed", "sealed_receipts": 1,
            "morning_state": "entry_unverifiable", "health_levels": [],
        }
        assert shift_outcomes["wrong_then_corrupt"] == {
            "watchdog_exit": 1, "marker_status": "alerted", "sealed_receipts": 0,
            "morning_state": "entry_unverifiable", "health_levels": ["WARN"],
        }
        summary["capture_shift_between_reads"] = shift_outcomes

        # 3. bts day-outcome across ALL end states + testing every clause of A6.5
        day_outcomes = {}

        # (a) entered (timely game proof, even after an earlier wrong_game observation — N10)
        w = ReviewWorld(root / "do_entered")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1, {"id": 78, "roundId": 1, "feedId": 101}])
        w.run_watchdog("18:30", [make_row(77, 1, 1)])
        w.run_watchdog("18:45", [make_row(78, 1, 1)])
        day_outcomes["n10_fixed_entered"] = w.run_day_outcome()["outcome"]["state"]
        assert day_outcomes["n10_fixed_entered"] == "entered"

        # (b) entered_game_unverified (timely unit_unverified)
        w = ReviewWorld(root / "do_unverified")
        w.write_pick_and_decision()
        w.write_capture([{"id": 77, "roundId": 1, "feedId": None}])
        w.run_watchdog("18:30", [make_row(77, 1, 1)])
        day_outcomes["n4_unit_unverified"] = w.run_day_outcome()["outcome"]["state"]
        assert day_outcomes["n4_unit_unverified"] == "entered_game_unverified"

        # (c) What about wrong_round / ambiguous / duplicate_selection when confirmed by watchdog?
        for sub, dd, units_list, pend in [
            ("wrong_round", None, [{"id": 77, "roundId": 2, "feedId": 102}], [make_row(77, 1, 1)]),
            ("ambiguous", None, [UNIT_WRONG_1, {"id": 78, "roundId": 1, "feedId": 102}], [make_row(77, 1, 1), make_row(78, 1, 2)]),
            ("duplicate_selection", (1010, 101), [UNIT_WRONG_1], [make_row(77, 1, 1)]),
        ]:
            w = ReviewWorld(root / f"do_{sub}")
            w.write_pick_and_decision(dd=dd)
            w.write_capture(units_list)
            rw = w.run_watchdog("18:30", pend)
            assert rw["exit_code"] == 0 and rw["marker"]["status"] == "confirmed"
            day_outcomes[f"confirmed_{sub}_morning"] = w.run_day_outcome()["outcome"]["state"]
            # Notice day-outcome returns entry_unverifiable (NOT entered_game_unverified)!
            assert day_outcomes[f"confirmed_{sub}_morning"] == "entry_unverifiable"

        # (d) entered_late (only post-cutoff game proof, after earlier wrong_game)
        w = ReviewWorld(root / "do_late")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1, {"id": 78, "roundId": 1, "feedId": 101}])
        w.run_watchdog("18:30", [make_row(77, 1, 1)])
        w.run_watchdog("19:34", [make_row(78, 1, 1)], elapsed_sec=90.0)
        day_outcomes["late_proof_after_wrong_game"] = w.run_day_outcome()["outcome"]["state"]
        assert day_outcomes["late_proof_after_wrong_game"] == "entered_late"

        # (e) not_entered: BOTH no_pick + wrong_game AND mismatch + wrong_game!
        for abs_kind, abs_pend in [("no_pick", []), ("mismatch", [make_row(77, 2, 1)])]:
            w = ReviewWorld(root / f"do_abs_{abs_kind}")
            w.write_pick_and_decision()
            w.write_capture([UNIT_WRONG_1])
            w.run_watchdog("18:30", abs_pend)
            w.run_watchdog("18:45", [make_row(77, 1, 1)])
            day_outcomes[f"{abs_kind}_then_wrong_game"] = w.run_day_outcome()["outcome"]["state"]
            assert day_outcomes[f"{abs_kind}_then_wrong_game"] == "not_entered"

        # (f) entry_unverifiable: only wrong_game observations, and M10 raising capture
        w = ReviewWorld(root / "do_only_wg")
        w.write_pick_and_decision()
        w.write_capture([UNIT_WRONG_1])
        w.run_watchdog("18:30", [make_row()])
        w.run_watchdog("19:05", [make_row()])
        day_outcomes["only_wrong_game"] = w.run_day_outcome()["outcome"]["state"]
        assert day_outcomes["only_wrong_game"] == "entry_unverifiable"

        w = ReviewWorld(root / "do_m10")
        w.write_pick_and_decision()
        (w.units_dir / "20270324T210000Z.json.gz").write_bytes(b"not gzip")
        w.run_watchdog("18:30", [make_row()])
        day_outcomes["m10_raising_capture"] = w.run_day_outcome()["outcome"]["state"]
        assert day_outcomes["m10_raising_capture"] == "entry_unverifiable"

        summary["day_outcomes"] = day_outcomes

        # 4. Compare M9 (later unit_unverified after earlier wrong_game) on BASE vs F1!
        b_m9 = root / "b_m9.json"
        h_m9 = root / "h_m9.json"
        call_worker(BASE_SRC, "m9_compare", root / "b_m9_w", b_m9)
        call_worker(HEAD_SRC, "m9_compare", root / "h_m9_w", h_m9)
        summary["m9_base_vs_f1"] = {
            "base": json.loads(b_m9.read_text()),
            "f1": json.loads(h_m9.read_text()),
        }

        # 5. Test A6.4 rollback from F1 to BASE across all 3 F1 non-terminal marker states:
        #    (i) alerted/wrong_game
        #    (ii) dm_failed/wrong_game (escalations=[])
        #    (iii) present_unverified after wrong_game (escalations=["wrong_game"])
        rollback_cases = {}
        for rb_name, setup_steps in [
            ("from_alerted_wrong_game", [("18:30", [make_row()], "success")]),
            ("from_dm_failed_wrong_game", [("18:30", [make_row()], "failure")]),
            ("from_present_unverified_with_wg_tier", [("18:30", [make_row()], "success"), ("18:40", [make_row(77, 99, 1)], "success")]),
        ]:
            w_rb = ReviewWorld(root / f"rb_{rb_name}")
            w_rb.write_pick_and_decision()
            w_rb.write_capture([UNIT_WRONG_1])
            for hhmm, pend, dmm in setup_steps:
                w_rb.run_watchdog(hhmm, pend, dm_mode=dmm)
            f1_marker = w_rb.read_marker()
            rb_out = root / f"rb_{rb_name}.json"
            call_worker(BASE_SRC, "rollback_step", w_rb.root, rb_out)
            rollback_cases[rb_name] = {
                "f1_marker_before_rollback": f1_marker,
                "base_after_rollback": json.loads(rb_out.read_text()),
            }
        summary["a6_4_rollback_cases"] = rollback_cases

    (HERE / "probe_04_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_side_worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        main()

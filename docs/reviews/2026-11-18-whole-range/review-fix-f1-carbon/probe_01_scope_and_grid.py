"""Probe 01 (Brief Item 1): Changed-file scope check and base (1e69d0f0) vs F1 (1d30de86) grid comparison.

Runs identical grid scenarios in two isolated subprocesses (one importing base_pkg/src, one importing src),
prints bts.__file__ of each side, verifies byte-for-byte identity outside A1's two cases (and Amendment 3),
and verifies exact contract behavior inside A1's two cases.
"""
from __future__ import annotations

from itertools import combinations, product
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


def build_marker_cases(day: str):
    cases = [
        ("none", None, None),
        ("unreadable", None, b"{broken json"),
        ("other_date_confirmed", {"date": "2027-03-23", "status": "confirmed", "reason": "match"}, None),
        ("other_date_wrong_game", {"date": "2027-03-23", "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"]}, None),
        ("confirmed_match", {"date": day, "status": "confirmed", "reason": "match", "receipt": "prior-rec-id"}, None),
        ("confirmed_unchecked", {"date": day, "status": "confirmed", "reason": "match_game_unchecked", "receipt": "prior-rec-id"}, None),
        ("dm_failed_no_pick", {"date": day, "status": "dm_failed", "reason": "no_pick"}, None),
        ("dm_failed_wrong_game_empty", {"date": day, "status": "dm_failed", "reason": "wrong_game", "escalations": []}, None),
        ("dm_failed_wrong_game_init", {"date": day, "status": "dm_failed", "reason": "wrong_game", "escalations": ["initial"]}, None),
        ("present_unverified_empty", {"date": day, "status": "present_unverified", "reason": "present_unverified", "escalations": []}, None),
        ("present_unverified_init", {"date": day, "status": "present_unverified", "reason": "present_unverified", "escalations": ["initial"]}, None),
        ("present_unverified_wg", {"date": day, "status": "present_unverified", "reason": "present_unverified", "escalations": ["wrong_game"]}, None),
        ("present_unverified_both", {"date": day, "status": "present_unverified", "reason": "present_unverified", "escalations": ["initial", "t30", "wrong_game"]}, None),
        ("alerted_legacy_no_pick", {"date": day, "status": "alerted", "reason": "no_pick"}, None),
        ("alerted_legacy_wrong_game", {"date": day, "status": "alerted", "reason": "wrong_game"}, None),
    ]
    tiers = ("initial", "wrong_game", "t30", "t15")
    for r in range(len(tiers) + 1):
        for sub in combinations(tiers, r):
            tag = "+".join(sub) if sub else "empty"
            cases.append((f"alerted_np_{tag}", {"date": day, "status": "alerted", "reason": "no_pick", "escalations": list(sub)}, None))
            cases.append((f"alerted_wg_{tag}", {"date": day, "status": "alerted", "reason": "wrong_game", "escalations": list(sub)}, None))
    return cases


def worker_run(out_path: Path):
    import bts
    from bts import entry_receipt
    import httpx
    from bts.leaderboard.auth import AuthError
    from bts.contest_fetch import ContestFetchError
    from review_harness import (
        DEFAULT_DAY,
        DEFAULT_PITCH,
        UNIT_RIGHT_1,
        UNIT_RIGHT_2,
        UNIT_WRONG_1,
        UNIT_WRONG_2,
        ReviewWorld,
        make_row,
        strip_volatile,
    )

    records = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Part A: Full factor grid over verifier/qualification category x prior marker x dm_mode x delivery x time
        # To keep runtime under ~10s per side while covering every combination of factors:
        # 1) Full cross-product of:
        #    6 core scenarios (no_pick, mismatch_wrong_batter, mismatch_missing_dd, present_unverified, match_right, match_wrong)
        #    x 26 representative prior markers (all statuses, another date, unreadable, legacy, all 16 tier subsets on no_pick + key wrong_game markers)
        #    x 3 dm modes (success, failure, none)
        #    x 4 delivery modes (None/legacy, delivered, private_locked, locked_unconfirmed)
        #    = 1,872 cells at 18:30, PLUS full time sweep (5 window/cutoff bands) across all 6 scenarios x key markers x dm x delivery!
        all_markers = build_marker_cases(DEFAULT_DAY)
        rep_markers = [
            c for c in all_markers
            if not c[0].startswith("alerted_wg_") or c[0] in {
                "alerted_wg_empty", "alerted_wg_wrong_game", "alerted_wg_initial+wrong_game",
                "alerted_wg_wrong_game+t30", "alerted_wg_initial+wrong_game+t30+t15",
            }
        ]

        w = ReviewWorld(root / "grid")
        scenarios = [
            ("no_pick", [UNIT_RIGHT_1], [], None, 101),
            ("mismatch_batter", [UNIT_RIGHT_1], [make_row(unit=77, player=2)], None, 101),
            ("mismatch_dd_missing", [UNIT_RIGHT_1, UNIT_RIGHT_2], [make_row(unit=77, player=1)], (1020, 201), 101),
            ("present_unverified", [UNIT_WRONG_1], [make_row(unit=77, player=99)], None, 101),
            ("match_right", [UNIT_RIGHT_1], [make_row(unit=77, player=1)], None, 101),
            ("match_unit_unverified", [{"id": 77, "roundId": 1, "feedId": None}], [make_row(unit=77, player=1)], None, 101),
            ("match_wrong", [UNIT_WRONG_1], [make_row(unit=77, player=1)], None, 101),
        ]

        for scen_name, units_list, pending, dd, gpk in scenarios:
            w.clear_captures()
            w.write_capture(units_list)
            for delivery in (None, "delivered", "private_locked", "locked_unconfirmed"):
                w.write_pick_and_decision(game_pk=gpk, dd=dd, delivery=delivery)
                for m_name, m_obj, m_raw in rep_markers:
                    for dm_mode in ("success", "failure", "none"):
                        for p in (w.picks_dir.parent / "health_state" / "pick_entry_receipts" / DEFAULT_DAY).glob("*"):
                            p.unlink()
                        w.set_marker(m_obj, m_raw)
                        obs = w.run_watchdog("18:30", pending, dm_mode=dm_mode)
                        key = f"A|{scen_name}|deliv={delivery}|prior={m_name}|dm={dm_mode}"
                        records[key] = strip_volatile(obs, w.picks_dir)

        # Part B: Full time-band sweep across window, exact t30 (19:05), between (19:12), exact t15 (19:20),
        # inside t15 (19:30), cutoff passed during fetch (19:34+61s), exact cutoff start (19:35), outside early (12:00)
        time_bands = [
            ("outside_early", "12:00", 0.0),
            ("early_window", "18:30", 0.0),
            ("exact_t30", "19:05", 0.0),
            ("mid_t30_t15", "19:12", 0.0),
            ("exact_t15", "19:20", 0.0),
            ("inside_t15", "19:30", 0.0),
            ("cutoff_during_fetch", "19:34", 61.0),
            ("exact_cutoff_start", "19:35", 0.0),
        ]
        time_markers = [c for c in all_markers if c[0] in {
            "none", "confirmed_match", "dm_failed_no_pick", "dm_failed_wrong_game_empty",
            "present_unverified_wg", "alerted_np_initial", "alerted_np_initial+t30",
            "alerted_np_initial+t30+t15", "alerted_wg_wrong_game", "alerted_wg_wrong_game+t30",
            "alerted_wg_wrong_game+t30+t15", "alerted_wg_initial+wrong_game",
        }]
        for scen_name, units_list, pending, dd, gpk in scenarios:
            w.clear_captures()
            w.write_capture(units_list)
            w.write_pick_and_decision(game_pk=gpk, dd=dd, delivery="delivered")
            for m_name, m_obj, m_raw in time_markers:
                for band_name, hhmm, elapsed in time_bands:
                    for dm_mode in ("success", "failure"):
                        for p in (w.picks_dir.parent / "health_state" / "pick_entry_receipts" / DEFAULT_DAY).glob("*"):
                            p.unlink()
                        w.set_marker(m_obj, m_raw)
                        obs = w.run_watchdog(hhmm, pending, dm_mode=dm_mode, elapsed_sec=elapsed)
                        key = f"B|{scen_name}|prior={m_name}|band={band_name}|dm={dm_mode}"
                        records[key] = strip_volatile(obs, w.picks_dir)

        # Part C: Every qualify state (8 single + all 64 double pairs) x all 4 verifier results,
        # tested both with prior=none AND with prior=alerted_wg_wrong_game!
        qual_states = (
            "confirmed", "unit_unverified", "wrong_round", "ambiguous",
            "missing", "player_unverified", "unsupported_duplicate_selection", "wrong_game",
        )
        w_q = ReviewWorld(root / "qual")
        w_q.write_capture([UNIT_RIGHT_1, UNIT_WRONG_2, {"id": 99, "roundId": 1, "feedId": 999}])
        combos = [(s,) for s in qual_states] + list(product(qual_states, repeat=2))
        for prior_tag, prior_marker in (
            ("none", None),
            ("prior_wg", {"date": DEFAULT_DAY, "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"]}),
        ):
            for q_tuple in combos:
                dd = (1020, 201) if len(q_tuple) == 2 else None
                w_q.write_pick_and_decision(game_pk=101, dd=dd, delivery="delivered")
                for ver_ok, ver_reason in ((True, "match"), (False, "no_pick"), (False, "mismatch"), (True, "present_unverified")):
                    for p in (w_q.picks_dir.parent / "health_state" / "pick_entry_receipts" / DEFAULT_DAY).glob("*"):
                        p.unlink()
                    w_q.set_marker(prior_marker)

                    def fake_qualify(slots, rows, cw, units, _qt=q_tuple):
                        out_slots = []
                        for s, st in zip(slots, _qt, strict=True):
                            r = {"roundId": 1, "unitId": 99, "playerId": 1, "number": 1} if st == "wrong_game" else None
                            out_slots.append({"role": s["role"], "batter_id": s["batter_id"], "game_pk": s["game_pk"], "row": r, "state": st})
                        return {"slots": out_slots, "all_confirmed": all(st == "confirmed" for st in _qt)}

                    obs = w_q.run_watchdog(
                        "18:30",
                        [make_row()],
                        extra_patches=[
                            ("bts.contest_fetch.pick_entry_status", lambda *a, _ok=ver_ok, _r=ver_reason: (_ok, _r)),
                            ("bts.entry_receipt.qualify", fake_qualify),
                        ],
                    )
                    key = f"C|prior={prior_tag}|ver={ver_reason}|qual={'+'.join(q_tuple)}"
                    records[key] = strip_volatile(obs, w_q.picks_dir)

        # Part D: Real-capture edge states (shared batter single/double rows, null/bool/str/float game_pk,
        # pre-fetch exits, fetch failures)
        w_e = ReviewWorld(root / "edges")
        obs = w_e.run_watchdog("18:30")
        records["D|no_pick_file"] = strip_volatile(obs, w_e.picks_dir)

        w_e.write_pick_and_decision(delivery=None, notification_sent=False)
        records["D|not_committed_preview"] = strip_volatile(w_e.run_watchdog("18:30"), w_e.picks_dir)

        w_e.write_pick_and_decision(delivery="delivered", scoreable=False, notification_sent=False)
        records["D|not_committed_unscoreable"] = strip_volatile(w_e.run_watchdog("18:30"), w_e.picks_dir)

        w_e.write_pick_and_decision(delivery="delivered")
        w_e.write_capture([UNIT_WRONG_1])
        records["D|identity_mismatch"] = strip_volatile(
            w_e.run_watchdog("18:30", [make_row()], session_username="intruder"), w_e.picks_dir
        )
        for leaf, exc in (
            ("login", AuthError("expired")),
            ("profile", httpx.ConnectError("down")),
            ("pending", ContestFetchError("bad envelope")),
            ("rounds", ValueError("bad json")),
            ("crosswalk", KeyError("missing")),
        ):
            w_e.set_marker(None)
            for p in (w_e.picks_dir.parent / "health_state" / "pick_entry_receipts" / DEFAULT_DAY).glob("*"):
                p.unlink()
            records[f"D|fetch_error_{leaf}"] = strip_volatile(
                w_e.run_watchdog("18:30", [make_row()], fetch_error=(leaf, exc)), w_e.picks_dir
            )

        # Non-int game_pk types (None, True, False, 101.0, "101") with wrong_game capture, with and without prior_wg
        for gpk_label, gpk_val in (("null", None), ("bool_true", True), ("bool_false", False), ("float", 101.0), ("str", "101")):
            for prior_tag, prior_m in (
                ("none", None),
                ("prior_wg", {"date": DEFAULT_DAY, "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"]}),
            ):
                for p in (w_e.picks_dir.parent / "health_state" / "pick_entry_receipts" / DEFAULT_DAY).glob("*"):
                    p.unlink()
                w_e.write_pick_and_decision(game_pk=gpk_val, delivery="delivered")
                w_e.set_marker(prior_m)
                records[f"D|gpk_{gpk_label}|prior={prior_tag}"] = strip_volatile(
                    w_e.run_watchdog("18:30", [make_row()]), w_e.picks_dir
                )

        # Shared-batter double-down across single/two rows and right/wrong bindings, with and without prior_wg
        w_e.clear_captures()
        w_e.write_capture([UNIT_RIGHT_1, UNIT_WRONG_2])
        for rows_label, pend in (
            ("one_right_primary", [make_row(unit=77, player=1, number=1)]),
            ("one_wrong_primary", [make_row(unit=78, player=1, number=1)]),
            ("two_rows", [make_row(unit=77, player=1, number=1), make_row(unit=78, player=1, number=2)]),
        ):
            for prior_tag, prior_m in (
                ("none", None),
                ("prior_wg", {"date": DEFAULT_DAY, "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"]}),
            ):
                for p in (w_e.picks_dir.parent / "health_state" / "pick_entry_receipts" / DEFAULT_DAY).glob("*"):
                    p.unlink()
                w_e.write_pick_and_decision(batter_id=1010, game_pk=101, dd=(1010, 201, DEFAULT_PITCH, "Batter 1010"))
                w_e.set_marker(prior_m)
                records[f"D|shared_batter_{rows_label}|prior={prior_tag}"] = strip_volatile(
                    w_e.run_watchdog("18:30", pend), w_e.picks_dir
                )

    out_path.write_text(json.dumps({"bts_file": bts.__file__, "cell_count": len(records), "records": records}))


def is_new_f1_behavior_expected(key: str, base_rec: dict, head_rec: dict) -> tuple[bool, str]:
    """Determine from the scenario specification alone whether F1 is permitted/required to differ from base."""
    # Any prior marker carrying reason=="wrong_game" or "wrong_game" in escalations is F1-introduced marker state
    prior_has_wg = any(
        tok in key
        for tok in (
            "prior=dm_failed_wrong_game",
            "prior=present_unverified_wg",
            "prior=present_unverified_both",
            "prior=alerted_legacy_wrong_game",
            "prior=alerted_wg_",
            "prior=prior_wg",
            "+wrong_game",
            "alerted_np_wrong_game",
        )
    )
    # Pre-fetch or already_confirmed exits never change even if prior had wg (unless prior wasn't confirmed)
    if "prior=confirmed_" in key or "band=outside_early" in key or "band=exact_cutoff_start" in key:
        return False, "pre_fetch_or_confirmed"
    if key.startswith("D|no_pick_file") or key.startswith("D|not_committed") or key.startswith("D|identity_mismatch") or key.startswith("D|fetch_error_"):
        return False, "early_exit_or_fetch_error"

    # Case (a): match + distinct batters + exact int game_pk + at least one wrong_game slot
    if "A|match_wrong|" in key or "B|match_wrong|" in key:
        return True, "case_a_match_wrong"
    if key.startswith("C|") and "|ver=match|" in key and "wrong_game" in key.split("|qual=")[1]:
        return True, "case_a_injected_wrong_game"

    # Case (b) or prior wrong_game marker interactions (e.g. preserving escalations on confirmed/dm_failed,
    # legacy wrong_game marker not fabricating 'initial', or keeping unresolved when not all_confirmed)
    if prior_has_wg:
        return True, "prior_wrong_game_state"

    return False, "outside_a1_and_prior_wg"


def main():
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        base_json = tmp_dir / "base.json"
        head_json = tmp_dir / "head.json"

        for src_dir, out_file in ((BASE_SRC, base_json), (HEAD_SRC, head_json)):
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{src_dir}:{HERE}"
            env["TZ"] = "America/New_York"
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", str(out_file)],
                env=env,
                timeout=90,
                capture_output=True,
                text=True,
                check=True,
            )
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)

        base_data = json.loads(base_json.read_text())
        head_data = json.loads(head_json.read_text())

    print(f"base bts.__file__ = {base_data['bts_file']}")
    print(f"head bts.__file__ = {head_data['bts_file']}")
    assert str(BASE_SRC) in base_data["bts_file"], base_data["bts_file"]
    assert str(HEAD_SRC) in head_data["bts_file"], head_data["bts_file"]
    assert base_data["cell_count"] == head_data["cell_count"]

    unexpected_diffs = []
    expected_diffs = 0
    expected_same_inside = 0
    outside_checked = 0

    for key in sorted(base_data["records"]):
        b_rec = base_data["records"][key]
        h_rec = head_data["records"][key]
        allowed, why = is_new_f1_behavior_expected(key, b_rec, h_rec)
        if not allowed:
            outside_checked += 1
            if b_rec != h_rec:
                unexpected_diffs.append({"key": key, "why": why, "base": b_rec, "head": h_rec})
        else:
            if b_rec != h_rec:
                expected_diffs += 1
                # Verify key A1-A4 invariants on every changed cell!
                if why in ("case_a_match_wrong", "case_a_injected_wrong_game"):
                    assert h_rec["exit_code"] == 1, (key, h_rec)
                    assert h_rec["marker"]["reason"] == "wrong_game", (key, h_rec)
                    assert h_rec["marker"]["status"] in ("alerted", "dm_failed"), (key, h_rec)
                    assert len(h_rec["receipts"]) == 1, (key, h_rec)
                    r = h_rec["receipts"][0]
                    assert r["verifier"] == b_rec["receipts"][0]["verifier"], (key, r)
                    assert r["qualification"] == b_rec["receipts"][0]["qualification"], (key, r)
                    assert r["marker_status"] == h_rec["marker"]["status"], (key, r)
                    assert r["exit_code"] == 1, (key, r)
            else:
                expected_same_inside += 1

    summary = {
        "base_bts_file": base_data["bts_file"],
        "head_bts_file": head_data["bts_file"],
        "total_cells_per_side": base_data["cell_count"],
        "outside_a1_cells_checked_identical": outside_checked,
        "unexpected_diffs_count": len(unexpected_diffs),
        "unexpected_diffs_sample": unexpected_diffs[:5],
        "inside_f1_changed_cells": expected_diffs,
        "inside_f1_unchanged_cells": expected_same_inside,
    }
    (HERE / "probe_01_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert not unexpected_diffs, f"Found {len(unexpected_diffs)} unexpected diffs outside F1 scope!"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_run(Path(sys.argv[2]))
    else:
        main()

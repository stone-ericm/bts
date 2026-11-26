"""Items 4 and 10: Round-1 defect reproduction on 1d30de86, closure on 15b27474, and exhaustive sequence hunt."""
from __future__ import annotations

from datetime import date
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from probe_common import (
    DAY,
    HERE,
    PITCH,
    SandboxWorld,
    UNIT_RIGHT_1,
    UNIT_RIGHT_2,
    UNIT_WRONG_1,
    UNIT_WRONG_2,
    bts_source_path,
    make_row,
)

R1_SRC = HERE / "r1_archive" / "src"
CAND_SRC = HERE.parents[1] / "src"
ALL_CLEAR_MATCH = f"✅ BTS pick entry confirmed for {DAY} (match)."


def run_codex_r1_probe_worker():
    failures = []
    with tempfile.TemporaryDirectory(prefix="f1_r1repro_") as tmp:
        root = Path(tmp)
        for has_initial in (True, False):
            for bridge in ("crosswalk", "removal", "mismatch"):
                w = SandboxWorld(root / f"{has_initial}_{bridge}")
                w.commit()
                w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
                if has_initial:
                    w.check("18:25", [])
                w.check("18:35", [make_row(77, 1, 1)], dm="failure")
                if bridge == "crosswalk":
                    w.check("18:45", [make_row(77, 1, 1)], crosswalk={})
                elif bridge == "removal":
                    w.check("18:45", [])
                else:
                    w.check("18:45", [make_row(77, 3, 1)])
                w.capture([UNIT_RIGHT_1], f"{DAY}T210000Z")
                last = w.check("19:00", [make_row(77, 1, 1)])
                if (
                    last["exit"] != 1
                    or (last["marker"] or {}).get("status") == "confirmed"
                    or (last["marker"] or {}).get("reason") != "wrong_game"
                    or ALL_CLEAR_MATCH in last["attempted_dm"]
                ):
                    failures.append(f"{'M11' if has_initial else 'M12'}:{bridge}")
    print(json.dumps({"bts_file": bts_source_path(), "failed_cases": failures}))


def run_sequence_hunt():
    repro = {}
    for label, src_dir in (("r1_1d30de86", R1_SRC), ("r2_15b27474", CAND_SRC)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-r1"],
            env={**os.environ, "PYTHONPATH": str(src_dir) + ":" + str(HERE), "TZ": "America/New_York"},
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        repro[label] = json.loads(proc.stdout)

    assert len(repro["r1_1d30de86"]["failed_cases"]) == 6, repro["r1_1d30de86"]
    assert repro["r2_15b27474"]["failed_cases"] == [], repro["r2_15b27474"]

    violations = []
    sequences_checked = 0

    with tempfile.TemporaryDirectory(prefix="f1_hunt_") as tmp:
        root = Path(tmp)

        openers = (
            "wg_dm_fail",
            "wg_dm_ok",
            "wg_dm_none",
            "init_then_wg_fail",
            "wg_cutoff_first",
            "legacy_r1_alerted_no_bit",
            "legacy_r1_dm_failed_no_bit",
            "legacy_r1_escalations_only",
        )

        step_catalog = (
            ("crosswalk_gap", "success"),
            ("no_pick", "success"),
            ("no_pick", "failure"),
            ("no_pick", "none"),
            ("mismatch", "success"),
            ("mismatch", "failure"),
            ("fetch_fail", "success"),
            ("identity_mismatch", "success"),
            ("outside_window", "success"),
            ("cutoff_pass_nopick", "success"),
            ("wg_still", "failure"),
            ("wg_still", "success"),
        )

        unproved_endings = (
            "unit_unverified_conflict",
            "unit_unverified_null",
            "wrong_round",
            "ambiguous_two_rows",
            "player_unverified_str",
            "shared_batter_switch",
            "double_leg1_proved_leg2_unbound",
            "double_leg1_unbound_leg2_proved",
        )

        def setup_opener(w: SandboxWorld, opener: str):
            w.commit(pitch=PITCH)
            w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
            if opener == "wg_dm_fail":
                w.check("18:20", [make_row(77, 1, 1)], dm="failure", window_min=90)
            elif opener == "wg_dm_ok":
                w.check("18:20", [make_row(77, 1, 1)], dm="success", window_min=90)
            elif opener == "wg_dm_none":
                w.check("18:20", [make_row(77, 1, 1)], dm="none", window_min=90)
            elif opener == "init_then_wg_fail":
                w.check("18:15", [], dm="success", window_min=90)
                w.check("18:20", [make_row(77, 1, 1)], dm="failure", window_min=90)
            elif opener == "wg_cutoff_first":
                w.check("18:20", [make_row(77, 1, 1)], elapsed=4600, window_min=90)
            elif opener == "legacy_r1_alerted_no_bit":
                w.set_marker({"date": DAY, "status": "alerted", "reason": "wrong_game", "escalations": []})
            elif opener == "legacy_r1_dm_failed_no_bit":
                w.set_marker({"date": DAY, "status": "dm_failed", "reason": "wrong_game", "escalations": []})
            elif opener == "legacy_r1_escalations_only":
                w.set_marker({"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["wrong_game"]})

        def run_step(w: SandboxWorld, at: str, kind: str, dm_mode: str):
            if kind == "crosswalk_gap":
                return w.check(at, [make_row(77, 1, 1)], crosswalk={}, dm=dm_mode, window_min=90)
            if kind == "no_pick":
                return w.check(at, [], dm=dm_mode, window_min=90)
            if kind == "mismatch":
                return w.check(at, [make_row(77, 3, 1)], dm=dm_mode, window_min=90)
            if kind == "fetch_fail":
                return w.check(at, [make_row(77, 1, 1)], fetch_failure="pending", dm=dm_mode, window_min=90)
            if kind == "identity_mismatch":
                return w.check(at, [make_row(77, 1, 1)], username="other", dm=dm_mode, window_min=90)
            if kind == "outside_window":
                return w.check("17:00", [make_row(77, 1, 1)], dm=dm_mode, window_min=90)
            if kind == "cutoff_pass_nopick":
                return w.check(at, [], elapsed=4600, dm=dm_mode, window_min=90)
            if kind == "wg_still":
                return w.check(at, [make_row(77, 1, 1)], dm=dm_mode, window_min=90)
            raise AssertionError(kind)

        def run_ending(w: SandboxWorld, at: str, ending: str, dm_mode: str = "success"):
            for p in w.units_dir.glob("*.json.gz"):
                p.unlink()
            if ending == "unit_unverified_conflict":
                w.capture([UNIT_WRONG_1], f"{DAY}T200000Z")
                w.capture([UNIT_RIGHT_1], f"{DAY}T210000Z")
                return w.check(at, [make_row(77, 1, 1)], dm=dm_mode, window_min=90)
            if ending == "unit_unverified_null":
                w.capture([{"id": 77, "roundId": 1, "feedId": None}], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, 1, 1)], dm=dm_mode, window_min=90)
            if ending == "wrong_round":
                w.capture([{"id": 77, "roundId": 2, "feedId": 101}], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, 1, 1)], dm=dm_mode, window_min=90)
            if ending == "ambiguous_two_rows":
                w.capture([UNIT_RIGHT_1, {"id": 79, "roundId": 1, "feedId": 101}], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, 1, 1), make_row(79, 1, 2)], dm=dm_mode, window_min=90)
            if ending == "player_unverified_str":
                w.capture([UNIT_RIGHT_1], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, "1", 1)], dm=dm_mode, window_min=90)
            if ending == "shared_batter_switch":
                w.commit(dd=(1010, 201, PITCH))
                w.capture([UNIT_RIGHT_1, UNIT_RIGHT_2], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, 1, 1), make_row(78, 1, 2)], dm=dm_mode, window_min=90)
            if ending == "double_leg1_proved_leg2_unbound":
                w.commit(dd=(1020, 201, PITCH))
                w.capture([UNIT_RIGHT_1, {"id": 78, "roundId": 1, "feedId": None}], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, 1, 1), make_row(78, 2, 2)], dm=dm_mode, window_min=90)
            if ending == "double_leg1_unbound_leg2_proved":
                w.commit(dd=(1020, 201, PITCH))
                w.capture([{"id": 77, "roundId": 1, "feedId": None}, UNIT_RIGHT_2], f"{DAY}T200000Z")
                return w.check(at, [make_row(77, 1, 1), make_row(78, 2, 2)], dm=dm_mode, window_min=90)
            raise AssertionError(ending)

        idx = 0
        for opener in openers:
            for s1_idx, step1 in enumerate(step_catalog):
                ending = unproved_endings[(idx + s1_idx) % len(unproved_endings)]
                dm_end = ("success", "failure", "none")[idx % 3]
                w = SandboxWorld(root / f"s1_{idx}")
                setup_opener(w, opener)
                r_mid = run_step(w, "18:35", step1[0], step1[1])
                m_mid = r_mid["marker"]
                if m_mid is None or m_mid.get("status") == "confirmed" or ALL_CLEAR_MATCH in r_mid["attempted_dm"]:
                    violations.append(("mid_confirmed", opener, step1, r_mid))
                r_end = run_ending(w, "19:05", ending, dm_end)
                m_end = r_end["marker"]
                if (
                    r_end["exit"] != 1
                    or m_end is None
                    or m_end.get("status") == "confirmed"
                    or m_end.get("game_unresolved") is not True
                    or m_end.get("reason") != "wrong_game"
                    or ALL_CLEAR_MATCH in r_end["attempted_dm"]
                ):
                    violations.append(("end_confirmed_without_proof", opener, step1, ending, dm_end, r_end))
                w.clear()
                idx += 1
                sequences_checked += 1

        chain_pairs = list(product(
            (("crosswalk_gap", "success"), ("no_pick", "failure"), ("no_pick", "success"), ("mismatch", "success"), ("fetch_fail", "success")),
            (("crosswalk_gap", "success"), ("no_pick", "success"), ("wg_still", "failure"), ("cutoff_pass_nopick", "success")),
        ))
        for p_idx, (c1, c2) in enumerate(chain_pairs):
            opener = openers[p_idx % len(openers)]
            ending = unproved_endings[p_idx % len(unproved_endings)]
            w = SandboxWorld(root / f"s2_{p_idx}")
            setup_opener(w, opener)
            run_step(w, "18:35", c1[0], c1[1])
            run_step(w, "19:05", c2[0], c2[1])
            r_unproved = run_ending(w, "19:20", ending, "success")
            if (
                r_unproved["exit"] != 1
                or r_unproved["marker"]["status"] == "confirmed"
                or r_unproved["marker"].get("game_unresolved") is not True
                or ALL_CLEAR_MATCH in r_unproved["attempted_dm"]
            ):
                violations.append(("chain2_unproved", opener, c1, c2, ending, r_unproved))

            for p in w.units_dir.glob("*.json.gz"):
                p.unlink()
            if ending == "shared_batter_switch":
                w.commit(pitch=PITCH)
                w.capture([UNIT_RIGHT_1], f"{DAY}T220000Z")
                r_proved = w.check("19:25", [make_row(77, 1, 1)], window_min=90)
            elif ending.startswith("double_"):
                w.capture([UNIT_RIGHT_1, UNIT_RIGHT_2], f"{DAY}T220000Z")
                r_proved = w.check("19:25", [make_row(77, 1, 1), make_row(78, 2, 2)], window_min=90)
            else:
                w.capture([UNIT_RIGHT_1], f"{DAY}T220000Z")
                r_proved = w.check("19:25", [make_row(77, 1, 1)], window_min=90)
            if (
                r_proved["exit"] != 0
                or r_proved["marker"]["status"] != "confirmed"
                or "game_unresolved" in r_proved["marker"]
            ):
                violations.append(("chain2_full_proof_failed", opener, c1, c2, ending, r_proved))
            w.clear()
            sequences_checked += 1

        # Full tier lifecycle + terminal confirmation + midnight reset
        w = SandboxWorld(root / "tier_lifecycle")
        w.commit()
        w.capture([UNIT_WRONG_1])
        s_a = w.check("18:30", [make_row(77, 1, 1)], dm="failure")
        assert s_a["marker"]["status"] == "dm_failed" and s_a["marker"]["escalations"] == []
        s_b = w.check("18:32", [make_row(77, 1, 1)], dm="success")
        assert s_b["marker"]["escalations"] == ["wrong_game"] and len(s_b["dm"]) == 1
        assert w.check("18:35", [make_row(77, 1, 1)])["dm"] == []
        s_c = w.check("18:45", [], dm="failure")
        assert s_c["marker"]["status"] == "alerted" and s_c["marker"]["escalations"] == ["wrong_game"]
        s_d = w.check("18:46", [], dm="success")
        assert s_d["marker"]["escalations"] == ["initial", "wrong_game"] and "NOT entered" in s_d["dm"][0]
        assert w.check("18:50", [])["dm"] == []
        s_e = w.check("19:05", [make_row(77, 1, 1)], dm="failure")
        assert s_e["marker"]["escalations"] == ["initial", "wrong_game"]
        s_f = w.check("19:06", [make_row(77, 1, 1)], dm="success")
        assert s_f["marker"]["escalations"] == ["initial", "t30", "wrong_game"] and len(s_f["dm"]) == 1
        assert w.check("19:07", [])["dm"] == []
        s_g = w.check("19:20", [make_row(77, 1, 1)], dm="success")
        assert s_g["marker"]["escalations"] == ["initial", "t15", "t30", "wrong_game"] and len(s_g["dm"]) == 1
        assert w.check("19:21", [make_row(77, 1, 1)])["dm"] == []
        w.capture([UNIT_RIGHT_1])
        s_h = w.check("19:25", [make_row(77, 1, 1)])
        assert s_h["exit"] == 0 and s_h["marker"]["status"] == "confirmed" and s_h["dm"] == [ALL_CLEAR_MATCH]
        w.capture([UNIT_WRONG_1])
        s_i = w.check("19:30", [make_row(77, 1, 1)])
        assert s_i["exit"] == 0 and s_i["fetches"] == [] and s_i["dm"] == []

        m_root = root / "midnight_bleed"
        w_d1 = SandboxWorld(m_root, "2027-03-24")
        w_d2 = SandboxWorld(m_root, "2027-03-25")
        w_d1.commit(pitch="2027-03-25T00:45:00-04:00")
        w_d2.commit(pitch="2027-03-25T00:45:00-04:00")
        w_d1.capture([UNIT_WRONG_1, {"id": 88, "roundId": 2, "feedId": None}], "20270324T230000Z")
        rounds = {1: date(2027, 3, 24), 2: date(2027, 3, 25)}
        r_d1 = w_d1.check("23:55", [make_row(77, 1, 1, rnd=1)], rounds=rounds, dm="failure")
        assert r_d1["marker"]["game_unresolved"] is True and r_d1["marker"]["date"] == "2027-03-24"
        r_d2 = w_d2.check("00:05", [make_row(88, 1, 1, rnd=2)], rounds=rounds)
        assert r_d2["exit"] == 0 and r_d2["marker"]["date"] == "2027-03-25" and r_d2["marker"]["status"] == "confirmed"
        assert "game_unresolved" not in r_d2["marker"]

    summary = {
        "r1_reproduction": repro,
        "sequences_checked": sequences_checked,
        "violations": len(violations),
        "tier_lifecycle_and_midnight": "passed",
    }
    (HERE / "log_3_tiers_and_r1_hunt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert not violations, f"Violations found: {violations[:3]}"


if __name__ == "__main__":
    if "--worker-r1" in sys.argv:
        run_codex_r1_probe_worker()
    else:
        run_sequence_hunt()

"""Runner for the 1,600-cell parity grid in two fast foreground steps (<9s each)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_1_parity_grid import (
    BASE_SRC,
    CAND_SRC,
    generate_cells,
    prior_has_same_day_wrong_game,
    run_one_side,
)

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "grid_cache"


def verify_comparison():
    base_data = json.loads((STATE_DIR / "base.json").read_text())
    cand_data = json.loads((STATE_DIR / "cand.json").read_text())

    print(f"base bts.__file__      = {base_data['bts_file']}")
    print(f"candidate bts.__file__ = {cand_data['bts_file']}")
    assert str(BASE_SRC) in base_data["bts_file"]
    assert str(CAND_SRC) in cand_data["bts_file"]

    cells = {c["id"]: c for c in generate_cells()}
    b_res = base_data["results"]
    c_res = cand_data["results"]
    assert set(b_res.keys()) == set(c_res.keys()) == set(cells.keys())

    identical = 0
    changed_a1a = 0
    changed_a1b = 0
    changed_prior_wg_meta = 0
    unexpected_diffs = []
    false_alarms = []
    missed_wrong_games = []

    for cid, cell in cells.items():
        b = b_res[cid]
        c = c_res[cid]
        has_prior_wg = prior_has_same_day_wrong_game(cell["prior"])

        c_rec = c["receipts"][-1] if c["receipts"] else None
        reached_match = bool(
            c_rec and c_rec.get("outcome") == "observed" and (c_rec.get("verifier") or {}).get("reason") == "match"
        )
        shared_batter = bool(cell["dd"] is not None and cell["dd"][0] == 1010)
        q_slots = ((c_rec.get("qualification") or {}).get("slots") or []) if reached_match else []
        slot_gpks = [cell["game1"]] + ([cell["dd"][1]] if cell["dd"] else [])

        is_a1a = False
        is_a1b = False
        if reached_match:
            if not shared_batter:
                is_a1a = any(
                    qs["state"] == "wrong_game" and type(gpk) is int
                    for qs, gpk in zip(q_slots, slot_gpks, strict=True)
                )
            if not is_a1a and has_prior_wg:
                all_conf = (not shared_batter) and bool(q_slots) and all(qs["state"] == "confirmed" for qs in q_slots)
                if not all_conf:
                    is_a1b = True

        if reached_match and not (is_a1a or is_a1b):
            if c["exit"] != 0 or (c["marker"] or {}).get("status") != "confirmed":
                false_alarms.append((cid, c))

        if reached_match and not shared_batter:
            if any(qs["state"] == "wrong_game" and type(gpk) is int for qs, gpk in zip(q_slots, slot_gpks, strict=True)):
                if (
                    c["exit"] != 1
                    or (c["marker"] or {}).get("status") == "confirmed"
                    or (c["marker"] or {}).get("reason") != "wrong_game"
                ):
                    missed_wrong_games.append((cid, c))

        if b == c:
            identical += 1
            if is_a1a or is_a1b:
                unexpected_diffs.append((cid, "expected A1 change, got identical", b, c))
            continue

        if not (is_a1a or is_a1b or has_prior_wg):
            unexpected_diffs.append((cid, "changed outside A1(a)/A1(b)/prior_wrong_game", b, c))
            continue

        if is_a1a or is_a1b:
            if is_a1a:
                changed_a1a += 1
            else:
                changed_a1b += 1
            assert c["exit"] == 1, (cid, c)
            assert c["marker"]["reason"] == "wrong_game", (cid, c)
            assert c["marker"]["status"] in ("alerted", "dm_failed"), (cid, c)
            assert c["marker"].get("game_unresolved") is True, (cid, c)
            assert all("✅ BTS pick entry" not in m for m in c["attempted_dm"]), (cid, c)
            b_rec_cmp = {k: v for k, v in b["receipts"][-1].items() if k not in ("marker_status", "exit_code")}
            c_rec_cmp = {k: v for k, v in c["receipts"][-1].items() if k not in ("marker_status", "exit_code")}
            if b_rec_cmp != c_rec_cmp:
                unexpected_diffs.append((cid, "receipt differed outside marker_status/exit_code", b_rec_cmp, c_rec_cmp))
        else:
            changed_prior_wg_meta += 1
            if c["marker"] is not None and c["marker"] != "<UNREADABLE>":
                if c["marker"]["status"] == "confirmed":
                    assert "game_unresolved" not in c["marker"], (cid, c)
                else:
                    assert c["marker"].get("game_unresolved") is True, (cid, c)

    # Clean up temporary JSON dumps after comparison
    for f in STATE_DIR.glob("*.json"):
        f.unlink()
    STATE_DIR.rmdir()

    summary = {
        "base_bts_file": base_data["bts_file"],
        "candidate_bts_file": cand_data["bts_file"],
        "total_cells": len(cells),
        "clean_family_cells": sum(1 for c in cells.values() if c["family"] == "clean_parity"),
        "identical_to_base": identical,
        "changed_a1a": changed_a1a,
        "changed_a1b": changed_a1b,
        "changed_prior_wg_meta": changed_prior_wg_meta,
        "unexpected_diffs": len(unexpected_diffs),
        "false_alarms": len(false_alarms),
        "missed_wrong_games": len(missed_wrong_games),
    }
    (HERE / "log_1_parity_grid.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    assert not unexpected_diffs, f"Unexpected diffs: {unexpected_diffs[:3]}"
    assert not false_alarms, f"False alarms: {false_alarms[:3]}"
    assert not missed_wrong_games, f"Missed wrong games: {missed_wrong_games[:3]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("base", "candidate", "verify"), required=True)
    args = ap.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "base":
        run_one_side(STATE_DIR / "base.json")
        print("base grid complete")
    elif args.step == "candidate":
        run_one_side(STATE_DIR / "cand.json")
        print("candidate grid complete")
    else:
        verify_comparison()

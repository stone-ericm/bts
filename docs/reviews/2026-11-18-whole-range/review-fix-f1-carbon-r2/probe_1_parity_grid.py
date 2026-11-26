"""Items 1, 3, and 11: Full parity grid and false-alarm / missed-case checks (base 1e69d0f0 vs 15b27474)."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SRC = HERE / "base_archive" / "src"
CAND_SRC = REPO / "src"

SLOT_STATES = (
    "confirmed",
    "wrong_game",
    "unit_unverified",
    "wrong_round",
    "ambiguous",
    "player_unverified",
    "missing_null_dd",
)
TIERS = ("initial", "t30", "t15", "wrong_game")
ALL_TIER_SETS = [sorted(c) for r in range(len(TIERS) + 1) for c in combinations(TIERS, r)]


def setup_slot_states(world, states: tuple[str, ...], *, source_shape: str = "pending"):
    """Build real capture + response payloads yielding exact qualify states on a (True, 'match') run."""
    from probe_common import make_row

    pending_rows = []
    profile_preds = []
    units = []
    for idx, st in enumerate(states):
        unit_id, player_id, right_game = (77, 1, 101) if idx == 0 else (78, 2, 201)
        if st == "missing_null_dd":
            # Only used for idx == 1 when dd batter_id is None: no row needed for slot 2.
            continue
        p_val = str(player_id) if st == "player_unverified" else player_id
        r1 = make_row(unit=unit_id, player=p_val, number=idx + 1, rnd=1)
        rows_for_leg = [r1]
        if st == "ambiguous":
            rows_for_leg.append(make_row(unit=unit_id + 10, player=player_id, number=idx + 1, rnd=1))
        if source_shape == "pending":
            pending_rows.extend(rows_for_leg)
        elif source_shape == "profile":
            profile_preds.extend(
                {"unitId": r["unitId"], "playerId": r["playerId"], "number": r["number"], "result": "hit"}
                for r in rows_for_leg
            )
        else:  # "both": same (roundId, unitId, playerId, number) in both endpoints deduplicates in qualify!
            pending_rows.extend(rows_for_leg)
            profile_preds.extend(
                {"unitId": r["unitId"], "playerId": r["playerId"], "number": r["number"], "result": "hit"}
                for r in rows_for_leg
            )

        if st == "confirmed":
            units.append({"id": unit_id, "roundId": 1, "feedId": right_game})
        elif st == "wrong_game":
            units.append({"id": unit_id, "roundId": 1, "feedId": right_game + 1})
        elif st == "unit_unverified":
            units.append({"id": unit_id, "roundId": 1, "feedId": None})
        elif st == "wrong_round":
            units.append({"id": unit_id, "roundId": 2, "feedId": right_game + 1})
        elif st == "ambiguous":
            # Even when BOTH candidate rows bind to wrong games, qualify returns 'ambiguous'!
            units.append({"id": unit_id, "roundId": 1, "feedId": right_game + 1})
            units.append({"id": unit_id + 10, "roundId": 1, "feedId": right_game + 2})
        elif st == "player_unverified":
            units.append({"id": unit_id, "roundId": 1, "feedId": right_game + 1})

    world.capture(units)
    profile = {"predictions": [{"roundId": 1, "roundPredictions": profile_preds}]} if profile_preds else {"predictions": []}
    return pending_rows, profile


def apply_prior_marker(world, spec):
    from probe_common import DAY

    if spec == "none":
        return
    if spec == "unreadable":
        world.marker.parent.mkdir(parents=True, exist_ok=True)
        world.marker.write_text("{corrupt json")
        return
    status, reason, escalations, m_day, unresolved = spec
    marker = {"date": m_day, "status": status, "reason": reason}
    if escalations is not None:
        marker["escalations"] = list(escalations)
    if unresolved is not None:
        marker["game_unresolved"] = unresolved
    world.set_marker(marker)


def generate_cells():
    from probe_common import DAY

    cells = []

    # Family 1: Qualifier & slot-state matrix (Single + Double + game_pk types + source shapes + shared batter)
    priors_f1 = [
        ("none", "none"),
        ("base_alerted", ("alerted", "no_pick", ["initial"], DAY, None)),
        ("r1_wg_alerted", ("alerted", "wrong_game", ["wrong_game"], DAY, None)),
        ("r2_unresolved_present", ("present_unverified", "present_unverified", [], DAY, True)),
        ("r2_unresolved_dm_failed", ("dm_failed", "no_pick", [], DAY, True)),
        ("stale_unresolved", ("alerted", "wrong_game", ["wrong_game"], "2027-03-23", True)),
    ]

    # 1a: Single slot states across priors and source shapes
    for st in SLOT_STATES[:-1]:
        for shape in ("pending", "profile", "both"):
            for p_name, p_spec in priors_f1:
                cells.append({
                    "id": f"single:{st}:{shape}:{p_name}",
                    "family": "qualifier",
                    "kind": "match_slots",
                    "states": (st,),
                    "shape": shape,
                    "game1": 101,
                    "dd": None,
                    "prior": p_spec,
                    "delivery": "delivered",
                    "dm": "success",
                    "at": "18:45",
                    "elapsed": 0.0,
                })

    # 1b: Single non-int & edge game_pk values with wrong_game / confirmed bindings
    for gpk_label, gpk in (
        ("null", None),
        ("str", "101"),
        ("float_same", 101.0),
        ("float_diff", 100.0),
        ("bool_true", True),
        ("bool_false", False),
        ("zero", 0),
        ("neg", -5),
    ):
        for st in ("wrong_game", "confirmed"):
            for p_name, p_spec in priors_f1[:4]:
                cells.append({
                    "id": f"gpk:{gpk_label}:{st}:{p_name}",
                    "family": "gpk_types",
                    "kind": "match_slots",
                    "states": (st,),
                    "shape": "pending",
                    "game1": gpk,
                    "dd": None,
                    "prior": p_spec,
                    "delivery": "delivered",
                    "dm": "success",
                    "at": "18:45",
                    "elapsed": 0.0,
                })

    # 1c: Double all 7x7 ordered pairs across clean vs unresolved prior
    usable_s1 = SLOT_STATES[:-1]
    usable_s2 = SLOT_STATES
    for s1, s2 in product(usable_s1, usable_s2):
        # Note: if s1 is player_unverified, string playerId makes unresolved=True so s2 missing_null_dd also becomes player_unverified
        for p_name, p_spec in (("none", "none"), ("unresolved", ("present_unverified", "present_unverified", ["wrong_game"], DAY, True))):
            dd_batter = None if s2 == "missing_null_dd" else 1020
            cells.append({
                "id": f"double:{s1}+{s2}:{p_name}",
                "family": "double_pairs",
                "kind": "match_slots",
                "states": (s1, s2),
                "shape": "pending",
                "game1": 101,
                "dd": (dd_batter, 201),
                "prior": p_spec,
                "delivery": "delivered",
                "dm": "success",
                "at": "18:45",
                "elapsed": 0.0,
            })

    # 1d: Shared batter & exact duplicate selection (unsupported_duplicate_selection)
    for dd_game, tag in ((201, "shared_diff_game"), (101, "exact_dup_selection")):
        for p_name, p_spec in priors_f1:
            cells.append({
                "id": f"dup:{tag}:{p_name}",
                "family": "shared_batter",
                "kind": "shared_batter",
                "states": ("wrong_game", "wrong_game"),
                "shape": "pending",
                "game1": 101,
                "dd": (1010, dd_game),
                "prior": p_spec,
                "delivery": "delivered",
                "dm": "success",
                "at": "18:45",
                "elapsed": 0.0,
            })

    # Family 2: Full parity grid across non-wrong-game paths + wrong-game paths
    # Every tier subset x representative statuses/verdicts/DM/delivery/times
    times = [
        ("outside_early", "18:39", 0.0),   # 61 min to 19:40 pitch (> 60 window)
        ("win_edge", "18:40", 0.0),        # 60 min to pitch (inside)
        ("mid_win", "18:45", 0.0),         # 50 min to cutoff
        ("t30_above", "19:04", 0.0),       # 31 min to cutoff
        ("t30_exact", "19:05", 0.0),       # 30 min to cutoff
        ("t15_above", "19:19", 0.0),       # 16 min to cutoff
        ("t15_exact", "19:20", 0.0),       # 15 min to cutoff
        ("cutoff_pass", "19:34", 61.0),    # starts 6m to pitch, 61s elapsed -> -0.017m to cutoff
        ("cutoff_exact", "19:35", 0.0),    # 5 min to pitch -> outside_window
    ]

    verdicts = (
        "no_pick",
        "mismatch_wrong_batter",
        "mismatch_missing_dd",
        "present_unverified_empty",
        "present_unverified_partial",
        "match_confirmed",
        "match_unit_unverified",
        "match_wrong_game",
    )
    deliveries = ("delivered", "locked_unconfirmed", "private_locked", "legacy_no_decision")
    dms = ("success", "failure", "none")

    # 2a: Exhaustive tier-set x time x DM x verdict sweep with deterministic status/delivery rotation
    for idx, (tiers, (t_tag, at, el), dm, verd) in enumerate(
        product(ALL_TIER_SETS, times[2:8], dms, ("no_pick", "mismatch_wrong_batter", "present_unverified_empty", "match_confirmed", "match_unit_unverified", "match_wrong_game"))
    ):
        # Subsample deterministically to ~580 dense cells covering every tier set, every time, every DM, every verdict
        if (idx % 3) != 0 and "wrong_game" not in tiers and verd != "match_wrong_game":
            continue
        status = ("alerted", "dm_failed", "present_unverified")[idx % 3]
        has_wg = "wrong_game" in tiers or (idx % 5 == 0)
        reason = "wrong_game" if (has_wg and idx % 2 == 0) else "no_pick"
        unresolved = True if (has_wg and idx % 4 != 0) else None
        deliv = deliveries[idx % len(deliveries)]
        cells.append({
            "id": f"ladder:{idx}:{verd}:{'-'.join(tiers) or 'empty'}:{status}:{t_tag}:{dm}:{deliv}",
            "family": "ladder",
            "kind": verd,
            "states": ("wrong_game",) if verd == "match_wrong_game" else ("unit_unverified",) if verd == "match_unit_unverified" else ("confirmed",),
            "shape": "pending",
            "game1": 101,
            "dd": (1020, 201) if verd == "mismatch_missing_dd" else None,
            "prior": (status, reason, tiers, DAY, unresolved),
            "delivery": deliv,
            "dm": dm,
            "at": at,
            "elapsed": el,
        })

    # 2b: Clean-day parity sweep (NO prior wrong_game, NO current wrong_game -> MUST BE 100% IDENTICAL TO BASE)
    clean_priors = [
        ("none", "none"),
        ("unreadable", "unreadable"),
        ("stale_wg", ("alerted", "wrong_game", ["wrong_game"], "2027-03-23", True)),
        ("confirmed_prior", ("confirmed", "match", None, DAY, None)),
        ("legacy_alerted", ("alerted", "no_pick", None, DAY, None)),
        ("alerted_init", ("alerted", "no_pick", ["initial"], DAY, None)),
        ("alerted_t30", ("alerted", "no_pick", ["initial", "t30"], DAY, None)),
        ("alerted_all", ("alerted", "mismatch", ["initial", "t30", "t15"], DAY, None)),
        ("dm_failed_prior", ("dm_failed", "no_pick", None, DAY, None)),
        ("present_prior", ("present_unverified", "present_unverified", ["initial"], DAY, None)),
    ]
    clean_verdicts = [
        "no_pick",
        "mismatch_wrong_batter",
        "mismatch_missing_dd",
        "present_unverified_empty",
        "present_unverified_partial",
        "match_confirmed",
        "match_unit_unverified",
        "no_pick_file",
        "not_committed_preview",
        "not_committed_skip",
        "fetch_auth",
        "fetch_profile",
        "fetch_pending",
        "fetch_rounds",
        "fetch_crosswalk",
        "identity_mismatch",
    ]
    for c_idx, (v, (p_name, p_spec)) in enumerate(product(clean_verdicts, clean_priors)):
        t_tag, at, el = times[c_idx % len(times)]
        dm = dms[c_idx % len(dms)]
        deliv = deliveries[c_idx % len(deliveries)]
        cells.append({
            "id": f"clean:{v}:{p_name}:{t_tag}:{dm}:{deliv}",
            "family": "clean_parity",
            "kind": v,
            "states": ("unit_unverified",) if v == "match_unit_unverified" else ("confirmed",),
            "shape": "pending",
            "game1": 101,
            "dd": (1020, 201) if v in ("mismatch_missing_dd", "present_unverified_partial") else None,
            "prior": p_spec,
            "delivery": None if v == "not_committed_preview" else "not_applicable" if v == "not_committed_skip" else deliv,
            "dm": dm,
            "at": at,
            "elapsed": el,
        })

    return cells


def run_one_side(out_path: Path):
    from probe_common import SandboxWorld, bts_source_path, make_row, normalize_obs, UNIT_WRONG_1, UNIT_WRONG_2

    src_file = bts_source_path()
    cells = generate_cells()
    results = {}

    with tempfile.TemporaryDirectory(prefix="f1_grid_") as tmp:
        base_dir = Path(tmp)
        for idx, cell in enumerate(cells):
            world = SandboxWorld(base_dir / str(idx))
            kind = cell["kind"]
            if kind != "no_pick_file":
                world.commit(
                    game=cell["game1"],
                    dd=(cell["dd"][0], cell["dd"][1], "2027-03-24T19:40:00-04:00") if cell["dd"] else None,
                    delivery=cell["delivery"],
                )
            apply_prior_marker(world, cell["prior"])

            pending = []
            profile = {"predictions": []}
            kwargs = {"dm": cell["dm"], "elapsed": cell["elapsed"]}

            if kind in ("match_slots", "match_confirmed", "match_unit_unverified", "match_wrong_game"):
                pending, profile = setup_slot_states(world, cell["states"], source_shape=cell["shape"])
            elif kind == "shared_batter":
                world.capture([UNIT_WRONG_1, UNIT_WRONG_2])
                pending = [make_row(77, 1, 1), make_row(78, 1, 2)]
            elif kind == "no_pick":
                world.capture([UNIT_WRONG_1])
                pending = []
            elif kind == "mismatch_wrong_batter":
                world.capture([UNIT_WRONG_1])
                pending = [make_row(77, 3, 1)]
            elif kind == "mismatch_missing_dd":
                world.capture([UNIT_WRONG_1, UNIT_WRONG_2])
                pending = [make_row(77, 1, 1)]
            elif kind == "present_unverified_empty":
                world.capture([UNIT_WRONG_1])
                pending = [make_row(77, 1, 1)]
                kwargs["crosswalk"] = {}
            elif kind == "present_unverified_partial":
                world.capture([UNIT_WRONG_1, UNIT_WRONG_2])
                pending = [make_row(77, 1, 1), make_row(78, 99, 2)]
            elif kind.startswith("fetch_"):
                world.capture([UNIT_WRONG_1])
                pending = [make_row(77, 1, 1)]
                kwargs["fetch_failure"] = kind.removeprefix("fetch_")
            elif kind == "identity_mismatch":
                world.capture([UNIT_WRONG_1])
                pending = [make_row(77, 1, 1)]
                kwargs["username"] = "wrong_user"
            else:
                world.capture([UNIT_WRONG_1])

            obs = world.check(cell["at"], pending, profile=profile, **kwargs)
            results[cell["id"]] = normalize_obs(obs, world.root)
            world.clear()

    out_path.write_text(json.dumps({"bts_file": src_file, "count": len(cells), "results": results}))


def prior_has_same_day_wrong_game(prior_spec) -> bool:
    from probe_common import DAY

    if not isinstance(prior_spec, tuple):
        return False
    status, reason, escalations, m_day, unresolved = prior_spec
    if m_day != DAY or status == "confirmed":
        return False
    esc = escalations or []
    return bool(unresolved is True or reason == "wrong_game" or "wrong_game" in esc)


def compare_sides():
    with tempfile.TemporaryDirectory(prefix="f1_cmp_") as tmp:
        base_json = Path(tmp) / "base.json"
        cand_json = Path(tmp) / "cand.json"

        for label, src_dir, out_file in (("base", BASE_SRC, base_json), ("candidate", CAND_SRC, cand_json)):
            env = {**os.environ, "PYTHONPATH": str(src_dir), "TZ": "America/New_York"}
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--side", label, "--out", str(out_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=True,
            )
            if proc.stdout:
                print(proc.stdout, end="")

        base_data = json.loads(base_json.read_text())
        cand_data = json.loads(cand_json.read_text())

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

        # Determine whether candidate's match branch was reached and whether A1(a) or A1(b) applies
        c_rec = c["receipts"][-1] if c["receipts"] else None
        reached_match = bool(c_rec and c_rec.get("outcome") == "observed" and (c_rec.get("verifier") or {}).get("reason") == "match")
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

        # Item 3 check 1: False alarms (did candidate refuse confirmation on a match outside A1(a) and A1(b)?)
        if reached_match and not (is_a1a or is_a1b):
            if c["exit"] != 0 or (c["marker"] or {}).get("status") != "confirmed":
                false_alarms.append((cid, c))

        # Item 3 check 2: Missed wrong games (distinct batters, exact int game_pk, unambiguous wrong_game state -> must NOT confirm)
        if reached_match and not shared_batter:
            if any(qs["state"] == "wrong_game" and type(gpk) is int for qs, gpk in zip(q_slots, slot_gpks, strict=True)):
                if c["exit"] != 1 or (c["marker"] or {}).get("status") == "confirmed" or (c["marker"] or {}).get("reason") != "wrong_game":
                    missed_wrong_games.append((cid, c))

        if b == c:
            identical += 1
            if is_a1a or is_a1b:
                unexpected_diffs.append((cid, "expected A1 change, got identical", b, c))
            continue

        # Output differed from base: ONLY permitted when is_a1a, is_a1b, or has_prior_wg!
        if not (is_a1a or is_a1b or has_prior_wg):
            unexpected_diffs.append((cid, "changed outside A1(a)/A1(b)/prior_wrong_game", b, c))
            continue

        if is_a1a or is_a1b:
            if is_a1a:
                changed_a1a += 1
            else:
                changed_a1b += 1
            # Verify A1/A2/R1 invariants on every A1(a)/A1(b) cell
            assert c["exit"] == 1, (cid, c)
            assert c["marker"]["reason"] == "wrong_game", (cid, c)
            assert c["marker"]["status"] in ("alerted", "dm_failed"), (cid, c)
            assert c["marker"].get("game_unresolved") is True, (cid, c)
            assert all("confirmed" not in m for m in c["attempted_dm"]), (cid, c)
            # Receipt differs from base ONLY in marker_status and exit_code
            b_rec_cmp = {k: v for k, v in b["receipts"][-1].items() if k not in ("marker_status", "exit_code")}
            c_rec_cmp = {k: v for k, v in c["receipts"][-1].items() if k not in ("marker_status", "exit_code")}
            if b_rec_cmp != c_rec_cmp:
                unexpected_diffs.append((cid, "receipt differed outside marker_status/exit_code", b_rec_cmp, c_rec_cmp))
        else:
            # has_prior_wg is True, and run was NOT A1(a)/A1(b):
            # e.g., nonterminal run carrying game_unresolved/escalations, or full-proof confirmation preserving escalations / sending all-clear
            changed_prior_wg_meta += 1
            if c["marker"] is not None and c["marker"] != "<UNREADABLE>":
                if c["marker"]["status"] == "confirmed":
                    assert "game_unresolved" not in c["marker"], (cid, c)
                else:
                    assert c["marker"].get("game_unresolved") is True, (cid, c)

    summary = {
        "base_bts_file": base_data["bts_file"],
        "candidate_bts_file": cand_data["bts_file"],
        "total_cells": len(cells),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("base", "candidate", "compare"), default="compare")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.side == "compare":
        compare_sides()
    else:
        run_one_side(args.out)

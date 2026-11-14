"""Fast adversarial probes against scripts/audit/c2_framing/f26_studio.py."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["BTS_LGBM_DETERMINISTIC"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_probe"

import numpy as np
import pandas as pd

from scripts.audit.c2_framing import f26 as F
from scripts.audit.c2_framing import f26_studio as ST
from scripts.audit.c2_framing import screen as S


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_pa_2026(dates: list[str], resumed_only_dates: set[str] | None = None) -> pd.DataFrame:
    rows = []
    resumed_only = resumed_only_dates or set()
    pk_base = 800000
    for i, d in enumerate(dates):
        for b in range(1, 5):
            # If date is in resumed_only, every row is resumed; otherwise all False
            is_res = d in resumed_only
            rows.append({
                "game_pk": pk_base + i,
                "date": pd.Timestamp(d),
                "season": 2026,
                "batter_id": 1000 + b,
                "pitcher_id": 2000 + (i % 3),
                "is_home": bool(b % 2),
                "is_hit": int(b % 2 == 0),
                "is_resumed_portion": is_res,
            })
    return pd.DataFrame(rows)


def _write_complete_seed(
    out: Path,
    seed: int,
    cal: list[str],
    pins: dict,
    hits: dict[str, list[int]],
    *,
    p_game_hit_r1: float = 0.7,
    p_game_hit_r2: float = 0.6,
    rank2_val: int = 2,
    add_tie_row: bool = False,
    basis: str = S.BASIS,
    season_val: int = F.TEST_SEASON,
    batter_offset_by_arm: dict[str, int] | None = None,
    manifest_overrides: dict | None = None,
    results_overrides: dict | None = None,
    a_posted_identified: int = 10,
    raw_rank1_hits: bool = False,
):
    d = out / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    offsets = batter_offset_by_arm or {}
    for arm in F.ARMS:
        off = offsets.get(arm, 0)
        rows = []
        for idx, day in enumerate(cal):
            rows.append({
                "date": pd.Timestamp(day),
                "rank": 1,
                "batter_id": 1001 + off,
                "game_pk": 800000 + idx,
                "p_game_hit": p_game_hit_r1,
                "actual_hit": hits[arm][idx],
                "n_pas": 4,
                "p_game_hit_basis": basis,
                "season": season_val,
            })
            rows.append({
                "date": pd.Timestamp(day),
                "rank": rank2_val,
                "batter_id": 1002 + off,
                "game_pk": 800000 + idx,
                "p_game_hit": p_game_hit_r2,
                "actual_hit": 0,
                "n_pas": 4,
                "p_game_hit_basis": basis,
                "season": season_val,
            })
            if add_tie_row:
                rows.append({
                    "date": pd.Timestamp(day),
                    "rank": rank2_val,
                    "batter_id": 1003 + off,
                    "game_pk": 800000 + idx,
                    "p_game_hit": p_game_hit_r2,
                    "actual_hit": 0,
                    "n_pas": 4,
                    "p_game_hit_basis": basis,
                    "season": season_val,
                })
        pd.DataFrame(rows).to_parquet(d / f"profiles_{arm}_2026.parquet", index=False)

    units = []
    for arm in F.ARMS:
        u = {"arm": arm, "season": F.TEST_SEASON, "cpu_s": 3600.0, "wall_s": 1800.0,
             "labels": {"changed": 0, "void_dropped": 0}}
        if arm != "baseline":
            ident = a_posted_identified if arm == "A_posted" else 10
            no_cat = 0 if ident > 0 else len(cal)
            u["catcher"] = {
                "counts": {
                    "side_games": {"identified": ident, "no_catcher": no_cat, "too_few_rates": 0},
                    "pa_rows": {"identified": ident * 10, "no_catcher": no_cat * 10, "too_few_rates": 0},
                },
                "identified_ids": [50] if ident > 0 else [],
            }
        units.append(u)

    man = {
        "schema": ST.SCHEMA,
        "seed": seed,
        "input_pins": pins,
        "test_season": F.TEST_SEASON,
        "arms": list(F.ARMS),
        "basis": S.BASIS,
        "retrain_every": S.RETRAIN_EVERY,
        "lgb_params": F.reviewed_lgb_params(),
        "feature_settings": dict(S.SETTINGS),
        "scoring": dict(S.SCORING),
        "env": {"BTS_LGBM_DETERMINISTIC": "1", "BTS_LGBM_RANDOM_STATE": str(seed), "TZ": "America/New_York"},
        "self_check": {"identical": True, "max_abs_diff": None, "rows": 100},
        "calendar": cal,
    }
    if manifest_overrides:
        man.update(manifest_overrides)
    (d / "manifest.json").write_text(json.dumps(man))

    # Mimic F.rank1 truncation when results.json is generated
    recorded_rank1 = hits if raw_rank1_hits else {
        arm: F.rank1(pd.read_parquet(d / f"profiles_{arm}_2026.parquet"), cal)
        for arm in F.ARMS
    }
    res = {
        "schema": ST.SCHEMA,
        "seed": seed,
        "units": units,
        "total_cpu_s": 10800.0,
        "calendar_sha256": ST._sha(F.canonical(cal)),
        "rank1": recorded_rank1,
        "arms": {a: {"p_at_1_delta": 0.0, "p_57_exact": 0.0, "mean_max_streak": 0.0,
                     "longest_replay_streak": 0} for a in F.ARMS[1:]},
    }
    if results_overrides:
        res.update(results_overrides)
    (d / "results.json").write_text(json.dumps(res))


def probe_1_calendar_omission_and_missing_pa():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 1a: Shared omitted calendar (5 of 20 days)
        inputs, out = root / "in1a", root / "out1a"
        inputs.mkdir()
        full_cal = [f"2026-04-{d:02d}" for d in range(1, 21)]
        _make_pa_2026(full_cal).to_parquet(inputs / F.FROZEN_PA, index=False)
        pins = {"pa_2026.parquet": _sha((inputs / F.FROZEN_PA).read_bytes())}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))
        short_cal = full_cal[:5]
        for s in F.SEEDS:
            _write_complete_seed(out, s, short_cal, pins,
                                 {"baseline": [0] * 5, "A_posted": [1] * 5, "A_projected": [0] * 5})
        agg = ST.aggregate(inputs, out)
        print("PROBE 1a (shared 5/20 days):",
              "calendar_2026=", len(ST.calendar_2026(inputs, pins)),
              "agg.n_days=", agg["n_days"],
              "agg.disposition=", agg["A_posted"]["disposition"],
              "agg.m=", agg["A_posted"]["m"])

        # 1b: Missing inputs/pa_2026.parquet completely
        inputs, out = root / "in1b", root / "out1b"
        inputs.mkdir()
        pins_bad = {"pa_2026.parquet": "0" * 64}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins_bad}))
        for s in F.SEEDS:
            _write_complete_seed(out, s, full_cal, pins_bad,
                                 {"baseline": [0] * 20, "A_posted": [1] * 20, "A_projected": [0] * 20})
        agg = ST.aggregate(inputs, out)
        print("PROBE 1b (missing inputs/pa_2026.parquet):",
              "agg.n_days=", agg["n_days"],
              "agg.disposition=", agg["A_posted"]["disposition"])


def probe_2_a_posted_zero_identified():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs, out = root / "in", root / "out"
        inputs.mkdir()
        cal = [f"2026-04-{d:02d}" for d in range(1, 21)]
        _make_pa_2026(cal).to_parquet(inputs / F.FROZEN_PA, index=False)
        pins = {"pa_2026.parquet": _sha((inputs / F.FROZEN_PA).read_bytes())}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))
        for s in F.SEEDS:
            _write_complete_seed(out, s, cal, pins,
                                 {"baseline": [1] * 20, "A_posted": [0] * 20, "A_projected": [1] * 20},
                                 a_posted_identified=0)
        agg = ST.aggregate(inputs, out)
        print("PROBE 2a (A_posted identified=0, no catcher parquet):",
              "ident=", [c["side_games"]["identified"] for c in agg["catcher_coverage"]["A_posted"]],
              "agg.disposition=", agg["A_posted"]["disposition"])


def probe_3_profile_validation_gaps():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cal = [f"2026-04-{d:02d}" for d in range(1, 21)]
        pa26 = _make_pa_2026(cal)

        # 3a: Non-binary actual_hit=2 and fractional actual_hit=0.9 (truncated to 0 by F.rank1)
        inputs, out = root / "c3a_in", root / "c3a_out"
        inputs.mkdir()
        pa26.to_parquet(inputs / F.FROZEN_PA, index=False)
        pins = {"pa_2026.parquet": _sha((inputs / F.FROZEN_PA).read_bytes())}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))
        for s in F.SEEDS:
            _write_complete_seed(out, s, cal, pins,
                                 {"baseline": [0.9] * 20, "A_posted": [2] * 20, "A_projected": [0] * 20})
        agg3a = ST.aggregate(inputs, out)
        print("PROBE 3a (actual_hit=0.9 baseline, 2 A_posted):",
              "baseline_rate=", agg3a["baseline_rank1_hit_rate_by_seed"][0],
              "m=", agg3a["A_posted"]["m"],
              "disposition=", agg3a["A_posted"]["disposition"])

        # 3b: NaN p_game_hit, out-of-range p_game_hit, rank ties/gaps, wrong basis & season, different batters per arm
        inputs, out = root / "c3b_in", root / "c3b_out"
        inputs.mkdir()
        pa26.to_parquet(inputs / F.FROZEN_PA, index=False)
        pins = {"pa_2026.parquet": _sha((inputs / F.FROZEN_PA).read_bytes())}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))
        for s in F.SEEDS:
            _write_complete_seed(
                out, s, cal, pins,
                {"baseline": [0] * 20, "A_posted": [1] * 20, "A_projected": [0] * 20},
                p_game_hit_r1=float("nan"), p_game_hit_r2=1.5, rank2_val=99, add_tie_row=True,
                basis="actual_pa", season_val=2025,
                batter_offset_by_arm={"baseline": 0, "A_posted": 500, "A_projected": 900},
            )
        agg3b = ST.aggregate(inputs, out)
        print("PROBE 3b (NaN/1.5 p_game_hit, rank=99 tie, basis=actual_pa, season=2025, disjoint batters):",
              "disposition=", agg3b["A_posted"]["disposition"])

        # 3c: Rank 1 has LOWER p_game_hit than rank 2
        inputs, out = root / "c3c_in", root / "c3c_out"
        inputs.mkdir()
        pa26.to_parquet(inputs / F.FROZEN_PA, index=False)
        pins = {"pa_2026.parquet": _sha((inputs / F.FROZEN_PA).read_bytes())}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))
        for s in F.SEEDS:
            _write_complete_seed(out, s, cal, pins,
                                 {"baseline": [0] * 20, "A_posted": [1] * 20, "A_projected": [0] * 20},
                                 p_game_hit_r1=0.05, p_game_hit_r2=0.95)
        agg3c = ST.aggregate(inputs, out)
        print("PROBE 3c (rank-1 p=0.05 < rank-2 p=0.95):",
              "disposition=", agg3c["A_posted"]["disposition"])


def probe_4_settings_reconciliation_and_admission_pins():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs, out = root / "in", root / "out"
        inputs.mkdir()
        cal = [f"2026-04-{d:02d}" for d in range(1, 21)]
        _make_pa_2026(cal).to_parquet(inputs / F.FROZEN_PA, index=False)
        pins = {
            "pa_2026.parquet": _sha((inputs / F.FROZEN_PA).read_bytes()),
            "pa_2017.parquet": "deadbeef" * 8,  # mismatch vs admission.json
        }
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))
        for idx, s in enumerate(F.SEEDS):
            _write_complete_seed(
                out, s, cal, pins,
                {"baseline": [0] * 20, "A_posted": [1] * 20, "A_projected": [0] * 20},
                manifest_overrides={
                    "basis": "actual_pa" if idx % 2 == 0 else "estimated_pa",
                    "retrain_every": 30 + idx,
                    "feature_settings": {"ROOKIE_GATE_K": 0},
                    "lgb_params": {"deterministic": False},
                    "self_check": {"identical": False},
                    "env": {"BTS_LGBM_DETERMINISTIC": "0", "BTS_LGBM_RANDOM_STATE": "999"},
                },
                results_overrides={
                    "arms": {
                        "A_posted": {"p_at_1_delta": -0.99, "p_57_exact": -0.5,
                                     "mean_max_streak": -99.0, "longest_replay_streak": -50},
                        "A_projected": {"p_at_1_delta": 0.0, "p_57_exact": 0.0,
                                        "mean_max_streak": 0.0, "longest_replay_streak": 0},
                    }
                },
            )
        agg = ST.aggregate(inputs, out)
        print("PROBE 4 (tampered pins, mismatched settings, self_check=False, forged streak_metrics, missing scorecards/diffs):",
              "disposition=", agg["A_posted"]["disposition"],
              "streak_s0=", agg["streak_metrics_by_seed"][0]["A_posted"]["mean_max_streak"])


def probe_5_positive_completeness_controls():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs, out = root / "in", root / "out"
        inputs.mkdir()
        cal = [f"2026-04-{d:02d}" for d in range(1, 21)]
        pins = {"pa_2026.parquet": "a" * 64}
        (inputs / ST.PREPARED).write_text(json.dumps({"pins": pins}))

        def reset_clean():
            for s in F.SEEDS:
                _write_complete_seed(out, s, cal, pins,
                                     {"baseline": [0] * 20, "A_posted": [1] * 20, "A_projected": [0] * 20})
                st = out / f"seed_{s}" / "STOPPED.json"
                if st.exists():
                    st.unlink()

        reset_clean()
        # 5a: Unsorted calendar in 1 seed
        _write_complete_seed(out, F.SEEDS[0], list(reversed(cal)), pins,
                             {"baseline": [0] * 20, "A_posted": [1] * 20, "A_projected": [0] * 20},
                             raw_rank1_hits=True)
        try:
            ST.aggregate(inputs, out)
            print("PROBE 5a (unsorted calendar): UNEXPECTEDLY ACCEPTED")
        except F.RunInvalid as e:
            print("PROBE 5a (unsorted calendar): refused:", e)

        reset_clean()
        # 5b: Missing results.json
        (out / f"seed_{F.SEEDS[9]}" / "results.json").unlink()
        try:
            ST.aggregate(inputs, out)
            print("PROBE 5b (missing results.json): UNEXPECTEDLY ACCEPTED")
        except F.RunInvalid as e:
            print("PROBE 5b (missing results.json): refused:", e)

        reset_clean()
        # 5c: STOPPED.json present
        (out / f"seed_{F.SEEDS[4]}" / "STOPPED.json").write_text('{"reason": "test"}')
        try:
            ST.aggregate(inputs, out)
            print("PROBE 5c (STOPPED.json): refused unexpectedly? Wait no - UNEXPECTEDLY ACCEPTED")
        except F.RunInvalid as e:
            print("PROBE 5c (STOPPED.json): refused:", e)

        reset_clean()
        # 5d: Tampered results["rank1"] not reproduced by profile
        res_p = out / f"seed_{F.SEEDS[0]}" / "results.json"
        res_obj = json.loads(res_p.read_text())
        res_obj["rank1"]["A_posted"][0] = 0
        res_p.write_text(json.dumps(res_obj))
        try:
            ST.aggregate(inputs, out)
            print("PROBE 5d (unreproduced rank1): UNEXPECTEDLY ACCEPTED")
        except F.RunInvalid as e:
            print("PROBE 5d (unreproduced rank1): refused:", e)


def probe_6_calendar_2026():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = root / "in"
        inputs.mkdir()
        # Unsorted input + day 2 resumed-only + day 3 mixed original/resumed
        df = pd.DataFrame([
            {"game_pk": 3, "date": pd.Timestamp("2026-04-03"), "season": 2026, "batter_id": 1, "is_resumed_portion": False},
            {"game_pk": 3, "date": pd.Timestamp("2026-04-03"), "season": 2026, "batter_id": 2, "is_resumed_portion": True},
            {"game_pk": 2, "date": pd.Timestamp("2026-04-02"), "season": 2026, "batter_id": 1, "is_resumed_portion": True},
            {"game_pk": 1, "date": pd.Timestamp("2026-04-01"), "season": 2026, "batter_id": 1, "is_resumed_portion": False},
        ])
        p = inputs / F.FROZEN_PA
        df.to_parquet(p, index=False)
        pins = {F.FROZEN_PA: _sha(p.read_bytes())}
        got = ST.calendar_2026(inputs, pins)
        print("PROBE 6a (calendar_2026 mixed/resumed-only):", got)

        df_bad = df.drop(columns=["is_resumed_portion"])
        df_bad.to_parquet(p, index=False)
        pins_bad = {F.FROZEN_PA: _sha(p.read_bytes())}
        try:
            ST.calendar_2026(inputs, pins_bad)
            print("PROBE 6b (missing is_resumed_portion): UNEXPECTEDLY ACCEPTED")
        except F.InputRefused as e:
            print("PROBE 6b (missing is_resumed_portion): refused:", e)


if __name__ == "__main__":
    probe_1_calendar_omission_and_missing_pa()
    probe_2_a_posted_zero_identified()
    probe_3_profile_validation_gaps()
    probe_4_settings_reconciliation_and_admission_pins()
    probe_5_positive_completeness_controls()
    probe_6_calendar_2026()

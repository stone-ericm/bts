"""T8 end-to-end checks on fully synthetic worlds (no real profile is read): wiring, the stop paths and the caller's
evidential gate flags (code review r1 F1, F4, F7, F11)."""
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.audit import dd_p_policy_value_sensitivity as ddp
from scripts.audit.c1_r4b import run as RUN

SEASONS = (2021, 2022, 2023, 2024, 2025)
LENGTHS = {2021: 44, 2022: 46, 2023: 41, 2024: 48, 2025: 43}       # heterogeneous calendars (> the 30-day late phase)


def build_world(tmp_path, *, drop=None, drop_all=None, surplus_2026=False):
    """drop = (season, seed, day_index) removes one game day from one profile (an unknown-coverage date)."""
    root = tmp_path / "hetzner_results" / "mdp_estpa_run"
    sched_dir = tmp_path / "schedules"
    sched_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    lines, pins, cal_pins = [], {}, {}
    for s in SEASONS:
        start = date(s, 4, 1)
        dates = [start + timedelta(days=i) for i in range(LENGTHS[s]) if i != 5]
        listing = [{"date": str(d), "games": [{"gamePk": int(f"{s % 100}{d.timetuple().tm_yday:03d}{g}"),
                                                "officialDate": str(d), "status": {"detailedState": "Final"}}
                                               for g in range(6)]} for d in dates]
        b = json.dumps({"dates": listing}).encode()
        (sched_dir / f"sched_{s}.json").write_bytes(b)
        pins[s] = hashlib.sha256(b).hexdigest()
        cal_pins[s] = (str(dates[0]), str(dates[-1]), len(dates))
        for seed in range(24):
            rows = []
            for i, d in enumerate(dates):
                if drop == (s, seed, i) or drop_all == (s, i):
                    continue
                for rank in range(1, 6):
                    rows.append({"date": d, "rank": rank, "batter_id": 1000 + rank,
                                 "game_pk": int(f"{s % 100}{d.timetuple().tm_yday:03d}{rank % 6}"),
                                 "p_game_hit": float(np.clip(0.9 - 0.02 * rank + rng.normal(0, 0.03), 0, 1)),
                                 "actual_hit": int(rng.random() < 0.72)})
            f = root / f"box{seed % 3}" / f"simulation_seed{seed}" / f"backtest_{s}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(f, index=False)
    if surplus_2026:
        extra = root / "box0" / "simulation_seed0" / "backtest_2026.parquet"
        pd.DataFrame([{"date": date(2026, 4, 1), "rank": 1, "batter_id": 1, "game_pk": 1, "p_game_hit": 0.8,
                       "actual_hit": 1}]).to_parquet(extra, index=False)
    for name in RUN.RECIPE_FILES:
        (root / name).write_text(json.dumps({"synthetic": name}))
    for f in sorted(root.rglob("*")):
        if f.is_file():
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  mdp_estpa_run/{f.relative_to(root)}")
    man = tmp_path / "hetzner_results" / "mdp_estpa_run.sha256"
    man.write_text("\n".join(lines) + "\n")
    base = np.ones((58, 181, 2, 5), np.int8)
    base[:, :, :, 4] = 2
    bpath = tmp_path / "mdp_policy.npz"
    np.savez(bpath, policy_table=base, boundaries=np.array([0.80, 0.83, 0.85, 0.87]), season_length=np.int64(180),
             optimal_p57=np.float64(0.0))
    tpath = tmp_path / "mdp_tail_policy.npz"
    np.savez(tpath, policy_table=np.ones((58, 58, 29, 2, 1), np.int8), boundaries=np.zeros(0),
             base_policy_sha256=np.array(hashlib.sha256(bpath.read_bytes()).hexdigest()))
    gen_pin, gen = synthetic_generator_check(tmp_path, root)
    return {"root": root, "man": man, "sched": sched_dir, "base": bpath, "tail": tpath, "pins": pins, "cal": cal_pins,
            "runs": tmp_path / "runs", "gen": gen, "gen_pin": gen_pin}


GEN_BLOCK = ("2023-04-08", "2023-04-14")      # seven listed synthetic 2023 days; the synthetic world has ranks 1..5


def synthetic_generator_check(tmp_path, root, **overrides):
    """A generator-check manifest in the shape the real one has, bound to the synthetic seed-0 2023 profile."""
    rel = "box0/simulation_seed0/backtest_2023.parquet"
    m = {"generator_commit": RUN.PROFILE_RECIPE["generator_commit"], "seed": 0, "block": list(GEN_BLOCK),
         "retained": {"path": f"{root.name}/{rel}", "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()},
         "result": {"rows": 35, "rows_mine": 35, "columns_equal": True, "dtype_mismatch": [], "differing_cells": {},
                    "max_abs_diff": {}, "reproduced": True}}
    m.update(overrides)
    path = tmp_path / "generator_check_manifest.json"
    path.write_text(json.dumps(m))
    pin = {"path": "unused", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": 0, "season": 2023,
           "block": GEN_BLOCK, "ranks": 5}
    return pin, path


@pytest.fixture
def patched(monkeypatch):
    def apply(w, *, parity=None):
        monkeypatch.setattr(RUN, "admission_gate", lambda: "f" * 40)
        monkeypatch.setattr(RUN, "RUN_ROOT", w["runs"])
        monkeypatch.setattr(RUN, "GENERATOR_CHECK", w["gen_pin"])
        monkeypatch.setattr(RUN, "owner_gates", lambda text: [])
        monkeypatch.setattr(RUN, "dirty_tree", lambda: [])
        monkeypatch.setattr(RUN, "W0_MANIFEST_SHA", hashlib.sha256(w["man"].read_bytes()).hexdigest())
        monkeypatch.setattr(RUN, "A0_BASE_SHA", hashlib.sha256(w["base"].read_bytes()).hexdigest())
        monkeypatch.setattr(RUN, "A0_TAIL_SHA", hashlib.sha256(w["tail"].read_bytes()).hexdigest())
        monkeypatch.setattr(RUN, "SCHEDULE_PINS", w["pins"])
        monkeypatch.setattr(RUN, "CALENDAR_PINS", w["cal"])
        monkeypatch.setattr(RUN, "REPS", 3)
        monkeypatch.setattr(RUN, "DELTAS", (0.0, 0.10))
        monkeypatch.setattr(RUN, "C_GRID", (0.0,))
        monkeypatch.setattr(RUN, "H_GRID", (0.0,))
        monkeypatch.setattr(RUN, "PROJ_DELTAS", (0.0,))
        monkeypatch.setattr(ddp, "main", parity or (lambda argv: 0))
    return apply


def run_main(w, tmp_path, *extra):
    return RUN.main(["--profiles", str(w["root"]), "--w0-manifest", str(w["man"]), "--schedules", str(w["sched"]),
                     "--a0-base", str(w["base"]), "--a0-tail", str(w["tail"]), "--generator-check", str(w["gen"]),
                     *extra])


def only_run(tmp_path):
    (d,) = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    return d


@pytest.mark.slow
def test_end_to_end_outputs_and_evidential_flags(tmp_path, patched):
    w = build_world(tmp_path)
    patched(w)
    assert run_main(w, tmp_path) == 0
    d = only_run(tmp_path)
    res = json.loads((d / "results.json").read_text())
    assert res["coverage"] == {"complete": True, "by_season": {str(s): {} for s in SEASONS}}
    assert res["validation_ok"] is True and res["projections_ok"] is True and res["self_check"]["ok"] is True
    assert set(res["arms"]) == {"A0", "A1", "A2", "single", "double"}
    assert {"objective", "switch", "adaptation"} <= set(res["contrast"])
    assert res["final_object"]["d_final"] == max(LENGTHS.values())         # max horizon across uneven calendars
    for H in map(str, SEASONS):
        for src in ("fitting_rates", "held_out_rates"):
            (cell,) = res["projections"][H][src]
            assert set(cell) >= {"A0", "A1", "A2", "single", "double", "A2_minus_A0_p57_pp"}
    assert set(res["fix_ladder"]) == {"0_july_semantics", "1_calendar_clock", "2_legal_demotion",
                                      "3_m_state_plumbing", "4_a0_tail_routing"}
    assert res["achieved_haircuts"] and res["census"]["2021|0"]["known_days"] == LENGTHS[2021] - 1
    assert (d / "final_environment.json").exists() and not (d / "parity_inputs").exists()
    assert hashlib.sha256((d / "final_environment.json").read_bytes()).hexdigest() == res["final_object"]["environment_sha256"]
    # code review r2 F6: retained trade evidence, checked by its relations rather than key presence
    metrics = set(res["arms"]["A2"]["0.0"])
    assert "act_play" in metrics
    for arm in res["arms"]:
        for dl, row in res["arms"][arm].items():
            assert row["act_play"] == pytest.approx(row["act_single"] + row["act_double"])
            assert set(res["by_season"][arm][dl]) == metrics
            seas = res["by_season"][arm][dl]["max"]
            assert res["arms_range"][arm][dl]["max"] == [min(seas.values()), max(seas.values())]
    for name in ("objective", "switch", "adaptation"):
        d10 = res["contrast"][name]["d10"]
        assert len(d10["per_replicate"]) == RUN.REPS and len(d10["seasons"]) == len(SEASONS)
        assert np.mean(d10["seasons"]) == pytest.approx(d10["mean"])
        assert d10["range"] == [min(d10["seasons"]), max(d10["seasons"])]
    for b_ in ("A1", "A0"):
        r = res["reach20"]["d10_contrast"][b_]
        assert len(r["per_replicate"]) == RUN.REPS and np.mean(r["per_replicate"]) == pytest.approx(np.mean(r["seasons"]))
        assert np.mean(r["seasons"]) == pytest.approx(res["reach20"]["A2"]["d10"] - res["reach20"][b_]["d10"])
    for H in map(str, SEASONS):
        for src in ("fitting_rates", "held_out_rates"):
            (cell,) = res["projections"][H][src]
            for b_ in ("A0", "A1"):
                if "p57" in cell["A2"] and "p57" in cell[b_]:
                    assert cell[f"A2_minus_{b_}_p57"] == pytest.approx(cell["A2"]["p57"] - cell[b_]["p57"])
                    assert cell[f"A2_minus_{b_}_p57_pp"] == pytest.approx(100 * cell[f"A2_minus_{b_}_p57"])
    assert all("n_primary_hit" in h for h in res["achieved_haircuts"])
    man = json.loads((d / "manifest.json").read_text())
    (schema,) = man["profile_schema"].values()
    assert schema["files"] == 120 and [c for c, _ in schema["schema"]][:6] == ["date", "rank", "batter_id", "game_pk",
                                                                              "p_game_hit", "actual_hit"]
    assert man["generator_check"]["rows"] == 35 and man["generator_check"]["block_dates"][0] == GEN_BLOCK[0]
    assert set(man["closure"]) == set(RUN.CLOSURE) and (d / "CLAIM.json").exists()


@pytest.mark.slow
def test_a_gap_in_a_later_seed_makes_the_disposition_inconclusive(tmp_path, patched):
    w = build_world(tmp_path, drop=(2023, 17, 3))
    patched(w)
    assert run_main(w, tmp_path) == 0
    res = json.loads((only_run(tmp_path) / "results.json").read_text())
    assert res["coverage"]["by_season"]["2023"] == {str(date(2023, 4, 4)): [17]}
    assert res["disposition"]["disposition"] == "inconclusive"
    assert any("coverage" in r for r in res["disposition"]["reasons"])


def test_parity_reads_only_the_verified_files_and_a_failure_stops(tmp_path, patched):
    w = build_world(tmp_path, surplus_2026=True)
    seen = []

    def fake_parity(argv):
        root = argv[argv.index("--root") + 1]
        from pathlib import Path
        seen.extend(sorted(p.name for p in Path(root).rglob("*.parquet")))
        raise AssertionError("anchor mismatch")
    patched(w, parity=fake_parity)
    assert run_main(w, tmp_path) == 2
    d = only_run(tmp_path)
    assert (d / "STOPPED_parity.txt").exists() and not (d / "results.json").exists()
    assert len(seen) == 120 and "backtest_2026.parquet" not in seen


def test_a_failed_self_check_stops_before_any_outcome_read(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path)
    patched(w)
    monkeypatch.setattr(RUN, "self_check", lambda: {"ok": False, "errors": {"solver_emax": 0.1}})
    calls = []
    monkeypatch.setattr(RUN, "read_verified", lambda *a, **k: calls.append(1))
    assert run_main(w, tmp_path) == 2
    assert (only_run(tmp_path) / "STOPPED_self_check.txt").exists() and calls == []


def test_a_nan_solver_mutant_stops_at_the_self_check_before_the_verified_read(tmp_path, patched, monkeypatch):
    """r2 N6: the real self-check (not a stub) must catch NaN values and prevent read_verified."""
    w = build_world(tmp_path)
    patched(w)
    real = RUN.S.solve

    def nan_values(env, **kw):
        sol = real(env, **kw)
        sol.value[...] = np.nan
        return sol
    monkeypatch.setattr(RUN.S, "solve", nan_values)
    calls = []
    monkeypatch.setattr(RUN, "read_verified", lambda *a, **k: calls.append(1))
    assert run_main(w, tmp_path) == 2
    assert (only_run(tmp_path) / "STOPPED_self_check.txt").exists() and calls == []


@pytest.mark.slow
def test_a_zero_leg_rate_stops_at_the_preflight_before_any_replay(tmp_path, patched, monkeypatch):
    """r2 N6: a zero r_bar is caught in the preflight, not after the first corrected replay."""
    w = build_world(tmp_path)
    patched(w)
    real = RUN.F.r_bar
    monkeypatch.setattr(RUN.F, "r_bar", lambda days: real([{**d, "hit2": np.zeros_like(d["hit2"])} for d in days]))
    real_replay, corrected = RUN.R.replay, []

    def spy(sd, arms, **kw):                   # the self-check's synthetic replay passes; corrected ones carry A0
        if "A0" in arms:
            corrected.append(1)
        return real_replay(sd, arms, **kw)
    monkeypatch.setattr(RUN.R, "replay", spy)
    assert run_main(w, tmp_path) == 2
    assert (only_run(tmp_path) / "STOPPED_preflight.txt").exists() and corrected == []
    assert "r_bar" in (only_run(tmp_path) / "STOPPED_preflight.txt").read_text()


def interrupted_run(w, tmp_path, monkeypatch, where):
    """A first run killed after parity (in schema validation) or during the corrected replay."""
    if where == "after_parity":
        monkeypatch.setattr(RUN.D, "validate_profile", lambda df: (_ for _ in ()).throw(KeyboardInterrupt("killed")))
    else:
        real = RUN.R.replay
        monkeypatch.setattr(RUN.R, "replay", lambda sd, arms, **kw: (_ for _ in ()).throw(KeyboardInterrupt("killed"))
                            if "A0" in arms else real(sd, arms, **kw))
    with pytest.raises(KeyboardInterrupt):
        run_main(w, tmp_path)
    monkeypatch.undo()
    d = only_run(tmp_path)
    assert (d / "CLAIM.json").exists() and not list(d.glob("STOPPED_*")) and not (d / "results.json").exists()
    return d


@pytest.mark.parametrize("where", ["after_parity", "during_replay"])
def test_an_interrupted_claimed_run_blocks_another_run(tmp_path, patched, monkeypatch, where):
    """Code review r2 N5: a kill leaves no STOPPED_* or results.json, but its claim still blocks."""
    w = build_world(tmp_path)
    patched(w)
    d = interrupted_run(w, tmp_path, monkeypatch, where)
    patched(w)
    with pytest.raises(SystemExit, match="earlier claimed runs"):
        run_main(w, tmp_path)
    for rec in ({}, {"run": d.name}, {"run": d.name, "reason": "killed", "correction_commit": "0" * 40,
                                      "register_row": "C1-4b-review-r3", "approved_by": "Eric"}):
        (w["runs"] / f"INVALIDATION_{d.name}.json").write_text(json.dumps(rec))   # empty, partial, bad commit
        with pytest.raises(SystemExit, match="earlier claimed runs"):
            run_main(w, tmp_path)


def test_a_valid_invalidation_record_releases_a_claimed_run(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path)
    patched(w)
    d = interrupted_run(w, tmp_path, monkeypatch, "after_parity")
    head = RUN._git(RUN.REPO, "rev-parse", "HEAD").stdout.strip()
    rec = {"run": d.name, "reason": "killed by the operator", "correction_commit": head,
           "register_row": "C1-4b-review-r3", "approved_by": "Eric"}
    assert RUN.validate_invalidation(rec, d.name, RUN._git(RUN.REPO, "show", f"HEAD:{RUN.REGISTER_REL}").stdout) == []
    (w["runs"] / f"INVALIDATION_{d.name}.json").write_text(json.dumps(rec))
    assert RUN.claimed_runs(w["runs"], (RUN.REPO / RUN.REGISTER_REL).read_text()) == []


def test_a_stop_before_the_claim_does_not_block(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path)
    patched(w)
    monkeypatch.setattr(RUN, "self_check", lambda: {"ok": False, "errors": {"solver_emax": 0.1}})
    assert run_main(w, tmp_path) == 2
    assert not (only_run(tmp_path) / "CLAIM.json").exists()
    assert RUN.claimed_runs(w["runs"], "") == []


def test_an_alternate_output_root_is_not_accepted(tmp_path, patched):
    w = build_world(tmp_path)
    patched(w)
    with pytest.raises(SystemExit) as e:
        run_main(w, tmp_path, "--out", str(tmp_path / "elsewhere"))
    assert e.value.code == 2 and not (tmp_path / "elsewhere").exists()


def test_a_held_admission_lock_refuses_a_second_run(tmp_path, patched):
    import fcntl
    w = build_world(tmp_path)
    patched(w)
    w["runs"].mkdir(parents=True)
    with open(w["runs"] / ".admission.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="admission lock"):
            run_main(w, tmp_path)


def test_parity_consumes_the_verified_base_and_a_changed_original_stops(tmp_path, patched, monkeypatch):
    """r2 N4: July parity gets the verified A0 base bytes; an original changed after verification stops the run."""
    w = build_world(tmp_path)
    seen = {}

    def parity(argv):
        from pathlib import Path
        seen["base"] = hashlib.sha256(Path(argv[argv.index("--policy") + 1]).read_bytes()).hexdigest()
        return 0
    patched(w, parity=parity)
    real = RUN._load_a0

    def load_then_change(base, tail):
        out = real(base, tail)
        Path(base).write_bytes(Path(base).read_bytes() + b"changed")
        return out
    monkeypatch.setattr(RUN, "_load_a0", load_then_change)
    assert run_main(w, tmp_path) == 2
    assert seen["base"] == RUN.A0_BASE_SHA
    assert (only_run(tmp_path) / "STOPPED_provenance.txt").exists()


@pytest.mark.parametrize("override", [{"seed": 1}, {"block": ["2023-04-08", "2023-04-13"]},
                                      {"generator_commit": "0" * 40}, {"status": "unequal"},
                                      {"result": {"rows": 0, "rows_mine": 0, "columns_equal": True, "dtype_mismatch": [],
                                                  "differing_cells": {}, "max_abs_diff": {}, "reproduced": True}}])
def test_a_generator_check_that_does_not_witness_the_registered_block_stops(tmp_path, patched, override):
    """r2 N3: the runner consumes the pinned check manifest field by field (pin re-made so only the fields differ)."""
    w = build_world(tmp_path)
    w["gen_pin"], w["gen"] = synthetic_generator_check(tmp_path, w["root"], **override)
    patched(w)
    with pytest.raises(RUN.ProvenanceError, match="generator check"):
        run_main(w, tmp_path)


def test_a_generator_check_manifest_off_its_pin_stops(tmp_path, patched):
    w = build_world(tmp_path)
    patched(w)
    w["gen"].write_text(w["gen"].read_text() + " ")
    with pytest.raises(RUN.ProvenanceError, match="pin"):
        run_main(w, tmp_path)


def test_a_reference_block_missing_a_rank_stops(tmp_path, patched, monkeypatch):
    """The block's dates and ranks are read from the verified bytes (date/rank columns only)."""
    w = build_world(tmp_path)
    patched(w)
    monkeypatch.setitem(w["gen_pin"], "ranks", 6)        # the synthetic profile holds ranks 1..5
    w["gen"].write_text(w["gen"].read_text().replace('"rows": 35, "rows_mine": 35', '"rows": 42, "rows_mine": 42'))
    monkeypatch.setitem(w["gen_pin"], "sha256", hashlib.sha256(w["gen"].read_bytes()).hexdigest())
    with pytest.raises(RUN.ProvenanceError, match="ranks"):
        run_main(w, tmp_path)


def test_a_tampered_profile_stops_before_any_decode(tmp_path, patched):
    w = build_world(tmp_path)
    patched(w)
    f = next(w["root"].rglob("backtest_2023.parquet"))
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(RUN.ProvenanceError):
        run_main(w, tmp_path)


def test_an_unpinned_schedule_stops(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path)
    patched(w)
    monkeypatch.setattr(RUN, "SCHEDULE_PINS", {**w["pins"], 2023: "0" * 64})
    with pytest.raises(RUN.ProvenanceError):
        run_main(w, tmp_path)




@pytest.mark.slow
def test_an_owner_ruled_no_play_final_day_leaves_the_calendar_and_coverage_complete(tmp_path, patched, monkeypatch):
    """Register row C1-4b-2023-10-02's shape: the schedule lists a final game day that no profile covers. Excluded, the
    calendar ends a day earlier and coverage is complete; the manifest records the exclusion."""
    w = build_world(tmp_path, drop_all=(2023, LENGTHS[2023] - 2))   # the last listed day (day 5 is never listed)
    patched(w)
    opening, final, n = w["cal"][2023]
    last = date.fromisoformat(final)
    monkeypatch.setattr(RUN, "EXCLUDED_CONTEST_DATES", {final: "synthetic no-play ruling"})
    monkeypatch.setattr(RUN, "CALENDAR_PINS", {**w["cal"], 2023: (opening, str(last - timedelta(days=1)), n - 1)})
    assert run_main(w, tmp_path) == 0
    run = only_run(tmp_path)
    assert json.loads((run / "results.json").read_text())["coverage"]["complete"] is True
    man = json.loads((run / "manifest.json").read_text())
    assert man["excluded_contest_dates"] == {final: "synthetic no-play ruling"}
    assert man["calendars"]["2023"]["final"] == str(last - timedelta(days=1))


def test_without_the_exclusion_the_uncovered_final_day_does_not_match_the_pin(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path, drop_all=(2023, LENGTHS[2023] - 2))   # the last listed day (day 5 is never listed)
    patched(w)
    opening, final, n = w["cal"][2023]
    monkeypatch.setattr(RUN, "EXCLUDED_CONTEST_DATES", {})
    monkeypatch.setattr(RUN, "CALENDAR_PINS", {**w["cal"], 2023: (opening, str(date.fromisoformat(final) - timedelta(days=1)), n - 1)})
    with pytest.raises(RUN.ProvenanceError):
        run_main(w, tmp_path)

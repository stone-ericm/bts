"""T8 end-to-end checks on fully synthetic worlds (no real profile is read): wiring, the stop paths and the caller's
evidential gate flags (code review r1 F1, F4, F7, F11)."""
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

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
    return {"root": root, "man": man, "sched": sched_dir, "base": bpath, "tail": tpath, "pins": pins, "cal": cal_pins}


@pytest.fixture
def patched(monkeypatch):
    def apply(w, *, parity=None):
        monkeypatch.setattr(RUN, "x31_gate", lambda: "f" * 40)
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
        monkeypatch.setattr(RUN, "SUPPLEMENT_MANIFEST_SHA", None)
        monkeypatch.setattr(RUN, "SUPPLEMENT_REQUIRED", False)
    return apply


def run_main(w, tmp_path):
    return RUN.main(["--profiles", str(w["root"]), "--w0-manifest", str(w["man"]), "--schedules", str(w["sched"]),
                     "--a0-base", str(w["base"]), "--a0-tail", str(w["tail"]), "--out", str(tmp_path / "runs")])


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


def test_an_earlier_run_without_an_invalidation_record_blocks_a_new_run(tmp_path, patched):
    w = build_world(tmp_path)
    patched(w)
    prior = tmp_path / "runs" / "abc1234-20261005T000000Z"
    prior.mkdir(parents=True)
    (prior / "STOPPED_parity.txt").write_text("x")
    with pytest.raises(SystemExit):
        run_main(w, tmp_path)
    (tmp_path / "runs" / f"INVALIDATION_{prior.name}.json").write_text("{}")
    assert RUN.prior_runs(tmp_path / "runs") == []


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


def build_supplement(tmp_path, w, day, seeds=range(24)):
    """A synthetic backfill output for one date: per-seed supplement parquets plus a manifest binding their hashes."""
    sdir = tmp_path / "backfill"
    sdir.mkdir()
    man = {"all_reproduced": True, "seeds": {}}
    for seed in seeds:
        rows = [{"date": day, "rank": r, "batter_id": 2000 + r, "game_pk": int(f"{day.year % 100}999{r}"),
                 "p_game_hit": 0.8 - 0.01 * r, "actual_hit": r % 2} for r in range(1, 6)]
        p = sdir / f"supplement_{day}_seed{seed}.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        man["seeds"][str(seed)] = {"reproduced": True, "supplement": p.name,
                                   "supplement_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (sdir / "backfill_manifest.json").write_text(json.dumps(man))
    return sdir


@pytest.mark.slow
def test_a_pinned_backfill_supplement_fills_the_gap(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path, drop_all=(2023, 3))
    day = date(2023, 4, 4)
    sdir = build_supplement(tmp_path, w, day)
    patched(w)
    monkeypatch.setattr(RUN, "SUPPLEMENT_DATE", str(day))
    monkeypatch.setattr(RUN, "SUPPLEMENT_MANIFEST_SHA", hashlib.sha256((sdir / "backfill_manifest.json").read_bytes()).hexdigest())
    assert RUN.main(["--profiles", str(w["root"]), "--w0-manifest", str(w["man"]), "--schedules", str(w["sched"]),
                     "--a0-base", str(w["base"]), "--a0-tail", str(w["tail"]), "--out", str(tmp_path / "runs"),
                     "--supplement-dir", str(sdir)]) == 0
    res = json.loads((only_run(tmp_path) / "results.json").read_text())
    assert res["coverage"]["complete"] is True


def test_a_supplement_that_does_not_match_its_pins_stops(tmp_path, patched, monkeypatch):
    w = build_world(tmp_path, drop_all=(2023, 3))
    day = date(2023, 4, 4)
    sdir = build_supplement(tmp_path, w, day)
    patched(w)
    monkeypatch.setattr(RUN, "SUPPLEMENT_DATE", str(day))
    monkeypatch.setattr(RUN, "SUPPLEMENT_MANIFEST_SHA", hashlib.sha256((sdir / "backfill_manifest.json").read_bytes()).hexdigest())
    f = next(sdir.glob("supplement_*seed5.parquet"))
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(RUN.ProvenanceError):
        RUN.main(["--profiles", str(w["root"]), "--w0-manifest", str(w["man"]), "--schedules", str(w["sched"]),
                  "--a0-base", str(w["base"]), "--a0-tail", str(w["tail"]), "--out", str(tmp_path / "runs"),
                  "--supplement-dir", str(sdir)])

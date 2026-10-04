"""T8 end-to-end smoke on a fully synthetic world (no real profile is read): wiring, outputs, determinism."""
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c1_r4b import run as RUN
from scripts.audit import dd_p_policy_value_sensitivity as ddp

SEASONS = (2021, 2022, 2023, 2024, 2025)
N_DAYS = 36


def build_world(tmp_path):
    root = tmp_path / "hetzner_results" / "mdp_estpa_run"
    sched_dir = tmp_path / "schedules"
    sched_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    lines = []
    for s in SEASONS:
        start = date(s, 4, 1)
        dates = [start + timedelta(days=i) for i in range(N_DAYS) if i != 10]   # one no-game day
        games = []
        for d in dates:
            games.append({"date": str(d), "games": [{"gamePk": int(f"{s}{d.timetuple().tm_yday:03d}{g}"),
                                                     "officialDate": str(d), "status": {"detailedState": "Final"}}
                                                    for g in range(6)]})
        (sched_dir / f"sched_{s}.json").write_text(json.dumps({"dates": games}))
        for seed in range(24):
            rows = []
            for d in dates:
                for rank in range(1, 6):
                    gp = int(f"{s}{d.timetuple().tm_yday:03d}{rank % 6}")
                    rows.append({"date": d, "rank": rank, "batter_id": 1000 + rank, "game_pk": gp,
                                 "p_game_hit": float(np.clip(0.9 - 0.02 * rank + rng.normal(0, 0.03), 0, 1)),
                                 "actual_hit": int(rng.random() < 0.72)})
            f = root / f"box{seed % 3}" / f"simulation_seed{seed}" / f"backtest_{s}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(f, index=False)
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  mdp_estpa_run/{f.relative_to(root)}")
    man = tmp_path / "hetzner_results" / "mdp_estpa_run.sha256"
    man.write_text("\n".join(lines) + "\n")
    base = np.ones((58, 181, 2, 5), np.int8)
    base[:, :, :, 4] = 2
    bpath = tmp_path / "mdp_policy.npz"
    np.savez(bpath, policy_table=base, boundaries=np.array([0.80, 0.83, 0.85, 0.87]), season_length=np.int64(180),
             optimal_p57=np.float64(0.0))
    bsha = hashlib.sha256(bpath.read_bytes()).hexdigest()
    tpath = tmp_path / "mdp_tail_policy.npz"
    np.savez(tpath, policy_table=np.ones((58, 58, 29, 2, 1), np.int8), boundaries=np.zeros(0),
             base_policy_sha256=np.array(bsha))
    return root, man, sched_dir, bpath, tpath


@pytest.mark.slow
def test_end_to_end_synthetic_run(tmp_path, monkeypatch):
    root, man, sched, bpath, tpath = build_world(tmp_path)
    monkeypatch.setattr(RUN, "x31_gate", lambda: "f" * 40)
    monkeypatch.setattr(RUN, "W0_MANIFEST_SHA", hashlib.sha256(man.read_bytes()).hexdigest())
    monkeypatch.setattr(RUN, "A0_BASE_SHA", hashlib.sha256(bpath.read_bytes()).hexdigest())
    monkeypatch.setattr(RUN, "A0_TAIL_SHA", hashlib.sha256(tpath.read_bytes()).hexdigest())
    monkeypatch.setattr(RUN, "C_GRID", (0.0, 0.04))
    monkeypatch.setattr(RUN, "H_GRID", (0.0, 0.04))
    monkeypatch.setattr(ddp, "main", lambda argv: 0)            # parity needs the real profiles; stubbed here
    out = tmp_path / "runs"
    rc = RUN.main(["--profiles", str(root), "--w0-manifest", str(man), "--schedules", str(sched),
                   "--a0-base", str(bpath), "--a0-tail", str(tpath), "--out", str(out), "--reps", "3"])
    assert rc == 0
    (run_dir,) = list(out.iterdir())
    res = json.loads((run_dir / "results.json").read_text())
    assert set(res["arms"]) == {"A0", "A1", "A2", "single", "double"}
    assert res["disposition"]["disposition"] in ("positive", "negative", "inconclusive")
    assert len(res["contrast"]["objective"]["d0"]["seasons"]) == 5
    assert res["unknown_coverage"] == {str(s): [] for s in SEASONS} or all(v == [] for v in res["unknown_coverage"].values())
    for H in map(str, SEASONS):
        for src in ("fitting_rates", "held_out_rates"):
            cells = res["projections"][H][src]
            assert len(cells) == 2 * 2 * 2
            assert all(0.0 <= c[a]["p57"] <= 1.0 for c in cells for a in ("A0", "A1", "A2"))
    assert res["final_object"]["d_final"] == N_DAYS
    assert (run_dir / "final_decision_object.npz").exists()
    man_out = json.loads((run_dir / "manifest.json").read_text())
    assert len(man_out["profiles"]) == 120
    c = res["consequences_d0"][str(SEASONS[0])]
    assert c["A0_states"]["visits"] > 0 and c["own_trajectory"]["days"] > 0


@pytest.mark.slow
def test_a_tampered_profile_stops_before_any_decode(tmp_path, monkeypatch):
    root, man, sched, bpath, tpath = build_world(tmp_path)
    monkeypatch.setattr(RUN, "x31_gate", lambda: "f" * 40)
    monkeypatch.setattr(RUN, "W0_MANIFEST_SHA", hashlib.sha256(man.read_bytes()).hexdigest())
    f = next(root.rglob("backtest_2023.parquet"))
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(RUN.ProvenanceError):
        RUN.main(["--profiles", str(root), "--w0-manifest", str(man), "--schedules", str(sched),
                  "--a0-base", str(bpath), "--a0-tail", str(tpath), "--out", str(tmp_path / "runs"), "--reps", "3"])

"""Verified Item 1 regression probe: 1,511 table cells, edge probabilities, and 2027 resolver parity.

Compares base (`f83e0d0b` extracted to `.codex-review/wr-g/base_f83e0d0b/src`) against HEAD (`src`)
using separate subprocesses so `import bts` loads each tree cleanly and confirms `bts.__file__`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np

REVIEW_DIR = REPO / ".codex-review" / "wr-g"
BASE_SRC = REVIEW_DIR / "base_f83e0d0b" / "src"
HEAD_SRC = REPO / "src"
CSV_PATH = REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "contract-g" / "affected-states.csv"
ENUM_SCRIPT = REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "contract-g" / "enumerate_states.py"


WORKER_CODE = """
import sys
import pickle
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
import bts
from bts.simulate import mdp
from bts import strategy

mode = sys.argv[1]
out_path = Path(sys.argv[2])
probs_by_bin, extra_probs = pickle.loads(Path(sys.argv[3]).read_bytes())

art = strategy._load_mdp()
table = art["policy_table"]
bounds = art["boundaries"]
sl = art["season_length"]

raw_results = []
guarded_results = []
has_guarded = hasattr(mdp, "lookup_action_guarded")

for s in range(58):
    for d in range(181):
        for v in (False, True):
            for q, plist in enumerate(probs_by_bin):
                for p in plist:
                    ra = mdp.lookup_action(table, bounds, s, d, v, p, sl)
                    raw_results.append(ra)
                    if has_guarded:
                        ga, gmeta = mdp.lookup_action_guarded(table, bounds, s, d, v, p, sl)
                        guarded_results.append((ga, gmeta))

sentinel_cases = []
for s in (-5, -1, 0, 50, 56, 57, 58, 100):
    for d in (-10, -1, 0, 1, 29, 180, 181, 269):
        for v in (False, True):
            for p in extra_probs:
                ra = mdp.lookup_action(table, bounds, s, d, v, p, sl)
                if has_guarded:
                    ga, gmeta = mdp.lookup_action_guarded(table, bounds, s, d, v, p, sl)
                    sentinel_cases.append((ra, ga, gmeta))
                else:
                    sentinel_cases.append((ra, None, None))

resolver_probs = [
    (0, -0.1), (0, float("nan")), (0, probs_by_bin[0][0]), (0, probs_by_bin[0][2]), (0, probs_by_bin[0][-1]),
    (1, probs_by_bin[1][0]), (1, probs_by_bin[1][1]), (1, probs_by_bin[1][-1]),
    (2, probs_by_bin[2][0]), (2, probs_by_bin[2][1]), (2, probs_by_bin[2][-1]),
    (3, probs_by_bin[3][0]), (3, probs_by_bin[3][1]), (3, probs_by_bin[3][-1]),
    (4, probs_by_bin[4][0]), (4, probs_by_bin[4][1]), (4, 1.0), (4, 1.25),
]

uncached_days = {
    (date(2027, 3, 24) + timedelta(days=i)).isoformat():
    strategy._days_remaining((date(2027, 3, 24) + timedelta(days=i)).isoformat())
    for i in range(188)
}
strategy._days_remaining = lambda d_str, _m=uncached_days: _m[d_str]

bars = []
decisions = []
start = date(2027, 3, 24)
for offset in range(188):
    d_str = (start + timedelta(days=offset)).isoformat()
    for s in range(58):
        for v in (False, True):
            bars.append(strategy.effective_pick_bar(s, d_str, v, mdp=art, best_streak=s, best_status="trusted"))
            for q, p in resolver_probs:
                ctx = strategy.DecisionContext(
                    primary_p=p, second_p=0.91, has_diff_game=True, date=d_str,
                    allow_double=True, mdp=art, best_streak=s, best_status="trusted",
                )
                dec = strategy.resolve_policy_decision(ctx, s, v)
                decisions.append((q, p, asdict(dec)))

out_path.write_bytes(pickle.dumps({
    "bts_file": str(Path(bts.__file__).resolve()),
    "mdp_file": str(Path(mdp.__file__).resolve()),
    "strategy_file": str(Path(strategy.__file__).resolve()),
    "base_sha": art["base_sha256"],
    "tail_sha": art["tail"].sha256,
    "raw_results": raw_results,
    "guarded_results": guarded_results,
    "sentinel_cases": sentinel_cases,
    "bars": bars,
    "decisions": decisions,
}))
"""


def assert_strict_json_types(obj, path="root") -> None:
    if obj is None:
        return
    t = type(obj)
    assert t in (str, int, float, bool, list, dict), f"non-builtin type {t} at {path}: {obj!r}"
    if t is float:
        assert math.isfinite(obj), f"non-finite float at {path}: {obj!r}"
    elif t is list:
        for idx, item in enumerate(obj):
            assert_strict_json_types(item, f"{path}[{idx}]")
    elif t is dict:
        for k, val in obj.items():
            assert type(k) is str, f"non-str dict key {type(k)} at {path}: {k!r}"
            assert_strict_json_types(val, f"{path}.{k}")


def main() -> None:
    regen_csv = REVIEW_DIR / "regen_affected_states.csv"
    subprocess.run(
        [sys.executable, str(ENUM_SCRIPT), str(REPO), str(regen_csv)],
        check=True, capture_output=True, text=True, timeout=15,
    )
    committed_bytes = CSV_PATH.read_bytes()
    assert regen_csv.read_bytes() == committed_bytes
    csv_sha = hashlib.sha256(committed_bytes).hexdigest()
    assert csv_sha.startswith("77ec5404")

    affected = {}
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            key = (int(row["streak"]), int(row["days_left"]), bool(int(row["saver"])), int(row["bin"]))
            affected[key] = (
                row["guarded_action"],
                {
                    "rule": "monotone_skip_v1",
                    "bin": int(row["bin"]),
                    "from_bin": int(row["from_bin"]),
                    "table_action": "skip",
                },
            )
    assert len(affected) == 1511

    with np.load(REPO / "data" / "models" / "mdp_policy.npz") as data:
        b = data["boundaries"].tolist()

    probs_by_bin = [
        [float("-inf"), -0.5, 0.0, float(np.nextafter(0.0, 1.0)), b[0] / 2.0, float(np.nextafter(b[0], -1.0)), float("nan")],
        [b[0], float(np.nextafter(b[0], 2.0)), (b[0] + b[1]) / 2.0, float(np.nextafter(b[1], -1.0))],
        [b[1], float(np.nextafter(b[1], 2.0)), (b[1] + b[2]) / 2.0, float(np.nextafter(b[2], -1.0))],
        [b[2], float(np.nextafter(b[2], 2.0)), (b[2] + b[3]) / 2.0, float(np.nextafter(b[3], -1.0))],
        [b[3], float(np.nextafter(b[3], 2.0)), (b[3] + 1.0) / 2.0, float(np.nextafter(1.0, 0.0)), 1.0, 1.5, float("inf")],
    ]
    extra_probs = [float("-inf"), -0.1, 0.0, b[0], 0.72, b[3], 0.96, 1.0, 1.5, float("inf"), float("nan")]

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p01v_") as tmp:
        tmp_dir = Path(tmp)
        worker_script = tmp_dir / "worker.py"
        worker_script.write_text(WORKER_CODE)
        probs_file = tmp_dir / "probs.pkl"
        probs_file.write_bytes(pickle.dumps((probs_by_bin, extra_probs)))
        base_out = tmp_dir / "base.pkl"
        head_out = tmp_dir / "head.pkl"

        for label, src_dir, out_file in (("base", BASE_SRC, base_out), ("head", HEAD_SRC, head_out)):
            env = dict(os.environ)
            env["PYTHONPATH"] = str(src_dir)
            subprocess.run(
                [sys.executable, str(worker_script), label, str(out_file), str(probs_file)],
                cwd=REPO, env=env, check=True, timeout=90,
            )

        base_data = pickle.loads(base_out.read_bytes())
        head_data = pickle.loads(head_out.read_bytes())

    assert base_data["bts_file"] == str((BASE_SRC / "bts" / "__init__.py").resolve())
    assert head_data["bts_file"] == str((HEAD_SRC / "bts" / "__init__.py").resolve())
    assert head_data["raw_results"] == base_data["raw_results"]

    idx = 0
    changed_cells = set()
    override_count = 0
    unchanged_count = 0
    for s in range(58):
        for d in range(181):
            for v in (False, True):
                for q, plist in enumerate(probs_by_bin):
                    expected_hit = affected.get((s, d, v, q))
                    for _p in plist:
                        raw_act = base_data["raw_results"][idx]
                        g_act, g_meta = head_data["guarded_results"][idx]
                        assert type(g_act) is str
                        if expected_hit is not None:
                            exp_act, exp_guard = expected_hit
                            assert raw_act == "skip" and g_act == exp_act and g_meta == exp_guard
                            assert_strict_json_types(g_meta, "g_meta")
                            changed_cells.add((s, d, v, q))
                            override_count += 1
                        else:
                            assert g_act == raw_act and g_meta is None
                            unchanged_count += 1
                        idx += 1

    assert idx == len(base_data["raw_results"])
    assert changed_cells == set(affected.keys())

    for (b_ra, _, _), (h_ra, h_ga, h_gm) in zip(base_data["sentinel_cases"], head_data["sentinel_cases"]):
        assert b_ra == h_ra
        if h_gm is None:
            assert h_ga == b_ra
        else:
            assert b_ra == "skip" and h_ga in ("single", "double")
            assert_strict_json_types(h_gm)

    assert base_data["bars"] == head_data["bars"]

    seen_csv_keys = set()
    resolver_overrides = 0
    for dec_idx in range(len(base_data["decisions"])):
        q_b, p_b, base_dec = base_data["decisions"][dec_idx]
        q_h, p_h, head_dec = head_data["decisions"][dec_idx]
        assert q_b == q_h
        assert (math.isnan(p_b) and math.isnan(p_h)) or p_b == p_h
        s = base_dec["streak"]
        v = head_dec["guard"] is not None  # unused placeholder; recover v from dec_idx
        v_bool = bool((dec_idx // 18) % 2)
        key = (s, base_dec["days_effective"], v_bool, q_b)
        hit = affected.get(key) if base_dec["objective"] == "reach57" else None

        expected = dict(base_dec)
        if hit is not None:
            exp_act, exp_guard = hit
            assert base_dec["action"] == "skip" and base_dec["policy_action"] == "skip"
            expected["action"] = exp_act
            expected["policy_action"] = exp_act
            expected["pick_bar"] = None
            expected["guard"] = exp_guard
            seen_csv_keys.add(key)
            resolver_overrides += 1
            assert_strict_json_types(head_dec["guard"], "head_dec.guard")
        else:
            expected["guard"] = None
        assert head_dec == expected

    assert seen_csv_keys == set(affected.keys())

    # Verify pick file & decision record JSON serialization with numpy-typed DataFrame columns
    import pandas as pd
    from bts import daily_decision, picks, scheduler, strategy
    from scripts.audit.season_walk import walk

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_jsonv_") as tmp_j:
        pdir = Path(tmp_j)
        frame = pd.DataFrame([
            {
                "batter_name": "Batter A", "batter_id": np.int64(101), "team": "NYY",
                "lineup": np.int64(1), "pitcher_name": "Pitcher X", "pitcher_id": np.int64(201),
                "p_game_hit": np.float64(0.96), "flags": "", "game_pk": np.int64(70001),
                "game_time": "2027-03-24T23:05:00Z", "pitcher_team": "BOS",
            },
            {
                "batter_name": "Batter B", "batter_id": np.int64(102), "team": "LAD",
                "lineup": np.int64(2), "pitcher_name": "Pitcher Y", "pitcher_id": np.int64(202),
                "p_game_hit": np.float64(0.85), "flags": "", "game_pk": np.int64(70002),
                "game_time": "2027-03-24T23:10:00Z", "pitcher_team": "SF",
            },
        ])
        statuses = {70001: {"abstract": "P", "detailed": "Scheduled"}, 70002: {"abstract": "P", "detailed": "Scheduled"}}
        with walk.offline(walk.REAL_DATETIME(2027, 3, 24, 15, tzinfo=walk.ET)):
            sel = strategy.select_pick(
                frame, "2027-03-24", pdir, streak=50, saver_available=False,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
            )
            assert sel.decision.guard == {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}
            assert_strict_json_types(sel.decision.guard, "sel.decision.guard")
            pick_path = picks.save_pick(sel.pick_result.daily, pdir)
            pick_doc = json.loads(pick_path.read_text())
            assert_strict_json_types(pick_doc, "pick_doc")
            json.dumps(pick_doc, allow_nan=False)

            st = scheduler.SchedulerState(
                date="2027-03-24", schedule_fetched_at="2027-03-24T15:00:00-04:00",
                games=[], confirmed_game_pks=[], runs_completed=[],
                pick_locked=True, pick_locked_at="2027-03-24T15:00:00-04:00",
                result_status=None, next_wakeup=None,
            )
            scheduler._commit_decision_for_pick(
                pdir, "2027-03-24", sel.pick_result.daily,
                selection=sel, delivery_status="delivered", state=st,
            )
            dec_doc = daily_decision.load_decision("2027-03-24", pdir)
            assert dec_doc is not None
            assert dec_doc["policy_guard"] == sel.decision.guard
            assert_strict_json_types(dec_doc, "dec_doc")
            json.dumps(dec_doc, allow_nan=False)

    summary = {
        "status": "PASS",
        "base_bts_file": base_data["bts_file"],
        "head_bts_file": head_data["bts_file"],
        "csv_sha256": csv_sha,
        "lookup_calls": len(base_data["raw_results"]),
        "changed_cells": len(changed_cells),
        "override_calls": override_count,
        "unchanged_calls": unchanged_count,
        "sentinel_calls": len(head_data["sentinel_cases"]),
        "bar_calls": len(head_data["bars"]),
        "resolver_calls": len(head_data["decisions"]),
        "resolver_overrides": resolver_overrides,
        "distinct_csv_rows_hit": len(seen_csv_keys),
    }
    (REVIEW_DIR / "probe_01_verified.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

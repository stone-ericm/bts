"""Adversarial probe 3b: below-boundary and zero-day parity against archived base 0a2bec10 (A6)."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import csv
from dataclasses import asdict, replace
import datetime as _dtmod
from datetime import date, datetime, time, timedelta, timezone
from functools import cache
import hashlib
from itertools import product
import json
import logging
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch"
ET = ZoneInfo("America/New_York")
REAL_DATETIME = _dtmod.datetime


class _Clock(REAL_DATETIME):
    at: REAL_DATETIME | None = None

    @classmethod
    def now(cls, tz=None):
        val = cls.at
        return val.astimezone(tz) if tz else val.astimezone(ET).replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.at.astimezone(timezone.utc).replace(tzinfo=None)


def _refuse_net(*_a, **_kw):
    raise OSError("network refused")


@contextmanager
def _pinned_offline(at: REAL_DATETIME):
    import bts.picks, bts.strategy  # noqa: F401
    _Clock.at = at
    with ExitStack() as stack:
        stack.enter_context(patch.object(_dtmod, "datetime", _Clock))
        for name, mod in list(sys.modules.items()):
            if (name == "bts" or name.startswith("bts.")) and getattr(mod, "datetime", None) is REAL_DATETIME:
                stack.enter_context(patch.object(mod, "datetime", _Clock))
        for m in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, m, _refuse_net))
        stack.enter_context(patch.object(socket, "getaddrinfo", _refuse_net))
        yield


def _synthetic_frame(day: date, top_p: float) -> pd.DataFrame:
    start = REAL_DATETIME.combine(day, time(19, 5), ET).astimezone(timezone.utc)
    rows = []
    for i, p in enumerate((top_p, round(top_p - 0.01, 4))):
        rows.append({
            "batter_name": f"Batter {i}",
            "batter_id": 100 + i,
            "team": f"T{i}",
            "lineup": 1 + i,
            "pitcher_name": f"Pitcher {i}",
            "pitcher_id": 900 + i,
            "p_game_hit": p,
            "flags": "",
            "game_pk": 70000 + i,
            "game_time": start.isoformat().replace("+00:00", "Z"),
            "pitcher_team": f"U{i}",
        })
    return pd.DataFrame(rows)


def run_worker(src_root: Path, work_tmp: Path) -> None:
    import bts
    from bts import daily_decision, picks, scheduler, strategy
    from bts.simulate import tail_policy

    resolved_init = Path(bts.__file__).resolve()
    assert resolved_init == (src_root / "bts" / "__init__.py").resolve(), (resolved_init, src_root)
    print(str(resolved_init), flush=True)

    logging.disable(logging.CRITICAL)
    strategy._days_remaining = cache(strategy._days_remaining)
    artifact = strategy._load_mdp()
    tail = artifact["tail"]

    freq, hit, both = [0.5, 0.5], [0.75, 0.75], [0.5, 0.5]
    pol2 = tail_policy.solve_emax_season_best(freq, hit, both).policy
    man2 = tail_policy.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2])
    two_bin = replace(
        tail, boundaries=[0.82], policy_table=pol2, bin_freq=freq, bin_p_hit=hit, bin_p_both=both, manifest=man2
    )
    tail_policy.validate_tail_policy(two_bin, expected_base_sha=artifact["base_sha256"])

    modes = {
        "shipped": artifact,
        "tail_missing": artifact | {"tail": None, "tail_error": "tail missing"},
        "tail_corrupt": artifact | {"tail": None, "tail_error": "tail corrupt"},
        "base_missing_tail_present": artifact | {"policy_table": None, "base_sha256": None},
        "sha_mismatch": artifact | {"tail": replace(tail, base_policy_sha256="f" * 64)},
        "none": None,
        "two_bin_tail": artifact | {"tail": two_bin},
        "horizon_90": artifact | {"season_length": 90},
        "horizon_15": artifact | {"season_length": 15},
        "horizon_0": artifact | {"season_length": 0},
        "horizon_neg1": artifact | {"season_length": -1},
        "empty_bounds": artifact | {"boundaries": []},
    }

    digest = hashlib.sha256()
    counts = {
        "shipped_grid": 0,
        "guard_cells": 0,
        "heuristic_edges": 0,
        "tail_cells": 0,
        "horizons_years_zero_days": 0,
        "bar_edge_inputs": 0,
        "live_select_save_commit": 0,
    }
    raw_days_seen: set[int] = set()
    eff_days_seen: set[int] = set()

    def record_case(tag: str, s: int, d_str: str, saver: bool, p: float, p2: float | None,
                    has_diff: bool, allow: bool, mode_name: str, best: int | None, status: str | None):
        m = modes[mode_name]
        raw = strategy._days_remaining(d_str)
        ctx = strategy.DecisionContext(p, p2, has_diff, d_str, allow, m, best, status)
        dec = strategy.resolve_policy_decision(ctx, s, saver)
        bar = strategy.effective_pick_bar(s, d_str, saver, m, best, status)
        raw_days_seen.add(raw)
        eff_days_seen.add(dec.days_effective)
        payload = [
            tag, s, d_str, raw, saver, p, p2, has_diff, allow, mode_name, best, status,
            asdict(dec), bar, type(bar).__name__,
        ]
        digest.update(json.dumps(payload, sort_keys=True).encode())
        counts[tag] += 1
        return dec, bar

    # 1. Shipped below-boundary grid: every streak 0..56 x raw_days 0..271 x saver x bin edges
    end_2028 = date(2028, 9, 28)
    dates_272 = [(end_2028 - timedelta(days=r)).isoformat() for r in range(272)]
    edges = (*artifact["boundaries"], 0.80, 0.82)
    probs = [
        0.0, 0.01, 0.99, 1.0,
        *(x for e in edges for x in (math.nextafter(e, 0.0), e, math.nextafter(e, 1.0))),
    ]
    bests = [(None, None), *product((0, 10, 28, 55, 56, 57), ("trusted", "untrusted"))]

    for raw in range(272):
        d_str = dates_272[raw]
        for s in range(57):
            for saver in (False, True):
                for qi, p in enumerate(probs):
                    idx = raw * 7 + s * 3 + int(saver) + qi
                    best_v, best_st = bests[idx % len(bests)]
                    has_diff = (idx % 3) != 0
                    p2 = 0.88 if has_diff else None
                    allow = (idx % 2) == 0
                    record_case("shipped_grid", s, d_str, saver, p, p2, has_diff, allow, "shipped", best_v, best_st)

    # 2. All 1,511 G guard cells x all 4 clamp combinations = 6,044 checks
    guard_csv = REPO / "docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv"
    unique_guards = 0
    with guard_csv.open() as fh:
        for row in csv.DictReader(fh):
            s = int(row["streak"])
            raw = int(row["days_left"])
            saver = bool(int(row["saver"]))
            q = int(row["bin"])
            d_str = (date(2027, 9, 27) - timedelta(days=raw)).isoformat()
            p = artifact["boundaries"][q - 1]
            unique_guards += 1
            for allow, has_diff in product((True, False), (True, False)):
                dec, _ = record_case(
                    "guard_cells", s, d_str, saver, p, 0.92 if has_diff else None,
                    has_diff, allow, "shipped", s, "trusted",
                )
                assert dec.guard is not None and dec.policy_action == row["guarded_action"]
    assert unique_guards == 1511

    # 3. Heuristic 0.80 edge and partner-product thresholds (including primary_p < 1.0!)
    for mode_name in ("none", "base_missing_tail_present"):
        for s, cut in ((0, 0.55), (9, 0.55), (10, 0.60), (15, 0.60), (16, 0.65), (45, 0.65), (46, None), (56, None)):
            for p1 in (math.nextafter(0.80, 0.0), 0.80, math.nextafter(0.80, 1.0), 0.85, 1.0):
                p2_cands = [None, 0.50, 0.99]
                if cut is not None and p1 >= 0.80:
                    ratio = cut / p1
                    if ratio <= 1.0:
                        p2_cands.extend([
                            math.nextafter(ratio, 0.0),
                            ratio,
                            math.nextafter(ratio, 1.0),
                        ])
                for p2 in p2_cands:
                    for has_diff, allow, saver in product((True, False), (True, False), (False, True)):
                        record_case(
                            "heuristic_edges", s, "2027-07-15", saver, p1, p2,
                            has_diff and (p2 is not None), allow, mode_name, s, "trusted",
                        )

    # 4. Tail regime below boundary (stop/play partition + artifact modes + best/trust)
    tail_modes = ("shipped", "tail_missing", "tail_corrupt", "base_missing_tail_present",
                  "sha_mismatch", "none", "two_bin_tail")
    for s in (0, 1, 10, 15, 28, 40, 54, 55):
        for raw in (1, 5, 14, 28):
            if s + 2 * raw >= 57:
                continue
            d_str = (date(2027, 9, 27) - timedelta(days=raw)).isoformat()
            reach = min(57, s + 2 * raw)
            for best_v in (None, 0, s, max(0, reach - 1), reach, 56, 57):
                for best_st in (None, "trusted", "untrusted"):
                    for mode_name in tail_modes:
                        record_case(
                            "tail_cells", s, d_str, (raw % 2 == 0), 0.83, 0.81, True, True,
                            mode_name, best_v, best_st,
                        )

    # 5. Alternate horizons, configured/unconfigured years, and ZERO raw days at & above 57
    year_dates = [
        "2025-07-15", "2025-09-27", "2025-09-28",
        "2026-07-15", "2026-09-27", "2026-09-28", "2026-12-31",
        "2027-01-01", "2027-03-24", "2027-09-26", "2027-09-27", "2027-12-31",
        "2028-01-01", "2028-09-27", "2028-09-28", "2028-12-31",
    ]
    for mode_name in modes:
        for d_str in year_dates:
            raw = strategy._days_remaining(d_str)
            streaks_for_day = [0, 12, 43, 56]
            if raw == 0:
                streaks_for_day.extend([57, 58, 100, 10**9])
            for s in streaks_for_day:
                for saver in (False, True):
                    for p in (0.50, 0.80, 0.96):
                        record_case(
                            "horizons_years_zero_days", s, d_str, saver, p, 0.90, True, True,
                            mode_name, min(s, 57), "trusted",
                        )

    # 6. Direct effective_pick_bar edge inputs that did not raise on base
    for s in (-5, -1, 0, 25, 56):
        for d_str in ("2027-07-15", "2027-09-22", "2027-09-27"):
            for mode_name in ("shipped", "none", "empty_bounds", "tail_missing", "two_bin_tail"):
                bar = strategy.effective_pick_bar(s, d_str, False, modes[mode_name], 20, "trusted")
                digest.update(json.dumps(["bar_edge", s, d_str, mode_name, bar, type(bar).__name__]).encode())
                counts["bar_edge_inputs"] += 1

    # 7. Real select_pick + save_pick + _commit_decision_for_pick below boundary & at d=0 s=57
    live_cases = [
        ("2027-07-15", 12, True, 0.96),
        ("2027-03-24", 50, False, 0.96),
        ("2027-07-15", 56, False, 0.96),
        ("2027-09-20", 0, False, 0.96),
        ("2027-09-25", 0, False, 0.96),
        ("2027-09-27", 57, False, 0.96),
    ]
    for i, (d_str, s, saver, top_p) in enumerate(live_cases):
        day = date.fromisoformat(d_str)
        pdir = work_tmp / f"live_{i}"
        pdir.mkdir(parents=True, exist_ok=True)
        frame = _synthetic_frame(day, top_p)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        strategy._mdp_cache = None
        with _pinned_offline(REAL_DATETIME.combine(day, time(12), ET)):
            sel = strategy.select_pick(
                frame, d_str, pdir, streak=s, saver_available=saver, allow_double=True,
                game_statuses_detailed=statuses, best_streak=57 if s == 0 and d_str == "2027-09-25" else s,
                best_status="trusted",
            )
            saved_dict = None
            committed_rec = None
            if sel.pick_result and sel.pick_result.daily:
                picks.save_pick(sel.pick_result.daily, pdir)
                reloaded = picks.load_pick(d_str, pdir)
                saved_dict = asdict(reloaded)
                saved_dict["run_time"] = "<normalized>"
                st = scheduler.SchedulerState(d_str, "2027-01-01T00:00:00", [], [], [], False, None, None, None)
                scheduler._commit_decision_for_pick(
                    pdir, d_str, reloaded, selection=sel, delivery_status="delivered", state=st
                )
                committed_rec = daily_decision.load_decision(d_str, pdir)
                committed_rec["finalized_at"] = "<normalized>"
            digest.update(json.dumps({
                "action": sel.action,
                "source": sel.source,
                "decision": asdict(sel.decision) if sel.decision else None,
                "saved": saved_dict,
                "committed": committed_rec,
            }, sort_keys=True).encode())
            counts["live_select_save_commit"] += 1

    print(json.dumps({
        "bts_file": str(resolved_init),
        "counts": counts,
        "total_cases": sum(counts.values()),
        "raw_days_min_max": [min(raw_days_seen), max(raw_days_seen)],
        "raw_days_count": len(raw_days_seen),
        "eff_days_min_max": [min(eff_days_seen), max(eff_days_seen)],
        "eff_days_count": len(eff_days_seen),
        "digest": digest.hexdigest(),
    }), flush=True)


def run_comparison(tmp_dir: Path) -> dict:
    mdp_diff = subprocess.run(
        ["git", "diff", "0a2bec10", "813935bc", "--", "src/bts/simulate/mdp.py"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    assert mdp_diff == "", "mdp.py must be untouched!"

    base_dir = tmp_dir / "archived_0a2bec10"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "0a2bec10", "src"],
        cwd=REPO, capture_output=True, check=True, timeout=15,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(base_dir)], input=archive, check=True, timeout=15)

    outputs = []
    for label, src_root in (("base_0a2bec10", base_dir / "src"), ("head_813935bc", REPO / "src")):
        sub_tmp = tmp_dir / f"work_{label}"
        sub_tmp.mkdir(parents=True, exist_ok=True)
        env = os.environ | {"PYTHONPATH": str(src_root)}
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(src_root), str(sub_tmp)],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=90,
        )
        assert proc.returncode == 0, f"{label} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        lines = proc.stdout.strip().splitlines()
        assert len(lines) == 2, lines
        bts_path, summary_json = lines[0], json.loads(lines[1])
        assert Path(bts_path).resolve() == (src_root / "bts" / "__init__.py").resolve()
        outputs.append(summary_json)

    base_out, head_out = outputs
    assert base_out["bts_file"] != head_out["bts_file"]
    assert base_out["counts"] == head_out["counts"]
    assert base_out["raw_days_count"] == 272 and base_out["raw_days_min_max"] == [0, 271]
    assert base_out["eff_days_count"] == 181 and base_out["eff_days_min_max"] == [0, 180]
    assert base_out["digest"] == head_out["digest"], (base_out, head_out)
    return {
        "mdp_py_untouched": True,
        "base_bts_file": base_out["bts_file"],
        "head_bts_file": head_out["bts_file"],
        "counts": head_out["counts"],
        "total_cases": head_out["total_cases"],
        "raw_days": head_out["raw_days_min_max"],
        "eff_days": head_out["eff_days_min_max"],
        "digest": head_out["digest"],
    }


def test_probe_3b_parity(tmp_path):
    res = run_comparison(tmp_dir=tmp_path)
    assert res["mdp_py_untouched"] is True
    assert res["total_cases"] > 500000


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp_dir = SCRATCH / "p3b"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    summary = run_comparison(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    out_log = OUT_DIR / "probe_3_below_boundary_parity.log"
    out_log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

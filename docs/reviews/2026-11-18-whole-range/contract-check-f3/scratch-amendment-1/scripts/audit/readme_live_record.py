"""W1.6: the 2026 live recommendation record for the README, from the accepted W1.1 ledger (exposure row X-30).

Counts committed selections (commit_status committed_evidenced) whose contest slot grade is a graded hit / not_hit,
per slot, for the season and from the brief's current-recipe boundary (2026-04-30: the last production FEATURE_COLS
change, ee4190f). Every other selection row is an exclusion, counted. Wilson intervals are descriptive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ERA_START = "2026-04-30"


def wilson(k: int, n: int, z: float = 1.959964) -> list | None:
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def live_record(led: pd.DataFrame, era_start: str = ERA_START) -> dict:
    sel = led[led["row_kind"] == "selection"]
    committed = sel["commit_status"] == "committed_evidenced"
    graded = (sel["bts_outcome_status"] == "graded") & sel["bts_outcome"].isin(["hit", "not_hit"])
    use = sel[committed & graded]

    def block(df):
        return {slot: {"hit": int((g["bts_outcome"] == "hit").sum()), "graded": int(len(g))}
                for slot, g in df.groupby("slot")}
    return {"season": block(use), f"since_{era_start}": block(use[use["date"] >= era_start]),
            "exclusions": {"not_committed_evidenced": int((~committed).sum()),
                           "not_contest_graded": int((committed & ~graded).sum())},
            "dates": [str(use["date"].min()), str(use["date"].max())] if len(use) else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not (args.ledger_dir / "ACCEPTED.json").exists():
        raise SystemExit("not an accepted W1.1 build")
    raw = (args.ledger_dir / "season_2026_ledger.parquet").read_bytes()
    import io
    rec = live_record(pd.read_parquet(io.BytesIO(raw)))
    for scope in ("season", f"since_{ERA_START}"):
        for v in rec[scope].values():
            v["rate"] = round(v["hit"] / v["graded"], 4) if v["graded"] else None
            v["wilson95"] = wilson(v["hit"], v["graded"])
    rec["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
    rec["era_start_basis"] = "the 9/11 brief's current-recipe boundary: the 4/30 inference fix ee4190f, the last production FEATURE_COLS change"
    args.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

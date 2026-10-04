"""W1.4b due read: MDP quality-bin collapse (plan W1.4b; incident register E81/EX2).

Served game-hit probabilities of production selections (W1.1 ledger, primary and double-down slots) and of the
declined candidate on skip days (decision.json), classified against the saved reach-57 policy's quality-bin
boundaries with the production health check's own classifier, by epoch = calendar month × policy regime. Reads no
outcome. A bin histogram describes how the policy saw our probabilities; it is not evidence of useful
discrimination (plan)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bts.health.mdp_policy_alignment import _classify


def epoch(date: str, objective) -> str:
    return f"{date[:7]}/{objective if isinstance(objective, str) and objective else 'unknown'}"


def bin_table(df: pd.DataFrame, boundaries: list[float]) -> dict:
    out: dict = {}
    for ep, g in df.assign(ep=[epoch(d, o) for d, o in zip(df["date"], df["objective"])]).groupby("ep", sort=True):
        out[ep] = {}
        for slot, s in g.groupby("slot", sort=True):
            vals = s["p"].astype(float).tolist()
            counts = {i: 0 for i in range(len(boundaries) + 1)}
            for v in vals:
                counts[_classify(v, boundaries)] += 1
            out[ep][slot] = {"n": len(vals), "counts": counts, "below_lowest": int(sum(v < boundaries[0] for v in vals)),
                             "dominant_share": max(counts.values()) / len(vals) if vals else None,
                             "p_quantiles": {q: float(np.quantile(vals, q)) for q in (0.1, 0.5, 0.9)} if vals else {}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True, help="accepted W1.1 build dir")
    ap.add_argument("--picks", type=Path, required=True, help="data/picks (decision.json for skip-day candidates)")
    ap.add_argument("--policy", type=Path, required=True, help="data/models/mdp_policy.npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    lp = args.ledger / "season_2026_ledger.parquet"
    led = pd.read_parquet(lp, columns=["date", "row_kind", "slot", "objective", "p_stated"])
    sel = led[led["row_kind"] == "selection"].rename(columns={"p_stated": "p"})[["date", "slot", "objective", "p"]]
    skips, used = [], {}
    for d in sorted(led.loc[led["row_kind"] == "skip_day", "date"]):
        f = args.picks / d / "decision.json"
        if f.exists():
            used[f"decision:{d}"] = hashlib.sha256(f.read_bytes()).hexdigest()
            dec = json.loads(f.read_text())
            p = (dec.get("primary") or {}).get("p_game_hit")
            if p is not None:
                skips.append({"date": d, "slot": "declined_on_skip", "objective": dec.get("objective"), "p": float(p)})
    df = pd.concat([sel, pd.DataFrame(skips)], ignore_index=True)
    boundaries = [float(x) for x in np.load(args.policy)["boundaries"].tolist()]
    res = {"schema": "w14b_bin_collapse_v1", "boundaries": boundaries,
           "inputs": {"ledger": hashlib.sha256(lp.read_bytes()).hexdigest(),
                      "policy": hashlib.sha256(args.policy.read_bytes()).hexdigest(), **used},
           "rows": {k: int(v) for k, v in df["slot"].value_counts().to_dict().items()},
           "by_epoch": bin_table(df, boundaries),
           "note": "served probabilities only; no outcome read"}
    args.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

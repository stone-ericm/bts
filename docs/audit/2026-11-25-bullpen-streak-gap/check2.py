"""Per-seed paired new-minus-old rank-1/rank-2 realized and predicted rates (see README)."""
import math, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.audit.bullpen_r52_compare import SEEDS, SEASONS
OUT = Path.home() / "bts-runs/bullpen-bt/out"
rows = []
for s in SEEDS:
    for y in SEASONS:
        r = {}
        for arm in ("old", "new"):
            df = pd.read_parquet(OUT / arm / f"seed_{s}" / f"backtest_{y}.parquet")
            t = df[df["rank"] <= 2].pivot_table(index="date", columns="rank", values=["actual_hit", "p_game_hit"])
            h1, h2 = t[("actual_hit", 1)], t[("actual_hit", 2)]
            r[arm] = dict(p1=h1.mean(), p2=h2.mean(), both=(h1 * h2).mean(), mp1=t[("p_game_hit", 1)].mean(), mp2=t[("p_game_hit", 2)].mean())
        rows.append({k: r["new"][k] - r["old"][k] for k in r["new"]} | {"seed": s})
d = pd.DataFrame(rows).groupby("seed").mean()
for k in ("p1", "p2", "both", "mp1", "mp2"):
    x = d[k]
    print(f"{k:5s} new-old per-seed mean {x.mean()*100:+.3f} pt, se {x.std()/math.sqrt(len(x))*100:.3f}, seeds lower {int((x < 0).sum())}/24")

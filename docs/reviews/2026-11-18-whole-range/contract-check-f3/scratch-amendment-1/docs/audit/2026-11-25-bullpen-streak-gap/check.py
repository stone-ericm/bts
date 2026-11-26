"""Does the bullpen fix's lower mean max streak (R59 final test) come from the old arm's leak? (Eric 2026-11-25: "yes, run it")

The old arm's `opp_bullpen_hr_30g` was missing exactly when the opposing team used no reliever that day.
Here a pick is "no-reliever" when the batter's side faced exactly one pitcher in that game (from the PA files).
(1) On those picks, is the old arm's p_game_hit lower than the new arm's, as a model that saw the leak would be?
(2) Does the scorecard's mean-max-streak gap shrink when those dates are removed from both arms?
Outcomes 2021 to 2025 only. Reads ~/bts-runs/bullpen-bt/out/ (released after the R59 final test).
"""
import json, math, statistics as st, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.audit.bullpen_r52_compare import SEEDS, SEASONS
from bts.validate.scorecard import compute_streak_metrics

OUT = Path.home() / "bts-runs/bullpen-bt/out"
TRIALS = 2000

def no_reliever_games(season):
    pa = pd.read_parquet(f"data/processed/pa_{season}.parquet", columns=["game_pk", "is_home", "pitcher_id"])
    n = pa.groupby(["game_pk", "is_home"])["pitcher_id"].nunique()
    return n[n == 1].reset_index()[["game_pk", "is_home"]]

def batter_side(season):
    pa = pd.read_parquet(f"data/processed/pa_{season}.parquet", columns=["game_pk", "batter_id", "is_home"])
    return pa.drop_duplicates(["game_pk", "batter_id"])

res = {"by_seed_season": []}
p_gap_nr, p_gap_other, nr_share = [], [], []
streak = {"all": [], "without_nr_dates": []}
for season in SEASONS:
    nr = no_reliever_games(season); nr["nr"] = True
    side = batter_side(season)
    for seed in SEEDS:
        arms = {}
        for arm in ("old", "new"):
            df = pd.read_parquet(OUT / arm / f"seed_{seed}" / f"backtest_{season}.parquet")
            df = df.merge(side, on=["game_pk", "batter_id"], how="left").merge(nr, on=["game_pk", "is_home"], how="left")
            df["nr"] = df["nr"].fillna(False).astype(bool)
            arms[arm] = df
        top = {a: d[d["rank"] == 1].set_index("date") for a, d in arms.items()}
        both = top["old"].join(top["new"], lsuffix="_o", rsuffix="_n", how="inner")
        same = both[both["batter_id_o"] == both["batter_id_n"]]          # same pick in both arms: a clean p comparison
        g_nr = (same.loc[same["nr_o"], "p_game_hit_n"] - same.loc[same["nr_o"], "p_game_hit_o"]).mean()
        g_ot = (same.loc[~same["nr_o"], "p_game_hit_n"] - same.loc[~same["nr_o"], "p_game_hit_o"]).mean()
        # dates on which either arm's top pick was a no-reliever pick
        nr_dates = set(top["old"].index[top["old"]["nr"]]) | set(top["new"].index[top["new"]["nr"]])
        share = len(nr_dates) / max(1, len(set(top["old"].index)))
        row = {"seed": seed, "season": season, "p_gap_no_reliever": g_nr, "p_gap_other": g_ot, "nr_date_share": share}
        for label, keep in (("all", None), ("without_nr_dates", nr_dates)):
            sm = {}
            for arm, d in arms.items():
                dd = d if keep is None else d[~d["date"].isin(keep)]
                sm[arm] = compute_streak_metrics(dd.drop(columns=["nr", "is_home"]), n_trials=TRIALS)["mean_max_streak"]
            row[f"streak_gap_{label}"] = sm["new"] - sm["old"]
            streak[label].append(sm["new"] - sm["old"])
        res["by_seed_season"].append(row)
        if not math.isnan(g_nr): p_gap_nr.append(g_nr)
        p_gap_other.append(g_ot); nr_share.append(share)
        print(f"{season} seed {seed:>10}: p gap (new-old) on no-reliever picks {g_nr:+.4f}, others {g_ot:+.4f}; "
              f"streak gap all {row['streak_gap_all']:+.2f}, without those dates {row['streak_gap_without_nr_dates']:+.2f}", flush=True)
def summ(x): return {"mean": st.mean(x), "se": st.stdev(x) / math.sqrt(len(x)), "n": len(x)} if len(x) > 1 else {"n": len(x)}
res["summary"] = {"p_gap_no_reliever": summ(p_gap_nr), "p_gap_other": summ(p_gap_other),
                  "no_reliever_date_share": summ(nr_share),
                  "streak_gap_all": summ(streak["all"]), "streak_gap_without_nr_dates": summ(streak["without_nr_dates"])}
print(json.dumps(res["summary"], indent=1))
Path(__file__).with_name("result.json").write_text(json.dumps(res, indent=1, default=float) + "\n")

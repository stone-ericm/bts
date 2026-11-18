"""How often is opp_bullpen_hr_30g missing ONLY because the pitching team used no reliever that day? (2025 only.)"""
import numpy as np, pandas as pd
from bts.features import compute

df = pd.concat([pd.read_parquet(f"data/processed/pa_{y}.parquet") for y in (2025,)], ignore_index=True)
print("rows", len(df), "dates", df["date"].nunique())
out = compute.compute_all_features(df)
print("has team id col:", "opp_pitching_team_id" in out.columns, "| probable lookup non-empty:", out["opp_pitching_team_id"].notna().mean().round(4))
out["date"] = pd.to_datetime(out["date"])
team_dates = out.dropna(subset=["opp_pitching_team_id"]).groupby(["opp_pitching_team_id", "date"]).agg(
    n=("is_hit", "size"), hits=("is_hit", "sum"), has_val=("opp_bullpen_hr_30g", lambda s: s.notna().any())).reset_index()
team_dates = team_dates.sort_values(["opp_pitching_team_id", "date"])
team_dates["prior_valued_dates"] = team_dates.groupby("opp_pitching_team_id")["has_val"].cumsum() - team_dates["has_val"]
# A team-date with no value AFTER the team already had valued dates: the missing value is not the warm-up gate.
late_missing = team_dates[(~team_dates["has_val"]) & (team_dates["prior_valued_dates"] > 0)]
valued = team_dates[team_dates["has_val"]]
print("team-dates total", len(team_dates), "| valued", len(valued), "| missing after warm-up", len(late_missing),
      f"({len(late_missing)/max(1,len(valued)+len(late_missing)):.2%} of post-warm-up team-dates)")
print("PA rows affected:", int(late_missing["n"].sum()), f"({late_missing['n'].sum()/len(out):.3%} of all 2025 PA rows)")
print("hit rate per PA on affected team-dates: %.4f   on valued team-dates: %.4f" % (
    late_missing["hits"].sum()/max(1,late_missing["n"].sum()), valued["hits"].sum()/valued["n"].sum()))
print("PAs per affected team-date: %.1f   per valued team-date: %.1f" % (late_missing["n"].mean(), valued["n"].mean()))

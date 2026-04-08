import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")

from arch_eval import load_data, walk_forward_backtest, compute_metrics
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.simulate.backtest_blend import save_profiles
from bts.simulate.mdp import solve_mdp
from phase7_same_game_double import add_game_pk_to_profiles, build_game_pk_lookup, apply_different_game_rule

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

df = load_data()
profiles = walk_forward_backtest(df, TEST_SEASONS, BLEND_CONFIGS, LGB_PARAMS, game_level=False)

proc = Path("data/processed")
pa_dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
pa_data = pd.concat(pa_dfs, ignore_index=True)
gp_lookup = build_game_pk_lookup(pa_data)
profiles = add_game_pk_to_profiles(profiles, gp_lookup)
profiles = apply_different_game_rule(profiles)

out = Path("data/simulation")
for season in TEST_SEASONS:
    season_profiles = profiles[pd.to_datetime(profiles["date"]).dt.year == season]
    save_profiles(season_profiles, season, out)

metrics = compute_metrics(profiles)
print(f"P@1 avg: {metrics['p_at_1']['avg']:.4f}")
print(f"MDP P(57): {metrics['mdp_p57']:.6f}")

solution = solve_mdp(
    metrics["early_bins"], season_length=180,
    late_bins=metrics["late_bins"], late_phase_days=30,
)
solution.save("data/models/mdp_policy.npz")
print(f"Policy saved. P(57) = {solution.optimal_p57:.4%}")

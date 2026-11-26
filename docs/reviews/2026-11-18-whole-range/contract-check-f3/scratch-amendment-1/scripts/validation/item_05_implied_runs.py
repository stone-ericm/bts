"""
Item 05: Implied Run Total Investigation

Tests whether team-level Vegas implied offensive strength adds signal beyond
our existing pitcher/park/weather features.

The odds data (v2) contains batter_hits player props (0.5 over/under).
There are no game-level totals in this dataset, so we derive a team-level
"implied hit rate" by:
  1. Extracting implied P(hit>=1) for each player from the 0.5 over/under price
  2. Averaging across all listed players for each team in a game
  3. Using this "team implied hit rate" as a proxy for betting-market-consensus
     offensive environment (encodes pitcher quality + park + weather + lineup)

Join chain:
  - odds: player_name → implied_p_hit, associated with (date, game: away_team, home_team)
  - name→id map: player_name → batter_id (built from raw game JSONs)
  - PA data: (batter_id, game_pk, date, is_home) → which team the batter is on
  - game→teams map: game_pk → (home_team_name, away_team_name)
  - backtest: rank-1 pick (batter_id, date) → p_game_hit, actual_hit

Data coverage: v2 odds covers Sept 2023–Sept 2025 (only 2024 + partial 2023/2025
have backtest profiles), so analysis is effectively on 2024 + Sept 2023 + 2025.

Analyses:
  a. Coverage: what fraction of backtest rank-1 dates have odds data?
  b. Correlation: does team implied hit rate correlate with actual rank-1 P@1?
  c. P@1 by team implied hit rate quartile
  d. Residual signal: after controlling for p_game_hit, does implied hit rate add?
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ODDS_V2_DIR = DATA_DIR / "external" / "odds" / "v2"
RAW_DIR = DATA_DIR / "raw"
SIM_DIR = DATA_DIR / "simulation"
PROCESSED_DIR = DATA_DIR / "processed"

BACKTEST_SEASONS = [2023, 2024, 2025]


# ---------------------------------------------------------------------------
# Step 0: American odds → implied probability (no-vig for 0.5 line)
# ---------------------------------------------------------------------------

def american_to_prob(price: int) -> float:
    """Convert American odds to implied probability."""
    if price > 0:
        return 100.0 / (price + 100.0)
    else:
        return abs(price) / (abs(price) + 100.0)


# ---------------------------------------------------------------------------
# Step 1: Parse v2 odds files → (date, player_name) → implied_p_hit
#         Also build (date, player_name) → (event_id, away_team, home_team)
# ---------------------------------------------------------------------------

def parse_odds_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      player_odds: DataFrame with columns [date, player_name, implied_p_hit]
        (median across bookmakers for 0.5 over line)
      game_map: DataFrame with columns [date, event_id, away_team, home_team]
    """
    files = sorted(ODDS_V2_DIR.glob("*.json"))
    print(f"  {len(files)} v2 odds files: {files[0].stem} → {files[-1].stem}")

    player_rows = []
    game_rows = []

    for f in files:
        date_str = f.stem  # e.g. "2024-06-15"
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  WARNING: failed to parse {f.name}: {e}", file=sys.stderr)
            continue

        events = data.get("events", [])
        for event in events:
            if "error" in event:
                continue
            event_id = event.get("event_id", "")
            away_team = event.get("away_team", "")
            home_team = event.get("home_team", "")

            game_rows.append({
                "date": date_str,
                "event_id": event_id,
                "away_team": away_team,
                "home_team": home_team,
            })

            props = event.get("props", {})
            bookmakers = props.get("bookmakers", [])

            # Collect all 0.5-over prices per player across bookmakers
            player_prices: dict[str, list[float]] = {}
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market.get("key") != "batter_hits":
                        continue
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") != "Over":
                            continue
                        if outcome.get("point") != 0.5:
                            continue
                        name = outcome.get("description", "").strip()
                        price = outcome.get("price")
                        if name and price is not None:
                            if name not in player_prices:
                                player_prices[name] = []
                            player_prices[name].append(american_to_prob(int(price)))

            for player_name, probs in player_prices.items():
                # Median across bookmakers to reduce vig noise
                player_rows.append({
                    "date": date_str,
                    "event_id": event_id,
                    "player_name": player_name,
                    "implied_p_hit": float(np.median(probs)),
                    "n_books": len(probs),
                })

    player_odds = pd.DataFrame(player_rows)
    game_map = pd.DataFrame(game_rows)
    print(f"  {len(player_odds):,} player-date prop rows ({player_odds['player_name'].nunique():,} unique names)")
    print(f"  {len(game_map):,} game-date rows")
    return player_odds, game_map


# ---------------------------------------------------------------------------
# Step 2: Build player name → batter_id mapping from raw game JSONs
# ---------------------------------------------------------------------------

def build_name_id_map() -> dict[str, int]:
    """
    Scan all raw game JSON files (2023-2025) and build fullName → batter_id.
    We use all seasons to maximize coverage; conflicts resolved by most recent.
    """
    print("  Building name→batter_id map from raw game JSONs...")
    name_to_id: dict[str, int] = {}
    seasons = [2023, 2024, 2025]
    n_files = 0

    for season in seasons:
        season_dir = RAW_DIR / str(season)
        if not season_dir.exists():
            print(f"  WARNING: {season_dir} not found", file=sys.stderr)
            continue
        game_files = sorted(season_dir.glob("*.json"))
        print(f"    Season {season}: {len(game_files)} game files")
        for gf in game_files:
            try:
                feed = json.loads(gf.read_text())
            except Exception:
                continue
            n_files += 1
            ld = feed.get("liveData", {})
            bs = ld.get("boxscore", {})
            for side in ("away", "home"):
                players = bs.get("teams", {}).get(side, {}).get("players", {})
                for key, p in players.items():
                    pid = p.get("person", {}).get("id")
                    full_name = p.get("person", {}).get("fullName", "").strip()
                    if pid and full_name:
                        name_to_id[full_name] = int(pid)

    print(f"  Built name→id map: {len(name_to_id):,} unique player names from {n_files} games")
    return name_to_id


# ---------------------------------------------------------------------------
# Step 3: Build game_pk → (home_team_name, away_team_name) from raw JSONs
# ---------------------------------------------------------------------------

def build_game_team_map() -> pd.DataFrame:
    """
    Returns DataFrame: [game_pk, date, home_team, away_team]
    Used to match odds event (team names) to game_pk.
    """
    print("  Building game_pk→team map from raw game JSONs...")
    rows = []
    seasons = [2023, 2024, 2025]

    for season in seasons:
        season_dir = RAW_DIR / str(season)
        if not season_dir.exists():
            continue
        for gf in sorted(season_dir.glob("*.json")):
            try:
                feed = json.loads(gf.read_text())
            except Exception:
                continue
            gd = feed.get("gameData", {})
            game_pk = gd.get("game", {}).get("pk")
            date = gd.get("datetime", {}).get("officialDate")
            teams = gd.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            if game_pk and date:
                rows.append({
                    "game_pk": int(game_pk),
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["game_pk"])
    print(f"  {len(df):,} game→team mappings")
    return df


# ---------------------------------------------------------------------------
# Step 4: Load backtest and PA data
# ---------------------------------------------------------------------------

def load_backtest() -> pd.DataFrame:
    frames = []
    for season in BACKTEST_SEASONS:
        path = SIM_DIR / f"backtest_{season}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        df["season"] = season
        frames.append(df)
    bt = pd.concat(frames, ignore_index=True)
    bt["date"] = pd.to_datetime(bt["date"]).dt.strftime("%Y-%m-%d")
    bt["batter_id"] = bt["batter_id"].astype(int)
    return bt


def load_pa_data() -> pd.DataFrame:
    """Load PA data for batter_id → (game_pk, is_home) mapping."""
    frames = []
    for season in BACKTEST_SEASONS:
        path = PROCESSED_DIR / f"pa_{season}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found", file=sys.stderr)
            continue
        df = pd.read_parquet(path, columns=["game_pk", "date", "batter_id", "is_home"])
        df["season"] = season
        frames.append(df)
    pa = pd.concat(frames, ignore_index=True)
    pa["date"] = pd.to_datetime(pa["date"]).dt.strftime("%Y-%m-%d")
    pa["batter_id"] = pa["batter_id"].astype(int)
    return pa


# ---------------------------------------------------------------------------
# Step 5: Compute team-level implied hit rate per game-date
# ---------------------------------------------------------------------------

def compute_team_implied_hit_rate(
    player_odds: pd.DataFrame,
    game_map: pd.DataFrame,
    name_to_id: dict[str, int],
    game_team_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns DataFrame: [game_pk, date, team_role (home/away), team_name,
                         implied_team_hit_rate, n_players]
    """
    # Map player names to batter_ids
    player_odds = player_odds.copy()
    player_odds["batter_id"] = player_odds["player_name"].map(name_to_id)
    n_unmapped = player_odds["batter_id"].isna().sum()
    n_total = len(player_odds)
    print(f"  Name mapping: {n_total - n_unmapped}/{n_total} players mapped ({100*(n_total-n_unmapped)/n_total:.1f}%)")

    # Only keep mapped players
    player_odds = player_odds.dropna(subset=["batter_id"])
    player_odds["batter_id"] = player_odds["batter_id"].astype(int)

    # Normalize odds team names to match MLB API team names
    # The odds data uses "Oakland Athletics" / "Minnesota Twins" etc. — same as API.
    # Merge game_map (date, event_id, away_team, home_team) with game_team_map
    # (game_pk, date, home_team, away_team) on date + team names
    game_merged = game_map.merge(
        game_team_map,
        on=["date", "home_team", "away_team"],
        how="inner",
    )
    n_matched = len(game_merged)
    print(f"  Game matching: {n_matched}/{len(game_map)} odds events matched to game_pk")

    # Add game_pk to player_odds via event_id
    player_odds = player_odds.merge(
        game_merged[["event_id", "game_pk"]],
        on="event_id",
        how="left",
    )

    # Now determine which team each player is on:
    # We need (batter_id, date, game_pk) → is_home from PA data
    # Load PA for this step (we'll pass it in)
    return player_odds, game_merged


def compute_team_rates_from_pa(
    player_odds_with_pk: pd.DataFrame,
    pa: pd.DataFrame,
    game_team_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given player_odds with game_pk, join PA data to determine team membership.
    Returns [date, game_pk, batter_id, implied_p_hit, is_home, team_name].
    """
    # Get batter → is_home per game (deduplicated)
    batter_game = pa[["game_pk", "date", "batter_id", "is_home"]].drop_duplicates(
        subset=["game_pk", "batter_id"]
    )

    # Join player odds to PA to get is_home
    merged = player_odds_with_pk.merge(
        batter_game[["game_pk", "batter_id", "is_home"]],
        on=["game_pk", "batter_id"],
        how="inner",
    )

    # Add team name
    merged = merged.merge(
        game_team_map[["game_pk", "home_team", "away_team"]],
        on="game_pk",
        how="left",
    )
    merged["team_name"] = np.where(merged["is_home"], merged["home_team"], merged["away_team"])

    # Aggregate to (date, game_pk, team_name) → avg implied_p_hit
    team_rates = (
        merged.groupby(["date", "game_pk", "team_name", "is_home"])["implied_p_hit"]
        .agg(mean="mean", n_players="count")
        .reset_index()
        .rename(columns={"mean": "team_implied_hit_rate"})
    )
    return team_rates, merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def two_proportion_z_test(n1_hits, n1, n2_hits, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = n1_hits / n1
    p2 = n2_hits / n2
    p_pool = (n1_hits + n2_hits) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val


def main():
    import os
    os.chdir(PROJECT_ROOT)

    print("=" * 60)
    print("Item 05: Implied Run Total Investigation")
    print("=" * 60)

    # --- Parse odds files ---
    print("\n[1] Parsing v2 odds files...")
    player_odds, game_map = parse_odds_files()

    # --- Build name→ID map ---
    print("\n[2] Building player name→batter_id map...")
    name_to_id = build_name_id_map()

    # --- Build game→team map ---
    print("\n[3] Building game_pk→team name map...")
    game_team_map = build_game_team_map()

    # --- Load backtest + PA data ---
    print("\n[4] Loading backtest profiles and PA data...")
    backtest = load_backtest()
    rank1 = backtest[backtest["rank"] == 1].copy()
    print(f"  {len(rank1)} rank-1 picks across seasons {BACKTEST_SEASONS}")
    pa = load_pa_data()
    print(f"  {len(pa):,} PA rows across seasons {BACKTEST_SEASONS}")

    # --- Map player odds to game_pks ---
    print("\n[5] Matching odds events to game_pks...")
    player_odds_with_pk, game_merged = compute_team_implied_hit_rate(
        player_odds, game_map, name_to_id, game_team_map
    )

    # --- Compute team-level implied hit rate ---
    print("\n[6] Computing team-level implied hit rates...")
    team_rates, player_joined = compute_team_rates_from_pa(
        player_odds_with_pk, pa, game_team_map
    )
    print(f"  {len(team_rates):,} team-game-date rows")
    print(f"  {team_rates['date'].nunique()} unique dates")

    # --- Join to rank-1 backtest picks ---
    # rank1 has (batter_id, date) → need to find (game_pk, is_home) for each pick
    # Use PA data for this
    batter_game_lookup = pa[["game_pk", "date", "batter_id", "is_home"]].drop_duplicates(
        subset=["batter_id", "date"]
    )
    rank1 = rank1.merge(
        batter_game_lookup[["batter_id", "date", "game_pk", "is_home"]],
        on=["batter_id", "date"],
        how="left",
    )

    # Join team rates to rank1 via (game_pk, is_home)
    rank1 = rank1.merge(
        team_rates[["game_pk", "is_home", "team_implied_hit_rate", "n_players", "team_name"]],
        on=["game_pk", "is_home"],
        how="left",
    )

    # ── (a) Coverage ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(a) DATA COVERAGE")
    print("=" * 60)
    n_total = len(rank1)
    n_covered = rank1["team_implied_hit_rate"].notna().sum()
    odds_dates = set(player_odds["date"].unique())
    backtest_dates = set(rank1["date"].unique())
    overlap_dates = odds_dates & backtest_dates
    print(f"  Odds date range: {min(odds_dates)} → {max(odds_dates)}")
    print(f"  Backtest date range: {min(backtest_dates)} → {max(backtest_dates)}")
    print(f"  Overlapping dates: {len(overlap_dates)}/{len(backtest_dates)} ({100*len(overlap_dates)/len(backtest_dates):.1f}%)")
    print(f"  Rank-1 picks with team implied rate: {n_covered}/{n_total} ({100*n_covered/n_total:.1f}%)")

    # Per-season coverage
    print(f"\n  {'Season':<8} {'covered':>8} {'total':>7} {'%':>6}")
    for season in BACKTEST_SEASONS:
        s = rank1[rank1["season"] == season]
        cov = s["team_implied_hit_rate"].notna().sum()
        tot = len(s)
        print(f"  {season:<8} {cov:>8} {tot:>7} {100*cov/tot if tot>0 else 0:>6.1f}%")

    # Filter to covered picks for subsequent analyses
    rank1_covered = rank1.dropna(subset=["team_implied_hit_rate"]).copy()
    print(f"\n  Using {len(rank1_covered)} covered picks for analyses (b)-(d)")

    if len(rank1_covered) < 50:
        print("  ERROR: Insufficient coverage for meaningful analysis. Aborting.")
        return {}

    # ── (b) Correlation: implied hit rate vs actual P@1 ──────────────────────
    print("\n" + "=" * 60)
    print("(b) CORRELATION: team implied hit rate vs actual hit outcome")
    print("=" * 60)

    r_full, p_full = stats.pearsonr(
        rank1_covered["team_implied_hit_rate"],
        rank1_covered["actual_hit"],
    )
    r_pred, p_pred = stats.pearsonr(
        rank1_covered["team_implied_hit_rate"],
        rank1_covered["p_game_hit"],
    )
    print(f"  r(implied_hit_rate, actual_hit)  = {r_full:.4f}  p={p_full:.4f}")
    print(f"  r(implied_hit_rate, p_game_hit)  = {r_pred:.4f}  p={p_pred:.4f}")
    print(f"  r(p_game_hit, actual_hit)        = {stats.pearsonr(rank1_covered['p_game_hit'], rank1_covered['actual_hit'])[0]:.4f}")

    print(f"\n  Descriptive stats for team_implied_hit_rate:")
    print(f"  mean={rank1_covered['team_implied_hit_rate'].mean():.4f}  "
          f"std={rank1_covered['team_implied_hit_rate'].std():.4f}  "
          f"min={rank1_covered['team_implied_hit_rate'].min():.4f}  "
          f"max={rank1_covered['team_implied_hit_rate'].max():.4f}")

    # ── (c) P@1 by implied hit rate quartile ─────────────────────────────────
    print("\n" + "=" * 60)
    print("(c) P@1 BY TEAM IMPLIED HIT RATE QUARTILE")
    print("=" * 60)

    rank1_covered["quartile"] = pd.qcut(
        rank1_covered["team_implied_hit_rate"],
        q=4,
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
    )
    q_stats = (
        rank1_covered.groupby("quartile", observed=True)["actual_hit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    q_stats.columns = ["quartile", "hits", "n", "p1"]

    print(f"  {'Quartile':<12} {'P@1':>7} {'n':>6} {'hits':>6} {'avg_imp_hr':>12}")
    for _, row in q_stats.iterrows():
        avg_ihr = rank1_covered[rank1_covered["quartile"] == row["quartile"]]["team_implied_hit_rate"].mean()
        print(f"  {str(row['quartile']):<12} {row['p1']:>7.3f} {row['n']:>6.0f} {row['hits']:>6.0f} {avg_ihr:>12.4f}")

    # Q1 vs Q4 significance test
    q1 = rank1_covered[rank1_covered["quartile"] == "Q1 (low)"]
    q4 = rank1_covered[rank1_covered["quartile"] == "Q4 (high)"]
    z_q, p_q = two_proportion_z_test(
        q4["actual_hit"].sum(), len(q4),
        q1["actual_hit"].sum(), len(q1),
    )
    print(f"\n  Q4 vs Q1: z={z_q:.3f}  p={p_q:.4f}  ({'SIGNIFICANT p<0.05' if p_q < 0.05 else 'not significant'})")

    # ── (d) Residual signal: partial correlation controlling for p_game_hit ──
    print("\n" + "=" * 60)
    print("(d) RESIDUAL SIGNAL (partial correlation, controlling for p_game_hit)")
    print("=" * 60)

    # Residualize implied hit rate against p_game_hit
    from numpy.linalg import lstsq as np_lstsq
    X_pred = rank1_covered["p_game_hit"].values
    y_impl = rank1_covered["team_implied_hit_rate"].values
    y_actual = rank1_covered["actual_hit"].values

    # Partial correlation: implied_hit_rate with actual_hit, controlling for p_game_hit
    # Method: regress both on p_game_hit, take residuals, correlate residuals
    def resid(x, y):
        """Residuals of y ~ x."""
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np_lstsq(X, y, rcond=None)
        return y - X @ beta

    resid_implied = resid(X_pred, y_impl)
    resid_actual = resid(X_pred, y_actual)
    r_partial, p_partial = stats.pearsonr(resid_implied, resid_actual)
    print(f"  Partial r(implied_hit_rate, actual_hit | p_game_hit) = {r_partial:.4f}  p={p_partial:.4f}")

    # Logistic regression: baseline vs baseline + implied
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X_base = rank1_covered[["p_game_hit"]].values
        X_plus = rank1_covered[["p_game_hit", "team_implied_hit_rate"]].values
        y = rank1_covered["actual_hit"].values.astype(int)

        lr = LogisticRegression(random_state=42, max_iter=500)
        cv_base = cross_val_score(lr, X_base, y, cv=5, scoring="roc_auc")
        cv_plus = cross_val_score(lr, X_plus, y, cv=5, scoring="roc_auc")

        print(f"\n  Logistic regression AUC (5-fold CV):")
        print(f"  Baseline (p_game_hit only)        : {cv_base.mean():.4f} ± {cv_base.std():.4f}")
        print(f"  + team_implied_hit_rate            : {cv_plus.mean():.4f} ± {cv_plus.std():.4f}")
        auc_delta = cv_plus.mean() - cv_base.mean()
        print(f"  Delta AUC                          : {auc_delta:+.4f}")
    except ImportError:
        print("  sklearn not available — skipping logistic regression")
        auc_delta = float("nan")
        cv_base = np.array([float("nan")])
        cv_plus = np.array([float("nan")])

    # ── P@1 analysis: does rank-1 pick perform better when team implied rate is high?
    print("\n" + "=" * 60)
    print("(e) DOES FILTERING ON TEAM IMPLIED RATE IMPROVE P@1?")
    print("=" * 60)

    median_thr = rank1_covered["team_implied_hit_rate"].median()
    high = rank1_covered[rank1_covered["team_implied_hit_rate"] >= median_thr]
    low = rank1_covered[rank1_covered["team_implied_hit_rate"] < median_thr]
    z_hl, p_hl = two_proportion_z_test(
        high["actual_hit"].sum(), len(high),
        low["actual_hit"].sum(), len(low),
    )
    print(f"  Split at median implied hit rate ({median_thr:.4f}):")
    print(f"  High half: P@1={high['actual_hit'].mean():.3f}  n={len(high)}")
    print(f"  Low  half: P@1={low['actual_hit'].mean():.3f}  n={len(low)}")
    print(f"  z={z_hl:.3f}  p={p_hl:.4f}  ({'SIGNIFICANT p<0.05' if p_hl < 0.05 else 'not significant'})")

    # Per-season breakdown
    print(f"\n  Per-season P@1 by implied rate half:")
    print(f"  {'Season':<8} {'High P@1':>9} {'(n)':>6} {'Low P@1':>9} {'(n)':>6} {'Diff':>7}")
    for season in BACKTEST_SEASONS:
        s = rank1_covered[rank1_covered["season"] == season]
        if len(s) < 10:
            continue
        sh = s[s["team_implied_hit_rate"] >= s["team_implied_hit_rate"].median()]
        sl = s[s["team_implied_hit_rate"] < s["team_implied_hit_rate"].median()]
        ph = sh["actual_hit"].mean() if len(sh) > 0 else float("nan")
        pl = sl["actual_hit"].mean() if len(sl) > 0 else float("nan")
        diff = ph - pl
        print(f"  {season:<8} {ph:>9.3f} {len(sh):>6} {pl:>9.3f} {len(sl):>6} {diff:>+7.3f}")

    # ── Model's existing capture of implied rate ──────────────────────────────
    print("\n" + "=" * 60)
    print("(f) DOES OUR MODEL ALREADY CAPTURE IMPLIED OFFENSIVE CONTEXT?")
    print("=" * 60)

    # Average implied hit rate and p_game_hit by team, check correlation
    team_avg = rank1_covered.groupby("team_name").agg(
        avg_implied=("team_implied_hit_rate", "mean"),
        avg_p_game=("p_game_hit", "mean"),
        avg_actual=("actual_hit", "mean"),
        n=("actual_hit", "count"),
    ).reset_index()
    team_avg = team_avg[team_avg["n"] >= 5]  # Require at least 5 picks

    r_teams, p_teams = stats.pearsonr(team_avg["avg_implied"], team_avg["avg_p_game"])
    print(f"  Across teams: r(avg_implied, avg_p_game_hit) = {r_teams:.4f}  p={p_teams:.4f}")
    print(f"  ({len(team_avg)} teams with n>=5 rank-1 picks)")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    raw_corr_sig = abs(r_full) > 0.05 and p_full < 0.10
    q4_q1_sig = p_q < 0.05
    partial_sig = abs(r_partial) > 0.02 and p_partial < 0.10
    hl_sig = p_hl < 0.05

    print(f"  Correlation with actual hit (r={r_full:.4f}, p={p_full:.4f}): {'signal' if raw_corr_sig else 'noise'}")
    print(f"  Q4 vs Q1 P@1 gap: {'SIGNIFICANT' if q4_q1_sig else 'not significant'}")
    print(f"  Partial r after controlling for p_game_hit: {r_partial:.4f} ({'signal' if partial_sig else 'noise'})")
    print(f"  High vs low implied rate P@1 gap: {'SIGNIFICANT' if hl_sig else 'not significant'}")

    if q4_q1_sig and hl_sig and partial_sig:
        verdict = "ADD AS FEATURE — significant signal beyond p_game_hit"
    elif raw_corr_sig or q4_q1_sig:
        verdict = "MARGINAL — some raw signal but model may already capture it"
    else:
        verdict = "REJECT — no meaningful signal beyond existing features"

    print(f"\n  RECOMMENDATION: {verdict}")

    return {
        "n_covered": int(n_covered),
        "n_total": int(n_total),
        "coverage_pct": float(n_covered / n_total),
        "overlap_dates": len(overlap_dates),
        "r_implied_vs_actual": float(r_full),
        "p_implied_vs_actual": float(p_full),
        "r_implied_vs_p_game_hit": float(r_pred),
        "r_partial": float(r_partial),
        "p_partial": float(p_partial),
        "q4_p1": float(q4["actual_hit"].mean()),
        "q1_p1": float(q1["actual_hit"].mean()),
        "z_q4_vs_q1": float(z_q),
        "p_q4_vs_q1": float(p_q),
        "p1_high_implied": float(high["actual_hit"].mean()),
        "p1_low_implied": float(low["actual_hit"].mean()),
        "z_hl": float(z_hl),
        "p_hl": float(p_hl),
        "auc_baseline": float(cv_base.mean()),
        "auc_plus_implied": float(cv_plus.mean()),
        "auc_delta": float(auc_delta),
        "verdict": verdict,
    }


if __name__ == "__main__":
    results = main()
    print("\n── Summary dict ─────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")

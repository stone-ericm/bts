"""Daily BTS prediction: generate ranked picks for a given date."""

import json
import pickle  # noqa: S403 — used for caching trained ML models, not untrusted data
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from urllib.request import urlopen

from bts.features.compute import compute_all_features, FEATURE_COLS, STATCAST_COLS, TRAIN_START_YEAR

API_BASE = "https://statsapi.mlb.com"

# 12-model blend: baseline + single-Statcast variants + combos.
# Validated at 86.2% avg P@1 across 2024-2025 (vs 85.1% single model).
BLEND_CONFIGS = [
    ("baseline", FEATURE_COLS),
    ("barrel", FEATURE_COLS + ["batter_barrel_rate_30g"]),
    ("hard_hit", FEATURE_COLS + ["batter_hard_hit_rate_30g"]),
    ("sweet_spot", FEATURE_COLS + ["batter_sweet_spot_rate_30g"]),
    ("avg_ev", FEATURE_COLS + ["batter_avg_ev_30g"]),
    ("velo", FEATURE_COLS + ["pitcher_avg_velo_30g"]),
    ("spin", FEATURE_COLS + ["pitcher_avg_spin_30g"]),
    ("extension", FEATURE_COLS + ["pitcher_avg_extension_30g"]),
    ("break", FEATURE_COLS + ["pitcher_break_total_30g"]),
    ("velo_faced", FEATURE_COLS + ["batter_avg_velo_faced_30g"]),
    ("best_two", FEATURE_COLS + ["batter_sweet_spot_rate_30g", "pitcher_avg_extension_30g"]),
    ("all_statcast", FEATURE_COLS + STATCAST_COLS),
]

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    verbose=-1,
)

# Pitchers averaging < 3 IP over their recent appearances are likely openers/relievers
OPENER_IP_THRESHOLD = 3.0
OPENER_MIN_APPEARANCES = 5

# Players with this many days rest get flagged as IL return risk
IL_RETURN_DAYS_THRESHOLD = 7


def train_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """Train single LightGBM on historical PA data (2019+)."""
    train = df[df["season"] >= TRAIN_START_YEAR]
    train_X = train[FEATURE_COLS]
    train_y = train["is_hit"]
    mask = train_X.notna().any(axis=1)

    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
    model.fit(train_X[mask], train_y[mask])
    return model


def train_blend(df: pd.DataFrame) -> dict:
    """Train 12-model blend on historical PA data (2019+).

    Returns dict of {name: (model, feature_cols)} for each blend variant.
    """
    train = df[df["season"] >= TRAIN_START_YEAR]
    train_y = train["is_hit"]
    blend = {}

    for name, cols in BLEND_CONFIGS:
        train_X = train[cols]
        mask = train_X.notna().any(axis=1)
        model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
        model.fit(train_X[mask], train_y[mask])
        blend[name] = (model, cols)

    return blend


def save_blend(blend: dict, path) -> None:
    """Save trained blend models to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blend, f)


def load_blend(path) -> dict:
    """Load trained blend models from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — loading our own cached models


def _check_opener(pitcher_id: int, df: pd.DataFrame) -> dict:
    """Check if a pitcher is likely an opener based on recent usage.

    Returns dict with is_opener flag and avg innings per appearance.
    """
    pitcher_pas = df[df["pitcher_id"] == pitcher_id]
    if len(pitcher_pas) == 0:
        return {"is_opener": False, "avg_ip_approx": None, "appearances": 0, "note": "unknown pitcher"}

    # Approximate IP from PA count per game (rough: 3 PA ≈ 1 IP)
    game_pas = pitcher_pas.groupby(["date", "game_pk"]).size().reset_index(name="pa_count")
    game_pas["approx_ip"] = game_pas["pa_count"] / 3.0

    recent = game_pas.tail(10)  # last 10 appearances
    avg_ip = recent["approx_ip"].mean()
    n_apps = len(recent)

    is_opener = avg_ip < OPENER_IP_THRESHOLD and n_apps >= OPENER_MIN_APPEARANCES

    return {
        "is_opener": is_opener,
        "avg_ip_approx": round(avg_ip, 1),
        "appearances": n_apps,
        "note": "likely opener/reliever" if is_opener else "",
    }


def _build_feature_lookups(df: pd.DataFrame) -> dict:
    """Build lookup tables for latest feature values per entity."""
    lookups = {}

    # Batter rolling features
    batter_cols = ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g",
                   "batter_whiff_60g", "batter_count_tendency_30g", "batter_gb_hit_rate"]
    lookups["batter"] = {}
    for col in batter_cols:
        lookups["batter"][col] = df.dropna(subset=[col]).groupby("batter_id")[col].last().to_dict()

    # Platoon
    lookups["platoon"] = df.dropna(subset=["platoon_hr"]).groupby(
        ["batter_id", "pitch_hand"]
    )["platoon_hr"].last().to_dict()

    # Pitcher
    lookups["pitcher_hr"] = df.dropna(subset=["pitcher_hr_30g"]).groupby(
        "pitcher_id"
    )["pitcher_hr_30g"].last().to_dict()
    lookups["pitcher_ent"] = df.dropna(subset=["pitcher_entropy_30g"]).groupby(
        "pitcher_id"
    )["pitcher_entropy_30g"].last().to_dict()

    # Batter Statcast features
    batter_statcast_cols = ["batter_barrel_rate_30g", "batter_hard_hit_rate_30g",
                            "batter_sweet_spot_rate_30g", "batter_avg_ev_30g",
                            "batter_avg_velo_faced_30g"]
    for col in batter_statcast_cols:
        if col in df.columns:
            lookups["batter"][col] = df.dropna(subset=[col]).groupby("batter_id")[col].last().to_dict()

    # Catcher framing proxy
    if "pitcher_catcher_framing" in df.columns:
        lookups["pitcher_framing"] = df.dropna(subset=["pitcher_catcher_framing"]).groupby(
            "pitcher_id"
        )["pitcher_catcher_framing"].last().to_dict()

    # Pitcher Statcast features
    pitcher_statcast_cols = ["pitcher_avg_velo_30g", "pitcher_avg_spin_30g",
                             "pitcher_avg_extension_30g", "pitcher_break_total_30g"]
    lookups["pitcher_statcast"] = {}
    for col in pitcher_statcast_cols:
        if col in df.columns:
            lookups["pitcher_statcast"][col] = df.dropna(subset=[col]).groupby(
                "pitcher_id"
            )[col].last().to_dict()

    # Park factor
    lookups["park"] = df.dropna(subset=["park_factor"]).groupby(
        "venue_id"
    )["park_factor"].last().to_dict()

    # Batter last played date
    lookups["last_date"] = df.groupby("batter_id")["date"].max().to_dict()

    # --- Team bullpen composite ---
    # Identify relievers: pitchers who aren't the first pitcher faced by batters in a game
    import numpy as np
    game_starters = {}
    for gpk, game in df.groupby("game_pk"):
        for is_home in [True, False]:
            side = game[game["is_home"] == is_home]
            if len(side) > 0:
                game_starters[(gpk, is_home)] = side.iloc[0]["pitcher_id"]

    df_copy = df.copy()
    df_copy["_starter_id"] = df_copy.apply(
        lambda r: game_starters.get((r["game_pk"], r["is_home"])), axis=1
    )
    df_copy["_is_reliever"] = df_copy["pitcher_id"] != df_copy["_starter_id"]

    # Reliever PAs, grouped by opposing team (the team whose relievers are pitching)
    # The "opposing team" for a batter is identified by NOT is_home
    # We need team_id — approximate from venue or game structure
    # Simpler: group by (game_pk, pitcher side) to get team identity
    # Actually, we need team_id in the data. Let's use a simpler approach:
    # compute bullpen stats per pitcher, then average across a team's relievers
    reliever_pas = df_copy[df_copy["_is_reliever"]]
    if len(reliever_pas) > 0:
        # Per-pitcher reliever stats (last 30 dates)
        rp_dates = reliever_pas.groupby(["pitcher_id", "date"]).agg(
            rp_hits=("is_hit", "sum"), rp_pas=("is_hit", "count"),
        ).reset_index().sort_values(["pitcher_id", "date"])
        rp_dates["rp_hr"] = rp_dates["rp_hits"] / rp_dates["rp_pas"]
        rp_dates["rp_hr_30g"] = rp_dates.groupby("pitcher_id")["rp_hr"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=5).mean()
        )
        lookups["reliever_hr"] = rp_dates.dropna(subset=["rp_hr_30g"]).groupby(
            "pitcher_id"
        )["rp_hr_30g"].last().to_dict()

        # League-average reliever stats as fallback for debut pitchers
        all_reliever_hr = reliever_pas["is_hit"].mean()
        lookups["league_avg_reliever_hr"] = all_reliever_hr
    else:
        lookups["reliever_hr"] = {}
        lookups["league_avg_reliever_hr"] = 0.22

    # League-average starter stats for debut pitcher fallback
    starter_pas = df_copy[~df_copy["_is_reliever"]]
    lookups["league_avg_starter_hr"] = starter_pas["is_hit"].mean() if len(starter_pas) > 0 else 0.22
    lookups["league_avg_entropy"] = df.dropna(subset=["pitcher_entropy_30g"])["pitcher_entropy_30g"].mean()

    return lookups


def _fetch_prior_lineup(team_id: int) -> list[dict]:
    """Fetch a team's most recent game lineup as fallback.

    Returns list of {batter_id, batter_name, lineup} dicts.
    """
    try:
        sched = json.loads(urlopen(
            f"{API_BASE}/api/v1/schedule?sportId=1&teamId={team_id}"
            f"&startDate=2026-03-20&endDate=2026-12-31&gameType=R",
            timeout=15,
        ).read())
        # Walk backwards through dates to find most recent Final game
        all_games = []
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                if g["status"]["detailedState"] == "Final":
                    all_games.append(g)
        if not all_games:
            return []

        last_game = all_games[-1]
        feed = json.loads(urlopen(
            f"{API_BASE}/api/v1.1/game/{last_game['gamePk']}/feed/live", timeout=15
        ).read())
        bs = feed["liveData"]["boxscore"]

        # Figure out which side this team was on
        for side_name in ["away", "home"]:
            side = bs["teams"][side_name]
            players = side.get("players", {})
            lineup = []
            for key, player in players.items():
                bo = player.get("battingOrder")
                if bo and int(bo) <= 900:
                    lineup.append({
                        "batter_id": player["person"]["id"],
                        "batter_name": player["person"]["fullName"],
                        "lineup": int(bo) // 100,
                    })
            if lineup:
                # Verify this is the right team by checking team ID
                side_team_id = feed["gameData"]["teams"][side_name]["id"]
                if side_team_id == team_id:
                    return sorted(lineup, key=lambda x: x["lineup"])
    except Exception:
        pass
    return []


def _fetch_game_slots(date: str) -> list[dict]:
    """Fetch all batter-game slots for a date from MLB API.

    When lineups aren't posted yet (pre-game), falls back to each team's
    most recent lineup with a PROJECTED flag. Uses probable pitchers from
    the schedule when play data isn't available.

    Returns list of dicts with batter/pitcher/venue/weather info.
    """
    sched = json.loads(urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}"
        f"&hydrate=probablePitcher", timeout=15
    ).read())

    slots = []
    projected_count = 0
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            pk = g["gamePk"]
            status = g["status"]["detailedState"]
            game_time = g.get("gameDate", "")

            try:
                feed = json.loads(urlopen(
                    f"{API_BASE}/api/v1.1/game/{pk}/feed/live", timeout=15
                ).read())
            except Exception:
                continue

            gd = feed["gameData"]
            venue_id = gd["venue"]["id"]
            temp_str = gd.get("weather", {}).get("temp")
            temp = int(temp_str) if temp_str else None
            boxscore = feed["liveData"]["boxscore"]
            plays = feed["liveData"]["plays"]["allPlays"]

            # Build pitcher hand lookup from plays
            pitcher_hands = {}
            for play in plays:
                pid = play["matchup"]["pitcher"]["id"]
                ph = play["matchup"].get("pitchHand", {}).get("code")
                if ph:
                    pitcher_hands[pid] = ph

            for side in ["away", "home"]:
                opp = "home" if side == "away" else "away"
                team_abbr = gd["teams"][side]["abbreviation"]
                team_id = gd["teams"][side]["id"]
                target_half = "top" if side == "away" else "bottom"

                # Find opposing pitcher — from plays if available, else from schedule
                opp_pitcher_id = None
                opp_pitcher_name = "TBD"
                opp_pitcher_hand = None
                for play in plays:
                    if play["about"]["halfInning"] == target_half:
                        opp_pitcher_id = play["matchup"]["pitcher"]["id"]
                        opp_pitcher_name = play["matchup"]["pitcher"]["fullName"]
                        opp_pitcher_hand = pitcher_hands.get(opp_pitcher_id)
                        break

                if opp_pitcher_id is None:
                    # Use probable pitcher from schedule
                    pp = g["teams"][opp].get("probablePitcher", {})
                    if pp:
                        opp_pitcher_id = pp.get("id")
                        opp_pitcher_name = pp.get("fullName", "TBD")

                # Get lineup — from boxscore if posted, else fallback to prior game
                players = boxscore["teams"][side]["players"]
                lineup_players = []
                for key, player in players.items():
                    bo = player.get("battingOrder")
                    if bo and int(bo) <= 900:
                        lineup_players.append({
                            "batter_id": player["person"]["id"],
                            "batter_name": player["person"]["fullName"],
                            "lineup": int(bo) // 100,
                        })

                is_projected = False
                if not lineup_players:
                    # Fallback: use prior game's lineup
                    lineup_players = _fetch_prior_lineup(team_id)
                    is_projected = True
                    if lineup_players:
                        projected_count += 1

                for lp in lineup_players:
                    slot = {
                        "batter_id": lp["batter_id"],
                        "batter_name": lp["batter_name"],
                        "team": team_abbr,
                        "lineup": lp["lineup"],
                        "pitcher_id": opp_pitcher_id,
                        "pitcher_name": opp_pitcher_name,
                        "pitcher_hand": opp_pitcher_hand,
                        "venue_id": venue_id,
                        "weather_temp": temp,
                        "game_pk": pk,
                        "game_time": game_time,
                        "status": status,
                    }
                    if is_projected:
                        slot["projected"] = True
                    slots.append(slot)

    if projected_count > 0:
        print(f"  NOTE: {projected_count} teams using projected lineups (prior game)")

    return slots


def predict(
    date: str,
    df: pd.DataFrame,
    model,
    lookups: dict,
    check_openers: bool = True,
    blend: dict | None = None,
) -> pd.DataFrame:
    """Generate ranked BTS picks for a date.

    Args:
        date: YYYY-MM-DD
        df: Feature-enriched PA DataFrame (for opener checks)
        model: Trained LightGBM model (used when blend is None)
        lookups: Feature lookup tables from _build_feature_lookups
        check_openers: Whether to flag likely openers
        blend: Optional dict from train_blend. If provided, uses 12-model
            blend for ranking instead of single model.

    Returns:
        DataFrame of picks sorted by P(game hit), with flags for edge cases.
    """
    today = pd.Timestamp(date)
    slots = _fetch_game_slots(date)

    if not slots:
        return pd.DataFrame()

    # Build features for each slot
    rows = []
    for slot in slots:
        row = {}
        bid = slot["batter_id"]

        # Batter features from lookups (baseline + Statcast)
        for col in ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g",
                     "batter_whiff_60g", "batter_count_tendency_30g", "batter_gb_hit_rate",
                     "batter_barrel_rate_30g", "batter_hard_hit_rate_30g",
                     "batter_sweet_spot_rate_30g", "batter_avg_ev_30g",
                     "batter_avg_velo_faced_30g"]:
            row[col] = lookups["batter"].get(col, {}).get(bid)

        # Platoon
        row["platoon_hr"] = lookups["platoon"].get((bid, slot["pitcher_hand"]))

        # Pitcher — with debut fallback
        pitcher_hr = lookups["pitcher_hr"].get(slot["pitcher_id"])
        pitcher_ent = lookups["pitcher_ent"].get(slot["pitcher_id"])
        is_debut = pitcher_hr is None
        if is_debut:
            pitcher_hr = lookups.get("league_avg_starter_hr", 0.22)
            pitcher_ent = lookups.get("league_avg_entropy")

        row["pitcher_hr_30g"] = pitcher_hr
        row["pitcher_entropy_30g"] = pitcher_ent
        row["pitcher_catcher_framing"] = lookups.get("pitcher_framing", {}).get(slot["pitcher_id"])

        # Pitcher Statcast features
        for col in ["pitcher_avg_velo_30g", "pitcher_avg_spin_30g",
                     "pitcher_avg_extension_30g", "pitcher_break_total_30g"]:
            row[col] = lookups.get("pitcher_statcast", {}).get(col, {}).get(slot["pitcher_id"])

        # Context
        row["weather_temp"] = slot["weather_temp"]
        row["park_factor"] = lookups["park"].get(slot["venue_id"])

        # Days rest
        last = lookups["last_date"].get(bid)
        row["days_rest"] = (today - last).days if last else None

        # Flags
        flags = []
        if row["days_rest"] and row["days_rest"] > IL_RETURN_DAYS_THRESHOLD:
            flags.append(f"IL? ({int(row['days_rest'])}d rest)")

        is_opener = False
        if check_openers and slot["pitcher_id"]:
            opener_info = _check_opener(slot["pitcher_id"], df)
            if opener_info["is_opener"]:
                is_opener = True
                flags.append(f"OPENER ({opener_info['avg_ip_approx']}ip avg)")

        if is_debut:
            flags.append("DEBUT pitcher (using league avg)")

        if slot.get("projected"):
            flags.append("PROJECTED lineup")

        rows.append({**slot, **row, "flags": ", ".join(flags), "_is_opener": is_opener})

    pred_df = pd.DataFrame(rows)

    # --- Single-model prediction (used for display and as fallback) ---
    # Ensure numeric dtypes (lookups can produce mixed object columns)
    for col in FEATURE_COLS:
        if col in pred_df.columns:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

    feat_df = pred_df[FEATURE_COLS]
    valid = feat_df.notna().any(axis=1)
    pred_df.loc[valid, "p_hit_vs_starter"] = model.predict_proba(feat_df[valid])[:, 1]

    # Reliever features: swap pitcher features for league averages
    feat_reliever = feat_df.copy()
    feat_reliever["pitcher_hr_30g"] = lookups.get("league_avg_reliever_hr", 0.22)
    feat_reliever["pitcher_entropy_30g"] = lookups.get("league_avg_entropy")
    pred_df.loc[valid, "p_hit_vs_reliever"] = model.predict_proba(feat_reliever[valid])[:, 1]

    # Split aggregation: PAs 1-2.5 face starter, PAs 3+ face relievers
    pa_est = {1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0, 6: 3.9, 7: 3.8, 8: 3.7, 9: 3.6}
    pred_df["est_pas"] = pred_df["lineup"].map(pa_est).fillna(4.0)

    starter_pas = 2.5
    pred_df["reliever_pas"] = (pred_df["est_pas"] - starter_pas).clip(lower=0)
    pred_df["starter_pas"] = pred_df["est_pas"] - pred_df["reliever_pas"]

    # For openers, ALL PAs effectively face relievers
    if "_is_opener" in pred_df.columns:
        pred_df.loc[pred_df["_is_opener"], "reliever_pas"] = pred_df.loc[pred_df["_is_opener"], "est_pas"]
        pred_df.loc[pred_df["_is_opener"], "starter_pas"] = 0

    # P(>=1 hit) = 1 - P(no hit in starter PAs) * P(no hit in reliever PAs)
    pred_df["p_game_hit"] = 1 - (
        (1 - pred_df["p_hit_vs_starter"]) ** pred_df["starter_pas"] *
        (1 - pred_df["p_hit_vs_reliever"]) ** pred_df["reliever_pas"]
    )

    # Convenience: average PA probability for display
    pred_df["p_hit_pa"] = (
        pred_df["p_hit_vs_starter"] * pred_df["starter_pas"] +
        pred_df["p_hit_vs_reliever"] * pred_df["reliever_pas"]
    ) / pred_df["est_pas"]

    # --- Blend ranking (if blend is provided) ---
    if blend:
        # Each blend model predicts P(hit|PA) for starter features,
        # then we aggregate to game level and average across models.
        blend_game_scores = {}
        for name, (bmodel, bcols) in blend.items():
            bfeat = pred_df[bcols]
            bvalid = bfeat.notna().any(axis=1)
            p_starter = pd.Series(np.nan, index=pred_df.index)
            if bvalid.any():
                p_starter[bvalid] = bmodel.predict_proba(bfeat[bvalid])[:, 1]

            # Game-level: 1 - (1-p)^starter_pas * (1-p_reliever)^reliever_pas
            # Use single-model reliever estimate (blend is for starter matchup ranking)
            p_game = 1 - (
                (1 - p_starter) ** pred_df["starter_pas"] *
                (1 - pred_df["p_hit_vs_reliever"]) ** pred_df["reliever_pas"]
            )

            for idx, val in p_game.items():
                if pd.notna(val):
                    bid = pred_df.at[idx, "batter_id"]
                    if bid not in blend_game_scores:
                        blend_game_scores[bid] = []
                    blend_game_scores[bid].append(val)

        # Average across models per batter → blend rank
        blend_avg = {bid: np.mean(scores) for bid, scores in blend_game_scores.items()}
        pred_df["p_game_blend"] = pred_df["batter_id"].map(blend_avg)

        # Use blend score for ranking, keep single-model scores for display
        pred_df["p_game_hit"] = pred_df["p_game_blend"].fillna(pred_df["p_game_hit"])

    return pred_df.sort_values("p_game_hit", ascending=False).reset_index(drop=True)


def _refresh_season_data(date: str, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Pull recent game feeds and rebuild current season parquet.

    Downloads any new Final games from season start through yesterday,
    then rebuilds the season's parquet. Skips already-downloaded games.
    """
    from bts.data.pull import pull_feeds
    from bts.data.build import build_season
    from datetime import datetime, timedelta

    pred_date = datetime.strptime(date, "%Y-%m-%d")
    season = pred_date.year
    yesterday = (pred_date - timedelta(days=1)).strftime("%Y-%m-%d")
    season_start = f"{season}-03-20"

    if yesterday < season_start:
        return  # Too early in the year

    raw = Path(raw_dir)
    proc = Path(processed_dir)

    print(f"  Refreshing {season} data through {yesterday}...", file=__import__('sys').stderr)
    paths = pull_feeds(season_start, yesterday, raw, delay=0.3)
    print(f"  {len(paths)} game feeds ({sum(1 for _ in (raw / str(season)).glob('*.json'))} total)", file=__import__('sys').stderr)

    output_path = proc / f"pa_{season}.parquet"
    df = build_season(raw, output_path, season)
    print(f"  Rebuilt {output_path.name}: {len(df)} PAs", file=__import__('sys').stderr)


def run_pipeline(
    date: str,
    data_dir: str = "data/processed",
    check_openers: bool = True,
    cached_blend: dict | None = None,
    save_blend_path=None,
    refresh_data: bool = True,
) -> pd.DataFrame:
    """Run the full prediction pipeline for a date.

    Loads historical data, computes features, trains the 12-model blend,
    and returns ranked picks sorted by P(game hit).

    Args:
        cached_blend: Pre-trained blend dict to skip retraining. Must include
            a "_model" key with the single model.
        save_blend_path: If provided and no cached_blend, save the trained
            blend (including single model) to this path.
        refresh_data: Pull latest game feeds and rebuild current season
            parquet before running. Set False for backtesting.
    """
    if refresh_data:
        _refresh_season_data(date, processed_dir=data_dir)

    proc = Path(data_dir)
    dfs = []
    for parquet in sorted(proc.glob("pa_*.parquet")):
        dfs.append(pd.read_parquet(parquet))
    if not dfs:
        raise RuntimeError("No Parquet files found. Run 'bts data build' first.")

    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)
    df["date"] = pd.to_datetime(df["date"])

    if cached_blend:
        model = cached_blend.pop("_model")
        blend = cached_blend
    else:
        model = train_model(df)
        blend = train_blend(df)
        if save_blend_path:
            to_save = {**blend, "_model": model}
            save_blend(to_save, save_blend_path)

    lookups = _build_feature_lookups(df)

    return predict(
        date, df, model, lookups,
        check_openers=check_openers,
        blend=blend,
    )

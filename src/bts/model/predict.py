"""Daily BTS prediction: generate ranked picks for a given date."""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from urllib.request import urlopen

from bts.features.compute import compute_all_features, FEATURE_COLS, TRAIN_START_YEAR

API_BASE = "https://statsapi.mlb.com"

# Pitchers averaging < 3 IP over their recent appearances are likely openers/relievers
OPENER_IP_THRESHOLD = 3.0
OPENER_MIN_APPEARANCES = 5

# Players with this many days rest get flagged as IL return risk
IL_RETURN_DAYS_THRESHOLD = 7


def train_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """Train LightGBM on historical PA data (2019+)."""
    train = df[df["season"] >= TRAIN_START_YEAR]
    train_X = train[FEATURE_COLS]
    train_y = train["is_hit"]
    mask = train_X.notna().all(axis=1)

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(train_X[mask], train_y[mask])
    return model


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

    # Park factor
    lookups["park"] = df.dropna(subset=["park_factor"]).groupby(
        "venue_id"
    )["park_factor"].last().to_dict()

    # Batter last played date
    lookups["last_date"] = df.groupby("batter_id")["date"].max().to_dict()

    return lookups


def _fetch_game_slots(date: str) -> list[dict]:
    """Fetch all batter-game slots for a date from MLB API.

    Returns list of dicts with batter/pitcher/venue/weather info.
    """
    sched = json.loads(urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}", timeout=15
    ).read())

    slots = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            pk = g["gamePk"]
            status = g["status"]["detailedState"]

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

            # Build pitcher hand lookup
            pitcher_hands = {}
            for play in plays:
                pid = play["matchup"]["pitcher"]["id"]
                ph = play["matchup"].get("pitchHand", {}).get("code")
                if ph:
                    pitcher_hands[pid] = ph

            for side in ["away", "home"]:
                opp = "home" if side == "away" else "away"
                team_abbr = gd["teams"][side]["abbreviation"]
                target_half = "bottom" if side == "away" else "top"

                # Find opposing pitcher
                opp_pitcher_id = None
                opp_pitcher_name = "TBD"
                opp_pitcher_hand = None
                for play in plays:
                    if play["about"]["halfInning"] == target_half:
                        opp_pitcher_id = play["matchup"]["pitcher"]["id"]
                        opp_pitcher_name = play["matchup"]["pitcher"]["fullName"]
                        opp_pitcher_hand = pitcher_hands.get(opp_pitcher_id)
                        break

                players = boxscore["teams"][side]["players"]
                for key, player in players.items():
                    bo = player.get("battingOrder")
                    if not bo or int(bo) > 900:
                        continue

                    slots.append({
                        "batter_id": player["person"]["id"],
                        "batter_name": player["person"]["fullName"],
                        "team": team_abbr,
                        "lineup": int(bo) // 100,
                        "pitcher_id": opp_pitcher_id,
                        "pitcher_name": opp_pitcher_name,
                        "pitcher_hand": opp_pitcher_hand,
                        "venue_id": venue_id,
                        "weather_temp": temp,
                        "game_pk": pk,
                        "status": status,
                    })

    return slots


def predict(
    date: str,
    df: pd.DataFrame,
    model: lgb.LGBMClassifier,
    lookups: dict,
    check_openers: bool = True,
) -> pd.DataFrame:
    """Generate ranked BTS picks for a date.

    Args:
        date: YYYY-MM-DD
        df: Feature-enriched PA DataFrame (for opener checks)
        model: Trained LightGBM model
        lookups: Feature lookup tables from _build_feature_lookups
        check_openers: Whether to flag likely openers

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

        # Batter features from lookups
        for col in ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g",
                     "batter_whiff_60g", "batter_count_tendency_30g", "batter_gb_hit_rate"]:
            row[col] = lookups["batter"][col].get(bid)

        # Platoon
        row["platoon_hr"] = lookups["platoon"].get((bid, slot["pitcher_hand"]))

        # Pitcher
        row["pitcher_hr_30g"] = lookups["pitcher_hr"].get(slot["pitcher_id"])
        row["pitcher_entropy_30g"] = lookups["pitcher_ent"].get(slot["pitcher_id"])

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

        if check_openers and slot["pitcher_id"]:
            opener_info = _check_opener(slot["pitcher_id"], df)
            if opener_info["is_opener"]:
                flags.append(f"OPENER ({opener_info['avg_ip_approx']}ip avg)")

        rows.append({**slot, **row, "flags": ", ".join(flags)})

    pred_df = pd.DataFrame(rows)

    # Predict
    feat_df = pred_df[FEATURE_COLS]
    valid = feat_df.notna().all(axis=1)
    pred_df.loc[valid, "p_hit_pa"] = model.predict_proba(feat_df[valid])[:, 1]

    # Aggregate to game level
    pa_est = {1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0, 6: 3.9, 7: 3.8, 8: 3.7, 9: 3.6}
    pred_df["est_pas"] = pred_df["lineup"].map(pa_est).fillna(4.0)
    pred_df["p_game_hit"] = 1 - (1 - pred_df["p_hit_pa"]) ** pred_df["est_pas"]

    return pred_df.sort_values("p_game_hit", ascending=False).reset_index(drop=True)

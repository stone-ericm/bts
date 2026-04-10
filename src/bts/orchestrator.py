"""Pi5 orchestrator: cascade model runs across compute machines via SSH."""

import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_config(path: Path) -> dict:
    """Load orchestrator config from TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def ssh_predict(
    ssh_host: str,
    bts_dir: str,
    date: str,
    timeout_sec: int = 300,
    platform: str = "unix",
) -> pd.DataFrame | None:
    """Run bts predict-json on a remote machine via SSH.

    Returns predictions DataFrame on success, None on any failure.
    """
    if platform == "windows":
        cmd = (
            f"cd /d {bts_dir} && "
            f"git pull -q origin main && "
            f"%USERPROFILE%\\.local\\bin\\uv run bts predict-json --date {date}"
        )
    else:
        cmd = (
            f"export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && "
            f"cd {bts_dir} && "
            f"git pull -q origin main && "
            f"UV_CACHE_DIR=/tmp/uv-cache uv run bts predict-json --date {date}"
        )
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             ssh_host, cmd],
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{ssh_host}] Timeout after {timeout_sec}s", file=sys.stderr)
        return None
    except OSError as e:
        print(f"  [{ssh_host}] SSH error: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  [{ssh_host}] Exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  [{ssh_host}] Invalid JSON output", file=sys.stderr)
        return None

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def predict_local(
    date: str,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
) -> pd.DataFrame | None:
    """Run predictions locally in-process (no SSH cascade).

    Used when the scheduler runs on the same machine as the data and models
    (i.e., on the Fly cloud VM). Returns None on any failure, matching
    ssh_predict's contract.
    """
    from bts.model.predict import run_pipeline, load_blend
    from pathlib import Path

    models_path = Path(models_dir)
    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        print(f"  [local] Loading cached model from {cache_path}", file=sys.stderr)
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
        return predictions
    except Exception as e:
        print(f"  [local] Prediction failed: {e}", file=sys.stderr)
        return None


def predict_local_shadow(
    date: str,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
) -> pd.DataFrame | None:
    """Run shadow predictions locally with context_stack features.

    Same as predict_local but uses FEATURE_COLS + CONTEXT_COLS.
    Gets its own model cache (blend_{date}_shadow.pkl).
    """
    from bts.model.predict import run_pipeline, load_blend
    from bts.features.compute import FEATURE_COLS, CONTEXT_COLS
    from pathlib import Path

    shadow_cols = FEATURE_COLS + CONTEXT_COLS
    models_path = Path(models_dir)
    cache_path = models_path / f"blend_{date}_shadow.pkl"
    cached_blend = None
    if cache_path.exists():
        print(f"  [shadow] Loading cached shadow model from {cache_path}", file=sys.stderr)
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
            refresh_data=False,  # data already refreshed by production run
            feature_cols_override=shadow_cols,
        )
        return predictions
    except Exception as e:
        print(f"  [shadow] Shadow prediction failed: {e}", file=sys.stderr)
        return None


def run_cascade(
    tiers: list[dict],
    date: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """Try each tier in order until one succeeds.

    Returns (predictions_df, tier_name) or (None, None) if all fail.
    """
    for tier in tiers:
        name = tier["name"]
        tier_type = tier.get("type", "ssh")  # Default ssh for backward compat
        print(f"Trying {name} ({tier_type})...", file=sys.stderr)

        if tier_type == "local":
            df = predict_local(date=date)
        elif tier_type == "ssh":
            df = ssh_predict(
                tier["ssh_host"],
                tier["bts_dir"],
                date,
                timeout_sec=tier["timeout_min"] * 60,
                platform=tier.get("platform", "unix"),
            )
        else:
            print(f"  [{name}] Unknown tier type: {tier_type}", file=sys.stderr)
            continue

        if df is not None:
            print(f"  [{name}] Success — {len(df)} predictions", file=sys.stderr)
            return df, name

    return None, None


def run_and_pick(
    config: dict,
    date: str,
) -> tuple[pd.DataFrame | None, "PickResult | None", str | None]:
    """Run cascade and apply strategy. No posting, no DMs.

    Returns (predictions, pick_result, tier_name).
    predictions is None if all tiers fail.
    pick_result is None if skip or no games.
    """
    from bts.picks import load_streak
    from bts.strategy import select_pick

    picks_dir = Path(config["orchestrator"]["picks_dir"])

    predictions, tier_name = run_cascade(config["tiers"], date)
    if predictions is None or predictions.empty:
        return predictions, None, tier_name

    streak = load_streak(picks_dir)
    result = select_pick(predictions, date, picks_dir, streak=streak)

    return predictions, result, tier_name


def orchestrate(config_path: Path, date: str) -> bool:
    """Run the full orchestration: cascade -> strategy -> save -> post.

    Returns True if a pick was made, False otherwise.
    """
    from bts.dm import send_dm
    from bts.picks import save_pick, load_streak
    from bts.posting import format_post, format_skip_post, post_to_bluesky, should_post_now

    config = load_config(config_path)
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    dm_recipient = config["bluesky"]["dm_recipient"]

    predictions, result, tier_name = run_and_pick(config, date)

    if predictions is None:
        msg = f"BTS {date}: All compute tiers failed. No pick made."
        print(msg, file=sys.stderr)
        try:
            send_dm(dm_recipient, msg)
            print(f"  DM sent to {dm_recipient}", file=sys.stderr)
        except Exception as e:
            print(f"  DM failed: {e}", file=sys.stderr)
        return False

    if predictions.empty:
        print(f"No games found for {date}.", file=sys.stderr)
        return False

    if result is None:
        streak = load_streak(picks_dir)
        top = predictions.iloc[0] if not predictions.empty else None
        if top is not None:
            print(f"Skipping — {top['batter_name']} at {top['p_game_hit']:.1%} "
                  f"below threshold. Streak holds at {streak}.", file=sys.stderr)
            if should_post_now(top.get("game_time", ""), False):
                text = format_skip_post(top["batter_name"], top.get("team", "?"),
                                        top["p_game_hit"], streak)
                try:
                    uri = post_to_bluesky(text)
                    print(f"  Posted skip to Bluesky: {uri}", file=sys.stderr)
                except Exception as e:
                    print(f"  Bluesky skip post failed: {e}", file=sys.stderr)
        else:
            print(f"No valid picks. Streak holds at {streak}.", file=sys.stderr)
        return False

    if result.locked:
        print(f"Pick locked: {result.daily.pick.batter_name}", file=sys.stderr)
        # Catch-up posting
        if not result.daily.bluesky_posted:
            streak = load_streak(picks_dir)
            text = format_post(
                result.daily.pick.batter_name, result.daily.pick.team,
                result.daily.pick.pitcher_name, result.daily.pick.p_game_hit,
                streak,
                result.daily.double_down.batter_name if result.daily.double_down else None,
                result.daily.double_down.p_game_hit if result.daily.double_down else None,
                result.daily.double_down.team if result.daily.double_down else None,
                result.daily.double_down.pitcher_name if result.daily.double_down else None,
            )
            try:
                uri = post_to_bluesky(text)
                result.daily.bluesky_posted = True
                result.daily.bluesky_uri = uri
                save_pick(result.daily, picks_dir)
                print(f"  Posted to Bluesky (catch-up): {uri}", file=sys.stderr)
            except Exception as e:
                print(f"  Bluesky catch-up failed: {e}", file=sys.stderr)
        return True

    # New pick
    daily = result.daily
    save_pick(daily, picks_dir)
    print(
        f"Pick ({tier_name}): {daily.pick.batter_name} "
        f"({daily.pick.p_game_hit:.1%})",
        file=sys.stderr,
    )

    # Post to Bluesky
    streak = load_streak(picks_dir)
    if should_post_now(daily.pick.game_time, daily.bluesky_posted):
        text = format_post(
            daily.pick.batter_name, daily.pick.team,
            daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
            daily.double_down.batter_name if daily.double_down else None,
            daily.double_down.p_game_hit if daily.double_down else None,
            daily.double_down.team if daily.double_down else None,
            daily.double_down.pitcher_name if daily.double_down else None,
        )
        try:
            uri = post_to_bluesky(text)
            daily.bluesky_posted = True
            daily.bluesky_uri = uri
            save_pick(daily, picks_dir)
            print(f"  Posted to Bluesky: {uri}", file=sys.stderr)
        except Exception as e:
            print(f"  Bluesky post failed: {e}", file=sys.stderr)

    return True

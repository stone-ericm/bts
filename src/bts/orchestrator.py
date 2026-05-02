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


def _attach_conformal_lower_bounds(
    predictions: pd.DataFrame,
    conformal_dir: Path = Path("data/conformal"),
) -> pd.DataFrame:
    """Attach 6 conformal-lower-bound columns to predictions DataFrame.

    Gated by ``BTS_USE_CONFORMAL=1`` env var (default OFF; set to "1" in
    bts-hetzner .env after the validation gate passes). When OFF, returns
    predictions unchanged. When ON but no calibrator file exists, attaches
    the columns as all-None (graceful degradation; allows pre-shipping
    deploy of column infrastructure).
    """
    import os
    if os.environ.get("BTS_USE_CONFORMAL", "0") != "1":
        return predictions

    from bts.model.conformal import (
        apply_weighted_mondrian_conformal,
        apply_bucket_wilson,
    )

    # Find the most recent calibrator files
    if not conformal_dir.exists():
        # Attach all-None columns and return
        for method in ("conformal", "wilson"):
            for alpha_pct in (95, 90, 80):
                predictions[f"p_game_hit_lower_{method}_{alpha_pct}"] = None
        return predictions

    cal_files = sorted(conformal_dir.glob("calibrator_*.pkl"))
    wilson_files = sorted(conformal_dir.glob("wilson_calibrator_*.pkl"))
    if not cal_files or not wilson_files:
        for method in ("conformal", "wilson"):
            for alpha_pct in (95, 90, 80):
                predictions[f"p_game_hit_lower_{method}_{alpha_pct}"] = None
        return predictions

    import joblib
    cal = joblib.load(cal_files[-1])
    wilson = joblib.load(wilson_files[-1])

    for alpha_idx, alpha_pct in enumerate((95, 90, 80)):
        col_c = f"p_game_hit_lower_conformal_{alpha_pct}"
        col_w = f"p_game_hit_lower_wilson_{alpha_pct}"
        predictions[col_c] = predictions["p_game_hit"].apply(
            lambda p, ai=alpha_idx: apply_weighted_mondrian_conformal(cal, p, ai)
        )
        predictions[col_w] = predictions["p_game_hit"].apply(
            lambda p, ai=alpha_idx: apply_bucket_wilson(wilson, p, ai)
        )

    return predictions


def predict_local(
    date: str,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
    picks_dir: str = "data/picks",
) -> pd.DataFrame | None:
    """Run predictions locally in-process (no SSH cascade).

    Used when the scheduler runs on the same machine as the data and models
    (i.e., on the Fly cloud VM). Returns None on any failure, matching
    ssh_predict's contract.

    **Post-hoc calibration**: when env var ``BTS_USE_CALIBRATION=1`` is set,
    after run_pipeline produces predictions, this function fits an isotonic
    calibrator from the last 30 days of resolved picks (joined to actual
    day-level hit outcomes from the PA frame) and applies it to the
    ``p_game_hit`` column. Default OFF preserves identical-to-uncalibrated
    behavior. Enabled per project_bts_2026_05_01_morning_verdicts.md after
    the +6.6pp overall and +12.3pp [0.75, 0.80) over-confidence finding.
    """
    from bts.model.predict import run_pipeline, load_blend
    from pathlib import Path
    import os
    from datetime import date as _date

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
    except Exception as e:
        print(f"  [local] Prediction failed: {e}", file=sys.stderr)
        return None

    # Post-hoc calibration (opt-in via env var; default off).
    if os.environ.get("BTS_USE_CALIBRATION", "0") == "1" and predictions is not None and not predictions.empty:
        try:
            from bts.model.calibrate import fit_calibrator_from_picks, apply_calibrator_series
            # Fit calibrator from recent resolved picks against current PA frame.
            proc = Path(data_dir)
            current_year = int(date.split("-")[0])
            current_pa = proc / f"pa_{current_year}.parquet"
            if current_pa.exists():
                pa_df = pd.read_parquet(current_pa)
                today = _date.fromisoformat(date)
                cal = fit_calibrator_from_picks(Path(picks_dir), pa_df, today=today)
                if cal is not None:
                    raw = predictions["p_game_hit"].copy()
                    predictions["p_game_hit_raw"] = raw
                    predictions["p_game_hit"] = apply_calibrator_series(raw, cal)
                    n = len(predictions)
                    print(
                        f"  [local] Applied calibration to {n} predictions "
                        f"(top: raw={raw.max():.3f} → calibrated={predictions['p_game_hit'].max():.3f})",
                        file=sys.stderr,
                    )
                else:
                    print("  [local] Calibrator unavailable (insufficient resolved picks); using raw p", file=sys.stderr)
            else:
                print(f"  [local] No {current_pa.name}; calibration skipped", file=sys.stderr)
        except Exception as e:
            print(f"  [local] Calibration failed (non-fatal): {e}; using raw p", file=sys.stderr)

    # Conformal lower bounds (NEW 2026-05-01) — gated by BTS_USE_CONFORMAL
    predictions = _attach_conformal_lower_bounds(predictions)

    return predictions


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

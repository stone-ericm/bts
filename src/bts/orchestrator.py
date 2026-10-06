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

    return predictions


def shadow_feature_hash() -> str:
    """Short hash of the shadow feature set (FEATURE_COLS + CONTEXT_COLS).

    Baked into the shadow model cache filename so a cached model trained on a
    different context stack — or before/after the external park_drag table
    changed (including table-absent -> table-present same-day) — can never be
    loaded against today's inputs.
    """
    import hashlib
    from bts.features.compute import FEATURE_COLS, CONTEXT_COLS
    from bts.features import park_drag
    joined = ",".join(FEATURE_COLS + CONTEXT_COLS) + "|" + park_drag.artifact_fingerprint()
    return hashlib.md5(joined.encode()).hexdigest()[:8]


def shadow_cache_path(models_dir, date: str):
    """Canonical shadow blend cache path (feature-set-hashed)."""
    from pathlib import Path as _Path
    return _Path(models_dir) / f"blend_{date}_shadow_{shadow_feature_hash()}.pkl"


def predict_local_shadow(
    date: str,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
) -> pd.DataFrame | None:
    """Run shadow predictions locally with context_stack features.

    Same as predict_local but uses FEATURE_COLS + CONTEXT_COLS.
    Gets its own model cache (blend_{date}_shadow_{feature_hash}.pkl — the
    hash guards against loading a cache trained on a different context stack).
    """
    from bts.model.predict import run_pipeline, load_blend
    from bts.features.compute import FEATURE_COLS, CONTEXT_COLS
    from pathlib import Path

    shadow_cols = FEATURE_COLS + CONTEXT_COLS
    models_path = Path(models_dir)
    cache_path = shadow_cache_path(models_path, date)
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


def _contest_state_required(config: dict) -> bool:
    return bool(
        config.get("scheduler", {}).get("contest_state_required", False)
        or config.get("health_checks", {}).get("contest_state_expected", False)
    )


def run_and_pick(
    config: dict,
    date: str,
    *,
    require_detailed_statuses: bool = True,
    unavailable_game_pks: "set[int] | None" = None,
) -> tuple[pd.DataFrame | None, "SelectionResult | None", str | None]:
    """Run cascade and apply strategy. No posting, no DMs.

    Returns (predictions, sel, tier_name).
    predictions is None if all tiers fail.
    sel is None ONLY on the no-predictions / no-games early return (before
    select_pick is reached); otherwise it is a SelectionResult — check
    sel.pick_result for whether a pick was actually made.
    """
    from bts import progress
    from bts.contest_state import load_decision_streak_state
    from bts.picks import get_game_statuses_detailed
    from bts.strategy import select_pick

    picks_dir = Path(config["orchestrator"]["picks_dir"])

    progress.mark("running_cascade")
    predictions, tier_name = run_cascade(config["tiers"], date)
    if predictions is None or predictions.empty:
        return predictions, None, tier_name

    # Persist the full ranked slate (observability only — save_slate never
    # raises). Enables realized slate-level metrics; see bts/slate.py.
    from bts.slate import save_slate
    progress.mark("persisting_slate")
    save_slate(predictions, date, picks_dir, tier_name)

    progress.mark("loading_decision_state")
    decision_state = load_decision_streak_state(
        picks_dir,
        require_contest_state=_contest_state_required(config),
    )
    try:
        game_statuses_detailed = get_game_statuses_detailed(date)
    except Exception:
        game_statuses_detailed = None
    progress.mark("selecting_pick")
    sel = select_pick(
        predictions,
        date,
        picks_dir,
        streak=decision_state.streak,
        saver_available=decision_state.saver_available,
        allow_double=decision_state.allow_double,
        game_statuses_detailed=game_statuses_detailed,
        require_detailed_statuses=require_detailed_statuses,
        unavailable_game_pks=unavailable_game_pks,
        # Tail policy (2026-09-03): the contest season-best + its trust status.
        best_streak=getattr(decision_state, "best_streak", None),
        best_status=getattr(decision_state, "best_status", None),
    )
    # bts_daily_decision_v2 provenance: record WHICH state stream fed this
    # selection so decision records are exact by construction (the 8/09
    # boundary census found 31/44 v1 records state-null).
    sel.state_source = decision_state.source
    sel.state_status = decision_state.status
    sel.allow_double = decision_state.allow_double
    sel.contest_source_date = (
        decision_state.contest_source_date.isoformat()
        if decision_state.contest_source_date is not None else None
    )

    return predictions, sel, tier_name


def orchestrate(config_path: Path, date: str) -> bool:
    """Run the full orchestration: cascade -> strategy -> save -> post.

    Returns True if a pick was made, False otherwise.
    """
    from bts.dm import send_dm
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts.picks import save_pick
    from bts.posting import format_post, format_skip_post, post_to_bluesky, should_post_now

    config = load_config(config_path)
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    dm_recipient = config["bluesky"]["dm_recipient"]

    try:
        predictions, sel, tier_name = run_and_pick(config, date)
    except ContestStateError as e:
        msg = f"BTS {date}: contest-account streak state invalid. No pick made. ({e})"
        print(msg, file=sys.stderr)
        try:
            send_dm(dm_recipient, msg)
            print(f"  DM sent to {dm_recipient}", file=sys.stderr)
        except Exception as dm_error:
            print(f"  DM failed: {dm_error}", file=sys.stderr)
        return False

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

    result = sel.pick_result if sel is not None else None
    if result is None:
        decision_state = load_decision_streak_state(
            picks_dir,
            require_contest_state=_contest_state_required(config),
        )
        top = predictions.iloc[0] if not predictions.empty else None
        if top is not None:
            print(f"Skipping — {top['batter_name']} at {top['p_game_hit']:.1%} "
                  f"below threshold. Streak holds at {decision_state.streak}.", file=sys.stderr)
            if should_post_now(top.get("game_time", ""), False):
                text = format_skip_post(top["batter_name"], top.get("team", "?"),
                                        top["p_game_hit"], decision_state.streak)
                try:
                    uri = post_to_bluesky(text)
                    print(f"  Posted skip to Bluesky: {uri}", file=sys.stderr)
                except Exception as e:
                    print(f"  Bluesky skip post failed: {e}", file=sys.stderr)
        else:
            print(f"No valid picks. Streak holds at {decision_state.streak}.", file=sys.stderr)
        return False

    if result.locked:
        print(f"Pick locked: {result.daily.pick.batter_name}", file=sys.stderr)
        # Catch-up posting
        if not result.daily.bluesky_posted:
            decision_state = load_decision_streak_state(
                picks_dir,
                require_contest_state=_contest_state_required(config),
            )
            text = format_post(
                result.daily.pick.batter_name, result.daily.pick.team,
                result.daily.pick.pitcher_name, result.daily.pick.p_game_hit,
                decision_state.streak,
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

    # New pick — attach provenance v1 fields before saving (per Codex #168).
    daily = result.daily
    from bts.picks import attach_provenance
    from bts.simulate.mdp import DEFAULT_POLICY_PATH
    from bts.simulate.tail_policy import DEFAULT_TAIL_POLICY_PATH
    models_dir = config["orchestrator"].get("models_dir", "data/models")
    attach_provenance(
        daily,
        blend_path=Path(models_dir) / f"blend_{date}.pkl",
        policy_path=DEFAULT_POLICY_PATH,
        tail_path=DEFAULT_TAIL_POLICY_PATH,
    )
    save_pick(daily, picks_dir)
    print(
        f"Pick ({tier_name}): {daily.pick.batter_name} "
        f"({daily.pick.p_game_hit:.1%})",
        file=sys.stderr,
    )

    # Post to Bluesky
    decision_state = load_decision_streak_state(
        picks_dir,
        require_contest_state=_contest_state_required(config),
    )
    if should_post_now(daily.pick.game_time, daily.bluesky_posted):
        text = format_post(
            daily.pick.batter_name, daily.pick.team,
            daily.pick.pitcher_name, daily.pick.p_game_hit, decision_state.streak,
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

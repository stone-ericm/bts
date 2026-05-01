"""Post-hoc isotonic calibration for production p_game_hit.

Distribution shift between 2017-2025 training and 2026 production produces
systematic over-confidence in production picks. Diagnostic finding 2026-05-01:
the [0.75, 0.80) bucket realized 47.6% vs predicted 77.3% (gap +29.7pp),
and overconfidence scales monotonically with predicted P (gap +8.5pp at 0.65,
+20pp at 0.70, +29.7pp at 0.75-0.80).

This module fits an isotonic regression on (predicted_p, realized_hit) tuples
from a recent rolling window of resolved picks, then maps production output
through the learned mapping to produce a calibrated probability.

**Important rejection-history caveat**: a 2026-04-16 attempt at isotonic
calibration was REJECTED on backtest data — analytical evaluator showed
+1.14pp P(57) but MC bootstrap showed −1.12pp (t=−3.43). That rejection was
on BACKTEST data where calibration is OPPOSITE direction (under-confident on
2025). This module operates on PRODUCTION data where the direction is
inverted (over-confident on 2026), so the rejection doesn't directly apply
— but any deploy MUST validate via MC bootstrap, not analytical evaluator,
per the discipline established in `project_bts_2026_04_16_calibration_rejected.md`.

Usage:
    cal = fit_calibrator_from_picks(picks_dir, pa_df, today, lookback_days=30)
    if cal is not None:
        p_calibrated = apply_calibrator(p_raw, cal)
    else:
        p_calibrated = p_raw  # not enough data, fall through

Failsafe: returns None when fewer than `min_n` resolved picks fall within the
lookback window. Caller should treat None as identity (no calibration).
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_N = 30


def _resolve_pick_outcomes(
    picks_dir: Path,
    pa_df: pd.DataFrame,
    today: date,
    lookback_days: int,
) -> list[tuple[float, int]]:
    """Build (predicted_p, realized_hit) tuples from picks within the window.

    `pa_df` is the historical PA frame. We join (batter_id, date) → "did they
    have any hit that day" for each pick (primary + double_down).

    Returns empty list if no resolved picks found in the window.
    """
    if pa_df.empty:
        return []
    cutoff = today - timedelta(days=lookback_days)

    # Build a (batter_id, date) → had_hit lookup
    pa_local = pa_df.copy()
    pa_local["date"] = pd.to_datetime(pa_local["date"]).dt.date
    daily_hits = (
        pa_local.groupby(["batter_id", "date"])["is_hit"]
        .max()  # if any PA was a hit, day_hit = 1
        .reset_index()
        .rename(columns={"is_hit": "day_hit"})
    )
    lookup = {
        (row["batter_id"], row["date"]): int(row["day_hit"])
        for _, row in daily_hits.iterrows()
    }

    samples: list[tuple[float, int]] = []
    for f in sorted(picks_dir.glob("2*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            pick_date = date.fromisoformat(data.get("date", ""))
        except ValueError:
            continue
        if pick_date < cutoff or pick_date > today:
            continue
        # Only use days where the streak result is resolved (means we know per-PA hits)
        if data.get("result") not in ("hit", "miss"):
            continue
        for slot_key in ("pick", "double_down"):
            slot = data.get(slot_key)
            if not slot:
                continue
            p = slot.get("p_game_hit")
            bid = slot.get("batter_id")
            if p is None or bid is None:
                continue
            day_hit = lookup.get((bid, pick_date))
            if day_hit is None:
                # Pick not found in pa frame (unusual — could be late data). Skip.
                continue
            samples.append((float(p), int(day_hit)))
    return samples


def fit_calibrator_from_picks(
    picks_dir: Path,
    pa_df: pd.DataFrame,
    today: date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_n: int = DEFAULT_MIN_N,
):
    """Fit IsotonicRegression on resolved picks in the lookback window.

    Returns the fitted calibrator OR None if insufficient data. Caller should
    treat None as identity (apply_calibrator with None returns p unchanged).
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        log.warning("scikit-learn not available; calibration disabled")
        return None
    if today is None:
        today = date.today()
    samples = _resolve_pick_outcomes(picks_dir, pa_df, today, lookback_days)
    if len(samples) < min_n:
        log.info(
            f"calibrate: only {len(samples)} resolved picks in last {lookback_days}d "
            f"(need {min_n}); falling back to identity"
        )
        return None
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(xs, ys)
    log.info(f"calibrate: fit on n={len(samples)} samples (lookback={lookback_days}d)")
    return cal


def apply_calibrator(p: float, calibrator) -> float:
    """Apply calibrator to a single raw probability. Returns p unchanged if calibrator is None."""
    if calibrator is None:
        return p
    if p is None:
        return p
    try:
        return float(calibrator.predict([float(p)])[0])
    except Exception as e:
        log.warning(f"calibrator.predict failed on p={p}: {e}; returning raw p")
        return p


def apply_calibrator_series(s: pd.Series, calibrator) -> pd.Series:
    """Apply calibrator to a pandas Series of probabilities. Identity if None."""
    if calibrator is None:
        return s
    mask = s.notna()
    out = s.copy()
    if mask.any():
        out.loc[mask] = calibrator.predict(s.loc[mask].astype(float).values)
    return out

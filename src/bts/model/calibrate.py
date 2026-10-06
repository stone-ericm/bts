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

import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from bts.data.build import filter_out_resumed_portion
from bts.serving_witness import canon_sha256 as _canon_sha256
from bts.serving_witness import collect as _collect
from bts.serving_witness import note as _note
from bts.serving_witness import sha256_or_none as _sha256_or_none

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_N = 30


# Serving-witness collection (C2 step 2a, design §3.0/§3.2) uses the shared containment helpers: a witness problem
# nulls provenance and is recorded, and never changes a sample, the fit or the returned calibrator.


def _held_text(f: Path, inputs, errors):
    """Read one pick file once, for the resolver: (text, sha256 of the bytes decoded, or None).

    The held bytes are decoded with `Path.read_text()`'s exact semantics (`bts.picks._read_text_bytes`). A
    capture-preparation failure falls back to the original `read_text()` once; a genuine read error (OSError)
    propagates to the resolver's existing skip with no added retry, and a decode error propagates as before.
    """
    try:
        raw = f.read_bytes()
        reader = io.TextIOWrapper(io.BytesIO(raw), encoding=io.text_encoding(None))
    except OSError as e:
        _note(errors, f"pick {f.name}: unreadable, skipped: {e!r}")
        raise
    except Exception as e:
        _note(errors, f"pick {f.name}: capture preparation failed ({e!r}); parsed from path, not from the hashed buffer")
        _collect(inputs, {"file": f.name, "bytes": None, "sha256": None}, errors, f"pick {f.name}")
        return f.read_text(), None
    sha = _sha256_or_none(raw, errors, f"pick {f.name}")
    _collect(inputs, {"file": f.name, "bytes": len(raw), "sha256": sha}, errors, f"pick {f.name}")
    return reader.read(), sha


def _bind(bindings, errors, f: Path, file_sha, pick_date: date, slot_key: str, bid, sample) -> None:
    try:
        rec = {"file": f.name, "file_sha256": file_sha, "date": pick_date.isoformat(), "slot": slot_key,
               "batter_id": bid, "p": sample[0], "y": sample[1]}
    except Exception as e:
        _note(errors, f"pick {f.name} {slot_key}: binding failed: {e!r}")
        return
    _collect(bindings, rec, errors, f"pick {f.name} {slot_key}")


def _resolve_pick_outcomes(
    picks_dir: Path,
    pa_df: pd.DataFrame,
    today: date,
    lookback_days: int,
    bindings: list | None = None,
    inputs: list | None = None,
    errors: list | None = None,
) -> list[tuple[float, int]]:
    """Build (predicted_p, realized_hit) tuples from picks within the window.

    `pa_df` is the historical PA frame. We join (batter_id, date) → "did they
    have any hit that day" for each pick (primary + double_down).

    Returns empty list if no resolved picks found in the window.

    Optional witness collectors (design §3.2), observational only: `inputs` gets `{file, bytes, sha256}` for every
    pick file read, in read order; `bindings` gets one record per returned sample, in the same order; `errors` gets
    every provenance failure.
    """
    if pa_df.empty:
        return []
    cutoff = today - timedelta(days=lookback_days)

    # Build a (batter_id, date) → had_hit lookup. Exclude the resumed portion of a
    # suspended game -- never evaluated for BTS, so it must not flip a pick's outcome.
    pa_local = filter_out_resumed_portion(pa_df).copy()
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
            text, file_sha = _held_text(f, inputs, errors)
            data = json.loads(text)
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
        slot_results = data.get("slot_results") or {}
        for slot_key in ("pick", "double_down"):
            if slot_results.get(slot_key) == "void":
                continue
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
            sample = (float(p), int(day_hit))
            samples.append(sample)
            if bindings is not None:
                _bind(bindings, errors, f, file_sha, pick_date, slot_key, bid, sample)
    return samples


def _put(witness, key: str, value, errors) -> None:
    if witness is None:
        return
    try:
        witness[key] = value
    except Exception as e:
        _note(errors, f"witness {key}: assignment failed: {e!r}")


def _canon_or_none(obj, errors, what: str):
    try:
        return _canon_sha256(obj)
    except Exception as e:
        _note(errors, f"{what}: canonical sha256 failed: {e!r}")
        return None


def _fitted_map(cal, errors):
    try:
        return {"X_thresholds": cal.X_thresholds_.tolist(), "y_thresholds": cal.y_thresholds_.tolist(),
                "increasing": bool(cal.increasing_), "out_of_bounds": cal.out_of_bounds,
                "y_min": cal.y_min, "y_max": cal.y_max}
    except Exception as e:
        _note(errors, f"calibration map: extraction failed: {e!r}")
        return None


def fit_calibrator_from_picks(
    picks_dir: Path,
    pa_df: pd.DataFrame,
    today: date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_n: int = DEFAULT_MIN_N,
    witness: dict | None = None,
):
    """Fit IsotonicRegression on resolved picks in the lookback window.

    Returns the fitted calibrator OR None if insufficient data. Caller should
    treat None as identity (apply_calibrator with None returns p unchanged).

    When `witness` is a dict it is filled (design §3.2) with `status` (fitted | insufficient_support | no_sklearn),
    `n_fit`, `pick_inputs`, `samples` (the ordered bindings), `samples_sha256`, `map`, `map_sha256` and `errors`.
    Filling it never changes the return value.
    """
    errors = [] if witness is not None else None
    for key in ("status", "n_fit", "pick_inputs", "samples", "samples_sha256", "map", "map_sha256"):
        _put(witness, key, None, errors)
    _put(witness, "errors", errors, errors)
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        log.warning("scikit-learn not available; calibration disabled")
        _put(witness, "status", "no_sklearn", errors)
        return None
    if today is None:
        today = date.today()
    bindings, inputs = ([], []) if witness is not None else (None, None)
    samples = _resolve_pick_outcomes(picks_dir, pa_df, today, lookback_days,
                                     bindings=bindings, inputs=inputs, errors=errors)
    if witness is not None:
        _put(witness, "n_fit", len(samples), errors)
        _put(witness, "pick_inputs", inputs, errors)
        _put(witness, "samples", bindings, errors)
        _put(witness, "samples_sha256", _canon_or_none(bindings, errors, "calibration samples"), errors)
    if len(samples) < min_n:
        log.info(
            f"calibrate: only {len(samples)} resolved picks in last {lookback_days}d "
            f"(need {min_n}); falling back to identity"
        )
        _put(witness, "status", "insufficient_support", errors)
        return None
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(xs, ys)
    log.info(f"calibrate: fit on n={len(samples)} samples (lookback={lookback_days}d)")
    if witness is not None:
        cal_map = _fitted_map(cal, errors)
        _put(witness, "map", cal_map, errors)
        _put(witness, "map_sha256", None if cal_map is None else _canon_or_none(cal_map, errors, "calibration map"),
             errors)
        _put(witness, "status", "fitted", errors)
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

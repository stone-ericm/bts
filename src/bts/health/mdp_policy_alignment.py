"""Tier 2: MDP policy/probability-scale alignment.

The production MDP policy classifies today's `p_game_hit` against the saved
policy boundaries. If recent production picks all fall into one quality bin,
the MDP loses quality discrimination: different nominal probabilities map to
the same transition row and therefore the same action surface.

This check is diagnostic only. It does not change pick selection or policy
loading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "mdp_policy_alignment"


@dataclass(frozen=True)
class SlotBinMetrics:
    n: int
    counts: dict[int, int]
    dominant_bin: int | None
    dominant_count: int
    dominant_fraction: float | None
    p_min: float | None
    p_max: float | None


@dataclass(frozen=True)
class MDPBinMetrics:
    boundaries: list[float]
    primary: SlotBinMetrics
    double_down: SlotBinMetrics


DEFAULT_THRESHOLDS = {
    "lookback_days": 60,
    "recent_days": 21,
    "min_recent_days": 14,
    "dominant_warn_frac": 0.80,
}


def _classify(p_game_hit: float, boundaries: list[float]) -> int:
    q = 0
    for boundary in boundaries:
        if p_game_hit >= boundary:
            q += 1
    return q


def _slot_metrics(values: list[float], boundaries: list[float]) -> SlotBinMetrics:
    n_bins = len(boundaries) + 1
    counts = {i: 0 for i in range(n_bins)}
    for p in values:
        counts[_classify(p, boundaries)] += 1
    if not values:
        return SlotBinMetrics(
            n=0,
            counts=counts,
            dominant_bin=None,
            dominant_count=0,
            dominant_fraction=None,
            p_min=None,
            p_max=None,
        )
    dominant_bin, dominant_count = max(counts.items(), key=lambda item: item[1])
    return SlotBinMetrics(
        n=len(values),
        counts=counts,
        dominant_bin=dominant_bin,
        dominant_count=dominant_count,
        dominant_fraction=dominant_count / len(values),
        p_min=min(values),
        p_max=max(values),
    )


def compute_metrics(
    picks_dir: Path,
    policy_path: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> MDPBinMetrics | None:
    """Compute recent production-pick utilization of saved MDP bins."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if today is None:
        today = date.today()
    if not picks_dir.exists() or not policy_path.exists():
        return None

    try:
        policy = np.load(policy_path)
        boundaries = [float(x) for x in policy["boundaries"].tolist()]
    except Exception as e:
        log.warning("could not load MDP policy boundaries from %s: %s", policy_path, e)
        return None
    if not boundaries:
        log.warning("MDP policy at %s has no quality-bin boundaries", policy_path)
        return None

    cutoff = today - timedelta(days=int(t["lookback_days"]))
    rows: list[tuple[date, float, float | None]] = []
    for p in sorted(picks_dir.glob("*.json")):
        if "." in p.stem:
            continue
        try:
            pick_date = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if pick_date < cutoff or pick_date > today:
            continue
        try:
            body = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pick = body.get("pick") or {}
        p_game_hit = pick.get("p_game_hit")
        if p_game_hit is None:
            continue
        dd = body.get("double_down") or {}
        dd_p = dd.get("p_game_hit")
        try:
            primary_p = float(p_game_hit)
            dd_p_float = float(dd_p) if dd_p is not None else None
        except (TypeError, ValueError):
            log.warning("skipping malformed pick probabilities in %s", p)
            continue
        rows.append((pick_date, primary_p, dd_p_float))

    recent = rows[-int(t["recent_days"]):]
    primary_values = [r[1] for r in recent]
    dd_values = [r[2] for r in recent if r[2] is not None]
    return MDPBinMetrics(
        boundaries=boundaries,
        primary=_slot_metrics(primary_values, boundaries),
        double_down=_slot_metrics(dd_values, boundaries),
    )


def _dominance_warns(metrics: SlotBinMetrics, thresholds: dict) -> bool:
    if metrics.n < thresholds["min_recent_days"] or metrics.dominant_fraction is None:
        return False
    return metrics.dominant_fraction >= thresholds["dominant_warn_frac"]


def evaluate(metrics: MDPBinMetrics, thresholds: dict | None = None) -> list[Alert]:
    """Return an alert when recent primary picks collapse into one MDP bin."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not _dominance_warns(metrics.primary, t):
        return []

    p = metrics.primary
    dd = metrics.double_down
    lowest = metrics.boundaries[0] if metrics.boundaries else None
    highest = metrics.boundaries[-1] if metrics.boundaries else None
    q_label = f"Q{p.dominant_bin}" if p.dominant_bin is not None else "Q?"
    below_lowest = p.counts.get(0, 0)
    msg = (
        f"MDP quality-bin collapse: primary picks map {p.dominant_count}/{p.n} "
        f"to {q_label} over last {min(t['recent_days'], p.n)} picks "
        f"({below_lowest}/{p.n} below lowest boundary; p range "
        f"{p.p_min:.3f}-{p.p_max:.3f}, policy boundaries {lowest:.3f}-{highest:.3f}). "
        "MDP policy cannot distinguish recent "
        "pick quality; calibration/re-solve preflight required."
    )
    if dd.n:
        dd_label = f"Q{dd.dominant_bin}" if dd.dominant_bin is not None else "Q?"
        msg += (
            f" Double-down maps {dd.dominant_count}/{dd.n} to {dd_label} "
            f"(p range {dd.p_min:.3f}-{dd.p_max:.3f})."
        )
    return [Alert(level="WARN", source=SOURCE, message=msg)]


def check(
    picks_dir: Path,
    policy_path: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    try:
        metrics = compute_metrics(
            picks_dir=picks_dir,
            policy_path=policy_path,
            today=today,
            thresholds=thresholds,
        )
        if metrics is None:
            return []
        return evaluate(metrics, thresholds=thresholds)
    except Exception as e:
        log.exception("mdp_policy_alignment check failed: %s", e)
        return []

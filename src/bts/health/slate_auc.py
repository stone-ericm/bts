"""Rolling realized slate AUC — the M3 revisit trigger, made observable.

The M3 serving-staleness audit follow-up was closed HOLD
(docs/audit/2026-06-11-m3-serving-parity replay: pooled Δtop-1 −0.67pp,
CI [−3.4, +2.0]) with an explicit revisit trigger: if the model's
top-of-slate discrimination materially exceeds the ~0.59 replay baseline,
fixing serving freshness becomes worth its pipeline risk. Nothing in
production computed that number — slates weren't even persisted. Now they
are (bts.slate), and this check computes a rolling realized AUC over them.

Mechanics: load persisted slates for the lookback window, join each
candidate's p_game_hit to realized any-hit outcomes from the PA parquets
on (game_pk, batter_id), compute a tie-aware Mann-Whitney AUC over the
pooled window. Expensive-ish (parquet load), so results are cached in a
status JSON and recomputed at most every `recompute_every_days`.

Alert ladder:
  AUC < revisit_auc (0.61):  no alert (status JSON still written for dashboards)
  AUC >= revisit_auc:        WARN with the M3 pointer — re-run
                             scripts/replay_m3_serving_parity.py; the HOLD
                             may no longer be the right call.
No CRITICAL: this is a strategic nudge, not an operational failure. A
crashed check is escalated by the runner's _safe_run wrapper.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from bts.health.alert import Alert
from bts.util import atomic_write_text

log = logging.getLogger(__name__)

SOURCE = "slate_auc"

DEFAULT_THRESHOLDS = {
    "window_days": 60,
    "min_days": 20,
    "min_rows": 200,
    "revisit_auc": 0.61,
    "recompute_every_days": 7,
}


def _rank_auc(pos_scores, neg_scores) -> float | None:
    """Tie-aware Mann-Whitney AUC via average ranks. None if a class is empty."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    combined = sorted(
        [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores],
        key=lambda t: t[0],
    )
    rank_sum_pos = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # average of ranks i+1 .. j
        rank_sum_pos += avg_rank * sum(1 for k in range(i, j) if combined[k][1] == 1)
        i = j
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _repo_root_from_picks_dir(picks_dir: Path) -> Path:
    if picks_dir.name == "picks" and picks_dir.parent.name == "data":
        return picks_dir.parent.parent
    return picks_dir.parent


def _status_path(picks_dir: Path) -> Path:
    return _repo_root_from_picks_dir(Path(picks_dir)) / "data" / "health_state" / "slate_auc_status.json"


def _load_status(path: Path) -> dict | None:
    try:
        if path.exists():
            loaded = json.loads(path.read_text())
            return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None
    return None


def _alerts_for(auc: float | None, thresholds: dict) -> list[Alert]:
    if auc is None or auc < thresholds["revisit_auc"]:
        return []
    return [Alert(
        level="WARN",
        source=SOURCE,
        message=(
            f"rolling realized slate AUC {auc:.4f} >= {thresholds['revisit_auc']} — "
            f"M3 revisit trigger fired: the serving-staleness HOLD "
            f"(docs/audit/2026-06-11-m3-serving-staleness.md) assumed ~0.59 "
            f"discrimination; re-run scripts/replay_m3_serving_parity.py."
        ),
    )]


def _compute(picks_dir: Path, data_dir: Path, today: date, thresholds: dict) -> dict:
    import pandas as pd

    slates_dir = Path(picks_dir) / "slates"
    cutoff = today - timedelta(days=thresholds["window_days"])
    frames = []
    n_days = 0
    for f in sorted(slates_dir.glob("*.json")):
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if not (cutoff <= d < today):
            continue
        try:
            payload = json.loads(f.read_text())
            rows = pd.DataFrame(payload["rows"])
        except Exception as e:
            log.warning(f"unreadable slate {f.name}: {e}")
            continue
        if rows.empty or not {"batter_id", "game_pk", "p_game_hit"} <= set(rows.columns):
            continue
        frames.append(rows[["batter_id", "game_pk", "p_game_hit"]].assign(_slate_date=str(d)))
        n_days += 1

    base = {
        "computed_at": str(today),
        "window_days": thresholds["window_days"],
        "n_days": n_days,
    }
    if n_days < thresholds["min_days"]:
        return {**base, "n_rows": 0, "auc": None, "reason": "insufficient_days"}

    slate = pd.concat(frames, ignore_index=True).dropna(subset=["p_game_hit"])

    seasons = {today.year, (today - timedelta(days=thresholds["window_days"])).year}
    parts = []
    for y in sorted(seasons):
        p = Path(data_dir) / f"pa_{y}.parquet"
        if p.exists():
            try:
                parts.append(pd.read_parquet(p, columns=["batter_id", "game_pk", "is_hit"]))
            except Exception as e:
                log.warning(f"failed to load {p}: {e}")
    if not parts:
        return {**base, "n_rows": 0, "auc": None, "reason": "no_outcomes"}

    outcomes = (
        pd.concat(parts, ignore_index=True)
        .groupby(["game_pk", "batter_id"])["is_hit"].max().rename("actual_hit").reset_index()
    )
    joined = slate.merge(outcomes, on=["game_pk", "batter_id"], how="inner")
    if len(joined) < thresholds["min_rows"]:
        return {**base, "n_rows": int(len(joined)), "auc": None, "reason": "insufficient_rows"}

    auc = _rank_auc(
        joined.loc[joined["actual_hit"] == 1, "p_game_hit"].tolist(),
        joined.loc[joined["actual_hit"] == 0, "p_game_hit"].tolist(),
    )
    return {**base, "n_rows": int(len(joined)), "auc": auc, "reason": None}


def check(
    picks_dir: Path,
    data_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Rolling realized slate AUC vs the M3 revisit threshold."""
    today = today or date.today()
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    status_path = _status_path(picks_dir)
    cached = _load_status(status_path)
    if cached is not None and cached.get("computed_at"):
        try:
            age = (today - date.fromisoformat(cached["computed_at"])).days
        except ValueError:
            age = None
        if age is not None and 0 <= age < t["recompute_every_days"]:
            return _alerts_for(cached.get("auc"), t)

    if not (Path(picks_dir) / "slates").is_dir():
        return []

    status = _compute(picks_dir, Path(data_dir), today, t)
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(status_path, json.dumps(status, indent=2))
    except Exception as e:
        log.warning(f"slate_auc status write failed: {e}")
    return _alerts_for(status.get("auc"), t)

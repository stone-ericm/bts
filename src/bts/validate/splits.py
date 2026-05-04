"""Validation splits — purged blocked CV + lockbox manifest.

SOTA tracker item #5 phase 0/1. Provides a validation contract:
- LockboxSpec: explicit untouched date range stored in manifest
- FoldSpec: per-fold train + holdout date sets
- make_purged_blocked_cv: rolling-origin (forward-chaining) splits with
  strict max(train) < min(holdout) and a configurable purge gap
- save_manifest / load_manifest: deterministic JSON serialization
- apply_fold: filter a profiles DataFrame to a fold's train/holdout slices

References:
- López de Prado 2018, "Advances in Financial Machine Learning",
  ch. 7 (Cross-Validation in Finance)

Naming convention: `holdout` = the held-out evaluation slice for a
fold (avoids the Python builtin name conflict).

Phase 0/1 supports rolling_origin only. Symmetric blocked CV is a
deferred mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class LockboxSpec:
    """Explicit date range reserved for final evaluation. Stored in manifests
    so reproducibility doesn't depend on which files happen to be present
    in a local data directory."""

    start_date: date
    end_date: date
    description: str


@dataclass(frozen=True)
class FoldSpec:
    """A single fold: forward-chained train + holdout date sets."""

    fold_idx: int
    train_dates: frozenset[date]
    holdout_dates: frozenset[date]


def declare_lockbox(
    start_date: date, end_date: date, description: str
) -> LockboxSpec:
    if end_date < start_date:
        raise ValueError(
            f"end_date {end_date} must be >= start_date {start_date}"
        )
    return LockboxSpec(
        start_date=start_date, end_date=end_date, description=description
    )


def is_in_lockbox(d: date, lockbox: LockboxSpec) -> bool:
    return lockbox.start_date <= d <= lockbox.end_date


def make_purged_blocked_cv(
    available_dates: Iterable[date],
    *,
    n_folds: int = 5,
    purge_game_days: int = 7,
    embargo_game_days: int = 7,
    min_train_game_days: int = 365,
    lockbox: LockboxSpec | None = None,
    mode: str = "rolling_origin",
) -> list[FoldSpec]:
    """Build forward-chaining purged blocked CV folds.

    rolling_origin (default): each fold's train is the contiguous prefix
    BEFORE the holdout block (minus purge_game_days), guaranteeing
    max(train_dates) < min(holdout_dates).

    embargo_game_days is recorded for forward compatibility with a
    deferred symmetric blocked mode but has no effect here.
    """
    if mode != "rolling_origin":
        raise NotImplementedError(
            f"mode={mode!r} not implemented; "
            "only 'rolling_origin' is supported in P0/P1"
        )

    sorted_dates = sorted(set(available_dates))
    if lockbox is not None:
        sorted_dates = [
            d for d in sorted_dates if not is_in_lockbox(d, lockbox)
        ]

    n = len(sorted_dates)
    # Fold 0's holdout starts AFTER min_train_game_days + purge_game_days
    # so that fold 0's train has at least min_train_game_days dates.
    first_holdout_idx = min_train_game_days + purge_game_days
    if n < first_holdout_idx + n_folds:
        raise ValueError(
            f"Not enough non-lockbox dates ({n}) for "
            f"min_train_game_days={min_train_game_days} + "
            f"purge_game_days={purge_game_days} + "
            f"n_folds={n_folds} (need at least {first_holdout_idx + n_folds})"
        )

    holdout_portion_size = n - first_holdout_idx
    block_size = holdout_portion_size // n_folds
    remainder = holdout_portion_size % n_folds

    folds: list[FoldSpec] = []
    pos = first_holdout_idx
    for fold_idx in range(n_folds):
        cur_block_size = block_size + (1 if fold_idx < remainder else 0)
        holdout_start_idx = pos
        holdout_end_idx = pos + cur_block_size

        holdout_dates = frozenset(
            sorted_dates[holdout_start_idx:holdout_end_idx]
        )
        train_end_idx = max(0, holdout_start_idx - purge_game_days)
        train_dates = frozenset(sorted_dates[:train_end_idx])

        folds.append(
            FoldSpec(
                fold_idx=fold_idx,
                train_dates=train_dates,
                holdout_dates=holdout_dates,
            )
        )
        pos = holdout_end_idx

    return folds


def assert_no_lockbox_leakage(
    folds: list[FoldSpec], lockbox: LockboxSpec
) -> None:
    """Raise ValueError if any fold's train or holdout overlaps the lockbox."""
    for fold in folds:
        for d in fold.train_dates:
            if is_in_lockbox(d, lockbox):
                raise ValueError(
                    f"fold {fold.fold_idx} train_dates contains "
                    f"lockbox date {d}"
                )
        for d in fold.holdout_dates:
            if is_in_lockbox(d, lockbox):
                raise ValueError(
                    f"fold {fold.fold_idx} holdout_dates contains "
                    f"lockbox date {d}"
                )


def apply_fold(
    profiles_df: pd.DataFrame, fold: FoldSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a profiles DataFrame into (train_df, holdout_df) by date."""
    df = profiles_df.copy()
    if df.empty:
        return df, df.copy()

    first_val = df["date"].iloc[0]
    if isinstance(first_val, pd.Timestamp):
        date_col = df["date"].dt.date
    elif isinstance(first_val, date):
        date_col = df["date"]
    elif hasattr(first_val, "date"):
        date_col = df["date"].apply(
            lambda x: x.date() if hasattr(x, "date") else x
        )
    else:
        date_col = df["date"]

    train_mask = date_col.isin(fold.train_dates)
    holdout_mask = date_col.isin(fold.holdout_dates)
    return df[train_mask].copy(), df[holdout_mask].copy()


def save_manifest(
    folds: list[FoldSpec],
    lockbox: LockboxSpec,
    path: Path | str,
    *,
    purge_game_days: int,
    embargo_game_days: int,
    min_train_game_days: int,
    mode: str,
    universe_dates: Iterable[date],
    created_at: str | None = None,
) -> Path:
    """Serialize folds + lockbox + parameters to a deterministic JSON file."""
    universe_sorted = sorted(set(universe_dates))

    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    data = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "lockbox": {
            "start_date": lockbox.start_date.isoformat(),
            "end_date": lockbox.end_date.isoformat(),
            "description": lockbox.description,
        },
        "split_params": {
            "n_folds": len(folds),
            "purge_game_days": purge_game_days,
            "embargo_game_days": embargo_game_days,
            "min_train_game_days": min_train_game_days,
            "mode": mode,
        },
        "universe": {
            "n_dates": len(universe_sorted),
            "first_date": universe_sorted[0].isoformat()
            if universe_sorted
            else None,
            "last_date": universe_sorted[-1].isoformat()
            if universe_sorted
            else None,
        },
        "folds": [
            {
                "fold_idx": fold.fold_idx,
                "train_dates": sorted(
                    d.isoformat() for d in fold.train_dates
                ),
                "holdout_dates": sorted(
                    d.isoformat() for d in fold.holdout_dates
                ),
            }
            for fold in folds
        ],
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


def resolve_default_lockbox_season(
    available_dates_by_season: dict[int, list[date]],
    *,
    min_complete_season_dates: int = 150,
) -> int:
    """Pick the latest season with at least `min_complete_season_dates` game-dates.

    Skips seasons with fewer dates (presumed in-progress; e.g., a partial 2026
    backtest file alongside complete 2025). Raises if no season qualifies, so
    the caller is forced to specify `--lockbox-season` explicitly.
    """
    qualifying = sorted(
        (
            year
            for year, dates in available_dates_by_season.items()
            if len(dates) >= min_complete_season_dates
        ),
        reverse=True,
    )
    if not qualifying:
        raise ValueError(
            f"No season has >= {min_complete_season_dates} dates. "
            "Specify --lockbox-season explicitly."
        )
    return qualifying[0]


def collect_universe_dates(
    dates_by_season: dict[int, list[date]],
    lockbox: LockboxSpec,
    *,
    include_post_lockbox: bool = False,
) -> list[date]:
    """Combine all season dates into a single sorted universe.

    Default restricts to dates <= lockbox.end_date so that partial-current-season
    files (e.g., an in-progress 2026 backtest sitting alongside a complete 2025)
    cannot leak post-lockbox dates into the manifest. The lockbox itself is
    INCLUDED here; `make_purged_blocked_cv` removes it from fold train/holdout.

    Set include_post_lockbox=True to opt out of the filter (use only when
    intentionally evaluating across post-lockbox data).
    """
    all_dates = sorted(
        d for season_dates in dates_by_season.values() for d in season_dates
    )
    if not include_post_lockbox:
        all_dates = [d for d in all_dates if d <= lockbox.end_date]
    return all_dates


def default_lockbox_for_season(
    available_dates_by_season: dict[int, list[date]],
    season: int,
    *,
    n_game_days: int = 30,
) -> LockboxSpec:
    """Carve the last N game-days of `season` from available dates as the lockbox."""
    if season not in available_dates_by_season:
        raise ValueError(
            f"Season {season} not in available data; "
            f"available: {sorted(available_dates_by_season)}"
        )
    season_dates = sorted(available_dates_by_season[season])
    if len(season_dates) < n_game_days:
        raise ValueError(
            f"Season {season} has only {len(season_dates)} dates; "
            f"can't carve {n_game_days}-day lockbox"
        )
    lockbox_dates = season_dates[-n_game_days:]
    return declare_lockbox(
        start_date=lockbox_dates[0],
        end_date=lockbox_dates[-1],
        description=f"last {n_game_days} game-days of {season}",
    )


def load_manifest(
    path: Path | str,
) -> tuple[list[FoldSpec], LockboxSpec]:
    """Load a manifest JSON and reconstruct folds + lockbox.

    Rejects unknown schema_version values so that future manifest format
    changes surface explicit migration errors rather than silent
    misinterpretation of the data.
    """
    data = json.loads(Path(path).read_text())
    found_version = data.get("schema_version")
    if found_version != SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema_version {found_version!r} not supported "
            f"(expected {SCHEMA_VERSION!r})"
        )
    lb = LockboxSpec(
        start_date=date.fromisoformat(data["lockbox"]["start_date"]),
        end_date=date.fromisoformat(data["lockbox"]["end_date"]),
        description=data["lockbox"]["description"],
    )
    folds = [
        FoldSpec(
            fold_idx=f["fold_idx"],
            train_dates=frozenset(
                date.fromisoformat(d) for d in f["train_dates"]
            ),
            holdout_dates=frozenset(
                date.fromisoformat(d) for d in f["holdout_dates"]
            ),
        )
        for f in data["folds"]
    ]
    return folds, lb

"""Tests for splits module — SOTA tracker item #5 phase 0/1.

TDD: tests written before implementation. Module under test is
`bts.validate.splits`.

Scope per Codex sign-off (bus msg #76):
- Lockbox: explicit date range stored in manifest, generator default =
  last 30 game-days of latest TRACKED COMPLETE season.
- Splits: rolling-origin / forward-chaining default. Strict
  max(train_dates) < min(holdout_dates) (respecting purge_game_days).
- Symmetric blocked CV is deferred to a later phase.
- purge_game_days / embargo_game_days are GAME-DAY windows, not
  calendar-day. Embargo is parameterized but only meaningful in
  deferred symmetric mode; recorded but inactive in rolling_origin.
- min_train_game_days guard ensures first fold has enough training
  history.
- Manifest JSON: schema_version, created_at, lockbox spec, split params,
  universe metadata, folds with sorted ISO date lists. Deterministic +
  diffable.

Naming convention: holdout = the held-out evaluation slice for a fold,
following the López de Prado convention. (Avoids the Python builtin name.)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from bts.validate.splits import (
    LockboxSpec,
    FoldSpec,
    declare_lockbox,
    is_in_lockbox,
    make_purged_blocked_cv,
    apply_fold,
    save_manifest,
    load_manifest,
    assert_no_lockbox_leakage,
    resolve_default_lockbox_season,
    default_lockbox_for_season,
    collect_universe_dates,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game_dates(start: date, n: int) -> list[date]:
    """Return n consecutive calendar dates starting at `start`."""
    return [start + timedelta(days=i) for i in range(n)]


def _make_profiles_df(dates_list, top_n=10):
    """Synthetic profile DataFrame with the given dates."""
    import pandas as pd
    rows = []
    for d in dates_list:
        for rank in range(1, top_n + 1):
            rows.append({
                "date": d,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": 0.78,
                "actual_hit": 1,
                "n_pas": 4,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LockboxSpec / declare_lockbox
# ---------------------------------------------------------------------------

class TestLockboxSpec:
    def test_declare_lockbox_returns_spec_with_dates_and_description(self):
        lb = declare_lockbox(
            start_date=date(2025, 8, 30),
            end_date=date(2025, 9, 28),
            description="last 30 game-days of 2025 (regular-season finale)",
        )
        assert lb.start_date == date(2025, 8, 30)
        assert lb.end_date == date(2025, 9, 28)
        assert "2025" in lb.description

    def test_lockbox_spec_is_frozen(self):
        lb = declare_lockbox(date(2025, 8, 30), date(2025, 9, 28), "test")
        with pytest.raises((AttributeError, Exception)):
            lb.start_date = date(2024, 1, 1)  # type: ignore[misc]

    def test_is_in_lockbox_inclusive_bounds(self):
        lb = declare_lockbox(date(2025, 8, 30), date(2025, 9, 28), "test")
        assert is_in_lockbox(date(2025, 8, 30), lb) is True
        assert is_in_lockbox(date(2025, 9, 28), lb) is True
        assert is_in_lockbox(date(2025, 9, 1), lb) is True

    def test_is_in_lockbox_excludes_outside(self):
        lb = declare_lockbox(date(2025, 8, 30), date(2025, 9, 28), "test")
        assert is_in_lockbox(date(2025, 8, 29), lb) is False
        assert is_in_lockbox(date(2025, 9, 29), lb) is False
        assert is_in_lockbox(date(2024, 9, 1), lb) is False

    def test_declare_lockbox_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            declare_lockbox(date(2025, 9, 28), date(2025, 8, 30), "test")


# ---------------------------------------------------------------------------
# make_purged_blocked_cv (rolling_origin default)
# ---------------------------------------------------------------------------

class TestRollingOriginCV:
    def _setup(self):
        dates = _game_dates(date(2022, 1, 1), 1000)
        lockbox = declare_lockbox(date(2024, 8, 30), date(2024, 9, 28), "test lockbox")
        return dates, lockbox

    def test_returns_n_folds(self):
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        assert len(folds) == 5

    def test_strict_forward_chaining_max_train_lt_min_holdout(self):
        """The core forward-chaining invariant: max(train_dates) is strictly
        before min(holdout_dates)."""
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, purge_game_days=7,
                                        lockbox=lb, min_train_game_days=200)
        for fold in folds:
            train_max = max(fold.train_dates)
            holdout_min = min(fold.holdout_dates)
            assert train_max < holdout_min, \
                f"fold {fold.fold_idx}: max train {train_max} >= min holdout {holdout_min}"

    def test_purge_window_respected(self):
        """purge_game_days enforced: gap between max(train) and
        min(holdout) >= purge_game_days game-dates."""
        dates, lb = self._setup()
        purge = 14
        folds = make_purged_blocked_cv(dates, n_folds=5, purge_game_days=purge,
                                        lockbox=lb, min_train_game_days=200)
        sorted_universe = sorted(d for d in dates if not is_in_lockbox(d, lb))
        for fold in folds:
            train_max = max(fold.train_dates)
            holdout_min = min(fold.holdout_dates)
            gap = sum(1 for d in sorted_universe if train_max < d < holdout_min)
            assert gap >= purge, \
                f"fold {fold.fold_idx}: gap {gap} < purge {purge}"

    def test_disjoint_holdout_blocks(self):
        """Each non-lockbox holdout date appears in exactly one fold."""
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        holdout_sets = [fold.holdout_dates for fold in folds]
        for i in range(len(holdout_sets)):
            for j in range(i + 1, len(holdout_sets)):
                assert not (holdout_sets[i] & holdout_sets[j]), \
                    f"folds {i} and {j} overlap in holdout"

    def test_holdout_blocks_are_contiguous(self):
        """Within each fold, holdout dates form a contiguous game-date block."""
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        sorted_universe = sorted(d for d in dates if not is_in_lockbox(d, lb))
        for fold in folds:
            holdout_sorted = sorted(fold.holdout_dates)
            positions = [sorted_universe.index(d) for d in holdout_sorted]
            for i in range(len(positions) - 1):
                assert positions[i + 1] == positions[i] + 1, \
                    f"fold {fold.fold_idx} holdout not contiguous"

    def test_no_lockbox_leakage(self):
        """Lockbox dates appear in NO fold's train or holdout."""
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        for fold in folds:
            for d in fold.train_dates:
                assert not is_in_lockbox(d, lb), \
                    f"fold {fold.fold_idx} train leaks lockbox: {d}"
            for d in fold.holdout_dates:
                assert not is_in_lockbox(d, lb), \
                    f"fold {fold.fold_idx} holdout leaks lockbox: {d}"

    def test_min_train_game_days_guard_first_fold(self):
        """First fold's train must have at least min_train_game_days dates."""
        dates, lb = self._setup()
        min_train = 300
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=min_train)
        assert len(folds[0].train_dates) >= min_train

    def test_train_size_grows_across_folds(self):
        """Forward-chaining: each fold's train >= previous fold's train."""
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        for i in range(len(folds) - 1):
            assert len(folds[i + 1].train_dates) > len(folds[i].train_dates)

    def test_assert_no_lockbox_leakage_helper_passes(self):
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        assert_no_lockbox_leakage(folds, lb)

    def test_assert_no_lockbox_leakage_raises_on_corrupt_fold(self):
        dates, lb = self._setup()
        folds = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                        min_train_game_days=200)
        corrupt = [
            FoldSpec(
                fold_idx=folds[0].fold_idx,
                train_dates=folds[0].train_dates | {date(2024, 9, 1)},
                holdout_dates=folds[0].holdout_dates,
            )
        ] + list(folds[1:])
        with pytest.raises(ValueError, match="lockbox"):
            assert_no_lockbox_leakage(corrupt, lb)

    def test_reproducibility_same_inputs_same_folds(self):
        dates, lb = self._setup()
        folds_a = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                          min_train_game_days=200)
        folds_b = make_purged_blocked_cv(dates, n_folds=5, lockbox=lb,
                                          min_train_game_days=200)
        for a, b in zip(folds_a, folds_b):
            assert a.train_dates == b.train_dates
            assert a.holdout_dates == b.holdout_dates


# ---------------------------------------------------------------------------
# apply_fold
# ---------------------------------------------------------------------------

class TestApplyFold:
    def test_filters_train_and_holdout_correctly(self):
        dates = _game_dates(date(2022, 1, 1), 100)
        lb = declare_lockbox(date(2022, 4, 1), date(2022, 4, 10), "lb")
        folds = make_purged_blocked_cv(dates, n_folds=3, lockbox=lb,
                                        min_train_game_days=20)
        df = _make_profiles_df(dates)
        train_df, holdout_df = apply_fold(df, folds[0])
        train_dates_in_df = set(train_df["date"].tolist())
        holdout_dates_in_df = set(holdout_df["date"].tolist())
        assert train_dates_in_df == folds[0].train_dates
        assert holdout_dates_in_df == folds[0].holdout_dates

    def test_apply_fold_disjoint_train_holdout(self):
        dates = _game_dates(date(2022, 1, 1), 100)
        lb = declare_lockbox(date(2022, 4, 1), date(2022, 4, 10), "lb")
        folds = make_purged_blocked_cv(dates, n_folds=3, lockbox=lb,
                                        min_train_game_days=20)
        df = _make_profiles_df(dates)
        for fold in folds:
            train_df, holdout_df = apply_fold(df, fold)
            assert not set(train_df["date"]) & set(holdout_df["date"])


# ---------------------------------------------------------------------------
# Manifest persistence (JSON)
# ---------------------------------------------------------------------------

class TestManifest:
    def _setup(self, tmp_path):
        dates = _game_dates(date(2022, 1, 1), 1000)
        lb = declare_lockbox(date(2024, 8, 30), date(2024, 9, 28), "test lockbox")
        folds = make_purged_blocked_cv(
            dates, n_folds=5, purge_game_days=7, embargo_game_days=7,
            lockbox=lb, min_train_game_days=200,
        )
        path = tmp_path / "manifest.json"
        return dates, lb, folds, path

    def test_save_manifest_writes_file(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        assert path.exists()

    def test_manifest_has_required_top_level_keys(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        data = json.loads(path.read_text())
        for key in ("schema_version", "created_at", "lockbox", "split_params",
                    "universe", "folds"):
            assert key in data, f"missing top-level key {key}"

    def test_manifest_schema_version_constant(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        data = json.loads(path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_manifest_records_all_split_params(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=14, embargo_game_days=21,
                      min_train_game_days=300, mode="rolling_origin",
                      universe_dates=dates)
        data = json.loads(path.read_text())
        params = data["split_params"]
        assert params["purge_game_days"] == 14
        assert params["embargo_game_days"] == 21
        assert params["min_train_game_days"] == 300
        assert params["mode"] == "rolling_origin"
        assert params["n_folds"] == len(folds)

    def test_manifest_dates_iso_strings(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        data = json.loads(path.read_text())
        assert data["lockbox"]["start_date"] == "2024-08-30"
        assert data["lockbox"]["end_date"] == "2024-09-28"
        for fold in data["folds"]:
            for d_str in fold["train_dates"]:
                date.fromisoformat(d_str)
            for d_str in fold["holdout_dates"]:
                date.fromisoformat(d_str)

    def test_manifest_fold_dates_sorted(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        data = json.loads(path.read_text())
        for fold in data["folds"]:
            assert fold["train_dates"] == sorted(fold["train_dates"])
            assert fold["holdout_dates"] == sorted(fold["holdout_dates"])

    def test_manifest_round_trips(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        save_manifest(folds, lb, path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        loaded_folds, loaded_lb = load_manifest(path)
        assert loaded_lb.start_date == lb.start_date
        assert loaded_lb.end_date == lb.end_date
        assert len(loaded_folds) == len(folds)
        for orig, loaded in zip(folds, loaded_folds):
            assert orig.train_dates == loaded.train_dates
            assert orig.holdout_dates == loaded.holdout_dates

    def test_manifest_deterministic_repeated_save(self, tmp_path):
        dates, lb, folds, path = self._setup(tmp_path)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        save_manifest(folds, lb, path_a,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates, created_at="2026-05-04T12:00:00Z")
        save_manifest(folds, lb, path_b,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates, created_at="2026-05-04T12:00:00Z")
        assert path_a.read_text() == path_b.read_text()


# ---------------------------------------------------------------------------
# Lockbox season resolution (partial-current-season pitfall, Codex #76)
# ---------------------------------------------------------------------------

class TestResolveDefaultLockboxSeason:
    """Pick latest TRACKED COMPLETE season; skip partial in-progress."""

    def test_picks_latest_qualifying_season(self):
        dates_2024 = _game_dates(date(2024, 4, 1), 180)
        dates_2025 = _game_dates(date(2025, 4, 1), 180)
        result = resolve_default_lockbox_season(
            {2024: dates_2024, 2025: dates_2025},
            min_complete_season_dates=150,
        )
        assert result == 2025

    def test_skips_partial_current_season(self):
        """2026 has only 50 dates (in-progress); 2025 has 180 (complete).
        Default must pick 2025, NOT silently slide to 2026."""
        dates_2025 = _game_dates(date(2025, 4, 1), 180)
        dates_2026 = _game_dates(date(2026, 4, 1), 50)
        result = resolve_default_lockbox_season(
            {2025: dates_2025, 2026: dates_2026},
            min_complete_season_dates=150,
        )
        assert result == 2025

    def test_raises_when_no_season_qualifies(self):
        """If no season has >= min_complete_season_dates, force the caller
        to specify --lockbox-season explicitly."""
        with pytest.raises(ValueError, match="lockbox-season"):
            resolve_default_lockbox_season(
                {2026: _game_dates(date(2026, 4, 1), 50)},
                min_complete_season_dates=150,
            )


class TestDefaultLockboxForSeason:
    def test_carves_last_n_game_days(self):
        season_dates = _game_dates(date(2025, 4, 1), 180)
        lb = default_lockbox_for_season(
            {2025: season_dates}, season=2025, n_game_days=30
        )
        assert lb.start_date == season_dates[-30]
        assert lb.end_date == season_dates[-1]
        assert "30" in lb.description
        assert "2025" in lb.description

    def test_raises_on_unknown_season(self):
        season_dates = _game_dates(date(2025, 4, 1), 180)
        with pytest.raises(ValueError, match="not in available"):
            default_lockbox_for_season(
                {2025: season_dates}, season=2030, n_game_days=30
            )

    def test_raises_when_season_too_short(self):
        partial = _game_dates(date(2025, 9, 1), 10)
        with pytest.raises(ValueError, match="only 10 dates"):
            default_lockbox_for_season(
                {2025: partial}, season=2025, n_game_days=30
            )


# ---------------------------------------------------------------------------
# collect_universe_dates (post-lockbox filter, Codex #79)
# ---------------------------------------------------------------------------

class TestCollectUniverseDates:
    """Default behavior: exclude dates after lockbox.end_date so partial
    in-progress seasons can't leak into the manifest universe."""

    def test_default_filters_post_lockbox_dates(self):
        dates_2025 = _game_dates(date(2025, 4, 1), 180)
        dates_2026 = _game_dates(date(2026, 4, 1), 50)  # in-progress
        lb = declare_lockbox(
            date(2025, 8, 30), date(2025, 9, 28), "last 30 game-days of 2025"
        )
        universe = collect_universe_dates({2025: dates_2025, 2026: dates_2026}, lb)
        assert max(universe) <= lb.end_date
        assert all(d <= date(2025, 9, 28) for d in universe)
        assert date(2026, 4, 1) not in universe

    def test_lockbox_dates_included_in_universe(self):
        """The lockbox itself is in the universe; make_purged_blocked_cv
        removes it from fold train/holdout."""
        dates_2025 = _game_dates(date(2025, 4, 1), 200)  # 200 days covers Sep 28
        lb = declare_lockbox(
            date(2025, 8, 30), date(2025, 9, 28), "lb"
        )
        universe = collect_universe_dates({2025: dates_2025}, lb)
        assert date(2025, 8, 30) in universe
        assert date(2025, 9, 28) in universe

    def test_include_post_lockbox_keeps_all_dates(self):
        dates_2025 = _game_dates(date(2025, 4, 1), 180)
        dates_2026 = _game_dates(date(2026, 4, 1), 50)
        lb = declare_lockbox(
            date(2025, 8, 30), date(2025, 9, 28), "lb"
        )
        universe = collect_universe_dates(
            {2025: dates_2025, 2026: dates_2026},
            lb,
            include_post_lockbox=True,
        )
        assert date(2026, 4, 1) in universe

    def test_synthetic_2025_complete_2026_partial_no_post_lockbox_in_folds(self):
        """End-to-end: with default filtering, no fold's train or holdout
        has a date > lockbox.end_date."""
        dates_2025 = _game_dates(date(2025, 4, 1), 180)
        dates_2026 = _game_dates(date(2026, 4, 1), 50)
        lb = declare_lockbox(
            date(2025, 8, 30), date(2025, 9, 28), "last 30 of 2025"
        )
        universe = collect_universe_dates({2025: dates_2025, 2026: dates_2026}, lb)
        folds = make_purged_blocked_cv(
            universe, n_folds=3, lockbox=lb, min_train_game_days=30,
        )
        for fold in folds:
            for d in fold.train_dates:
                assert d <= lb.end_date, f"train leaks post-lockbox: {d}"
            for d in fold.holdout_dates:
                assert d <= lb.end_date, f"holdout leaks post-lockbox: {d}"


# ---------------------------------------------------------------------------
# save_manifest mkdir + load_manifest schema_version validation (Codex #79)
# ---------------------------------------------------------------------------

class TestManifestRobustness:
    def _setup(self):
        dates = _game_dates(date(2022, 1, 1), 1000)
        lb = declare_lockbox(date(2024, 8, 30), date(2024, 9, 28), "test")
        folds = make_purged_blocked_cv(
            dates, n_folds=3, lockbox=lb, min_train_game_days=200,
        )
        return dates, lb, folds

    def test_save_manifest_creates_parent_directories(self, tmp_path):
        dates, lb, folds = self._setup()
        nested_path = tmp_path / "subdir" / "deeper" / "manifest.json"
        save_manifest(folds, lb, nested_path,
                      purge_game_days=7, embargo_game_days=7,
                      min_train_game_days=200, mode="rolling_origin",
                      universe_dates=dates)
        assert nested_path.exists()

    def test_load_manifest_rejects_unknown_schema_version(self, tmp_path):
        path = tmp_path / "bad_schema.json"
        path.write_text(json.dumps({
            "schema_version": "v99",
            "lockbox": {
                "start_date": "2024-08-30",
                "end_date": "2024-09-28",
                "description": "test",
            },
            "folds": [],
        }))
        with pytest.raises(ValueError, match="schema_version"):
            load_manifest(path)

    def test_load_manifest_rejects_missing_schema_version(self, tmp_path):
        path = tmp_path / "no_schema.json"
        path.write_text(json.dumps({
            "lockbox": {
                "start_date": "2024-08-30",
                "end_date": "2024-09-28",
                "description": "test",
            },
            "folds": [],
        }))
        with pytest.raises(ValueError, match="schema_version"):
            load_manifest(path)

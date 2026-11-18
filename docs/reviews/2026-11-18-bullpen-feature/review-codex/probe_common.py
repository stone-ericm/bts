"""Shared inputs and checks for the isolated official-date review."""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "base"
DATA = Path("/Users/eric/projects/bts/data/processed")
LOOKUP_PATH = Path("/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json")
LOOKUP_HASH = "4a8989587f97412bcb7498da3a27d8cd1ff0321a5186028fc17f8ce9f4536454"
FEATURE = "opp_bullpen_hr_30g"
TEAM = "opp_pitching_team_id"
EXACT = dict(check_exact=True, check_index_type=True, check_column_type=True)


def stage_base():
    BASE.mkdir(exist_ok=True)
    archive = subprocess.check_output(
        ["git", "archive", "43fc5628", "src", "scripts"], cwd=ROOT
    )
    subprocess.run(["tar", "-xf", "-", "-C", str(BASE)], input=archive, check=True)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "base_compute", BASE / "src/bts/features/compute.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lookup_frozen():
    data = LOOKUP_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOOKUP_HASH
    return {int(key): row for key, row in json.loads(data).items()}


def season(year):
    assert 2017 <= year <= 2025
    return pd.read_parquet(DATA / f"pa_{year}.parquet")


@contextmanager
def inputs(module, lookup):
    from bts.features import park_drag

    with (
        patch.object(module, "_build_probable_pitcher_lookup", return_value=lookup),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        yield


def run(module, frame, lookup):
    with inputs(module, lookup):
        return module.compute_all_features(frame)


def in_pipeline_order(frame):
    return frame.assign(date=pd.to_datetime(frame["date"])).sort_values(
        ["date", "game_pk"]
    ).reset_index(drop=True)


def attach_probe_ids(frame):
    return frame.assign(review_row=np.arange(len(frame)))


def assert_aligned(result, frame):
    pd.testing.assert_frame_equal(
        result[frame.columns], in_pipeline_order(frame), **EXACT,
    )


def compare_versions(old, new):
    assert old.columns.equals(new.columns)
    assert old[FEATURE].dtype == new[FEATURE].dtype
    pd.testing.assert_frame_equal(
        old.drop(columns=FEATURE), new.drop(columns=FEATURE), **EXACT,
    )
    valued = old[FEATURE].notna()
    prev = old.loc[valued, FEATURE].to_numpy(dtype=np.float64)
    curr = new.loc[valued, FEATURE].to_numpy(dtype=np.float64)
    changed_bits = np.count_nonzero(prev.view(np.uint64) != curr.view(np.uint64))
    fills = (~valued & new[FEATURE].notna()).sum()
    losses = (valued & new[FEATURE].isna()).sum()
    max_abs = np.max(np.abs(prev - curr), initial=0)
    print(
        f"rows={len(old)} columns={len(old.columns)} "
        f"missing={sum(~valued)}->{new[FEATURE].isna().sum()} "
        f"fills={fills} losses={losses} valued_bit_changes={changed_bits} max_abs={max_abs:.16g}",
        flush=True,
    )
    assert not (losses or changed_bits)


def classify(row, lookup):
    game = lookup.get(row.game_pk, {})
    side = "away" if row.is_home else "home"
    return game.get(side + "_tid"), game.get(side)


def reference(frame, lookup):
    """Chronological sweep over PA rows, committing a date only after its targets."""
    history = defaultdict(list)
    expected = np.full(len(frame), np.nan)
    ordered = in_pipeline_order(frame)
    for date, rows in ordered.groupby("date", sort=True):
        today = {}
        values = {}
        for row in rows.itertuples():
            team, probable = classify(row, lookup)
            if pd.isna(team):
                continue
            if team not in values:
                defined = [rate for rate in history[team][-30:] if rate is not None]
                values[team] = sum(defined) / len(defined) if len(defined) >= 10 else np.nan
            expected[row.Index] = values[team]
            if not pd.isna(probable) and row.pitcher_id != probable:
                hits, count = today.get(team, (0, 0))
                if not pd.isna(row.is_hit):
                    hits += row.is_hit
                    count += 1
                today[team] = hits, count
        for team, (hits, count) in today.items():
            history[team].append(hits / count if count else None)
    return expected


def assert_reference(actual, frame, lookup):
    expected = reference(frame, lookup)
    actual_values = actual[FEATURE].to_numpy()
    mask = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual_values), mask)
    np.testing.assert_allclose(actual_values[~mask], expected[~mask], atol=1e-12, rtol=0)
    return np.max(np.abs(actual_values[~mask] - expected[~mask]), initial=0)


def toy(rows):
    """Input rows are (date, team, probable, pitcher, hit), one PA per game."""
    metadata = dict(
        batter_id=11, is_home=True, pitch_hand="R", venue_id=31,
        pitch_calls=["X"], pitch_types=["FF"], pitch_speeds=[93.0],
        pitch_spin_rates=[2300.0], pitch_extensions=[6.0],
        pitch_break_vertical=[15.0], pitch_break_horizontal=[5.0],
        pitch_px=[0.0], pitch_pz=[2.5], sz_top=3.5, sz_bottom=1.5,
        final_count_balls=0, final_count_strikes=0, launch_speed=91.0, launch_angle=10.0,
    )
    frame = []
    lookup = {}
    for index, (date, team, probable, pitcher, hit) in enumerate(rows):
        game = 1000 + index
        frame.append(dict(metadata, review_row=index, game_pk=game, date=date,
                          pitcher_id=pitcher, is_hit=hit))
        lookup[game] = {"away": probable, "away_tid": team, "home_tid": 999}
    return pd.DataFrame(frame), lookup

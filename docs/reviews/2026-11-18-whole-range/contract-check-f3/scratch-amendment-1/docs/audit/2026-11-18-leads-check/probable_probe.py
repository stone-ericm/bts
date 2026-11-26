import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from bts.features import compute, park_drag


def feed(root, pk, probable, *, team=51):
    path = root / "2027" / f"{pk}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gameData": {
        "probablePitchers": {"away": {"id": probable}} if probable is not None else {},
        "teams": {"away": {"id": team}, "home": {"id": 52}},
    }}))
    return path


def disk_lookup(root):
    raw_a, raw_b, cache = root / "a", root / "b", root / "lookup.json"
    feed(raw_a, 100, None)
    feed(raw_b, 100, 22, team=61)
    feed(raw_b, 101, 23, team=61)
    with patch.object(compute, "_probable_pitcher_cache_path", cache):
        empty_away = compute._build_probable_pitcher_lookup(raw_a)[100]
        assert empty_away["away"] is None and empty_away["away_tid"] == 51
        feed(raw_a, 100, 21)
        assert compute._build_probable_pitcher_lookup(raw_a)[100] == empty_away
        mixed = compute._build_probable_pitcher_lookup(raw_b)
        assert mixed[100] == empty_away and mixed[101]["away"] == 23

        real_write = Path.write_text

        def interrupted_write(path, data, *args, **kwargs):
            assert path == cache
            real_write(path, data[:7], *args, **kwargs)
            raise OSError("synthetic interruption after truncation")

        feed(raw_b, 102, 24)
        with patch.object(Path, "write_text", interrupted_write):
            try:
                compute._build_probable_pitcher_lookup(raw_b)
            except OSError as exc:
                assert str(exc).startswith("synthetic interruption")
            else:
                raise AssertionError("cache write did not reach interruption")
        torn = cache.read_text()
        assert compute._build_probable_pitcher_lookup(root / "absent") == {}
        assert cache.read_text() == torn
        rebuilt = compute._build_probable_pitcher_lookup(raw_b)
        assert rebuilt[100]["away"] == 22 and rebuilt[100]["away_tid"] == 61
        assert set(rebuilt) == {100, 101, 102} and len(json.loads(cache.read_text())) == 3
    print("disk: same ID stays stale across updates and raw roots; different ID loads; "
          "interruption tears cache; absent root returns empty, present root rebuilds")


def pa(pk, day, index, hit):
    return {
        "pa_id": index, "date": day, "game_pk": pk, "batter_id": 11, "pitcher_id": 25,
        "pitch_hand": "R", "venue_id": 31, "is_home": True, "is_hit": hit,
        "pitch_calls": ["X"], "pitch_types": ["FF"], "pitch_speeds": [93.0],
        "pitch_spin_rates": [2300.0], "pitch_extensions": [6.0],
        "pitch_break_vertical": [15.0], "pitch_break_horizontal": [5.0],
        "pitch_px": [0.0], "pitch_pz": [2.5], "sz_top": 3.5, "sz_bottom": 1.5,
        "final_count_balls": 0, "final_count_strikes": 0,
        "launch_speed": 91.0, "launch_angle": 10.0,
    }


def history(root):
    raw = root / "data/raw"
    dates = pd.date_range("2027-04-01", periods=12)
    rows = [pa(200 + i, date, i, i < 10) for i, date in enumerate(dates)]
    # Second game on the target date, with an independent gamePk and an actual
    # reliever PA. Its rate is 1, the formerly unknown side's rate is 0.
    rows.append(pa(300, dates[10], 99, True))
    frame = pd.DataFrame(rows).astype({"is_hit": "int64"})
    for i in range(12):
        feed(raw, 200 + i, None if i == 10 else 21)
    feed(raw, 300, 22)
    # The default cache and raw paths are exercised under two synthetic CWDs.
    feature = "opp_bullpen_hr_30g"
    original_dir = Path.cwd()
    with patch.object(park_drag, "get_table", return_value=None):
        os.chdir(root)
        try:
            compute._build_probable_pitcher_lookup()
            feed(raw, 210, 21)
            stale = compute.compute_all_features(frame).set_index("pa_id")
            fresh_root = root / "fresh"
            for i in range(12):
                feed(fresh_root / "data/raw", 200 + i, 21)
            feed(fresh_root / "data/raw", 300, 22)
            os.chdir(fresh_root)
            fresh = compute.compute_all_features(frame).set_index("pa_id")
        finally:
            os.chdir(original_dir)
    assert stale.loc[[10, 99], feature].tolist() == fresh.loc[[10, 99], feature].tolist() == [1.0, 1.0]
    assert stale.loc[11, feature] == 1.0
    assert fresh.loc[11, feature] == 10.5 / 11
    print(f"features: same warmed value for both doubleheader games; next date stale={stale.loc[11, feature]:.6f}, "
          f"fresh={fresh.loc[11, feature]:.6f}")


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        disk_lookup(Path(tmp) / "lookup")
        history(Path(tmp) / "pipeline")

"""Probe preseason build and the raw-feed cache used by the bullpen feature."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from harness import IS_BASE, World, write_json
from opening import D1


def observations(index):
    return dict(pitch_calls=["X"], pitch_types=["FF"], pitch_speeds=[93.0],
                pitch_spin_rates=[2300.0], pitch_extensions=[6.0], pitch_break_vertical=[15.0],
                pitch_break_horizontal=[5.0], pitch_px=[0.0], pitch_pz=[2.5],
                sz_top=3.5, sz_bottom=1.5, final_count_balls=0, final_count_strikes=0,
                launch_speed=91.0, launch_angle=10.0, pa_id=index, game_pk=1300 + index,
                batter_id=100, pitcher_id=222 if index < 10 else 221,
                pitch_hand="R", venue_id=31, is_home=True, is_hit=index % 2)


def probe(root):
    from bts.data import pull
    from bts.features import compute, park_drag
    from bts.model import predict
    world = World(root, D1, health=False)
    raw = world.root / "data/raw"
    processed = world.root / "data/processed"
    cache = world.root / "fixture-models/probable_pitcher_lookup.json"
    # Use prior-season synthetic dates only. The cache writer is pointed into
    # scratch since the harness symlinks shipped policy artifacts read-only.
    for index in range(11):
        write_json(raw / "2025" / f"{1300 + index}.json",
                   {"gameData": {"teams": {"away": {"id": 51}, "home": {"id": 52}},
                                 "probablePitchers": {"away": {"id": 221}, "home": {"id": 223}}}})
    date_rows = [
        dict(observations(index), date=day)
        for index, day in enumerate(pd.date_range("2025-03-27", periods=11))
    ]
    date_rows.append(dict(date_rows[-1], pa_id=11))
    only_starter = pd.DataFrame(date_rows)
    plus_reliever = only_starter.copy()
    plus_reliever.loc[plus_reliever["pa_id"] == 11, "pitcher_id"] = 222
    with (
        world.switched_on(),
        patch.object(pull, "pull_feeds", lambda *a, **kw: []),
        patch.object(compute, "_probable_pitcher_cache_path", cache),
        patch.object(park_drag, "get_table", lambda *a, **kw: None),
    ):
        for state in ("missing", "spring_only", "malformed"):
            if state == "spring_only":
                write_json(raw / "2027/7701.json", {"gameData": {"game": {"type": "S"}}})
            if state == "malformed":
                (raw / "2027/7701.json").write_text("invalid JSON")
            world.invoke("data", "build", "--seasons", 2027, "--raw-dir", raw, "--out-dir", processed)
            try:
                predict._refresh_season_data(D1, raw, processed)
            except (OSError, ValueError, RuntimeError) as error:
                world.event("refresh", inputs=state, result=repr(error))
            else:
                world.event("refresh", inputs=state, result="success",
                            wrote_season=bool(list(processed.glob("*"))))
        cache.parent.mkdir()
        cache.write_bytes(b"invalid JSON")
        before = compute.compute_all_features(only_starter).set_index("pa_id")
        after = compute.compute_all_features(plus_reliever).set_index("pa_id")
        def value(frame):
            result = frame.loc[10, "opp_bullpen_hr_30g"]
            return float(result) if pd.notna(result) else None
        world.event("bullpen", starter_only=value(before), plus_reliever=value(after),
                    cache_keys=sorted(json.loads(cache.read_text())),
                    evidence=sorted(str(path.relative_to(root)) for path in
                                    root.glob("health_state/probable_pitcher_cache/**/*.json")))
    return {"archived": IS_BASE, "events": world.events}


if __name__ == "__main__":
    with TemporaryDirectory(prefix="wrblind-cold-") as temp:
        result = probe(Path(temp))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2))

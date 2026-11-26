"""Build and refresh a synthetic preseason, then compute bullpen features."""

import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from bts.data import pull
from bts.features import compute, park_drag
from bts.health import probable_pitcher_cache

from offline import World, at
from opening import OPEN


def check():
    from bts.model import predict

    real_pipeline = predict.run_pipeline
    with World(OPEN) as w:
        raw = w.root / "data/raw"
        spring = raw / "2027"
        spring.mkdir(parents=True)
        (spring / "700.json").write_text(json.dumps({"gameData": {"game": {"type": "S"}}}))
        (spring / "700_weather.json").write_text(json.dumps({"temp": 63}))
        w.invoke("data", "build", "--seasons", "2027", expect=0)
        w.patch(pull, "urlopen", lambda *args, **kwargs: BytesIO(b'{"dates":[]}'))
        w.patch(pull, "retry_urlopen", lambda *args, **kwargs: BytesIO(b'{"dates":[]}'))
        w.now = at(OPEN, "03:00")
        w.record("refresh", predict._refresh_season_data(OPEN), list(map(str, Path("data/processed").glob("*"))))

        history = raw / "2025"
        history.mkdir()
        probable = {"away": 21, "home": 22, "away_tid": 51, "home_tid": 52}
        path = w.models / "probable_pitcher_lookup.json"
        path.write_text(json.dumps({**{str(100+i): probable for i in range(10)}, "110": [], "111": 42}))
        (history / "110.json").write_text(json.dumps({"gameData": {
            "teams": {"away": {"id": 51}, "home": {"id": 52}},
            "probablePitchers": {"away": {"id": 21}, "home": {"id": 22}}}}))
        lookup = compute._build_probable_pitcher_lookup(str(raw))
        w.record("cache_after", lookup, json.loads(path.read_bytes()))
        status = w.root / "data/health_state/probable_pitcher_cache"
        w.record("cache_health", [alarm.message for alarm in probable_pitcher_cache.check(status, w.now.date())])
        w.patch(park_drag, "get_table", lambda: None)
        rows = []
        for i, day in enumerate(pd.date_range("2025-03-27", periods=11)):
            rows.append(dict(pa_id=i, game_pk=100+i, date=day, batter_id=11,
                             pitcher_id=22 if i < 10 else 21, pitch_hand="R", venue_id=31,
                             is_home=True, is_hit=i % 2, pitch_calls=["X"], pitch_types=["FF"],
                             pitch_speeds=[93.0], pitch_spin_rates=[2300.0], pitch_extensions=[6.0],
                             pitch_break_vertical=[15.0], pitch_break_horizontal=[5.0], pitch_px=[0.0],
                             pitch_pz=[2.5], sz_top=3.5, sz_bottom=1.5, final_count_balls=0,
                             final_count_strikes=0, launch_speed=91.0, launch_angle=10.0))
        rows.append(dict(rows[-1], pa_id=11))
        starter = pd.DataFrame(rows)
        reliever = starter.copy()
        reliever.loc[11, "pitcher_id"] = 22
        before = compute.compute_all_features(starter).set_index("pa_id")
        after = compute.compute_all_features(reliever).set_index("pa_id")
        a, b = (float(frame.loc[10, "opp_bullpen_hr_30g"]) for frame in (before, after))
        w.record("starter_feature", a, b)
        assert a == b == 0.5
        processed = w.root / "data/processed"
        processed.mkdir(exist_ok=True)
        reliever.to_parquet(processed / "pa_2025.parquet", index=False)

        class StopBeforeFit(Exception):
            pass

        def stop_at_training(frame, *args, **kwargs):
            w.record("real_pipeline_before_first_fit", len(frame), sorted(frame["date"].dt.year.unique().tolist()),
                     float(frame.loc[frame["pa_id"] == 10, "opp_bullpen_hr_30g"].iloc[0]))
            raise StopBeforeFit()

        w.patch(predict, "train_model", stop_at_training)
        try:
            real_pipeline(OPEN, data_dir=str(processed))
        except StopBeforeFit:
            pass
        else:
            raise AssertionError("prediction did not reach the expected first-fit boundary")

        path.write_text("[42]")
        w.record("cache_from_whole_invalid", compute._build_probable_pitcher_lookup(str(raw)))
        w.record("cache_health_after_whole_invalid",
                 [alarm.message for alarm in probable_pitcher_cache.check(status, w.now.date())])
        w.dump("cold")


if __name__ == "__main__":
    check()

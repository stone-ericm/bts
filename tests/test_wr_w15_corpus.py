"""Real feature fixtures for the probable-pitcher cache corpus."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from bts.features import compute, park_drag


SEED = dict(away=1, home=22, away_tid=51, home_tid=52)
RECOVERY = dict(away=2, home=23, away_tid=61, home_tid=62)
RECOVERED = [(2, True, 61), (23, True, 62)]
MISSING = [(None, False, None), (None, False, None)]
BAD_IDS = {
    "string": "1", "integral-float": 1.0, "fraction": 1.5,
    "true": True, "false": False, "list": [1], "object": {"id": 1},
    "nan": float("nan"), "infinity": float("inf"), "negative-infinity": float("-inf"),
}
BAD_FIELDS = {
    f"{field}-{shape}": dict(SEED, **{
        field: 51.0 if field == "away_tid" and shape == "integral-float" else value
    })
    for field in ("away", "away_tid")
    for shape, value in BAD_IDS.items()
}
CACHE_CASES = {
    "entry-null": None, "entry-number": 1, "entry-string": "x",
    "entry-list": [], "entry-true": True, "entry-false": False,
    "empty": {}, "extra": dict(SEED, ballast={"unused": True}),
    **{f"missing-{field}": {k: v for k, v in SEED.items() if k != field} for field in SEED},
    "missing-away-and-home_tid": {"home": 22, "away_tid": 51},
    **BAD_FIELDS,
}


def feed_payload(entry):
    return json.dumps({"gameData": {
        "probablePitchers": {side: {"id": entry[side]} for side in ("away", "home")},
        "teams": {side: {"id": entry[f"{side}_tid"]} for side in ("away", "home")},
    }})


RAW_CASES = {
    **{name: feed_payload(value) for name, value in BAD_FIELDS.items()},
    "combined": feed_payload(dict(away="2", home=True, away_tid=61.0, home_tid={"id": 62})),
    "null-side": '{"gameData":{"probablePitchers":{"away":null}}}',
    "malformed-side": '{"gameData":{"teams":{"home":[]}}}',
    "broken-json": "{",
    "non-object": "[]",
}


def pa(game, day, pitcher, home, hit=0):
    return dict(
        pa_id=0, date=day, game_pk=game, batter_id=11, pitcher_id=pitcher,
        pitch_hand="R", venue_id=31, is_home=home, is_hit=hit, pitch_calls=["X"],
        pitch_types=["FF"], pitch_speeds=[93.0], pitch_spin_rates=[2300.0],
        pitch_extensions=[6.0], pitch_break_vertical=[15.0],
        pitch_break_horizontal=[5.0], pitch_px=[0.0], pitch_pz=[2.5],
        sz_top=3.5, sz_bottom=1.5, final_count_balls=0, final_count_strikes=0,
        launch_speed=91.0, launch_angle=10.0,
    )


ROWS = [pa(100, "2025-04-01", 1, True), pa(100, "2025-04-01", 22, False)]


def feature(rows=ROWS):
    columns = ["_probable_pid", "_is_reliever_pa", "_pitcher_team_id"]
    observed = []
    code = compute.compute_all_features.__code__
    capture_line = next(
        i for i, line in enumerate(Path(code.co_filename).read_text().splitlines(), 1)
        if line.strip().startswith("reliever_pas = df[")
    )

    def trace(frame, event, _arg):
        if frame.f_code is code:
            if event == "line" and frame.f_lineno == capture_line:
                observed.extend(frame.f_locals["df"][columns].itertuples(index=False, name=None))
            return trace
        return None

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        frame = pd.DataFrame(rows)
        frame["pa_id"] = range(len(frame))
        result = compute.compute_all_features(frame)
    except (AttributeError, TypeError) as error:
        return observed, error, None
    finally:
        sys.settrace(previous)
    return observed, None, result


@pytest.fixture
def corpus_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "data/models/probable_pitcher_lookup.json"
    cache.parent.mkdir(parents=True)
    raw_feed = tmp_path / "data/raw/2025/100.json"
    raw_feed.parent.mkdir(parents=True)
    # Keep the feature entry point's default raw_dir inside tmp_path.
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", cache)
    monkeypatch.setattr(park_drag, "get_table", lambda: None)
    return cache, raw_feed


@pytest.mark.parametrize("name", CACHE_CASES)
@pytest.mark.parametrize("with_feed", [False, True])
def test_cache_corpus(name, with_feed, corpus_env):
    cache, raw_feed = corpus_env
    cache.write_text(json.dumps({"100": CACHE_CASES[name], "999": SEED}))
    if with_feed:
        raw_feed.write_text(feed_payload(RECOVERY))
    observed, error, _ = feature()
    assert error is None
    assert observed == (RECOVERED if with_feed else MISSING)


@pytest.mark.parametrize("name", RAW_CASES)
def test_raw_corpus(name, corpus_env, capsys):
    cache, raw_feed = corpus_env
    cache.write_text(json.dumps({"999": SEED}))
    before = cache.read_bytes()
    raw_feed.write_text(RAW_CASES[name])
    observed, error, _ = feature()
    assert error is None
    assert observed == MISSING
    assert cache.read_bytes() == before
    assert capsys.readouterr().err == ""
    assert not (cache.parent.parent / "health_state").exists()


def test_missing_game_with_empty_lookup_has_no_team_column(corpus_env):
    cache, _ = corpus_env
    cache.write_text('{"100": null}')
    observed, error, result = feature()
    assert error is None and observed == []
    assert "opp_pitching_team_id" not in result
    assert result["opp_bullpen_hr_30g"].isna().all()


def test_w15_base_corpus_report(corpus_env):
    """Opt-in observation of the unchanged tree for the author handover."""
    import os

    if os.environ.get("BTS_W15_RECORD_BASE") != "1":
        pytest.skip("one-time base corpus report")
    cache, raw_feed = corpus_env

    def report(section, name):
        observed, error, _ = feature()
        print("W15_BASE", section, name, repr(observed), type(error).__name__, sep="|")

    for name, value in CACHE_CASES.items():
        cache.write_text(json.dumps({"100": value, "999": SEED}))
        report("cache", name)
    for name, payload in RAW_CASES.items():
        cache.write_text(json.dumps({"999": SEED}))
        raw_feed.write_text(payload)
        report("raw", name)
    raw_feed.unlink()
    for name, seed in [
        ("valid", {"100": SEED}),
        ("valid-null", {"100": dict(SEED, away=None, away_tid=None)}),
        ("absent-with-seed", {"999": SEED}),
        ("absent-empty", {}),
    ]:
        cache.write_text(json.dumps(seed))
        report("control", name)
    cache.write_text(json.dumps({"999": SEED}))
    raw_feed.write_text(feed_payload(RECOVERY))
    report("control", "recovered")
    cache.write_text(json.dumps({"999": SEED}))
    raw_feed.write_text("{}")
    report("control", "all-null-feed")

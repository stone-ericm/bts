"""The 2026 framing test's walk-forward hook (design `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` §4):
`blend_walk_forward(..., predict_day_transform=...)` replaces values on the copy of the predicted day's rows only, before
the blend predicts and before the estimated-PA step builds its starter and reliever rows; training frames, labels and
the caller's frame are untouched. `strict_predict=True` re-raises a model's prediction failure instead of scoring it
as missing (review r1 finding 4: no silent averaging of fewer models)."""
import pandas as pd
import pytest

import bts.simulate.backtest_blend as bb


def _world():
    """Two 2024 training days and two 2025 test days, two games a day, starters and a reliever."""
    rows = []
    for date, season in (("2024-04-01", 2024), ("2024-04-02", 2024), ("2025-04-01", 2025), ("2025-04-02", 2025)):
        base = int(date.replace("-", ""))
        for g, (home, starter, reliever) in enumerate(((True, 80, 81), (False, 90, 91))):
            pk = base * 10 + g
            for b, lineup in ((1, 1), (2, 9)):
                batter = 100 * (g + 1) + b
                for pitcher in (starter, starter, reliever):
                    rows.append({"date": date, "season": season, "game_pk": pk, "is_home": home, "batter_id": batter,
                                 "pitcher_id": pitcher, "lineup_position": lineup,
                                 "catcher_framing": 0.1 * (g + 1), "pitcher_hr_30g": 0.2, "pitcher_entropy_30g": 0.5,
                                 "is_hit": int(b == 1)})
    return pd.DataFrame(rows)


def _stub(monkeypatch, seen, fail=False):
    """The blend is one model whose prediction is the row's `catcher_framing` (so an override is visible in every
    score) and whose training records the frame it was given. With `fail`, the blend's own prediction on the day's rows
    fails (their `pitcher_hr_30g` is 0.2); the estimated-PA reliever rows carry the training window's reliever rate
    (0.5 here) and are scored, as the reliever path has no failure handling of its own."""
    def predict(_model, day_data, _cols):
        if fail and (day_data["pitcher_hr_30g"] == 0.2).all():
            raise RuntimeError("model failed")
        return pd.Series(day_data["catcher_framing"].to_numpy(dtype=float), index=day_data.index)

    def train(available, _configs, _params, cached_models=None):
        seen["train"].append(available[["date", "batter_id", "catcher_framing"]].copy())
        return {"baseline": (object(), ["catcher_framing", "pitcher_hr_30g", "pitcher_entropy_30g"], predict)}, set()

    monkeypatch.setattr(bb, "_train_blend_for_day", train)


def _run(df, **kw):
    return bb.blend_walk_forward(df, 2025, retrain_every=1, blend_configs=[("baseline", ["catcher_framing"])],
                                 lgb_params={}, game_probability_mode=bb.GAME_PROBABILITY_ESTIMATED_PA, **kw)


def test_without_a_transform_the_walk_forward_is_unchanged(monkeypatch):
    seen = {"train": []}
    _stub(monkeypatch, seen)
    default = _run(_world())
    explicit = _run(_world(), predict_day_transform=None)
    pd.testing.assert_frame_equal(default, explicit)


def test_the_override_reaches_the_starter_and_the_reliever_predictions(monkeypatch):
    seen = {"train": []}
    _stub(monkeypatch, seen)

    def override(day_data, day):
        out = day_data.copy()
        out["catcher_framing"] = 0.7
        return out

    out = _run(_world(), predict_day_transform=override)
    # Both the starter score and the reliever score are the model's prediction of 0.7, so every batter-game is
    # 1 - 0.3 ** est_pas exactly; without the override they would be 0.1 or 0.2.
    expected = 1 - 0.3 ** out["est_pas"]
    assert out["p_game_hit"].tolist() == pytest.approx(expected.tolist())


def test_training_frames_and_the_callers_frame_are_untouched(monkeypatch):
    plain, hooked = {"train": []}, {"train": []}
    _stub(monkeypatch, plain)
    _run(_world())
    _stub(monkeypatch, hooked)
    df = _world()
    before = df.copy()

    def override(day_data, day):
        out = day_data.copy()
        out["catcher_framing"] = 0.7
        return out

    _run(df, predict_day_transform=override)
    assert len(hooked["train"]) == len(plain["train"]) == 2
    for a, b in zip(hooked["train"], plain["train"]):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    pd.testing.assert_frame_equal(df, before)


def test_the_transform_sees_each_predicted_day_once_with_its_date(monkeypatch):
    seen = {"train": []}
    _stub(monkeypatch, seen)
    days = []

    def record(day_data, day):
        days.append((pd.Timestamp(day).date().isoformat(), set(pd.to_datetime(day_data["date"]).dt.date.astype(str))))
        return day_data

    _run(_world(), predict_day_transform=record)
    assert days == [("2025-04-01", {"2025-04-01"}), ("2025-04-02", {"2025-04-02"})]


@pytest.mark.parametrize("bad", ["drop_row", "reorder", "new_column", "drop_column", "not_a_frame"])
def test_a_transform_that_changes_the_frames_shape_is_refused(monkeypatch, bad):
    seen = {"train": []}
    _stub(monkeypatch, seen)

    def damage(day_data, day):
        if bad == "drop_row":
            return day_data.iloc[1:]
        if bad == "reorder":
            return day_data.iloc[::-1]
        if bad == "new_column":
            return day_data.assign(extra=1)
        if bad == "drop_column":
            return day_data.drop(columns=["pitcher_hr_30g"])
        return None

    with pytest.raises(ValueError, match="predict_day_transform"):
        _run(_world(), predict_day_transform=damage)


def test_a_prediction_failure_is_scored_missing_by_default_and_raised_when_strict(monkeypatch):
    seen = {"train": []}
    _stub(monkeypatch, seen, fail=True)
    out = _run(_world())                       # existing behavior: the failure is printed and scored as missing
    assert out["p_game_hit"].isna().all()
    with pytest.raises(RuntimeError, match="model failed"):
        _run(_world(), strict_predict=True)


def test_strict_mode_refuses_a_model_that_returns_missing_scores(monkeypatch):
    def predict(_model, day_data, _cols):
        if (day_data["pitcher_hr_30g"] == 0.2).all():
            return pd.Series(float("nan"), index=day_data.index)       # no exception, but no score either
        return pd.Series(day_data["catcher_framing"].to_numpy(dtype=float), index=day_data.index)

    def train(available, _configs, _params, cached_models=None):
        return {"baseline": (object(), ["catcher_framing", "pitcher_hr_30g", "pitcher_entropy_30g"], predict),
                "second": (object(), ["catcher_framing"], lambda m, d, c: pd.Series(0.5, index=d.index))}, set()

    monkeypatch.setattr(bb, "_train_blend_for_day", train)
    _run(_world())                              # default: the blend averages what it has, as before
    with pytest.raises(ValueError, match="non-finite"):
        _run(_world(), strict_predict=True)

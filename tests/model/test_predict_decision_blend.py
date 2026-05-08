from __future__ import annotations

import pandas as pd


def test_train_blend_accepts_decision_weighted_classifier_config(monkeypatch):
    import bts.simulate.backtest_blend as bb
    from bts.model.predict import train_blend

    seen = {}
    fake_model = object()

    def fake_train_lgbm_classifier(available, cols, merged_params):
        seen["cols"] = cols
        seen["merged_params"] = dict(merged_params)
        return fake_model

    monkeypatch.setattr(bb, "_train_lgbm_classifier", fake_train_lgbm_classifier)
    df = pd.DataFrame({
        "season": [2024, 2024],
        "date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
        "batter_id": [1, 2],
        "game_pk": [100, 101],
        "feature": [0.1, 0.2],
        "is_hit": [0, 1],
    })

    blend = train_blend(
        df,
        blend_configs=[
            ("baseline", ["feature"], {"decision_weight_mode": "top_slate_v0"})
        ],
        lgb_params={"n_estimators": 1},
    )

    assert blend == {"baseline": (fake_model, ["feature"])}
    assert seen["cols"] == ["feature"]
    assert seen["merged_params"]["n_estimators"] == 1
    assert seen["merged_params"]["decision_weight_mode"] == "top_slate_v0"

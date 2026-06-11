"""Tests for slate persistence (bts.slate.save_slate).

Production previously discarded the full ranked slate every cycle — only
pick/double_down/runner_up survived in {date}.json. save_slate persists the
candidate-level predictions so realized slate-level metrics (rolling AUC,
sub-top-1 ranking, feature attribution on live data) become computable.
"""

import json

import numpy as np
import pandas as pd

from bts.slate import SCHEMA_VERSION, save_slate


def _predictions(n=3):
    return pd.DataFrame({
        "batter_id": [100 + i for i in range(n)],
        "batter_name": [f"Batter {i}" for i in range(n)],
        "team": ["NYY"] * n,
        "game_pk": [778899] * n,
        "lineup": [i + 1 for i in range(n)],
        "pitcher_id": [200] * n,
        "pitcher_name": ["Some Pitcher"] * n,
        "p_game_hit": [0.8 - 0.05 * i for i in range(n)],
        "p_hit_vs_starter": [0.3 - 0.01 * i for i in range(n)],
        "p_hit_vs_reliever": [0.28] * n,
        "est_pas": [4.5 - 0.1 * i for i in range(n)],
        "flags": [""] * n,
    })


def test_writes_slate_with_schema(tmp_path):
    path = save_slate(_predictions(), "2026-06-11", tmp_path, "hetzner")

    assert path == tmp_path / "slates" / "2026-06-11.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["date"] == "2026-06-11"
    assert payload["tier"] == "hetzner"
    assert payload["n_rows"] == 3
    assert len(payload["rows"]) == 3
    row = payload["rows"][0]
    assert row["batter_id"] == 100
    assert row["game_pk"] == 778899
    assert row["p_game_hit"] == 0.8
    assert "written_at" in payload


def test_nan_serialized_as_null(tmp_path):
    preds = _predictions()
    preds.loc[0, "p_game_hit"] = np.nan

    path = save_slate(preds, "2026-06-11", tmp_path, "hetzner")

    payload = json.loads(path.read_text())
    assert payload["rows"][0]["p_game_hit"] is None


def test_optional_columns_included_when_present(tmp_path):
    preds = _predictions()
    preds["p_game_blend"] = 0.79

    path = save_slate(preds, "2026-06-11", tmp_path, "hetzner")

    payload = json.loads(path.read_text())
    assert payload["rows"][0]["p_game_blend"] == 0.79


def test_missing_optional_columns_tolerated(tmp_path):
    preds = _predictions().drop(columns=["flags", "pitcher_name", "est_pas"])

    path = save_slate(preds, "2026-06-11", tmp_path, "hetzner")

    payload = json.loads(path.read_text())
    assert payload["n_rows"] == 3
    assert "flags" not in payload["rows"][0]


def test_last_write_wins(tmp_path):
    save_slate(_predictions(3), "2026-06-11", tmp_path, "hetzner")
    path = save_slate(_predictions(5), "2026-06-11", tmp_path, "mac")

    payload = json.loads(path.read_text())
    assert payload["n_rows"] == 5
    assert payload["tier"] == "mac"


def test_never_raises_on_pathological_input(tmp_path):
    # slate persistence is observability — it must never break picking
    assert save_slate(pd.DataFrame(), "2026-06-11", tmp_path, None) is None
    assert save_slate(None, "2026-06-11", tmp_path, None) is None
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("")  # slates/ mkdir under a file must fail -> swallowed
    assert save_slate(_predictions(), "2026-06-11", blocked, "x") is None

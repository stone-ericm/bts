"""Tests for the park_drag daily producer (arming item 2)."""
import gzip
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.features import park_drag_producer as pdp
from bts.leaderboard.scraper import RateLimitedError

ANALYSIS_REPO = Path("/Users/eric/projects/juiced-ball-analysis")


class _FakeResp:
    def __init__(self, status_code=200, text="", json_obj=None):
        self.status_code = status_code
        self.text = text
        self._json = json_obj

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._json


def _pitch_csv(n_games=1, per_game=30, game_date="2026-07-07"):
    rows = []
    for g in range(n_games):
        for _ in range(per_game):
            rows.append({
                "game_pk": 900000 + g, "game_date": game_date, "home_team": "NYY",
                "release_speed": 95.0, "vx0": 2.0, "vy0": -135.0, "vz0": -5.0,
                "ax": -2.0, "ay": 30.0, "az": -20.0,
            })
    return pd.DataFrame(rows)


class TestFetch:
    def test_savant_ok(self):
        csv = _pitch_csv().to_csv(index=False)
        calls = []

        def fake(url, *, accept, timeout=180.0):
            calls.append(url)
            return _FakeResp(200, text=csv)

        df = pdp.fetch_savant_ff(2026, date(2026, 7, 5), date(2026, 7, 7), http_get=fake)
        assert len(df) == 30 and list(df.columns) == pdp.REQUIRED_PITCH_COLS
        assert len(calls) == 1

    def test_savant_403_kill_switch_no_retry(self):
        calls = []

        def fake(url, *, accept, timeout=180.0):
            calls.append(url)
            return _FakeResp(403, text="denied")

        with pytest.raises(RateLimitedError):
            pdp.fetch_savant_ff(2026, date(2026, 7, 5), date(2026, 7, 7), http_get=fake)
        assert len(calls) == 1  # never hammer a 403

    def test_savant_html_retries_then_fails(self, monkeypatch):
        monkeypatch.setattr(pdp.time, "sleep", lambda *_: None)
        calls = []

        def fake(url, *, accept, timeout=180.0):
            calls.append(url)
            return _FakeResp(200, text="<html>block page</html>")

        with pytest.raises(pdp.ProducerError):
            pdp.fetch_savant_ff(2026, date(2026, 7, 5), date(2026, 7, 7), http_get=fake)
        assert len(calls) == 4


class TestPhysics:
    def test_trimmed_mean_matches_reference(self):
        x = np.arange(100, dtype=float)
        # 5% trim each side of 0..99 -> mean of 5..94
        assert pdp.trimmed_mean(x) == pytest.approx(np.mean(np.arange(5, 95)))

    def test_cd_in_range_and_altitude_direction(self):
        pitches = _pitch_csv(n_games=2)
        pitches.loc[pitches.game_pk == 900001, "home_team"] = "COL"
        meta = pd.DataFrame([
            {"game_pk": 900000, "venue_id": 3313, "venue": "Yankee Stadium",
             "temp_f": 70, "condition": "Clear", "wind": "5 mph, In From CF"},
            {"game_pk": 900001, "venue_id": 19, "venue": "Coors Field",
             "temp_f": 70, "condition": "Clear", "wind": "5 mph, In From CF"},
        ])
        gl = pdp.compute_game_level(pitches, meta, 2026)
        assert len(gl) == 2
        nyy = gl[gl.venue_id == 3313].cd_trim.iloc[0]
        col = gl[gl.venue_id == 19].cd_trim.iloc[0]
        assert 0.24 < nyy < 0.50 and 0.24 < col < 0.50
        # same kinematics in thinner air -> HIGHER Cd after adjustment
        assert col > nyy

    def test_small_games_dropped(self):
        pitches = _pitch_csv(per_game=10)
        meta = pd.DataFrame([{"game_pk": 900000, "venue_id": 3313,
                              "venue": "Yankee Stadium", "temp_f": 70,
                              "condition": "Clear", "wind": ""}])
        assert len(pdp.compute_game_level(pitches, meta, 2026)) == 0


class TestBuildExport:
    def _game_level(self, n_dates=20):
        rows = []
        for i in range(n_dates):
            rows.append({"game_pk": 1000 + i, "season": 2026,
                         "game_date": f"2026-04-{i + 1:02d}", "venue_id": 1,
                         "venue": "Test Park", "n_pitch": 90,
                         "cd_trim": 0.32 - 0.001 * i})
        return pd.DataFrame(rows)

    def test_asof_excludes_same_date(self):
        export, manifest = pdp.build_export(self._game_level())
        # value on 04-16 must be the rolling-15 of dates 01..15 minus anchor(01..10)
        row = export[(export.venue_id == 1) & (export.date == "2026-04-16")]
        cds = [0.32 - 0.001 * i for i in range(15)]
        anchor = np.mean([0.32 - 0.001 * i for i in range(10)])
        w = (90 * 15) / (90 * 15 + pdp.SHRINK_K)
        expected = (np.mean(cds) - anchor) * w
        assert row.park_drag_delta.iloc[0] == pytest.approx(expected, abs=1e-12)

    def test_duplicate_key_raises(self):
        # data-error guard: rows mislabeled season=2025 but carrying 2026
        # calendar dates make both season groups emit the same (venue, date)
        gl = self._game_level()
        bad = gl.assign(season=2025, game_pk=gl.game_pk + 500)
        with pytest.raises(pdp.ProducerError):
            pdp.build_export(pd.concat([gl, bad], ignore_index=True))

    @pytest.mark.skipif(not (ANALYSIS_REPO / "data/game_level_full.csv").exists(),
                        reason="analysis repo not present on this machine")
    def test_matches_analysis_repo_builder(self):
        """Drift guard: this port must reproduce the analysis-repo export."""
        gl = pd.read_csv(ANALYSIS_REPO / "data/game_level_full.csv")
        export, _ = pdp.build_export(
            gl.rename(columns={})[["game_pk", "season", "game_date", "venue_id",
                                   "venue", "n_pitch", "cd_trim"]])
        ref = pd.read_csv(ANALYSIS_REPO / "data/park_drag_export.csv",
                          parse_dates=["date"])
        m = ref.merge(export, on=["venue_id", "date"], suffixes=("_ref", "_new"))
        assert len(m) == len(ref)
        a, b = m.park_drag_delta_ref.values, m.park_drag_delta_new.values
        both = ~(np.isnan(a) | np.isnan(b))
        assert (np.isnan(a) == np.isnan(b)).all()
        assert np.allclose(a[both], b[both], atol=1e-9)


class TestRefresh:
    def _seed(self, root: Path):
        prod = root / "producer"
        prod.mkdir(parents=True)
        store = _pitch_csv(n_games=1, game_date="2026-07-05")
        store.to_csv(prod / "pitches_current.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"game_pk": 900000, "start_utc": "2026-07-05T23:00:00Z",
                       "day_night": "night", "temp_f": 70, "condition": "Clear",
                       "wind": "", "venue_id": 3313, "venue": "Yankee Stadium"},
                      ]).to_csv(prod / "games_meta.csv", index=False)
        static = pd.DataFrame([{"game_pk": 1, "season": 2025,
                                "game_date": f"2025-04-{i + 1:02d}", "venue_id": 1,
                                "venue": "Test Park", "n_pitch": 90,
                                "cd_trim": 0.32} for i in range(20)])
        static.to_csv(prod / "game_level_static.csv", index=False)

    def _fake_http(self):
        new_csv = _pitch_csv(n_games=1, game_date="2026-07-06")
        new_csv["game_pk"] = 900007

        def fake(url, *, accept, timeout=180.0):
            if "baseballsavant" in url:
                return _FakeResp(200, text=new_csv.to_csv(index=False))
            return _FakeResp(200, json_obj={"gameData": {
                "datetime": {"dateTime": "2026-07-06T23:00:00Z", "dayNight": "night"},
                "weather": {"temp": "75", "condition": "Clear", "wind": "3 mph, Calm"},
                "venue": {"id": 3313, "name": "Yankee Stadium"},
            }})
        return fake

    def test_refresh_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pdp.time, "sleep", lambda *_: None)
        self._seed(tmp_path)
        summary = pdp.refresh(tmp_path, today=date(2026, 7, 7),
                              http_get=self._fake_http())
        assert summary["ok"], summary
        assert (tmp_path / "park_drag_export.csv").exists()
        manifest = json.loads((tmp_path / "park_drag_manifest.json").read_text())
        assert manifest["max_source_game_date"] == "2026-07-06"
        status = json.loads((tmp_path / "producer_status.json").read_text())
        assert status["ok"] is True

    def test_refresh_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pdp.time, "sleep", lambda *_: None)
        self._seed(tmp_path)
        pdp.refresh(tmp_path, today=date(2026, 7, 7), http_get=self._fake_http())
        first = (tmp_path / "park_drag_export.csv").read_text()
        pdp.refresh(tmp_path, today=date(2026, 7, 7), http_get=self._fake_http())
        assert (tmp_path / "park_drag_export.csv").read_text() == first

    def test_missing_seed_fails_with_instruction(self, tmp_path):
        summary = pdp.refresh(tmp_path, today=date(2026, 7, 7),
                              http_get=self._fake_http())
        assert summary["ok"] is False and "seed" in summary["error"]
        status = json.loads((tmp_path / "producer_status.json").read_text())
        assert status["ok"] is False

    def test_rate_limited_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pdp.time, "sleep", lambda *_: None)
        self._seed(tmp_path)

        def fake(url, *, accept, timeout=180.0):
            return _FakeResp(429, text="slow down")

        summary = pdp.refresh(tmp_path, today=date(2026, 7, 7), http_get=fake)
        assert summary["ok"] is False and summary.get("rate_limited") is True

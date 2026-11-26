"""Human-fidelity + throttle-safety behavior of the leaderboard scraper.

Covers the 2026-07-03 hardening: browser identity, jittered gaps, the 403/429
kill-switch that aborts the scrape (rather than hammering the account), and the
deep-board completeness flag.
"""
from __future__ import annotations

import json
import random
from datetime import date

import httpx
import pytest

import bts.leaderboard.scraper as scraper_mod
from bts.leaderboard.endpoints import browser_headers
from bts.leaderboard.ratelimit import next_gap
from bts.leaderboard.scraper import RateLimitedError, _scrape_active_streak_deep


class TestBrowserIdentity:
    def test_headers_look_like_chrome_not_a_bot(self):
        h = browser_headers()
        assert "Chrome/" in h["User-Agent"]
        assert "bts-leaderboard-watcher" not in h["User-Agent"]
        assert h["Accept-Language"].startswith("en-US")
        assert h["Referer"].endswith("beat-the-streak/game")
        assert h["sec-ch-ua-platform"] == '"macOS"'


class TestJitter:
    def test_gap_within_band(self):
        rng = random.Random(0)
        for _ in range(200):
            g = next_gap(2.0, 2.5, rng)
            assert 2.0 <= g <= 4.5

    def test_zero_jitter_is_fixed(self):
        assert next_gap(2.0, 0.0, random.Random(1)) == 2.0

    def test_jitter_actually_varies(self):
        rng = random.Random(0)
        vals = {round(next_gap(2.0, 2.5, rng), 4) for _ in range(50)}
        assert len(vals) > 40  # not a constant


def _rank(uid, rank, streak):
    return {"userId": uid, "rank": rank, "username": f"u{uid}",
            "activeStreak": streak, "streak": streak}


class TestKillSwitch:
    def test_get_json_raises_on_429(self, monkeypatch):
        class Resp:
            status_code = 429
            def raise_for_status(self): raise AssertionError("should not reach")
            def json(self): return {}
        monkeypatch.setattr(scraper_mod.httpx, "get", lambda *a, **k: Resp())
        with pytest.raises(RateLimitedError) as e:
            scraper_mod._get_json("http://x", cookies={})
        assert e.value.status_code == 429

    def test_get_json_raises_on_403(self, monkeypatch):
        class Resp:
            status_code = 403
            def raise_for_status(self): raise AssertionError
            def json(self): return {}
        monkeypatch.setattr(scraper_mod.httpx, "get", lambda *a, **k: Resp())
        with pytest.raises(RateLimitedError):
            scraper_mod._get_json("http://x", cookies={})

    def test_deep_propagates_rate_limit(self, monkeypatch):
        def _boom(url, cookies=None, **k):
            raise RateLimitedError(429, url)
        monkeypatch.setattr(scraper_mod, "_get_json", _boom)
        monkeypatch.setattr(scraper_mod, "_deep_page_pause", lambda: None)
        with pytest.raises(RateLimitedError):
            _scrape_active_streak_deep(cookies={}, xsid="x", season=2026,
                                       deep_limit=3, deep_max_pages=5, deep_min_streak=1)

    def test_run_aborts_and_alerts_on_throttle(self, monkeypatch, tmp_path):
        # active_streak page 1 throttles -> whole scrape aborts, status records
        # rate_limited, a DM fires, and NO snapshot is written.
        def fake_get_json(url, cookies=None, **k):
            if "ACTIVE_STREAK" in url:
                raise RateLimitedError(429, url)
            return {"success": {"ranks": []}}
        monkeypatch.setattr(scraper_mod, "_get_json", fake_get_json)
        monkeypatch.setattr(scraper_mod, "_deep_page_pause", lambda: None)
        monkeypatch.setattr(scraper_mod, "scrape_static_lookups",
                            lambda cookies: scraper_mod.StaticLookups())
        monkeypatch.setattr(scraper_mod, "scrape_user_profile",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no profiles after abort")))
        dms = []
        import bts.dm
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dms.append((h, m)))

        scraper_mod.run(cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
                        today=date(2026, 7, 4), deep_min_streak=1,
                        dm_recipient="x.bsky.social")

        status = json.loads((tmp_path / "scrape_status.json").read_text())
        assert status["rate_limited"] is True
        assert status["active_streak_complete"] is False
        assert len(dms) == 1 and "429" in dms[0][1]
        assert not (tmp_path / "leaderboard_snapshots" / "2026-07-04.parquet").exists()

    def test_throttle_dm_failure_leaves_alert_uncooled(self, monkeypatch, tmp_path):
        # If the throttle DM itself fails, last_alert_at must NOT be stamped, so a
        # later run can still alert (same discipline as check-pick-entered).
        def fake_get_json(url, cookies=None, **k):
            if "ACTIVE_STREAK" in url:
                raise RateLimitedError(429, url)
            return {"success": {"ranks": []}}
        monkeypatch.setattr(scraper_mod, "_get_json", fake_get_json)
        monkeypatch.setattr(scraper_mod, "_deep_page_pause", lambda: None)
        monkeypatch.setattr(scraper_mod, "scrape_static_lookups",
                            lambda cookies: scraper_mod.StaticLookups())
        import bts.dm
        monkeypatch.setattr(bts.dm, "send_dm",
                            lambda h, m: (_ for _ in ()).throw(RuntimeError("dm down")))
        scraper_mod.run(cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
                        today=date(2026, 7, 4), deep_min_streak=1,
                        dm_recipient="x.bsky.social")
        status = json.loads((tmp_path / "scrape_status.json").read_text())
        assert status["rate_limited"] is True
        assert "last_alert_at" not in status


class TestCompleteness:
    def _patch(self, monkeypatch, pages, fail=frozenset()):
        def fake_get_json(url, cookies=None, **k):
            if "ACTIVE_STREAK" in url:
                page = int(url.split("page=")[1].split("&")[0])
                if page in fail:
                    raise RuntimeError("transient")
                return {"success": {"ranks": pages.get(page, [])}}
            return {"success": {"ranks": []}}
        monkeypatch.setattr(scraper_mod, "_get_json", fake_get_json)
        monkeypatch.setattr(scraper_mod, "_deep_page_pause", lambda: None)

    def test_clean_stop_is_complete(self, monkeypatch):
        self._patch(monkeypatch, {1: [_rank(1, 1, 30), _rank(2, 2, 2)]})
        _, _, complete = _scrape_active_streak_deep(
            cookies={}, xsid="x", season=2026, deep_limit=3, deep_max_pages=5, deep_min_streak=3)
        assert complete is True

    def test_transient_error_marks_incomplete(self, monkeypatch):
        self._patch(monkeypatch,
                    {1: [_rank(i, i, 30) for i in range(1, 4)]}, fail={2})
        rows, _, complete = _scrape_active_streak_deep(
            cookies={}, xsid="x", season=2026, deep_limit=3, deep_max_pages=5, deep_min_streak=1)
        assert complete is False
        assert len(rows) == 3  # page-1 rows kept

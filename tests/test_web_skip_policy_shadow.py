"""Dashboard surfacing for the skip-policy shadow (bts/web.py)."""
import json

import bts.web as web


def test_render_section_shows_verdict_and_breakeven():
    status = {
        "counts": {"divergent_days": 5, "resolved_divergent": 3, "pending": 2},
        "shadow_band_hit_rate": {"resolved": 3, "hits": 2, "rate": 0.667,
                                 "wilson_ci": [0.2, 0.94], "breakeven_p": 0.744,
                                 "verdict": "insufficient_n"},
    }
    html = web.render_skip_policy_shadow_section(status)
    assert "pick-the-band" in html.lower() or "skip-policy" in html.lower()
    assert "0.744" in html
    assert "insufficient" in html.lower()


def test_render_section_empty_when_no_divergent_days():
    assert web.render_skip_policy_shadow_section({"counts": {"divergent_days": 0},
                                                  "shadow_band_hit_rate": {}}) == ""
    assert web.render_skip_policy_shadow_section({}) == ""


def test_load_all_picks_excludes_policy_shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
    (tmp_path / "2026-06-18.json").write_text(json.dumps({"date": "2026-06-18"}))
    (tmp_path / "2026-06-18.policy_shadow.json").write_text(
        json.dumps({"date": "2026-06-18", "divergent": True}))
    (tmp_path / "2026-06-18.shadow.json").write_text(json.dumps({"date": "2026-06-18"}))
    picks = web.load_all_picks()
    assert [p.get("date") for p in picks] == ["2026-06-18"]   # shadow + policy_shadow excluded
    assert all("divergent" not in p for p in picks)

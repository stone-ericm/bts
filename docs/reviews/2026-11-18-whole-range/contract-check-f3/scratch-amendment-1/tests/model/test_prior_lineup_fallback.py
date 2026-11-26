"""_fetch_prior_lineup must derive its schedule window from the prediction year,
not a hardcoded 2026 window (audit M6) — otherwise in 2027 the projected-lineup
fallback returns last year's lineups.
"""
import json

import pytest

try:  # lightgbm is an optional extra; skip (not error) when it/libomp is absent
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip(
        "lightgbm/libomp unavailable; skipping model tests",
        allow_module_level=True,
    )

from bts.model.predict import _fetch_prior_lineup


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()


def test_fetch_prior_lineup_uses_prediction_year(monkeypatch):
    urls = []

    def fake_urlopen(url, timeout=15):
        urls.append(url)
        return _Resp({"dates": []})  # no games -> returns []

    monkeypatch.setattr("bts.model.predict.urlopen", fake_urlopen)

    assert _fetch_prior_lineup(123, 2027) == []
    assert any("startDate=2027-03-20" in u and "endDate=2027-12-31" in u for u in urls), urls

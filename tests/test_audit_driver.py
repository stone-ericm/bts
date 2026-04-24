"""Tests for scripts/audit_driver.py secret lookup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestKeychainFallback:
    """_keychain must work on non-macOS hosts (bts-hetzner, Pi5) where the
    `security` command is absent. Falls back to a BTS_SECRET_<UPPER>_<UNDER>
    env var derived from the service name.
    """

    def test_fallback_to_env_var_when_keychain_misses(self, monkeypatch):
        from audit_driver import _keychain

        # Service name guaranteed NOT to exist in any real keychain.
        service = "unit-test-fake-service-for-keychain-fallback"
        env_name = "BTS_SECRET_UNIT_TEST_FAKE_SERVICE_FOR_KEYCHAIN_FALLBACK"
        monkeypatch.setenv(env_name, "sentinel-value-12345")

        assert _keychain(service) == "sentinel-value-12345"

    def test_raises_when_neither_keychain_nor_env_has_secret(self, monkeypatch):
        from audit_driver import _keychain

        service = "another-nonexistent-service-lmnop"
        env_name = "BTS_SECRET_ANOTHER_NONEXISTENT_SERVICE_LMNOP"
        monkeypatch.delenv(env_name, raising=False)

        with pytest.raises(RuntimeError):
            _keychain(service)


# ---------------------------------------------------------------------------
# teardown_retrieved + teardown_all tests
# ---------------------------------------------------------------------------

class FakeProvider:
    """Captures delete() calls instead of hitting a real API."""

    name = "fake"

    def __init__(self, raise_on_ids: set[str] | None = None) -> None:
        self.deleted: list[str] = []
        self._raise_on = raise_on_ids or set()

    def delete(self, box_id: str) -> None:
        if box_id in self._raise_on:
            raise RuntimeError(f"fake API failure for {box_id}")
        self.deleted.append(box_id)


@pytest.fixture
def captured_log(monkeypatch):
    """Replace audit_driver.log with a list-appender; returns the list."""
    from audit_driver import log as _original_log  # noqa: F401 — force import first
    import audit_driver
    captured: list[str] = []
    monkeypatch.setattr(audit_driver, "log", captured.append)
    return captured


@pytest.fixture
def boxes():
    from audit_driver import Box
    return [
        Box(id="1", name="b1", ipv4="10.0.0.1", region=""),
        Box(id="2", name="b2", ipv4="10.0.0.2", region=""),
        Box(id="3", name="b3", ipv4="10.0.0.3", region=""),
    ]


class TestTeardownRetrieved:
    def test_all_ok_tears_down_everything(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": "ok", "b2": "ok", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "2", "3"]
        assert selected == 3
        assert deleted == 3

    def test_one_partial_preserves_only_that_box(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": "ok", "b2": "partial", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "3"]
        assert selected == 2
        assert deleted == 2
        joined = "\n".join(captured_log)
        assert "PRESERVED b2" in joined
        assert "ip=10.0.0.2" in joined
        assert "retrieve_status=partial" in joined

    def test_all_partial_preserves_all(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": "partial", "b2": "partial", "b3": "partial"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0
        preserved_lines = [l for l in captured_log if "PRESERVED" in l]
        assert len(preserved_lines) == 3

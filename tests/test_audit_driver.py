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

    def test_missing_key_defaults_to_preserve(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        # b2 is missing from the dict
        results = {"b1": "ok", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "3"]
        assert selected == 2
        assert deleted == 2
        joined = "\n".join(captured_log)
        assert "PRESERVED b2" in joined
        assert "retrieve_status=not-attempted" in joined

    def test_empty_results_preserves_all(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()

        selected, deleted = teardown_retrieved(provider, boxes, {})

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0
        not_attempted = [l for l in captured_log if "not-attempted" in l]
        assert len(not_attempted) == 3

    def test_empty_boxes_list_noop(self, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()

        selected, deleted = teardown_retrieved(provider, [], {"b1": "ok"})

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0

    def test_malformed_values_preserve(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": None, "b2": True, "b3": "weird"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0

    def test_provider_delete_raises_on_one_box(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        # teardown_all's try/except will swallow this and keep going
        provider = FakeProvider(raise_on_ids={"2"})
        results = {"b1": "ok", "b2": "ok", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        # selected = 3 (picked all three), deleted = 2 (b2's API call failed)
        assert provider.deleted == ["1", "3"]
        assert selected == 3
        assert deleted == 2
        joined = "\n".join(captured_log)
        assert "FAILED to delete b2" in joined

    def test_retrieve_results_none_raises_typeerror(self, boxes):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()

        with pytest.raises(TypeError):
            teardown_retrieved(provider, boxes, None)

    def test_stray_key_logged_no_crash(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        # "b99" isn't in boxes — stray key
        results = {"b1": "ok", "b2": "ok", "b3": "ok", "b99": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "2", "3"]
        assert selected == 3
        assert deleted == 3
        joined = "\n".join(captured_log)
        assert "unrecognized key" in joined
        assert "b99" in joined


class TestTeardownAllReturn:
    def test_teardown_all_returns_count_of_successful_deletes(self, boxes, captured_log):
        from audit_driver import teardown_all
        provider = FakeProvider(raise_on_ids={"2"})

        deleted = teardown_all(provider, boxes)

        assert deleted == 2  # b1 and b3 succeeded; b2 raised
        assert provider.deleted == ["1", "3"]

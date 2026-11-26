"""Dashboard must not bind the public interface by default (audit S2).

:3003 is already firewalled externally, but the bind was 0.0.0.0 with no
app-layer auth, so a firewall change would silently expose pre-publication
picks + the audit-progress SSH fan-out. Default to the tailnet, never 0.0.0.0.
"""
from unittest.mock import patch

import bts.web as web


def test_bind_host_respects_override(monkeypatch):
    monkeypatch.setenv("BTS_DASHBOARD_HOST", "0.0.0.0")
    assert web._dashboard_bind_host() == "0.0.0.0"  # explicit opt-in still works


def test_bind_host_defaults_to_tailscale(monkeypatch):
    monkeypatch.delenv("BTS_DASHBOARD_HOST", raising=False)
    with patch("bts.web._tailscale_ipv4", return_value="100.100.43.24"):
        assert web._dashboard_bind_host() == "100.100.43.24"


def test_bind_host_never_public_without_override(monkeypatch):
    monkeypatch.delenv("BTS_DASHBOARD_HOST", raising=False)
    with patch("bts.web._tailscale_ipv4", return_value=None):
        host = web._dashboard_bind_host()
    assert host == "127.0.0.1"
    assert host != "0.0.0.0"

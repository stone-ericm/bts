"""Tests for /api/audit-progress response builder.

Tests ``audit_progress_response`` — the pure function behind the route.
Mirrors the pattern used for ``health_check``: keep the response logic
independent of BaseHTTPRequestHandler so it can be unit-tested without a
socket. Dependency-injects both ``project_root`` (so we can point at a
temp fixture) and ``scan`` (so we don't exercise real SSH).
"""
from __future__ import annotations

import json
from pathlib import Path

from bts.web import audit_progress_response


def _mk_fake_scan(recorded: list):
    """Return a fake scanner that records its args and returns a stub dict."""
    def fake(audit_dir: Path, seeds_file: Path | None = None, **kwargs) -> dict:
        recorded.append(
            {"audit_dir": audit_dir, "seeds_file": seeds_file, **kwargs}
        )
        return {
            "audit_dir": audit_dir.name,
            "scanned_at": "2026-04-24T18:00:00+00:00",
            "boxes": [],
            "overall": {
                "completed": 0,
                "expected": None,
                "boxes_done": 0,
                "boxes_total": 0,
                "pct_seeds": None,
            },
        }
    return fake


def _setup_audit_dir(root: Path, provider: str, name: str) -> Path:
    d = root / "data" / f"{provider}_results" / name
    d.mkdir(parents=True)
    (d / "boxes.json").write_text("[]")
    return d


class TestAuditProgressResponse:
    def test_missing_provider_returns_400(self, tmp_path: Path) -> None:
        status, body = audit_progress_response({"dir": ["x"]}, project_root=tmp_path)
        assert status == 400
        assert "required" in body["error"]

    def test_missing_dir_returns_400(self, tmp_path: Path) -> None:
        status, body = audit_progress_response({"provider": ["vultr"]}, project_root=tmp_path)
        assert status == 400

    def test_unknown_provider_returns_400(self, tmp_path: Path) -> None:
        status, body = audit_progress_response(
            {"provider": ["aws"], "dir": ["x"]}, project_root=tmp_path
        )
        assert status == 400
        assert "unknown provider" in body["error"]

    def test_missing_boxes_json_returns_404(self, tmp_path: Path) -> None:
        status, body = audit_progress_response(
            {"provider": ["vultr"], "dir": ["nope"]}, project_root=tmp_path
        )
        assert status == 404
        assert "boxes.json" in body["error"]

    def test_happy_path_passes_correct_audit_dir(self, tmp_path: Path) -> None:
        _setup_audit_dir(tmp_path, "vultr", "audit_ext_n100_v4")
        recorded: list = []
        status, body = audit_progress_response(
            {"provider": ["vultr"], "dir": ["audit_ext_n100_v4"]},
            project_root=tmp_path,
            scan=_mk_fake_scan(recorded),
        )
        assert status == 200
        assert len(recorded) == 1
        assert recorded[0]["audit_dir"].name == "audit_ext_n100_v4"
        assert recorded[0]["audit_dir"].parent.name == "vultr_results"
        assert recorded[0]["seeds_file"] is None
        assert body["audit_dir"] == "audit_ext_n100_v4"

    def test_seeds_file_relative_resolved_under_project_root(self, tmp_path: Path) -> None:
        _setup_audit_dir(tmp_path, "vultr", "audit_ext_n100_v4")
        (tmp_path / "scripts").mkdir()
        seeds = tmp_path / "scripts" / "seeds.txt"
        seeds.write_text("1,2,3")
        recorded: list = []
        status, _ = audit_progress_response(
            {
                "provider": ["vultr"],
                "dir": ["audit_ext_n100_v4"],
                "seeds_file": ["scripts/seeds.txt"],
            },
            project_root=tmp_path,
            scan=_mk_fake_scan(recorded),
        )
        assert status == 200
        assert recorded[0]["seeds_file"] == seeds

    def test_seeds_file_absolute_passed_through(self, tmp_path: Path) -> None:
        _setup_audit_dir(tmp_path, "vultr", "audit_ext_n100_v4")
        seeds = tmp_path / "absolute_seeds.txt"
        seeds.write_text("1")
        recorded: list = []
        status, _ = audit_progress_response(
            {
                "provider": ["vultr"],
                "dir": ["audit_ext_n100_v4"],
                "seeds_file": [str(seeds)],
            },
            project_root=tmp_path,
            scan=_mk_fake_scan(recorded),
        )
        assert status == 200
        assert recorded[0]["seeds_file"] == seeds

    def test_missing_seeds_file_returns_404(self, tmp_path: Path) -> None:
        _setup_audit_dir(tmp_path, "vultr", "audit_ext_n100_v4")
        status, body = audit_progress_response(
            {
                "provider": ["vultr"],
                "dir": ["audit_ext_n100_v4"],
                "seeds_file": ["scripts/missing.txt"],
            },
            project_root=tmp_path,
            scan=_mk_fake_scan([]),
        )
        assert status == 404
        assert "seeds_file" in body["error"]

    def test_query_string_values_can_be_raw_strings(self, tmp_path: Path) -> None:
        """parse_qs returns {key: [val]} but callers may pass {key: val}.
        Handle both gracefully."""
        _setup_audit_dir(tmp_path, "hetzner", "audit_full")
        recorded: list = []
        status, body = audit_progress_response(
            {"provider": "hetzner", "dir": "audit_full"},
            project_root=tmp_path,
            scan=_mk_fake_scan(recorded),
        )
        assert status == 200
        assert body["audit_dir"] == "audit_full"

    def test_endpoint_requests_audit_attach(self, tmp_path: Path) -> None:
        """The HTTP route should request audit_attach visibility by default —
        that's the value-add over calling the scanner from a notebook."""
        _setup_audit_dir(tmp_path, "vultr", "audit_ext_n100_v4")
        recorded: list = []
        audit_progress_response(
            {"provider": ["vultr"], "dir": ["audit_ext_n100_v4"]},
            project_root=tmp_path,
            scan=_mk_fake_scan(recorded),
        )
        assert recorded[0].get("include_audit_attach") is True

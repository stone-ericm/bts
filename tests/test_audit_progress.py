"""Tests for bts.audit_progress — live in-flight audit monitor.

Tests the SSH-based live progress scanner that powers /api/audit-progress.
ssh_runner is dependency-injected so tests don't actually open SSH sockets;
results are keyed by IP (not call order) so ThreadPoolExecutor dispatch order
doesn't make the tests flaky.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from bts.audit_progress import load_boxes, scan_audit_progress


SAMPLE_BOXES = [
    {"id": "1", "name": "bts-audit-vultr-1", "ipv4": "10.0.0.1", "region": "ewr"},
    {"id": "2", "name": "bts-audit-vultr-2", "ipv4": "10.0.0.2", "region": "ewr"},
    {"id": "3", "name": "bts-audit-vultr-3", "ipv4": "10.0.0.3", "region": "ewr"},
]


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audit_ext_n100_v4"
    d.mkdir()
    (d / "boxes.json").write_text(json.dumps(SAMPLE_BOXES))
    return d


def _mk_run(stdout: str, rc: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def _running_stdout(count: int, last: str = "=== seed=1234 done at Thu ===") -> str:
    return f"STATE:RUN\nCOUNT:{count}\nLAST:{last}\n"


def _done_stdout(count: int, last: str = "queue done at Thu") -> str:
    return f"STATE:DONE\nCOUNT:{count}\nLAST:{last}\n"


def _ssh_runner_from_map(
    responses: dict[str, subprocess.CompletedProcess],
) -> Callable[[str, str, int], subprocess.CompletedProcess]:
    def ssh_runner(ip: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        if ip not in responses:
            raise AssertionError(f"unexpected SSH to {ip}")
        return responses[ip]

    return ssh_runner


class TestLoadBoxes:
    def test_reads_boxes_json(self, audit_dir: Path) -> None:
        boxes = load_boxes(audit_dir)
        assert len(boxes) == 3
        assert boxes[0]["name"] == "bts-audit-vultr-1"
        assert boxes[0]["ipv4"] == "10.0.0.1"

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_boxes(tmp_path / "nope")


class TestScanAuditProgress:
    def test_all_running(self, audit_dir: Path) -> None:
        responses = {
            "10.0.0.1": _mk_run(_running_stdout(1)),
            "10.0.0.2": _mk_run(_running_stdout(1)),
            "10.0.0.3": _mk_run(_running_stdout(0)),
        }
        result = scan_audit_progress(
            audit_dir, ssh_runner=_ssh_runner_from_map(responses)
        )
        assert result["audit_dir"] == "audit_ext_n100_v4"
        assert "scanned_at" in result
        assert len(result["boxes"]) == 3
        assert all(b["state"] == "running" for b in result["boxes"])
        by_name = {b["name"]: b for b in result["boxes"]}
        assert by_name["bts-audit-vultr-1"]["completed_seeds"] == 1
        assert by_name["bts-audit-vultr-3"]["completed_seeds"] == 0
        assert result["overall"]["completed"] == 2
        assert result["overall"]["boxes_done"] == 0
        assert result["overall"]["boxes_total"] == 3

    def test_boxes_json_order_preserved(self, audit_dir: Path) -> None:
        responses = {ip: _mk_run(_running_stdout(0)) for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.3"]}
        result = scan_audit_progress(
            audit_dir, ssh_runner=_ssh_runner_from_map(responses)
        )
        names = [b["name"] for b in result["boxes"]]
        assert names == ["bts-audit-vultr-1", "bts-audit-vultr-2", "bts-audit-vultr-3"]

    def test_one_done(self, audit_dir: Path) -> None:
        responses = {
            "10.0.0.1": _mk_run(_done_stdout(2)),
            "10.0.0.2": _mk_run(_running_stdout(1)),
            "10.0.0.3": _mk_run(_running_stdout(0)),
        }
        result = scan_audit_progress(
            audit_dir, ssh_runner=_ssh_runner_from_map(responses)
        )
        by_name = {b["name"]: b for b in result["boxes"]}
        assert by_name["bts-audit-vultr-1"]["state"] == "done"
        assert by_name["bts-audit-vultr-2"]["state"] == "running"
        assert result["overall"]["boxes_done"] == 1

    def test_ssh_error_isolated(self, audit_dir: Path) -> None:
        responses = {
            "10.0.0.1": _mk_run("", rc=255, stderr="ssh: connect: Connection refused"),
            "10.0.0.2": _mk_run(_running_stdout(1)),
            "10.0.0.3": _mk_run(_running_stdout(0)),
        }
        result = scan_audit_progress(
            audit_dir, ssh_runner=_ssh_runner_from_map(responses)
        )
        by_name = {b["name"]: b for b in result["boxes"]}
        assert by_name["bts-audit-vultr-1"]["state"] == "error"
        assert "Connection refused" in by_name["bts-audit-vultr-1"]["error"]
        assert by_name["bts-audit-vultr-2"]["state"] == "running"
        # Aggregate sums only boxes that reported a count
        assert result["overall"]["completed"] == 1

    def test_ssh_runner_raising_exception_is_caught(self, audit_dir: Path) -> None:
        def ssh_runner(ip: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
            if ip == "10.0.0.1":
                raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=timeout)
            return _mk_run(_running_stdout(1))

        result = scan_audit_progress(audit_dir, ssh_runner=ssh_runner)
        by_name = {b["name"]: b for b in result["boxes"]}
        assert by_name["bts-audit-vultr-1"]["state"] == "error"
        assert "timeout" in by_name["bts-audit-vultr-1"]["error"].lower()

    def test_with_seeds_file_round_robin(self, audit_dir: Path, tmp_path: Path) -> None:
        # 5 seeds across 3 boxes round-robin → box1=[s0,s3], box2=[s1,s4], box3=[s2]
        seeds_file = tmp_path / "seeds.txt"
        seeds_file.write_text("100,200,300,400,500")
        responses = {
            "10.0.0.1": _mk_run(_running_stdout(1)),
            "10.0.0.2": _mk_run(_running_stdout(2)),
            "10.0.0.3": _mk_run(_running_stdout(0)),
        }
        result = scan_audit_progress(
            audit_dir, seeds_file=seeds_file, ssh_runner=_ssh_runner_from_map(responses)
        )
        expected = {b["name"]: b["expected_seeds"] for b in result["boxes"]}
        assert expected == {
            "bts-audit-vultr-1": 2,
            "bts-audit-vultr-2": 2,
            "bts-audit-vultr-3": 1,
        }
        assert result["overall"]["expected"] == 5
        assert result["overall"]["completed"] == 3
        assert result["overall"]["pct_seeds"] == pytest.approx(60.0)

    def test_seeds_file_accepts_newline_separated(self, audit_dir: Path, tmp_path: Path) -> None:
        seeds_file = tmp_path / "seeds.txt"
        seeds_file.write_text("100\n200\n300\n")  # newline-separated also valid
        responses = {ip: _mk_run(_running_stdout(0)) for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.3"]}
        result = scan_audit_progress(
            audit_dir, seeds_file=seeds_file, ssh_runner=_ssh_runner_from_map(responses)
        )
        assert result["overall"]["expected"] == 3

    def test_without_seeds_file(self, audit_dir: Path) -> None:
        responses = {
            "10.0.0.1": _mk_run(_running_stdout(1)),
            "10.0.0.2": _mk_run(_running_stdout(1)),
            "10.0.0.3": _mk_run(_running_stdout(0)),
        }
        result = scan_audit_progress(
            audit_dir, ssh_runner=_ssh_runner_from_map(responses)
        )
        for b in result["boxes"]:
            assert b["expected_seeds"] is None
        assert result["overall"]["expected"] is None
        assert result["overall"]["pct_seeds"] is None

    def test_last_seed_event_captured(self, audit_dir: Path) -> None:
        responses = {
            "10.0.0.1": _mk_run(_running_stdout(1, last="=== seed=42 done at X ===")),
            "10.0.0.2": _mk_run(_running_stdout(0, last="")),
            "10.0.0.3": _mk_run(_running_stdout(0, last="")),
        }
        result = scan_audit_progress(
            audit_dir, ssh_runner=_ssh_runner_from_map(responses)
        )
        by_name = {b["name"]: b for b in result["boxes"]}
        assert by_name["bts-audit-vultr-1"]["last_seed_event"] == "=== seed=42 done at X ==="
        assert by_name["bts-audit-vultr-2"]["last_seed_event"] == ""

    def test_done_box_completed_equals_expected_when_known(
        self, audit_dir: Path, tmp_path: Path
    ) -> None:
        seeds_file = tmp_path / "seeds.txt"
        seeds_file.write_text("100,200,300")  # 1 seed per box (3 boxes, 3 seeds)
        responses = {
            "10.0.0.1": _mk_run(_done_stdout(1)),
            "10.0.0.2": _mk_run(_done_stdout(1)),
            "10.0.0.3": _mk_run(_done_stdout(1)),
        }
        result = scan_audit_progress(
            audit_dir, seeds_file=seeds_file, ssh_runner=_ssh_runner_from_map(responses)
        )
        assert result["overall"]["completed"] == 3
        assert result["overall"]["boxes_done"] == 3
        assert result["overall"]["pct_seeds"] == pytest.approx(100.0)

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


class TestPollResilience:
    """A single box's SSH timeout must not kill the whole poll cycle.

    Regression for 2026-04-25 09:36 ET incident: audit_attach crashed when
    one box (80.240.17.54) hit a transient SSH timeout, abandoning the
    other 25 still-running boxes mid-audit.
    """

    def test_one_timeout_doesnt_kill_poll(self, boxes, captured_log, monkeypatch):
        import subprocess as _sub
        import audit_driver

        def fake_ssh_run(ip, cmd, timeout=60):
            if ip == "10.0.0.2":
                raise _sub.TimeoutExpired(cmd=["ssh"], timeout=timeout)
            # Other boxes return a "still running" response
            return _sub.CompletedProcess(
                args=[], returncode=0, stdout="3\n=== seed=42 done at X ===", stderr=""
            )

        monkeypatch.setattr(audit_driver, "ssh_run", fake_ssh_run)

        done_count, lines = audit_driver.poll(boxes)

        assert done_count == 0  # nobody's done
        assert len(lines) == 3  # all three boxes reported, none lost
        b2_line = next(l for l in lines if l[0] == "b2")
        assert b2_line[1] is False  # not done
        assert "ssh-timeout" in b2_line[2]
        # Other boxes' results still captured
        b1_line = next(l for l in lines if l[0] == "b1")
        assert "seed=42 done" in b1_line[2]

    def test_one_generic_exception_doesnt_kill_poll(self, boxes, captured_log, monkeypatch):
        import subprocess as _sub
        import audit_driver

        def fake_ssh_run(ip, cmd, timeout=60):
            if ip == "10.0.0.2":
                raise OSError("Connection refused")
            return _sub.CompletedProcess(
                args=[], returncode=0, stdout="0", stderr=""
            )

        monkeypatch.setattr(audit_driver, "ssh_run", fake_ssh_run)

        done_count, lines = audit_driver.poll(boxes)

        assert done_count == 0
        assert len(lines) == 3
        b2_line = next(l for l in lines if l[0] == "b2")
        assert "ssh-error" in b2_line[2]
        assert "OSError" in b2_line[2]

    def test_done_marker_still_recognized(self, boxes, captured_log, monkeypatch):
        """A DONE response must still increment done_count post-fix."""
        import subprocess as _sub
        import audit_driver

        def fake_ssh_run(ip, cmd, timeout=60):
            if ip == "10.0.0.1":
                return _sub.CompletedProcess(
                    args=[], returncode=0, stdout="DONE\nqueue done at Sat", stderr=""
                )
            return _sub.CompletedProcess(args=[], returncode=0, stdout="2", stderr="")

        monkeypatch.setattr(audit_driver, "ssh_run", fake_ssh_run)

        done_count, lines = audit_driver.poll(boxes)

        assert done_count == 1
        b1_line = next(l for l in lines if l[0] == "b1")
        assert b1_line[1] is True


# ---------------------------------------------------------------------------
# _stage1_seed_dirs_via_symlinks tests
# ---------------------------------------------------------------------------

def _seed_layout(stage1_out: Path, box: str, seed: int, exp: str, payload: dict) -> None:
    """Helper: write a synthetic <stage1_out>/<box>/phase1_seed{seed}/<exp>/diff.json."""
    import json
    seed_dir = stage1_out / box / f"phase1_seed{seed}" / exp
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "diff.json").write_text(json.dumps(payload))


class TestStage1SeedDirsViaSymlinks:
    """Cover _stage1_seed_dirs_via_symlinks bridging logic.

    Helper takes the audit-output layout
        <stage1_out>/<box>/phase1_seed{N}/<exp>/diff.json
    and produces virtual seed dirs
        <stage1_out>/_seeds_view/<box>__seed{N}/phase1 -> phase1_seed{N}
    that aggregate_stage_one_results can consume directly.
    """

    def test_stage1_seed_dirs_clean_creation(self, tmp_path):
        from audit_driver import _stage1_seed_dirs_via_symlinks

        # Box1 has seeds 42 and 99; box2 has seed 42 only.
        _seed_layout(tmp_path, "box1", 42, "exp_a", {"precision": {"1": {"delta": 0.005}}})
        _seed_layout(tmp_path, "box1", 99, "exp_a", {"precision": {"1": {"delta": 0.002}}})
        _seed_layout(tmp_path, "box2", 42, "exp_a", {"precision": {"1": {"delta": -0.001}}})

        seed_dirs = _stage1_seed_dirs_via_symlinks(tmp_path)

        assert len(seed_dirs) == 3
        # Each virt dir has a working phase1 symlink resolving to the right
        # phase1_seed{N} target directory.
        for virt in seed_dirs:
            phase1 = virt / "phase1"
            assert phase1.is_symlink()
            target = phase1.resolve()
            assert target.exists()
            assert target.name.startswith("phase1_seed")
            # diff.json under the link must be reachable.
            assert (phase1 / "exp_a" / "diff.json").is_file()

    def test_stage1_seed_dirs_idempotent(self, tmp_path):
        from audit_driver import _stage1_seed_dirs_via_symlinks

        _seed_layout(tmp_path, "box1", 42, "exp_a", {"precision": {"1": {"delta": 0.005}}})
        _seed_layout(tmp_path, "box1", 99, "exp_a", {"precision": {"1": {"delta": 0.002}}})
        _seed_layout(tmp_path, "box2", 42, "exp_a", {"precision": {"1": {"delta": -0.001}}})

        first = _stage1_seed_dirs_via_symlinks(tmp_path)
        first_targets = {p: (p / "phase1").resolve() for p in first}

        second = _stage1_seed_dirs_via_symlinks(tmp_path)
        second_targets = {p: (p / "phase1").resolve() for p in second}

        # Identical paths AND identical resolved symlink targets.
        assert sorted(first) == sorted(second)
        assert first_targets == second_targets

    def test_stage1_seed_dirs_integration_with_aggregate(self, tmp_path):
        from audit_driver import _stage1_seed_dirs_via_symlinks
        from bts.experiment.two_stage import aggregate_stage_one_results

        # 4 seeds × 2 experiments. good_exp positive, bad_exp negative.
        for box, seed in [("box1", 42), ("box1", 99), ("box2", 42), ("box2", 99)]:
            _seed_layout(tmp_path, box, seed, "good_exp",
                         {"precision": {"1": {"delta": 0.005}}})
            _seed_layout(tmp_path, box, seed, "bad_exp",
                         {"precision": {"1": {"delta": -0.003}}})

        seed_dirs = _stage1_seed_dirs_via_symlinks(tmp_path)
        assert len(seed_dirs) == 4

        results = aggregate_stage_one_results(seed_dirs, ["good_exp", "bad_exp"])

        assert "good_exp" in results
        assert "bad_exp" in results
        assert results["good_exp"].wins == 4
        assert results["bad_exp"].wins == 0
        assert results["good_exp"].seeds_run == 4
        assert results["bad_exp"].seeds_run == 4

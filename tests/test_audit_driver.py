"""Tests for scripts/audit_driver.py secret lookup."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

    def test_fallback_to_env_alias_when_primary_env_misses(self, monkeypatch):
        from audit_driver import _keychain

        service = "unit-test-fake-service-for-keychain-alias"
        primary_env = "BTS_SECRET_UNIT_TEST_FAKE_SERVICE_FOR_KEYCHAIN_ALIAS"
        alias_env = "UNIT_TEST_ALIAS_SECRET"
        monkeypatch.delenv(primary_env, raising=False)
        monkeypatch.setenv(alias_env, "alias-sentinel-67890")

        assert _keychain(service, env_aliases=(alias_env,)) == "alias-sentinel-67890"

    def test_error_mentions_env_aliases(self, monkeypatch):
        from audit_driver import _keychain

        service = "unit-test-missing-service-with-alias"
        primary_env = "BTS_SECRET_UNIT_TEST_MISSING_SERVICE_WITH_ALIAS"
        alias_env = "UNIT_TEST_MISSING_ALIAS"
        monkeypatch.delenv(primary_env, raising=False)
        monkeypatch.delenv(alias_env, raising=False)

        with pytest.raises(RuntimeError) as exc:
            _keychain(service, env_aliases=(alias_env,))

        assert primary_env in str(exc.value)
        assert alias_env in str(exc.value)


class TestOCIReadinessHelpers:
    def _provider(self, ads: list[str], cursor: int = 0):
        from audit_driver import OCIProvider

        provider = object.__new__(OCIProvider)
        provider._ad_fallbacks = ads
        provider._next_ad_idx = cursor
        return provider

    def test_ad_order_rotates_from_next_cursor(self):
        provider = self._provider(["AD-1", "AD-2", "AD-3"], cursor=1)

        assert provider._ordered_ads_for_create() == ["AD-2", "AD-3", "AD-1"]

    def test_mark_ad_used_advances_next_cursor(self):
        provider = self._provider(["AD-1", "AD-2", "AD-3"])

        provider._mark_ad_used("AD-2")

        assert provider._next_ad_idx == 2
        assert provider._ordered_ads_for_create() == ["AD-3", "AD-1", "AD-2"]

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (SimpleNamespace(code="LimitExceeded", status=400), True),
            (SimpleNamespace(code="OutOfCapacity", status=400), True),
            (SimpleNamespace(code=None, status=503), True),
            (SimpleNamespace(code="NotAuthorizedOrNotFound", status=404), False),
        ],
    )
    def test_service_error_classification_for_next_ad(self, error, expected):
        from audit_driver import OCIProvider

        assert OCIProvider._should_try_next_ad(error) is expected

    def test_create_tries_next_ad_after_limit_exceeded(
        self, monkeypatch, tmp_path,
    ):
        import audit_driver

        class FakeServiceError(Exception):
            def __init__(self, status, code):
                super().__init__(f"{status} {code}")
                self.status = status
                self.code = code

        class KwargsModel:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        attempts: list[str] = []

        class FakeComposite:
            def launch_instance_and_wait_for_work_request(
                self, launch, operation_kwargs,
            ):
                attempts.append(launch.availability_domain)
                if launch.availability_domain == "AD-1":
                    raise FakeServiceError(status=400, code="LimitExceeded")
                return SimpleNamespace(
                    data=SimpleNamespace(
                        id="wr-ok",
                        resources=[
                            SimpleNamespace(
                                entity_type="instance",
                                identifier="ocid1.instance.oc1",
                            )
                        ],
                    )
                )

        pubkey = tmp_path / "id_ed25519.pub"
        pubkey.write_text("ssh-ed25519 unit-test\n")
        monkeypatch.setattr(
            audit_driver.os.path,
            "expanduser",
            lambda path: str(pubkey) if path.startswith("~/.ssh/") else path,
        )

        fake_models = SimpleNamespace(
            LaunchInstanceDetails=KwargsModel,
            LaunchInstanceShapeConfigDetails=KwargsModel,
            InstanceSourceViaImageDetails=KwargsModel,
            CreateVnicDetails=KwargsModel,
        )
        provider = self._provider(["AD-1", "AD-2"])
        provider._resolve_once = lambda: None
        provider._oci = SimpleNamespace(
            core=SimpleNamespace(models=fake_models),
            exceptions=SimpleNamespace(ServiceError=FakeServiceError),
        )
        provider.compartment_id = "compartment"
        provider.subnet_id = "subnet"
        provider._image_id = "image"
        provider.retry_strategy = object()
        provider.composite = FakeComposite()

        box = provider.create("bts-audit-oci")

        assert attempts == ["AD-1", "AD-2"]
        assert box.id == "ocid1.instance.oc1"
        assert box.region == "AD-2"
        assert provider._next_ad_idx == 2


# ---------------------------------------------------------------------------
# split-aware remote command rendering tests
# ---------------------------------------------------------------------------


class TestValidationSplit:
    def test_default_legacy_split_renders_test_seasons(self):
        from audit_driver import render_screen_command, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons=None,
            outer_eval_seasons=None,
        )

        assert split.metadata()["split_mode"] == "legacy_test_seasons"
        assert render_screen_command("exp_a,exp_b", split) == (
            'uv run bts experiment screen --subset "exp_a,exp_b" \\\n'
            "    --test-seasons 2024,2025"
        )

    def test_split_flags_render_without_legacy_test_seasons(self):
        from audit_driver import render_screen_command, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2025",
        )
        command = render_screen_command("exp_a", split)

        assert "--selection-seasons 2023,2024" in command
        assert "--outer-eval-seasons 2025" in command
        assert "--test-seasons" not in command
        assert split.metadata()["artifact_role"] == "selection_only"
        assert split.metadata()["production_deploy_claim"] is False

    def test_profile_seasons_default_to_full_surface_for_legacy_split(self):
        from audit_driver import resolve_profile_seasons, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons=None,
            outer_eval_seasons=None,
        )

        assert resolve_profile_seasons(profile_seasons=None, split=split) == [
            2021,
            2022,
            2023,
            2024,
            2025,
        ]

    def test_profile_seasons_default_to_full_surface_plus_split_seasons(self):
        from audit_driver import resolve_profile_seasons, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2026",
        )

        assert resolve_profile_seasons(profile_seasons=None, split=split) == [
            2021,
            2022,
            2023,
            2024,
            2025,
            2026,
        ]

    def test_profile_seasons_reject_missing_split_season(self):
        from audit_driver import resolve_profile_seasons, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2025",
        )

        with pytest.raises(ValueError, match="missing: \\[2025\\]"):
            resolve_profile_seasons(profile_seasons="2023,2024", split=split)

    def test_rejects_mixed_legacy_and_split_flags(self):
        from audit_driver import resolve_validation_split

        with pytest.raises(ValueError, match="cannot be combined"):
            resolve_validation_split(
                test_seasons="2024",
                selection_seasons="2023",
                outer_eval_seasons="2025",
            )

    def test_rejects_half_specified_split_flags(self):
        from audit_driver import resolve_validation_split

        with pytest.raises(ValueError, match="must be supplied together"):
            resolve_validation_split(
                test_seasons=None,
                selection_seasons="2023",
                outer_eval_seasons=None,
            )
        with pytest.raises(ValueError, match="must be supplied together"):
            resolve_validation_split(
                test_seasons=None,
                selection_seasons=None,
                outer_eval_seasons="2025",
            )

    def test_rejects_overlapping_split_flags(self):
        from audit_driver import resolve_validation_split

        with pytest.raises(ValueError, match="disjoint"):
            resolve_validation_split(
                test_seasons=None,
                selection_seasons="2024",
                outer_eval_seasons="2024,2025",
            )

    def test_writes_local_split_metadata(self, tmp_path):
        from audit_driver import _write_validation_split_metadata, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2025",
        )

        _write_validation_split_metadata(
            tmp_path,
            split,
            audit_driver={
                "provider": "hetzner",
                "requested_boxes": 5,
                "actual_boxes_obtained": 4,
                "label": "bts-audit-hetzner",
                "two_stage": False,
            },
        )

        payload = json.loads((tmp_path / "audit_validation_split.json").read_text())
        assert payload["split_mode"] == "season_level_selection_outer_eval"
        assert payload["selection_seasons"] == [2023, 2024]
        assert payload["outer_eval_seasons"] == [2025]
        assert payload["production_deploy_claim"] is False
        assert payload["audit_driver"]["provider"] == "hetzner"
        assert payload["audit_driver"]["requested_boxes"] == 5
        assert payload["audit_driver"]["actual_boxes_obtained"] == 4

    def test_seed_metadata_records_provider_box_and_determinism(self):
        from audit_driver import Box, _seed_audit_metadata, resolve_validation_split

        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2025",
        )

        payload = _seed_audit_metadata(
            split,
            provider_name="oci",
            box=Box(
                id="ocid1.instance.oc1.unit",
                name="bts-audit-oci-1",
                ipv4="10.0.0.1",
                region="AD-2",
            ),
        )

        assert payload["audit_driver"]["provider"] == "oci"
        assert payload["audit_driver"]["box_id"] == "ocid1.instance.oc1.unit"
        assert payload["audit_driver"]["box_name"] == "bts-audit-oci-1"
        assert payload["audit_driver"]["box_region"] == "AD-2"
        assert payload["audit_driver"]["box_region_semantics"] == "oci_availability_domain"
        assert payload["audit_driver"]["determinism_intent"] is True
        assert payload["audit_driver"]["launch_command_env"] == {
            "BTS_LGBM_DETERMINISTIC": "1",
            "BTS_LGBM_RANDOM_STATE": "per_seed_loop",
        }
        assert payload["audit_driver"]["cross_provider_pooling_validated"] is False

        hetzner_payload = _seed_audit_metadata(
            split,
            provider_name="hetzner",
            box=Box(id="123", name="bts-audit-hetzner-1", region="fsn1"),
        )
        assert hetzner_payload["audit_driver"]["box_region_semantics"] == "provider_region"

    def test_audit_driver_provenance_base_records_determinism_intent(self):
        from audit_driver import _audit_driver_provenance_base

        payload = _audit_driver_provenance_base()

        assert payload["determinism_intent"] is True
        assert payload["launch_command_env"] == {
            "BTS_LGBM_DETERMINISTIC": "1",
            "BTS_LGBM_RANDOM_STATE": "per_seed_loop",
        }
        assert payload["cross_provider_pooling_validated"] is False

    def test_launch_queue_command_uses_split_flags_and_seed_metadata(
        self, monkeypatch,
    ):
        import audit_driver
        from audit_driver import Box, launch_box_queue, resolve_validation_split

        captured: dict[str, str] = {}

        def fake_ssh_run(ip, cmd, timeout=60):
            captured["ip"] = ip
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="launched seeds=42",
                stderr="",
            )

        monkeypatch.setattr(audit_driver, "ssh_run", fake_ssh_run)
        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2025",
        )

        name, rc, out = launch_box_queue(
            Box(id="box-id", name="box1", ipv4="10.0.0.1"),
            [42],
            "exp_a",
            split,
            "oci",
        )

        assert name == "box1"
        assert rc == 0
        assert out == "launched seeds=42"
        assert captured["ip"] == "10.0.0.1"
        command = captured["cmd"]
        assert "--selection-seasons 2023,2024" in command
        assert "--outer-eval-seasons 2025" in command
        assert "--test-seasons" not in command
        assert "phase1_seed$SEED/audit_validation_split.json" in command
        assert "<<'JSON'" in command
        assert '"split_mode": "season_level_selection_outer_eval"' in command
        assert '"provider": "oci"' in command
        assert '"box_name": "box1"' in command
        assert '"determinism_intent": true' in command
        assert '"launch_command_env":' in command
        assert '"run_kind": "screen"' in command
        assert '"queue_mode": "screen"' in command
        assert '"BTS_LGBM_RANDOM_STATE": "per_seed_loop"' in command
        assert '"cross_provider_pooling_validated": false' in command

    def test_render_profile_command_logs_pa_predictions(self):
        from audit_driver import render_profile_command

        command = render_profile_command(
            [2023, 2024, 2025],
            log_pa_predictions=True,
        )

        assert command == (
            "uv run bts simulate backtest \\\n"
            "    --seasons 2023,2024,2025 \\\n"
            "    --output-dir data/simulation \\\n"
            "    --log-pa-predictions"
        )

    def test_launch_profile_queue_command_uses_profile_artifacts_and_metadata(
        self, monkeypatch,
    ):
        import audit_driver
        from audit_driver import Box, launch_profile_queue, resolve_validation_split

        captured: dict[str, str] = {}

        def fake_ssh_run(ip, cmd, timeout=60):
            captured["ip"] = ip
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="launched profile seeds=42",
                stderr="",
            )

        monkeypatch.setattr(audit_driver, "ssh_run", fake_ssh_run)
        split = resolve_validation_split(
            test_seasons=None,
            selection_seasons="2023,2024",
            outer_eval_seasons="2025",
        )

        name, rc, out = launch_profile_queue(
            Box(id="box-id", name="box1", ipv4="10.0.0.1", region="AD-1"),
            [42],
            split,
            "oci",
            [2023, 2024, 2025],
            log_pa_predictions=True,
        )

        assert name == "box1"
        assert rc == 0
        assert out == "launched profile seeds=42"
        assert captured["ip"] == "10.0.0.1"
        command = captured["cmd"]
        assert "uv run bts simulate backtest" in command
        assert "--seasons 2023,2024,2025" in command
        assert "--output-dir data/simulation" in command
        assert "--log-pa-predictions" in command
        assert "BTS_LGBM_RANDOM_STATE=$SEED BTS_LGBM_DETERMINISTIC=1" in command
        assert "data/simulation_seed$SEED/audit_validation_split.json" in command
        assert "bts experiment screen" not in command
        assert '"artifact_role": "raw_backtest_profile_surface"' in command
        assert '"profile_seasons": [2023, 2024, 2025]' in command
        assert '"provider": "oci"' in command
        assert '"box_region": "AD-1"' in command
        assert '"run_kind": "profiles"' in command
        assert '"queue_mode": "backtest"' in command
        assert '"cross_provider_pooling_validated": false' in command

    def test_retrieve_profile_one_fetches_simulation_seed_dirs(
        self, monkeypatch, tmp_path,
    ):
        import audit_driver
        from audit_driver import Box, retrieve_profile_one

        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(audit_driver.subprocess, "run", fake_run)

        name, status, errs = retrieve_profile_one(
            Box(id="box-id", name="box1", ipv4="10.0.0.1"),
            tmp_path,
            [42],
        )

        assert name == "box1"
        assert status == "ok"
        assert errs == []
        assert any("/root/audit.log" in part for call in calls for part in call)
        assert any(
            "/root/projects/bts/data/simulation_seed42/" in part
            for call in calls
            for part in call
        )
        assert (tmp_path / "box1" / "simulation_seed42").is_dir()


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

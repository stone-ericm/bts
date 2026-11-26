"""Run the eleven requested test files in one pytest process."""

from importlib.util import find_spec
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
TESTS = [
    "test_wr_w3.py",
    "test_incident_2026_08_30.py",
    "test_saved_pick_paths_cs5a.py",
    "test_fallback_plan.py",
    "test_scheduler.py",
    "test_scheduler_decision_record_integration.py",
    "test_e2_delivery_idempotency.py",
    "test_late_delivery_guard.py",
    "test_alert_status_cs6b2_carbon.py",
    "test_lightgbm_optional.py",
    "test_web_saver.py",
]


if __name__ == "__main__":
    assert Path(find_spec("bts").origin).resolve().is_relative_to(HERE / "patched" / "src")
    assert find_spec("sitecustomize").origin.endswith("offline_python/sitecustomize.py")
    raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short",
                                  *(f"tests/{name}" for name in TESTS)]))

"""The 2a mutant runner's classification: only a clean failure is RED (framing review r2 R2-4, applied here too)."""
import importlib.util
from pathlib import Path

import pytest


def _runner():
    path = Path(__file__).resolve().parents[2] / "docs/audit/2026-10-06-c2-2a-evidence/mutant_runner.py"
    spec = importlib.util.spec_from_file_location("c2_2a_mutant_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rc, out, verdict", [
    (1, "FAILED t.py::a - x\n1 failed, 3 passed in 0.1s", "RED"),
    (1, "FAILED t.py::a - x\nERROR t.py::b - setup\n1 failed, 1 error in 0.01s", "INCONCLUSIVE(exit 1, errors)"),
    (1, "FAILED t.py::a - x\n1 failed, 1 skipped in 0.1s", "INCONCLUSIVE(exit 1)"),
    (0, "4 passed in 0.1s", "SURVIVED"),
    (2, "1 error in 0.1s", "INCONCLUSIVE(exit 2)"),
    (1, "FAILED t.py::a - x\n1 failed, 71 deselected in 9.7s", "RED"),
])
def test_the_runner_classifies_only_clean_failures_as_red(rc, out, verdict):
    assert _runner().classify(rc, out)[0] == verdict

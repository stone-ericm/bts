"""The strict sweep's classifier reads pytest's JUnit XML, never free-text summary lines (Codex phase-1 r4 #10)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SWEEP = Path(__file__).resolve().parents[3] / "docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py"


def _sweep():
    spec = importlib.util.spec_from_file_location("w15_mutation_sweep", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # guarded main: importing runs nothing
    return mod


def _junit(tmp_path, body: str) -> tuple[str, int]:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_x.py").write_text(body)
    xml = tmp_path / "out.xml"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", "test_x.py"],
                       cwd=tmp_path, capture_output=True, text=True)
    return xml.read_text(), p.returncode


@pytest.mark.parametrize("body, verdict", [
    ("def test_a():\n    assert 1 == 2\n", "KILLED"),
    ("def test_a():\n    raise AssertionError('explicit')\n", "KILLED"),
    ("import pytest\ndef test_a():\n    with pytest.raises(ValueError):\n        pass\n", "KILLED"),
    ("def test_a():\n    raise RuntimeError('DID NOT RAISE')\n", "FAILED-OTHER"),                 # r4 #10
    ("def test_a():\n    class AssertionError(Exception):\n        pass\n    raise AssertionError('looks like one')\n",
     "FAILED-OTHER"),
    ("def test_a():\n    raise KeyError('assert 1')\n", "FAILED-OTHER"),
    ("import pytest\n@pytest.fixture\ndef f():\n    raise KeyError('x')\ndef test_a(f):\n    pass\n"
     "def test_b():\n    assert 1 == 2\n", "ERRORED"),
    ("def test_a():\n    pass\n", "SURVIVED"),
])
def test_each_failure_shape_is_classified(tmp_path, body, verdict):
    xml, rc = _junit(tmp_path, body)
    assert _sweep().classify(xml, rc)[0] == verdict


def test_a_run_short_of_the_baseline_count_is_incomplete(tmp_path):
    xml, rc = _junit(tmp_path, "def test_a():\n    assert 1 == 2\n")
    assert _sweep().classify(xml, rc, expected_cases=5)[0] == "INCOMPLETE"
    assert _sweep().classify(xml, rc, expected_cases=1) == ("KILLED", ["test_x::test_a"])

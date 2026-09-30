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


def test_a_run_whose_inventory_differs_from_the_baseline_is_incomplete(tmp_path):
    """The baseline's exact node ids, not only their count (Codex phase-1 r5 #8)."""
    xml, rc = _junit(tmp_path, "def test_a():\n    assert 1 == 2\n")
    assert _sweep().classify(xml, rc, expected_nodes=["test_x::test_a", "test_x::test_b"])[0] == "INCOMPLETE"
    assert _sweep().classify(xml, rc, expected_nodes=["test_x::test_b"])[0] == "INCOMPLETE"     # same count
    assert _sweep().classify(xml, rc, expected_nodes=["test_x::test_a"]) == ("KILLED", ["test_x::test_a"])


def test_a_skipped_baseline_is_not_clean(tmp_path):
    """r5 #8 measured: a real run with one skipped node exits 0 and was SURVIVED, i.e. a clean baseline."""
    xml, rc = _junit(tmp_path, "import pytest\ndef test_a():\n    pytest.skip('not exercised')\n")
    assert rc == 0 and _sweep().classify(xml, rc)[0] == "SKIPPED"


def test_an_assertion_beside_a_skip_is_not_a_kill(tmp_path):
    """A mutant that makes one node skip while another asserts is not a clean kill: the skipped node's
    verdict is unknown."""
    xml, rc = _junit(tmp_path, "import pytest\ndef test_a():\n    assert 1 == 2\n"
                               "def test_b():\n    pytest.skip('mutant made this skip')\n")
    assert _sweep().classify(xml, rc, expected_nodes=["test_x::test_a", "test_x::test_b"])[0] == "SKIPPED"

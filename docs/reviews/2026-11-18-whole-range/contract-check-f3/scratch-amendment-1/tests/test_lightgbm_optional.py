"""Collection must survive a no-model environment (audit T2).

lightgbm is an optional extra (`uv sync --extra model`). On the Pi5 (no extra)
or a fresh Mac (no libomp), importing it raises ImportError or OSError. If a
test module imports it at module level without guarding, pytest aborts the
ENTIRE collection (`Interrupted: N errors during collection`) and runs zero
tests — broken-red, not the intended skip. Note `pytest.importorskip("lightgbm")`
is insufficient: it catches ImportError but NOT the libomp dlopen OSError.

This test shadows lightgbm with a module that raises on import, then runs a full
`--collect-only` in a subprocess and asserts it succeeds (exit 0).
"""
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_collection_survives_without_lightgbm(tmp_path):
    # A fake top-level `lightgbm` that raises on import, shadowing the real one
    # (PYTHONPATH is searched before site-packages).
    (tmp_path / "lightgbm.py").write_text(
        "raise ImportError('simulated: lightgbm unavailable in this environment')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests"],
        cwd=_REPO, env=env, capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        "pytest collection aborted when lightgbm is unimportable (exit "
        f"{result.returncode}); model-dependent test modules must skip at module "
        "level via `try: import lightgbm / except (ImportError, OSError): "
        "pytest.skip(allow_module_level=True)`.\n\n"
        f"--- stdout tail ---\n{result.stdout[-2500:]}\n"
        f"--- stderr tail ---\n{result.stderr[-1500:]}"
    )

"""Use a clean archive to check base-only controls with today's interpreter."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


if __name__ == "__main__":
    mode = sys.argv[1]
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        paths = ["src"]
        if mode == "audit":
            paths += ["tests/test_audit_driver.py", "scripts/audit_driver.py", "pyproject.toml"]
        archive = subprocess.check_output(["git", "archive", "1e69d0f0", *paths], cwd=ROOT, timeout=4)
        subprocess.run(["tar", "xf", "-", "-C", tmp], input=archive, check=True, timeout=4)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(Path(tmp) / "src"), str(ROOT)))}
        if mode == "trickle":
            args, cwd = [str(HERE / "trickle.py")], ROOT
        else:
            assert mode == "audit"
            args, cwd = ["-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short",
                         "tests/test_audit_driver.py"], tmp
        raise SystemExit(subprocess.run([sys.executable, *args], env=env, cwd=cwd, timeout=12).returncode)

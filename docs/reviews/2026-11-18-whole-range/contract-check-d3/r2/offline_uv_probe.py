"""Check what offline no-sync does with a missing project venv."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    executable = shutil.which("uv")
    assert executable
    with TemporaryDirectory(dir=HERE) as directory:
        project = Path(directory)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "offline-d3-probe"\nversion = "0.0.1"\n'
            'requires-python = ">=3.12"\n'
        )
        env = {
            "PATH": os.defpath,
            "HOME": str(project),
            "UV_CACHE_DIR": str(project / "cache"),
            "UV_PYTHON_INSTALL_DIR": str(project / "interpreters"),
            "UV_PYTHON": str(Path(sys.executable).resolve()),
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
        result = subprocess.run(
            [executable, "--no-config", "run", "--no-sync", "python", "-c",
             "import sys; print(sys.prefix)"],
            env=env, cwd=project, capture_output=True, text=True, timeout=3,
        )
        print(f"offline no-sync rc={result.returncode}; created venv={(project / '.venv').exists()}")
        print(result.stdout.replace(directory, "<scratch>").strip())
        print(result.stderr.replace(directory, "<scratch>").strip())

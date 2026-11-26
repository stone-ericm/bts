"""Complete run of all 9 required scratch changes (plus control and both 9a/9b variants) on isolated scratch copies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from probe_scratch_mutations import MUTATIONS, ORIG_SRC, REPO_ROOT, apply_scratch_change

TARGET_TESTS = [
    "tests/test_contest_dated_cs4a.py",
    "tests/test_wr_f2_saver_flag.py",
    "tests/health/test_runner.py",
    "tests/health/test_contest_state.py",
]


def run_pytest(env: dict[str, str], args: list[str], timeout_s: int = 20) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=line", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
        return proc.returncode != 0, f"rc={proc.returncode}\n{tail}"
    except subprocess.TimeoutExpired:
        return True, f"TIMEOUT after {timeout_s}s (blocked)"


def main() -> int:
    summary = []
    with tempfile.TemporaryDirectory(prefix="f2_scratch2_") as tmp:
        tmp_root = Path(tmp)
        for name in MUTATIONS:
            mut_dir = tmp_root / name
            scratch_src = mut_dir / "src"
            shutil.copytree(ORIG_SRC / "bts", scratch_src / "bts")
            (scratch_src / "sitecustomize.py").write_text(
                f"import sys\n"
                f"from pathlib import Path\n"
                f"orig = Path({str(ORIG_SRC)!r}).resolve()\n"
                f"sys.path = [p for p in sys.path if Path(p or '.').resolve() != orig]\n"
                f"sys.meta_path = [f for f in sys.meta_path if 'editable' not in type(f).__name__.lower()]\n"
                f"sys.path.insert(0, {str(scratch_src.resolve())!r})\n",
                encoding="utf-8",
            )
            apply_scratch_change(scratch_src, name)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(scratch_src.resolve())
            env["TZ"] = "America/New_York"

            chk = subprocess.run(
                [sys.executable, "-c", "import bts; print(bts.__file__)"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            bts_file = chk.stdout.strip()
            assert str(scratch_src.resolve()) in str(Path(bts_file).resolve())

            noticed, detail = run_pytest(env, ["-x", *TARGET_TESTS], timeout_s=20)
            extra = None
            if name == "2_use_load_saver_state":
                # Also verify test_wr_f2_saver_flag.py alone catches it (blocks/times out on FIFO):
                f2_noticed, f2_detail = run_pytest(
                    env, ["tests/test_wr_f2_saver_flag.py", "-k", "fifo", "-x"], timeout_s=5
                )
                extra = f"f2_fifo_only: noticed={f2_noticed} ({f2_detail})"

            status = "KILLED" if noticed else "UNNOTICED (SURVIVED)"
            print(f"[{name}] -> {status}")
            for line in detail.splitlines():
                print(f"    {line}")
            if extra:
                print(f"    {extra}")

            summary.append({
                "mutation": name,
                "noticed": noticed,
                "detail": detail,
                "extra": extra,
            })

    print("\nSUMMARY_JSON:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

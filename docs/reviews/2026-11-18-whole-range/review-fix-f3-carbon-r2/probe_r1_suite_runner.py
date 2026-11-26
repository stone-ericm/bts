"""Run round 1's review-fix-f3-codex probes against archived base (5e0167b5) and F3 (490c5381)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
R1_DIR = REPO / "docs/reviews/2026-11-18-whole-range/review-fix-f3-codex"
OUT_DIR = Path(__file__).resolve().parent
BASE_SRC = Path("/tmp/wr-f3-r2-scratch").resolve() / "base" / "src"
F3_SRC = REPO / "src"


def run_r1_script_in_child(script_name: str, extra_args: list[str]) -> None:
    sys.path.insert(0, str(R1_DIR))
    import review_support  # type: ignore

    review_support.REPO = REPO
    sys.argv = [str(R1_DIR / script_name), *extra_args]
    runpy.run_path(str(R1_DIR / script_name), run_name="__main__")


def invoke_probe(script_name: str, *, base: bool, pass_base_flag: bool = False) -> str:
    src = BASE_SRC if base else F3_SRC
    env = os.environ | {
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{src}:{REPO}:{R1_DIR}",
    }
    extra = ["--base"] if (base and pass_base_flag) else []
    res = subprocess.run(
        [sys.executable, __file__, "--child", script_name, *extra],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=45, check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"{script_name} ({'base' if base else 'f3'}) failed ({res.returncode}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    first = res.stdout.splitlines()[0]
    assert first.startswith("bts.__file__="), first
    loaded = Path(first.split("=", 1)[1]).resolve()
    assert loaded.is_relative_to(src.resolve()), (loaded, src)
    return res.stdout


def run_parity() -> None:
    base_out = invoke_probe("probe_parity.py", base=True)
    f3_out = invoke_probe("probe_parity.py", base=False)
    (OUT_DIR / "r1-parity-base.log").write_text(base_out)
    (OUT_DIR / "r1-parity-f3.log").write_text(f3_out)
    base_lines = base_out.splitlines()
    f3_lines = f3_out.splitlines()
    assert len(base_lines) == len(f3_lines) == 19, (len(base_lines), len(f3_lines))
    assert base_lines[1:] == f3_lines[1:], "non-deliberate_skip parity differed!"
    print(f"base: {base_lines[0]}")
    print(f"f3:   {f3_lines[0]}")
    print(f"probe_parity: 18/18 non-deliberate_skip cases matched byte for byte, 0 evidence reads.")


def run_scheduler() -> None:
    base_out = invoke_probe("probe_scheduler.py", base=True, pass_base_flag=True)
    f3_out = invoke_probe("probe_scheduler.py", base=False, pass_base_flag=False)
    (OUT_DIR / "r1-scheduler-base.log").write_text(base_out)
    (OUT_DIR / "r1-scheduler-f3.log").write_text(f3_out)
    assert len(base_out.splitlines()) == 27 and len(f3_out.splitlines()) == 27
    print(f"base: {base_out.splitlines()[0]}")
    print(f"f3:   {f3_out.splitlines()[0]}")
    print("probe_scheduler: 26/26 paired scheduler + morning CLI cases passed.")


def run_archives_and_carbon() -> None:
    arch_out = invoke_probe("probe_archives.py", base=False)
    (OUT_DIR / "r1-archives-f3.log").write_text(arch_out)
    assert "Only 65537 bytes read of an 8388608-byte file." in arch_out

    cb_base = invoke_probe("probe_carbon_cases.py", base=True, pass_base_flag=True)
    cb_f3 = invoke_probe("probe_carbon_cases.py", base=False, pass_base_flag=False)
    (OUT_DIR / "r1-carbon-base.log").write_text(cb_base)
    (OUT_DIR / "r1-carbon-f3.log").write_text(cb_f3)
    print("probe_archives + probe_carbon_cases: all assertions passed.")


def run_inputs() -> None:
    in_base = invoke_probe("probe_inputs.py", base=True, pass_base_flag=True)
    in_f3 = invoke_probe("probe_inputs.py", base=False, pass_base_flag=False)
    (OUT_DIR / "r1-inputs-base.log").write_text(in_base)
    (OUT_DIR / "r1-inputs-f3.log").write_text(in_f3)

    edges = []
    for line in in_f3.splitlines():
        if line.startswith('{"edge":'):
            edges.append(json.loads(line))
    assert len(edges) == 7, len(edges)
    for row in edges[:6]:
        assert row["exit"] == 0 and json.loads(row["stdout"])["state"] == "deliberate_skip", row
    assert edges[6]["edge"] == "overflowing timestamp outranks a valid differing archive"
    assert edges[6]["exit"] == 1 and json.loads(edges[6]["stdout"])["state"] == "skip_state_review", edges[6]
    print("probe_inputs: all bad-file, standing, and E1 timestamp edge assertions passed.")


def run_scripts() -> None:
    chunks = []
    for sub in ("cron", "leaderboard", "pvr"):
        res = subprocess.run(
            [sys.executable, str(R1_DIR / "probe_script_matches.py"), sub],
            cwd=REPO, env=os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=20, check=True,
        )
        chunks.append(f"=== {sub} ===\n{res.stdout}")
    (OUT_DIR / "r1-scripts.log").write_text("\n".join(chunks))
    print("probe_script_matches: cron, leaderboard, and pvr offline checks passed.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        run_r1_script_in_child(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 2:
        dispatch = {
            "parity": run_parity,
            "scheduler": run_scheduler,
            "archives_and_carbon": run_archives_and_carbon,
            "inputs": run_inputs,
            "scripts": run_scripts,
        }
        dispatch[sys.argv[1]]()
    else:
        raise SystemExit(f"usage: {sys.argv[0]} <parity|scheduler|archives_and_carbon|inputs|scripts>")

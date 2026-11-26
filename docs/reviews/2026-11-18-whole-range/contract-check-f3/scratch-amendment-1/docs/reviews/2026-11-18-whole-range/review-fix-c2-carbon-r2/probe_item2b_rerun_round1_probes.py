"""Item 2 probe (v2, short /tmp tempdirs so path length stays <100 chars as in round 1 & prod):
re-run all Round 1 Codex (6) and Carbon (5) probes against HEAD (e897c766) and base (386aa846)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

ROOT = Path("/Users/eric/projects/bts-work/wr-c2-r2-carbon")
REVIEW_DIR = ROOT / ".codex-review" / "wr-c2-r2"
R1_CODEX = REVIEW_DIR / "round1" / "codex"
R1_CARBON = REVIEW_DIR / "round1" / "carbon"
STAGED = REVIEW_DIR / "r1_staged_v2"
BASE_DIR = REVIEW_DIR / "base_386aa846"
BASE_SRC = BASE_DIR / "src"
HEAD_SRC = ROOT / "src"
PYTHON = ROOT / ".venv" / "bin" / "python"


def stage_probes() -> tuple[Path, Path]:
    if STAGED.exists():
        shutil.rmtree(STAGED)
    codex_dst = STAGED / "codex"
    carbon_dst = STAGED / "carbon"
    codex_dst.mkdir(parents=True)
    carbon_dst.mkdir(parents=True)

    for py in sorted(R1_CODEX.glob("probe_*.py")):
        text = py.read_text(encoding="utf-8")
        if py.name == "probe_support.py":
            assert "ROOT = HERE.parents[1]" in text
            assert 'TemporaryDirectory(prefix=f"{label}-", dir=HERE)' in text
            text = text.replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[3]", 1)
            text = text.replace(
                'TemporaryDirectory(prefix=f"{label}-", dir=HERE)',
                'TemporaryDirectory(prefix=f"{label}-", dir="/tmp")',
                1,
            )
        (codex_dst / py.name).write_text(text, encoding="utf-8")

    carbon_finals = (
        "probe_1c_restart_and_base_parity.py",
        "probe_2c_isolation_and_dedup.py",
        "probe_3b_ack_and_legacy_parity.py",
        "probe_4b_damaged_receipts_and_interrupts.py",
        "probe_5g_operator_text_and_watchdog.py",
    )
    for name in carbon_finals:
        src_file = R1_CARBON / name
        text = src_file.read_text(encoding="utf-8")
        assert "ROOT = HERE.parents[1]" in text, name
        text = text.replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[3]", 1)
        (carbon_dst / name).write_text(text, encoding="utf-8")
    (carbon_dst / "base").symlink_to(BASE_DIR, target_is_directory=True)
    return codex_dst, carbon_dst


def make_env(pythonpath: str) -> dict[str, str]:
    Path("/tmp/mplconfig").mkdir(exist_ok=True)
    return dict(
        os.environ,
        PYTHONPATH=pythonpath,
        PYTHONDONTWRITEBYTECODE="1",
        TZ="America/New_York",
        MPLCONFIGDIR="/tmp/mplconfig",
    )


def run_cmd(label: str, args: list[str], pythonpath: str, timeout: int = 60) -> dict:
    proc = subprocess.run(args, env=make_env(pythonpath), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return {
        "returncode": proc.returncode,
        "stdout_lines": proc.stdout.strip().splitlines()[-5:],
    }


def main() -> None:
    signal.alarm(240)
    try:
        codex_dst, carbon_dst = stage_probes()
        results: dict[str, dict] = {"codex": {}, "carbon": {}}

        restart_cmp = codex_dst / "restart_cmp.json"
        results["codex"]["probe_restart_base"] = run_cmd(
            "codex/probe_restart(base)",
            [str(PYTHON), str(codex_dst / "probe_restart.py"), "base", "--comparison", str(restart_cmp)],
            f"{BASE_SRC}:{codex_dst}",
        )
        results["codex"]["probe_restart_head"] = run_cmd(
            "codex/probe_restart(head)",
            [str(PYTHON), str(codex_dst / "probe_restart.py"), "head", "--comparison", str(restart_cmp)],
            f"{HEAD_SRC}:{codex_dst}",
        )

        results["codex"]["probe_isolation"] = run_cmd(
            "codex/probe_isolation",
            [str(PYTHON), str(codex_dst / "probe_isolation.py")],
            f"{HEAD_SRC}:{codex_dst}",
        )

        legacy_cmp = codex_dst / "legacy_cmp.json"
        results["codex"]["probe_legacy_base"] = run_cmd(
            "codex/probe_legacy_dispatch(base)",
            [str(PYTHON), str(codex_dst / "probe_legacy_dispatch.py"), "base", str(legacy_cmp)],
            f"{BASE_SRC}:{codex_dst}",
        )
        results["codex"]["probe_legacy_head"] = run_cmd(
            "codex/probe_legacy_dispatch(head)",
            [str(PYTHON), str(codex_dst / "probe_legacy_dispatch.py"), "head", str(legacy_cmp)],
            f"{HEAD_SRC}:{codex_dst}",
        )

        rec_proc = subprocess.run(
            [str(PYTHON), str(codex_dst / "probe_receipts.py")],
            env=make_env(f"{HEAD_SRC}:{codex_dst}"),
            capture_output=True, text=True, timeout=60, check=True,
        )
        assert "CONTRACT MISMATCH" not in rec_proc.stdout, rec_proc.stdout
        assert "wrong_schema: sends=1 (contract 1), diagnostics=0." in rec_proc.stdout, rec_proc.stdout
        results["codex"]["probe_receipts"] = {
            "returncode": 0,
            "contract_mismatch_present": False,
            "wrong_schema_line": "wrong_schema: sends=1 (contract 1), diagnostics=0.",
            "stdout_tail": rec_proc.stdout.strip().splitlines()[-5:],
        }

        results["codex"]["probe_failures"] = run_cmd(
            "codex/probe_failures",
            [str(PYTHON), str(codex_dst / "probe_failures.py")],
            f"{HEAD_SRC}:{codex_dst}",
        )
        results["codex"]["probe_operator"] = run_cmd(
            "codex/probe_operator",
            [str(PYTHON), str(codex_dst / "probe_operator.py")],
            f"{HEAD_SRC}:{codex_dst}",
        )

        for name in (
            "probe_1c_restart_and_base_parity.py",
            "probe_2c_isolation_and_dedup.py",
            "probe_3b_ack_and_legacy_parity.py",
            "probe_4b_damaged_receipts_and_interrupts.py",
            "probe_5g_operator_text_and_watchdog.py",
        ):
            results["carbon"][name] = run_cmd(
                f"carbon/{name}",
                [str(PYTHON), str(carbon_dst / name)],
                str(HEAD_SRC),
                timeout=75,
            )

        out_file = REVIEW_DIR / "item2_round1_rerun.json"
        out_file.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print("ITEM 2B PROBE OK: all 6 Codex probes (with CONTRACT MISMATCH gone!) and all 5 Carbon probes passed against e897c766 & 386aa846.")
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

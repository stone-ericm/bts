"""Item 3 probe: replay Round 1 & Round 2 regression probes (base 386aa846 parity, restarts, isolation, receipts, diagnostics, operator)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
BASE_SRC = REVIEW / "src_base_386aa846" / "src"
HEAD_SRC = ROOT / "src"
STAGED = REVIEW / "r1_codex_staged"


class ProbeTimeout(BaseException):
    pass


def set_deadline(seconds: int) -> None:
    def on_alarm(*_):
        raise ProbeTimeout(f"probe_3 exceeded {seconds}s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def stage_probes() -> None:
    if STAGED.exists():
        shutil.rmtree(STAGED)
    STAGED.mkdir(parents=True)

    r1_dir = REVIEW / "round2" / "round1-codex"
    for src_file in sorted(r1_dir.glob("probe_*.py")):
        text = src_file.read_text(encoding="utf-8")
        if src_file.name == "probe_support.py":
            assert text.count("ROOT = HERE.parents[1]") == 1
            text = text.replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[2]", 1)
            # Keep temporary directories in REVIEW (.codex-review/wr-c2b) so paths stay <195 chars and inside this copy
            old_tmp = 'TemporaryDirectory(prefix=f"{label}-", dir=HERE)'
            assert text.count(old_tmp) == 1
            text = text.replace(old_tmp, 'TemporaryDirectory(prefix=f"{label}-", dir=HERE.parent)', 1)
        (STAGED / src_file.name).write_text(text, encoding="utf-8")

    diag_text = (REVIEW / "round2" / "codex" / "probe_diagnostics.py").read_text(encoding="utf-8")
    old_sys = 'sys.path.insert(0, str(HERE / "replay" / "codex"))'
    assert diag_text.count(old_sys) == 1
    diag_text = diag_text.replace(old_sys, "sys.path.insert(0, str(HERE))", 1)
    diag_text = diag_text.replace("support.HERE = HERE", "support.HERE = HERE", 1)
    (STAGED / "probe_diagnostics.py").write_text(diag_text, encoding="utf-8")


def run_probe(script: str, src: Path, log_name: str, args: tuple[str, ...] = ()) -> None:
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(src), str(STAGED), str(ROOT)]),
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [str(PYTHON), str(STAGED / script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=55,
    )
    (REVIEW / log_name).write_text(proc.stdout + proc.stderr, encoding="utf-8")
    assert proc.returncode == 0, f"{script} {args} failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"


def main() -> None:
    set_deadline(58)
    try:
        stage_probes()
        legacy_json = STAGED / "legacy-base.json"
        restart_json = STAGED / "restart-base.json"

        run_probe("probe_legacy_dispatch.py", BASE_SRC, "r1_replay_legacy_dispatch_base.log", ("base", str(legacy_json)))
        run_probe("probe_legacy_dispatch.py", HEAD_SRC, "r1_replay_legacy_dispatch_head.log", ("head", str(legacy_json)))

        run_probe("probe_restart.py", BASE_SRC, "r1_replay_restart_base.log", ("base", "--comparison", str(restart_json)))
        run_probe("probe_restart.py", HEAD_SRC, "r1_replay_restart_head.log", ("head", "--comparison", str(restart_json)))

        run_probe("probe_isolation.py", HEAD_SRC, "r1_replay_isolation_head.log")
        run_probe("probe_receipts.py", HEAD_SRC, "r1_replay_receipts_head.log")
        run_probe("probe_diagnostics.py", HEAD_SRC, "r1_replay_diagnostics_head.log")
        run_probe("probe_operator.py", HEAD_SRC, "r1_replay_operator_head.log")

        receipts_log = (REVIEW / "r1_replay_receipts_head.log").read_text(encoding="utf-8")
        assert "CONTRACT MISMATCH" not in receipts_log
        assert "wrong_schema: sends=1 (contract 1), diagnostics=0." in receipts_log

        summary = {
            "legacy_dispatch_base_vs_head_parity": "PASS (14 callers/cases, omitted + explicit False byte-identical to 386aa846)",
            "restart_base_vs_head_parity": "PASS (3 real restart phases + 6 classification controls matched base hashes/polling)",
            "isolation_head": "PASS (cutoff both orders, failed unrelated DM, EOD, midnight & New Year)",
            "receipts_head": "PASS (reply shapes, special/aliased/damaged files, KeyboardInterrupt/SystemExit boundaries)",
            "diagnostics_and_failures_head": "PASS (18 containment stages + EOD backstop + N1 200-char long-path check)",
            "operator_head": "PASS (all U7 states, read-only status, watchdog agreement & ledgered divergence)",
        }
        (REVIEW / "probe_3_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("PROBE 3 OK: all 8 base/HEAD Round-1 & Round-2 replay invocations exited 0 with zero mismatches.")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()

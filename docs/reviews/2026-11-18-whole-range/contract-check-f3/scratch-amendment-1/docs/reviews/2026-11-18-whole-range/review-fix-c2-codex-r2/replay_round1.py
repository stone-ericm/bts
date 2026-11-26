"""Replay archived probes from their new directory without editing the originals."""

import os
from pathlib import Path
import subprocess
import sys


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
REPLAY = REVIEW / "replay"
BASE = REVIEW / "base"

CARBON = (
    "probe_1c_restart_and_base_parity.py",
    "probe_2c_isolation_and_dedup.py",
    "probe_3b_ack_and_legacy_parity.py",
    "probe_4b_damaged_receipts_and_interrupts.py",
    "probe_5g_operator_text_and_watchdog.py",
)


def prepare():
    BASE.mkdir(exist_ok=True)
    archive = subprocess.run(["git", "archive", "386aa846", "src"], cwd=ROOT,
                             capture_output=True, check=True, timeout=10)
    subprocess.run(["tar", "-x", "-C", str(BASE)], input=archive.stdout,
                   check=True, timeout=10)
    for reviewer in ("codex", "carbon"):
        destination = REPLAY / reviewer
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted((REVIEW / "round1" / reviewer).glob("probe_*.py")):
            adapted = source.read_text().replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[3]")
            adapted = adapted.replace('BASE_SRC = HERE / "base" / "src"',
                                      'BASE_SRC = HERE.parents[1] / "base" / "src"')
            (destination / source.name).write_text(adapted)


def probe(reviewer, script, version="head", args=()):
    path = REPLAY / reviewer / f"{script}.py"
    source = BASE / "src" if version == "base" else ROOT / "src"
    env = os.environ | {"PYTHONPATH": os.pathsep.join(map(str, (source, ROOT))),
                        "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"}
    log = REVIEW / f"replay-{reviewer}-{script}-{version}.log"
    with log.open("w") as output:
        result = subprocess.run([sys.executable, str(path), *args], cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT, timeout=55)
    print(f"{log.name}: exit {result.returncode}", flush=True)
    return result.returncode == 0


def main():
    mode, *names = sys.argv[1:]
    if mode == "prepare":
        prepare()
        return
    if mode in {"restart", "legacy_dispatch"}:
        comparison = REPLAY / f"{mode}-base.json"
        assert probe("codex", f"probe_{mode}", "base", ("base", str(comparison)))
        passed = probe("codex", f"probe_{mode}", args=("head", str(comparison)))
    elif mode == "codex":
        passed = all([probe("codex", f"probe_{name}") for name in names])
    elif mode == "carbon":
        choices = [source.removesuffix(".py") for source in CARBON]
        passed = all([probe("carbon", name) for name in (names or choices)])
    else:
        raise SystemExit(f"Unknown mode {mode!r}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

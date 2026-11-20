"""Run one selected pytest invocation under the shared yield wrapper."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
IDENTITY = "tests/test_wr_c2_receipt_identity.py"
C2 = ["tests/test_wr_c2_recovery.py", "tests/test_wr_c2_carbon_v2.py",
      "tests/test_wr_c2_r2_carbon.py", IDENTITY]
COMPATIBILITY = ["tests/test_lightgbm_optional.py", "tests/test_web_saver.py",
                 "tests/test_alert_status_callers_cs6b2.py", "tests/test_late_delivery_guard.py",
                 "tests/test_stale_delivery_cs4c.py", "tests/test_stale_delivery_cs4c_carbon.py",
                 "tests/test_morning_state_cs6b1.py"]
MUTANTS = {
    "drop-keys": "extra-error",
    "accept-superset": "extra-unknown",
    "allow-duplicates": "duplicate-status or reader_rejects_duplicate_members_inside_nested_objects",
    "drop-hash-compare": "foreign-body",
    "hash-constant": "receipt_hash_matches_exact_transport_body",
    "hash-message": "receipt_hash_matches_exact_transport_body",
    "compare-day-only": "foreign-body",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("combined", *MUTANTS))
    mode = parser.parse_args().mode
    source = ROOT / "src"
    selection = C2 + COMPATIBILITY
    if mode != "combined":
        source = REVIEW / "mutants" / mode / "src"
        selection = [IDENTITY, "-k", MUTANTS[mode]]
    assert (source / "bts" / "health" / "alert.py").is_file()

    path = os.pathsep.join(map(str, (REVIEW / "local_http_only", source, ROOT)))
    env = os.environ | {"PYTHONPATH": path, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLCONFIGDIR": str(REVIEW / "mpl-cache")}
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=short", *selection]
    log = REVIEW / f"pytest-{mode}.log"
    with log.open("w", encoding="utf-8") as output:
        print(f"Source tree {source}\nSelection {selection!r}", file=output, flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=120)
    print(f"{mode} exit {result.returncode} ({log.name})", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

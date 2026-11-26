"""Pytest Family A (run under ~/bts-runs/yield_run.sh):
1. Runs the new/updated CS6b-2 tests against Base's src (59efa825) to establish exact
   RED vs GREEN control counts and classify every test passing against Base.
2. Mutates a scratch copy of Branch's src across all 13 mutations in brief.md item 6
   and verifies every mutation makes the test suite fail cleanly (including FIFO blocks).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "scratch_base" / "src"
MUT_DIR = REVIEW_DIR / "scratch_mut"
MUT_SRC = MUT_DIR / "src"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")


def base_env(prepend_src: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{prepend_src}:{REPO_ROOT}"
    return env


def run_base_red_green() -> dict:
    targets = [
        "tests/health/test_alert_status_cs6b2.py",
        "tests/test_alert_status_callers_cs6b2.py",
        "tests/test_saved_pick_paths_cs5a.py::test_refusal_survives_non_object_health_status",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-v", "--tb=no", *targets],
        cwd=REPO_ROOT,
        env=base_env(BASE_SRC),
        capture_output=True,
        text=True,
        timeout=180,
    )
    elapsed = round(time.monotonic() - t0, 2)
    passed_ids: list[str] = []
    failed_ids: list[str] = []
    for line in proc.stdout.splitlines():
        if " PASSED" in line:
            passed_ids.append(line.split(" PASSED")[0].strip())
        elif " FAILED" in line:
            failed_ids.append(line.split(" FAILED")[0].strip())

    named_green = [t for t in passed_ids if "green" in t]
    unprefixed_green = [t for t in passed_ids if "green" not in t]

    # Categorize unprefixed passing tests on base so we know exact reasons
    unprefixed_reasons: dict[str, int] = {
        "dangling_or_loop_symlink_oserror_already_caught": 0,
        "failed_count_1_0_neg1_already_handled": 0,
        "prior_sources_stale_day_or_non_sent_or_empty_null": 0,
        "persistent_directory_oserror_already_caught": 0,
        "classified_unconfirmed_no_dispatch_control": 0,
        "other": 0,
    }
    other_examples: list[str] = []
    for tid in unprefixed_green:
        if "test_status_shapes[" in tid and ("-dangling]" in tid or "-loop]" in tid):
            unprefixed_reasons["dangling_or_loop_symlink_oserror_already_caught"] += 1
        elif "test_failed_count[" in tid and any(f"-{c}]" in tid for c in ("1", "0", "-1")):
            unprefixed_reasons["failed_count_1_0_neg1_already_handled"] += 1
        elif "test_prior_sources[" in tid:
            unprefixed_reasons["prior_sources_stale_day_or_non_sent_or_empty_null"] += 1
        elif "test_dedup_after_corruption[" in tid and "-True]" in tid:
            unprefixed_reasons["persistent_directory_oserror_already_caught"] += 1
        elif "classified_unconfirmed-None-corrupt" in tid:
            unprefixed_reasons["classified_unconfirmed_no_dispatch_control"] += 1
        else:
            unprefixed_reasons["other"] += 1
            other_examples.append(tid)

    summary_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    print(f"[BASE RED/GREEN] rc={proc.returncode} ({elapsed}s): {summary_line}", flush=True)
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "summary_line": summary_line,
        "passed_count": len(passed_ids),
        "failed_count": len(failed_ids),
        "named_green_count": len(named_green),
        "unprefixed_green_count": len(unprefixed_green),
        "unprefixed_breakdown": unprefixed_reasons,
        "other_unprefixed_examples": other_examples,
    }


MUTATIONS: list[tuple[str, str, str, list[str]]] = [
    (
        "drop_dict_check",
        "    return data if isinstance(data, dict) else None",
        "    return data",
        ["tests/health/test_alert_status_cs6b2.py", "tests/test_alert_status_callers_cs6b2.py"],
    ),
    (
        "drop_reg_check_read",
        "        if not stat.S_ISREG(path.stat().st_mode):\n            return None",
        "        pass",
        ["tests/health/test_alert_status_cs6b2.py", "-k", "fifo"],
    ),
    (
        "drop_reg_check_write",
        (
            "            if not stat.S_ISREG(mode):\n"
            "                log.warning(\n"
            '                    "could not write health DM delivery status at %s: not a regular file",\n'
            "                    status_path,\n"
            "                )\n"
            "                return"
        ),
        "            pass",
        ["tests/health/test_alert_status_cs6b2.py", "-k", "fifo"],
    ),
    (
        "catch_only_os_value",
        "    except (OSError, ValueError, RecursionError):",
        "    except (OSError, ValueError):",
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "accept_bool_count",
        '    if data.get("status") == "failed" and type(n) is int and n > 0:',
        '    if data.get("status") == "failed" and isinstance(n, int) and n > 0:',
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "accept_ge_zero",
        '    if data.get("status") == "failed" and type(n) is int and n > 0:',
        '    if data.get("status") == "failed" and type(n) is int and n >= 0:',
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "truthiness_sent_sources",
        (
            "        if isinstance(sources, list):\n"
            "            prior_sent_sources = [source for source in sources if isinstance(source, str)]"
        ),
        (
            "        if sources:\n"
            "            prior_sent_sources = [source for source in list(sources) if isinstance(source, str)]"
        ),
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "keep_non_string_members",
        "            prior_sent_sources = [source for source in sources if isinstance(source, str)]",
        "            prior_sent_sources = list(sources)",
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "read_twice",
        '    if prior is not None and prior.get("status") == "sent" and prior.get("sent_day") == now_day:',
        (
            "    prior = _read_health_dm_delivery_status(status_path)\n"
            '    if prior is not None and prior.get("status") == "sent" and prior.get("sent_day") == now_day:'
        ),
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "prior_keys_non_sent",
        '    if prior is not None and prior.get("status") == "sent" and prior.get("sent_day") == now_day:',
        '    if prior is not None and prior.get("sent_day") == now_day:',
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "prior_keys_other_day",
        '    if prior is not None and prior.get("status") == "sent" and prior.get("sent_day") == now_day:',
        '    if prior is not None and prior.get("status") == "sent":',
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "drop_union_on_write",
        "            sent_sources=sorted(set(prior_sent_sources) | set(alert_key)),",
        "            sent_sources=sorted(set(alert_key)),",
        ["tests/health/test_alert_status_cs6b2.py"],
    ),
    (
        "swallow_send",
        "        send_dm(dm_recipient, body)",
        "        pass",
        ["tests/health/test_alert_status_cs6b2.py", "tests/test_alert_status_callers_cs6b2.py"],
    ),
]


def run_mutations() -> dict:
    if MUT_DIR.exists():
        shutil.rmtree(MUT_DIR)
    shutil.copytree(REPO_ROOT / "src", MUT_SRC)

    orig_alert_path = REPO_ROOT / "src" / "bts" / "health" / "alert.py"
    mut_alert_path = MUT_SRC / "bts" / "health" / "alert.py"
    orig_text = orig_alert_path.read_text()

    out: dict[str, dict] = {}
    try:
        for name, target, replacement, pytest_args in MUTATIONS:
            assert orig_text.count(target) == 1, f"Target for {name} appeared {orig_text.count(target)} times"
            mutated = orig_text.replace(target, replacement, 1)
            mut_alert_path.write_text(mutated)

            t0 = time.monotonic()
            proc = subprocess.run(
                [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=line", *pytest_args],
                cwd=REPO_ROOT,
                env=base_env(MUT_SRC),
                capture_output=True,
                text=True,
                timeout=90,
            )
            elapsed = round(time.monotonic() - t0, 2)
            last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            fifo_msg_seen = "dispatcher blocked on a FIFO" in proc.stdout
            print(f"[MUTATION {name}] rc={proc.returncode} ({elapsed}s): {last_line}", flush=True)
            assert proc.returncode == 1, (
                f"Mutation {name} WAS NOT KILLED! rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
            )
            if name in ("drop_reg_check_read", "drop_reg_check_write"):
                assert fifo_msg_seen, f"Expected bounded_fifo message for {name}:\n{proc.stdout}"
            m_fail = re.search(r"(\d+)\s+failed", last_line)
            out[name] = {
                "killed": True,
                "returncode": proc.returncode,
                "failed_count": int(m_fail.group(1)) if m_fail else None,
                "elapsed_sec": elapsed,
                "summary_line": last_line,
                "fifo_timeout_triggered": fifo_msg_seen if "fifo" in pytest_args else None,
            }
    finally:
        mut_alert_path.write_text(orig_text)
        shutil.rmtree(MUT_DIR, ignore_errors=True)

    return out


def main() -> None:
    base_res = run_base_red_green()
    mut_res = run_mutations()
    report = {
        "base_red_green": base_res,
        "mutations": mut_res,
    }
    (REVIEW_DIR / "pytest_mutations_and_base.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("ALL_MUTATIONS_AND_BASE_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()

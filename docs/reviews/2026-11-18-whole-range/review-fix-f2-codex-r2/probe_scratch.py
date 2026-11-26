"""Apply one change to a temporary source copy and run the real focused tests."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = (
    "tests/test_wr_f2_saver_flag.py",
    "tests/test_f2_b1_stat_oserror_regression.py",
    "tests/test_contest_dated_cs4a.py",
    "tests/health/test_contest_state.py",
    "tests/health/test_runner.py",
)
VARIANTS = (
    "control", "no-key", "live-reader", "wall-year", "warn-used", "in-digest", "mismatch-first",
    "shared-status", "no-recursion-catch", "no-oserror-catch", "hoisted-read",
    "no-ack", "no-fallback", "guarded-reorder",
)


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def mutate(src, variant):
    if variant == "control":
        return
    contest = src / "bts/health/contest_state.py"
    runner = src / "bts/health/runner.py"
    checks = contest.read_text()
    delivery = runner.read_text()
    if variant == "no-key":
        checks = swap(checks, "                    incident_key=DATED_USED_MISMATCH,\n", "")
    elif variant == "live-reader":
        checks = swap(checks, "            from bts.saver_state import _read_saver_state\n",
                      "            from bts.saver_state import load_saver_state\n")
        checks = swap(checks, "flag = _read_saver_state(picks_dir, season=year, historical=True)",
                      "flag = load_saver_state(picks_dir, season=year)")
    elif variant == "wall-year":
        checks = swap(checks, 'year = now.astimezone(ZoneInfo("America/New_York")).year\n            try:',
                      'year = datetime.now(ZoneInfo("America/New_York")).year\n            try:')
    elif variant == "warn-used":
        checks = swap(checks, 'if flag.state in {"active", "not_earned"}:',
                      'if flag.state in {"active", "not_earned", "used"}:')
    elif variant in {"no-recursion-catch", "no-oserror-catch"}:
        new = "OSError, ValueError" if variant == "no-recursion-catch" else "ValueError, RecursionError"
        checks = swap(checks, "except (OSError, ValueError, RecursionError):", f"except ({new}):")
    elif variant in {"guarded-reorder", "hoisted-read"}:
        dated = checks.index('        if state.saver_state == "not_earned" and 10 <= state.streak <= 15:')
        after_not_earned = checks.index('        if state.saver_state == "used":', dated)
        end = checks.index("\n        return alerts\n\n    # The flag", after_not_earned)
        guard = '        if state.saver_state == "used":\n'
        if variant == "guarded-reorder":
            use = checks[after_not_earned:end] + "\n"
            checks = checks[:dated] + use + checks[dated:after_not_earned].rstrip("\n") + checks[end:]
        else:
            read_end = checks.index('            if flag.state in {"active", "not_earned"}:', after_not_earned)
            read = checks[after_not_earned + len(guard):read_end]
            hoisted = "".join(line[4:] if line.startswith("    ") else line for line in read.splitlines(True))
            checks = checks[:dated] + hoisted + checks[dated:after_not_earned] + guard + checks[read_end:]
    elif variant == "in-digest":
        delivery = swap(delivery, "    if any(alert.incident_key == mismatch_key for alert in alerts):",
                        "    if False:")
    elif variant == "mismatch-first":
        start = delivery.index("    dispatch_dm_for_health_alerts(\n        digest_alerts,")
        second = delivery.index("    if mismatch_attention:", start)
        end = delivery.index("    return alerts\n", second)
        delivery = delivery[:start] + delivery[second:end] + delivery[start:second] + delivery[end:]
    elif variant == "shared-status":
        delivery = swap(delivery, "            status_path=mismatch_status,", "            status_path=status_path,")
    elif variant == "no-ack":
        delivery = swap(delivery, "            require_message_id=True,\n", "")
    elif variant == "no-fallback":
        start = delivery.index("        try:\n            mismatch_status =")
        end = delivery.index("        dispatch_dm_for_health_alerts(", start)
        delivery = (delivery[:start] + "        mismatch_status = status_path.with_name(status_path.name + suffix)\n"
                    + delivery[end:])
    else:
        raise ValueError(variant)
    contest.write_text(checks)
    runner.write_text(delivery)


if __name__ == "__main__":
    variant, = sys.argv[1:]
    assert variant in VARIANTS
    with TemporaryDirectory(prefix="f2-mutation-", dir="/tmp") as work:
        src = Path(work) / "src"
        shutil.copytree(ROOT / "src", src)
        mutate(src, variant)
        code = ("from pathlib import Path; import sys, bts, pytest; print('bts.__file__ =', bts.__file__, flush=True); "
                "assert Path(bts.__file__).is_relative_to(Path(sys.argv.pop(1))); "
                "raise SystemExit(pytest.main(sys.argv[1:]))")
        result = subprocess.run(
            [sys.executable, "-c", code, str(src), "-q", "-x", "--tb=short", "--basetemp",
             str(Path(work) / "pytest"), *TESTS],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{src}:{ROOT}"}, timeout=21,
        )
        assert result.returncode in (0, 1), result.returncode
        print(f"{variant}: {'tests caught the change' if result.returncode else 'all tests passed'}")

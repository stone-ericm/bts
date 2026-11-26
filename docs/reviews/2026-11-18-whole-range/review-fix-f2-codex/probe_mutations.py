import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECK = Path("bts/health/contest_state.py")
RUNNER = Path("bts/health/runner.py")
CASES = ("drop-key", "live-reader", "wall-year", "include-used", "fold-into-digest",
         "dispatch-before", "share-status", "no-recursion-catch", "move-before-old-warning")


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def mutation(source, variant):
    path = source / (RUNNER if variant in {"fold-into-digest", "dispatch-before", "share-status"} else CHECK)
    text = path.read_text()
    start = "        if state.saver_state == \"used\":\n"
    old_check = '        if state.saver_state == "not_earned" and 10 <= state.streak <= 15:\n'
    second = "    if mismatch_attention:\n"
    dispatch = "    dispatch_dm_for_health_alerts(\n"
    if variant == "drop-key":
        after = replace_once(text, "                    incident_key=DATED_USED_MISMATCH,\n", "")
    elif variant == "live-reader":
        after = replace_once(text, "            from bts.saver_state import _read_saver_state\n",
                             "            from bts.saver_state import load_saver_state\n")
        after = replace_once(after, "                flag = _read_saver_state(picks_dir, season=year, historical=True)\n",
                             "                flag = load_saver_state(picks_dir, season=year)\n")
    elif variant == "wall-year":
        after = replace_once(text, '            year = now.astimezone(ZoneInfo("America/New_York")).year\n',
                             '            year = datetime.now(ZoneInfo("America/New_York")).year\n')
    elif variant == "include-used":
        after = replace_once(text, '            if flag.state in {"active", "not_earned"}:\n',
                             '            if flag.state in {"active", "not_earned", "used"}:\n')
    elif variant == "fold-into-digest":
        after = replace_once(text, "        warn_attention=digest_attention,\n",
                             "        warn_attention=warn_attention,\n")
        second_start = after.index(second)
        second_end = after.index("    return alerts\n", second_start)
        after = after[:second_start] + after[second_end:]
    elif variant == "dispatch-before":
        first_start = text.index(dispatch, text.index("    log_alerts(alerts)\n"))
        second_start = text.index(second, first_start)
        second_end = text.index("    return alerts\n", second_start)
        after = text[:first_start] + text[second_start:second_end] + text[first_start:second_start] + text[second_end:]
    elif variant == "share-status":
        after = replace_once(text, 'status_path=status_path.with_name(status_path.name + ".dated_used_mismatch.json"),',
                             'status_path=status_path,')
    elif variant == "no-recursion-catch":
        after = replace_once(text, "            except (ValueError, RecursionError):\n",
                             "            except ValueError:\n")
    else:
        old_start = text.index(old_check)
        new_start = text.index(start, old_start)
        end = text.index("\n        return alerts\n", new_start) + 1
        after = text[:old_start] + text[new_start:end] + text[old_start:new_start] + text[end:]
    assert text != after
    path.write_text(after)


if __name__ == "__main__":
    variant, = sys.argv[1:]
    assert variant in CASES
    with TemporaryDirectory(prefix=f"mutation-{variant}-", dir=HERE) as directory:
        source = Path(directory) / "src"
        shutil.copytree(REPO / "src", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        mutation(source, variant)
        invoke = ("import bts, pytest, sys; print('bts.__file__ =', bts.__file__, flush=True); "
                  "raise SystemExit(pytest.main(sys.argv[1:]))")
        arguments = [sys.executable, "-c", invoke, "-p", "no:cacheprovider", "-q", "--tb=short",
                     "tests/test_wr_f2_saver_flag.py",
                     "tests/test_contest_dated_cs4a.py::test_health_observation_saver_without_flag_read"]
        with (HERE / f"mutation-{variant}.log").open("w") as logfile:
            test = subprocess.run(arguments, stdout=logfile, stderr=subprocess.STDOUT,
                                  cwd=REPO, env={**os.environ, "PYTHONPATH": f"{source}:{REPO}"},
                                  timeout=22, check=False)
    assert test.returncode in {0, 1}, (variant, test.returncode)
    print(variant, "NOT DETECTED" if test.returncode == 0 else "DETECTED")

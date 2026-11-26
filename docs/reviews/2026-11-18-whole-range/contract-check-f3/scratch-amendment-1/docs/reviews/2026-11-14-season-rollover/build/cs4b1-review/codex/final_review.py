"""Final sequential wrapper family: clean matrix, ledger readers, and regressions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from review_support import ROOT,REPO,write_json
from test_audit import NEW, run


def main():
    # Re-run serially: the first wrapper's nice failure still allowed its child
    # to start, overlapping the approved retry. Discard both earlier datasets.
    for script in ["fetch_matrix.py","ledger_consumers.py"]:
        with (ROOT/("final-"+script+".log")).open("w") as f:
            result=subprocess.run([sys.executable,str(ROOT/script)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        print(script,"rc",result.returncode,flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    target=ROOT/"mutants/empty_rejected/src"
    shutil.copytree(REPO/"src",target,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    path=target/"bts/cli.py"
    code=path.read_text();old="        if not isinstance(predictions, list):"
    assert code.count(old)==1
    path.write_text(code.replace(old,"        if not isinstance(predictions, list) or not predictions:"))
    run("mutant-empty_rejected",target,2027,NEW)
    summary=json.loads((ROOT/"test-audit-summary.json").read_text())
    empty=json.loads((ROOT/"tests-mutant-empty_rejected.json").read_text())["records"]
    for node in list(summary["survivors"]):
        if empty[node]["outcome"]=="failed":
            summary["survivors"].remove(node)
            summary["green_killed"][node]=["empty_rejected"]
    summary["all_new_can_fail"]=not summary["survivors"]
    write_json(ROOT/"test-audit-summary-final.json",summary)
    # Check the migrated saver assertions against a real helper bypass, too.
    migrated=["tests/test_contest_fetch.py::test_fetch_auto_earn_cold_file_at_10_stays_uninitialized",
              "tests/test_contest_fetch.py::test_fetch_auto_earn_below_10_inits_not_earned_then_promotes_at_10",
              "tests/test_saver_cs3.py::test_current_year_observation_initializes_then_earns",
              "tests/test_saver_cs3.py::test_other_year_observation_does_not_touch_flag",
              "tests/test_contest_dated_cs4a_review.py::test_all_three_row11_warning_commands_and_fetch_auto_earn_isolation"]
    run("migrated-saver-skip",ROOT/"mutants/saver_skip/src",2027,migrated)
    run("migrated-saver-always",ROOT/"mutants/saver_always/src",2027,[migrated[3]])
    relevant=set(NEW+["tests/test_cli_integration.py","tests/test_contest_fetch.py","tests/test_contest_ledger.py",
                     "tests/test_saver_cs3.py","tests/test_saver_cs3_review.py","tests/test_contest_dated_cs4a.py",
                     "tests/test_contest_dated_cs4a_review.py","tests/health/test_contest_state.py",
                     "tests/test_backup.py","tests/test_backup_cli.py","tests/scripts/season_ledger"])
    relevant.update(str(p.relative_to(REPO)) for p in (REPO/"tests").glob("test_contest_state*.py"))
    relevant=sorted(p for p in relevant if (REPO/p).exists())
    run("relevant",REPO/"src",2027,relevant)
    print("all families complete; all new tests can fail:",summary["all_new_can_fail"],flush=True)


if __name__=="__main__":main()

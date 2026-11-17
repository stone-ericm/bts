"""Resolve the frozen-clock fingerprint artifact, then run the focused suite normally."""
import collections
import json
from pathlib import Path
import subprocess
import sys

from review_support import ROOT,REPO,write_json
from test_audit import run

FINGERPRINT="tests/scripts/season_ledger/test_reconcile.py::test_candidate_rules_and_their_evaluator_are_fingerprinted"


def main():
    if len(sys.argv)>1:
        import pytest
        sys.path.insert(0,str(REPO/"src"))
        import bts.cli
        relevant=set(["tests/test_season_contract_cs4b1.py","tests/test_season_contract_cs4b1_edges.py",
            "tests/test_cli_integration.py","tests/test_contest_fetch.py","tests/test_contest_ledger.py",
            "tests/test_saver_cs3.py","tests/test_saver_cs3_review.py","tests/test_contest_dated_cs4a.py",
            "tests/test_contest_dated_cs4a_review.py","tests/health/test_contest_state.py",
            "tests/test_backup.py","tests/test_backup_cli.py","tests/scripts/season_ledger"])
        relevant.update(str(p.relative_to(REPO)) for p in (REPO/"tests").glob("test_contest_state*.py"))
        relevant=sorted(p for p in relevant if (REPO/p).exists())
        records={}
        class Recorder:
            def pytest_runtest_logreport(self,report):
                if report.when=="call" or report.failed:
                    records[report.nodeid]={"outcome":report.outcome,"failure":str(report.longrepr) if report.failed else None}
        rc=pytest.main(["-p","no:cacheprovider",*relevant,"-q","--tb=short"],plugins=[Recorder()])
        write_json(ROOT/"tests-relevant-normal.json",dict(rc=int(rc),counts=collections.Counter(r["outcome"] for r in records.values()),records=records))
        raise SystemExit(int(rc))
    run("base-fingerprint",ROOT/"base/src",2027,[FINGERPRINT])
    frozen=json.loads((ROOT/"tests-base-fingerprint.json").read_text())
    assert frozen["rc"]==1 and frozen["records"][FINGERPRINT]["outcome"]=="failed"
    # The closure binds outside callable names, so substituting Clock/Day
    # changes the declared datetime callable identity at both source versions.
    with (ROOT/"tests-relevant-normal.log").open("w") as f:
        subprocess.run([sys.executable,__file__,"normal"],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    print("base frozen fingerprint fails identically; normal relevant suite passes",flush=True)


if __name__=="__main__":main()

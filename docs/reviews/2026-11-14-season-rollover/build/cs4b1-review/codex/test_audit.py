"""One wrapper family: touched files in two years, base controls, meaningful mutants."""
import collections
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from review_support import ROOT, REPO, REAL_DT, REAL_DATE, ET, write_json

NEW=["tests/test_season_contract_cs4b1.py","tests/test_season_contract_cs4b1_edges.py"]
TOUCHED=NEW+["tests/test_contest_fetch.py","tests/test_cli_integration.py","tests/test_saver_cs3.py",
             "tests/test_contest_dated_cs4a_review.py","tests/test_contest_state_cs1_review.py"]


def child(label,source,year,paths):
    sys.path.insert(0,str(source))
    import pandas
    import pytest
    fixed=REAL_DT(year,7,15,12,tzinfo=ET)
    class Clock(REAL_DT):
        @classmethod
        def now(cls,tz=None): return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
        @classmethod
        def utcnow(cls): return fixed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        @classmethod
        def today(cls): return cls.now()
    class Day(REAL_DATE):
        @classmethod
        def today(cls): return cls(year,7,15)
    dt.datetime,dt.date=Clock,Day
    import bts.cli as cli
    assert Path(cli.__file__).is_relative_to(source),cli.__file__
    records={}
    class Recorder:
        def pytest_runtest_logreport(self,report):
            if report.when=="call" or report.failed:
                records[report.nodeid]={"outcome":report.outcome,"when":report.when,
                    "failure":str(report.longrepr) if report.failed else None}
    rc=pytest.main(["-p","no:cacheprovider",*paths,"-q","--tb=short"],plugins=[Recorder()])
    write_json(ROOT/("tests-"+label+".json"),dict(rc=int(rc),source=str(source),year=year,records=records,
        counts=collections.Counter(r["outcome"] for r in records.values())))
    print("TEST RUN",label,collections.Counter(r["outcome"] for r in records.values()),flush=True)


def run(label,source,year,paths):
    with (ROOT/("tests-"+label+".log")).open("w") as f:
        subprocess.run([sys.executable,__file__,"child",label,str(source),str(year),*paths],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    if len(sys.argv)>1:
        child(sys.argv[2],Path(sys.argv[3]),int(sys.argv[4]),sys.argv[5:]);return
    for year in [2027,2028]:run(str(year),REPO/"src",year,TOUCHED)
    run("base",ROOT/"base/src",2027,NEW)
    # Mutants address preservation controls that intentionally pass at the base.
    # Each is a production change, copied into scratch; tracked source is untouched.
    mutations={
        "failure_exit":("bts/cli.py","sys.exit(code)","sys.exit(0)"),
        "rounds_precedence":("bts/cli.py",'predictions = success.get("predictions", [])\n        rounds = _fetch_rounds()',
                             'rounds = _fetch_rounds()\n        predictions = success.get("predictions", [])'),
        "identity_bypass":("bts/cli.py",'if out_path.exists():','if False and out_path.exists():'),
        "dry_persists":("bts/cli.py",'    if dry_run:\n        if switched:','    if False and dry_run:\n        if switched:'),
        "saver_skip":("bts/cli.py",'if source_date is not None and source_date.year == season:',
                       'if False and source_date is not None and source_date.year == season:'),
        "saver_always":("bts/cli.py",'if source_date is not None and source_date.year == season:',
                         'if source_date is not None:'),
        "reader_switch":("bts/contest_state.py",'    now = now or datetime.now(timezone.utc)\n    state_dir = picks_dir / "account_state"',
                          '    (picks_dir / "account_state/season_contract.json").read_text()\n    now = now or datetime.now(timezone.utc)\n    state_dir = picks_dir / "account_state"'),
        "observation_error_swallowed":("bts/cli.py",'    if switched:\n        _atomic_write_json(out_path, observation)',
                                       '    if switched:\n        try:\n            _atomic_write_json(out_path, observation)\n        except OSError:\n            pass'),
        "audit_fatal":("bts/saver_state.py",'    except OSError as e:\n        # Never block the transition',
                       '    except OSError as e:\n        raise\n        # Never block the transition'),
        "switch_checks_skipped":("bts/season_contract.py",'            raise InvalidContract(check, text)','            return'),
        "cap_removed":("bts/contest_fetch.py",'    if best_streak > 57:', '    if False and best_streak > 57:'),
    }
    for label,(filename,old,new) in mutations.items():
        target=ROOT/"mutants"/label/"src"
        shutil.copytree(REPO/"src",target,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        path=target/filename
        text=path.read_text()
        assert text.count(old)==1,(label,text.count(old))
        path.write_text(text.replace(old,new))
        run("mutant-"+label,target,2027,NEW)
    base=json.loads((ROOT/"tests-base.json").read_text())["records"]
    green={k for k,v in base.items() if v["outcome"]=="passed"}
    killed={}
    for label in mutations:
        records=json.loads((ROOT/("tests-mutant-"+label+".json")).read_text())["records"]
        for k,v in records.items():
            if v["outcome"]=="failed":killed.setdefault(k,[]).append(label)
    write_json(ROOT/"test-audit-summary.json",dict(green_controls=sorted(green),
        counts={str(y):json.loads((ROOT/("tests-"+str(y)+".json")).read_text())["counts"] for y in [2027,2028]},
        base_counts=json.loads((ROOT/"tests-base.json").read_text())["counts"],
        green_killed={k:killed.get(k,[]) for k in sorted(green)},
        all_new_can_fail=all(v["outcome"]=="failed" or k in killed for k,v in base.items()),
        survivors=[k for k,v in base.items() if v["outcome"]!="failed" and k not in killed]))
    print((ROOT/"test-audit-summary.json").read_text(),flush=True)


if __name__=="__main__":main()

"""Demonstrate that the five pure preservation-control cases detect a production regression."""
import dataclasses
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas
import pytest
from bts.cli import cli
import bts.contest_state as cs

# Deliberate runtime mutations only: preserve real parsing/writing, alter its streak.
real_parse=cs._parse_state_file
def wrong_streak(*a,**kw):
    result=real_parse(*a,**kw)
    return dataclasses.replace(result,streak=result.streak+1)
cs._parse_state_file=wrong_streak
real_write=Path.write_text
def wrong_legacy(path,text,*a,**kw):
    if path.name=='contest_streak.manual.json' and 'bts_contest_streak_manual_v2' in text:
        data=json.loads(text); data['active_streak']+=1
        text=json.dumps(data,indent=2,sort_keys=True)+'\n'
    return real_write(path,text,*a,**kw)
Path.write_text=wrong_legacy
outcomes={}
class Recorder:
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed: outcomes[report.nodeid]=report.outcome
code=pytest.main(['tests/test_contest_dated_cs4a.py','-q','-p','no:cacheprovider','--tb=no',
    '--basetemp',str(HERE/'pytest_tmp_mutations'),'-k',
    'legacy_fixed_clock_bytes_and_output_control or wrong_schema_remains_lax_legacy_control or automatic_v3_schema_still_uses_automatic_parser_control'],plugins=[Recorder()])
Path.write_text=real_write; cs._parse_state_file=real_parse
(HERE/'control_mutations.json').write_text(json.dumps(outcomes,indent=2))
assert code==1 and len(outcomes)==5 and set(outcomes.values())=={'failed'},outcomes
print('All five pure preservation-control nodes detect deliberate parser/writer mutations.')

"""Compare full matrix and prior production evidence with source/fixture artifacts normalized."""
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent;R=H.parents[1]
def read(name):return [json.loads(x) for x in (H/name).read_text().splitlines()]
def norm(value,case=None):
    if isinstance(value,dict):
        return {norm(k,case):norm(v,case) for k,v in value.items()}
    if isinstance(value,list):return [norm(v,case) for v in value]
    if isinstance(value,str):
        # Hex snapshots contain pick date strings with default-object repr addresses.
        if len(value)>20 and len(value)%2==0 and re.fullmatch('[0-9a-f]+',value):
            try:value=bytes.fromhex(value).decode('utf8')
            except UnicodeDecodeError:pass
        value=re.sub(r'0x[0-9a-f]+','0xADDRESS',value)
        value=re.sub(r'/(?:r2-)?fixtures-(?:base|current)/','/FIXTURES/',value)
        value=re.sub(r'/(?:r2-)?exit-fixtures-(?:base|current)/','/EXIT-FIXTURES/',value)
        if case in ('long','long-tail'):value=value[:200]
    return value

old=read('probe-current.jsonl');new=read('r2-probe-current.jsonl');base=read('r2-probe-base.jsonl')
om={(x['case'],x['path']):x for x in old if x['family']=='date'}
nm={(x['case'],x['path']):x for x in new if x['family']=='date'}
bm={(x['case'],x['path']):x for x in base if x['family']=='date'}
assert len(om)==len(nm)==len(bm)==344
changed=[];message_only=[]
def semantic(row):
    o=dict(row['outcome']);o.pop('message',None)
    return {**{k:row[k] for k in ('reads','replies','decisions','before','after','pick_before','pick_after','stdout','stderr')},'outcome':o}
for key,b in nm.items():
    a=om[key];case,path=key
    if norm(semantic(a),case)!=norm(semantic(b),case):changed.append(key)
    elif norm(a['outcome'],case)!=norm(b['outcome'],case):message_only.append(key)
special={'class-trap','str-subclass-exit','str-subclass','str-subclass-unpadded'}
assert all(case in special for case,path in changed),changed
assert {case for case,path in message_only}<= {'date','datetime','bytes','int','bool'}

counterexamples=[]
for case in ('class-trap','str-subclass-exit'):
    for path in ('shadow','poll-void','poll-final-peer','poll-early-peer','select','update'):
        a,b=bm[case,path],nm[case,path]
        if path=='update':
            assert b['outcome']['error']=='ValueError' and b['reads']==[] and b['before']==b['after']
        else:
            assert a['outcome'].get('error')==b['outcome'].get('error')
            assert [r[2] for r in a['reads']]==[r[2] for r in b['reads']]
            assert a['decisions']==b['decisions'] and a['replies']==b['replies']
            assert norm(a['pick_after'],case)==norm(b['pick_after'],case)
        counterexamples.append(dict(case=case,path=path,base=a['outcome'],current=b['outcome']))

for case in ('str-subclass','str-subclass-unpadded'):
    for path in ('shadow','poll-void','poll-final-peer','poll-early-peer','select','update'):
        a,b=bm[case,path],nm[case,path]
        if path=='update':assert b['outcome']['error']=='ValueError' and b['reads']==[]
        else:
            assert a['outcome'].get('error')==b['outcome'].get('error')
            assert [r[2] for r in a['reads']]==[r[2] for r in b['reads']]
            assert a['decisions']==b['decisions'] and a['replies']==b['replies']

exits_old=read('exit-paths-current.jsonl');exits_new=read('r2-exit-paths-current.jsonl');exits_base=read('r2-exit-paths-base.jsonl')
assert len(exits_old)==len(exits_new)==len(exits_base)==91
for a,b,c in zip(exits_old,exits_new,exits_base):
    assert norm(a)==norm(b)==norm(c),(a['case'],a['path'])

corrupt_old=[r for r in old if r['family']=='corrupt']
corrupt_new=[r for r in new if r['family']=='corrupt']
assert len(corrupt_old)==len(corrupt_new)==105
for a,b in zip(corrupt_old,corrupt_new):
    assert norm(a)==norm(b),(a['case'],a['path'])
for family in ('markers','saver','sequence'):
    a=[r for r in old if r['family']==family];b=[r for r in new if r['family']==family]
    assert norm(a)==norm(b),family
extra_b=read('r2-extras-base.jsonl');extra_n=read('r2-extras-current.jsonl')
for rs in (extra_b,extra_n):
    races=[r for r in rs if r['family']=='race'];assert len(races)==5
    for r in races:
        assert json.loads(bytes.fromhex(r['after']))['streak']==6 and r['pick']=='hit'
        assert all(x['exit']==0 for x in r['jobs'])
more_b=read('r2-more-corrupt-base.jsonl');more_n=read('r2-more-corrupt-current.jsonl')
assert len(more_b)==len(more_n)==60
for a,b in zip(more_b,more_n):
    assert a['case']==b['case'] and a['path']==b['path']
    assert a['outcome'].get('error')==b['outcome'].get('error')
    assert (a['before']==a['after'])==(b['before']==b['after'])

test_summary={};green=[]
for mode in ('clock2027','clock2028','base'):
    tree=ET.parse(H/f'r2-{mode}.xml');counts=Counter();files={}
    for test in tree.iter('testcase'):
        status='failed' if test.find('failure') is not None or test.find('error') is not None else 'skipped' if test.find('skipped') is not None else 'passed'
        counts[status]+=1;file=test.attrib['classname'].split('.')[1]
        files.setdefault(file,Counter())[status]+=1
        if mode=='base' and status=='passed':green.append(test.attrib)
    test_summary[mode]={'totals':dict(counts),'files':files}
(H/'r2-base-green-tests.json').write_text(json.dumps(green,indent=2))

assertion_changes=[]
for file in ('tests/test_cli_integration.py','tests/test_streak_season_cs2.py'):
    a=subprocess.check_output(['git','show','dd915ac9:'+file],cwd=R,text=True)
    b=(R/file).read_text()
    def assertions(source):return Counter(ast.dump(x,include_attributes=False) for x in ast.walk(ast.parse(source)) if isinstance(x,ast.Assert))
    aa,bb=assertions(a),assertions(b)
    assertion_changes.append(dict(file=file,removed=list((aa-bb).elements()),added=list((bb-aa).elements())))
summary=dict(matrix_semantic_changes=changed,matrix_message_only_changes=message_only,
             round1_counterexamples=counterexamples,tests=test_summary,assertion_changes=assertion_changes)
(H/'r2-comparison.json').write_text(json.dumps(summary,indent=2))
print('MATRIX semantic changes',changed)
print('MATRIX message-only changes',message_only)
print('TESTS',json.dumps(test_summary,default=dict))
print('CORRUPTION, EXITS, SEQUENCES, MARKERS, SAVER AND FIVE REAL RACES: assertions passed')

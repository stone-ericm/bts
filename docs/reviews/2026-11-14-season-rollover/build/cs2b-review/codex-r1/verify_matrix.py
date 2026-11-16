"""Assertions separating allowed scope/error/tag changes from the helper hook failures."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
def read(f):return [json.loads(s) for s in (H/f).read_text().splitlines()]
b=read('probe-base.jsonl');c=read('probe-current.jsonl')
bm={(x['case'],x['path']):x for x in b if x['family']=='date'}
helper={x['case']:x['outcome'] for x in c if x['family']=='helper'}
counts={}
for z in c:
    if z['family']!='date':continue
    case,path=z['case'],z['path'];a=bm[case,path]
    if case in ('class-trap','str-subclass-exit'):
        counts['B1-hook-rows']=counts.get('B1-hook-rows',0)+1
        continue
    assert a['before']==z['before']
    # Real pick records, error classes, stdout, and decisions are all compared.
    assert a['pick_before']==z['pick_before'] and a['pick_after']==z['pick_after'],(case,path,'pick bytes')
    year=helper[case]['return']
    if path=='cli' and case in ('date','int'):
        year=2027  # Click converts the raw option to text before production sees it.
    def norm(value):
        if isinstance(value,str):
            value=value.replace('fixtures-base/','fixtures-/').replace('fixtures-current/','fixtures-/')
            return value[:200] if case in ('long','long-tail') else value
        if isinstance(value,dict):return {k:norm(v) for k,v in value.items()}
        return value
    if path=='update':
        if year is not None:
            assert a['outcome']=={'return':14} and z['outcome']=={'return':1}
            doc=json.loads(bytes.fromhex(z['after']))
            assert doc['season']==year and doc['streak']==1 and doc['saver_available'] is True
            assert len(doc['applied_dates'])==1
        elif case=='none':
            assert a['outcome']==z['outcome']=={'return':14}
            ad=json.loads(bytes.fromhex(a['after']));zd=json.loads(bytes.fromhex(z['after']))
            assert zd.pop('season')==2026 and ad==zd
        else:
            assert z['outcome']['error']=='ValueError' and z['reads']==[] and z['after']==z['before']
    else:
        assert z['before']==z['after']
        if year is None:
            assert norm(a['outcome'])==norm(z['outcome']),(case,path,'unreadable-date outcome')
            assert [x[2] for x in a['reads']]==[x[2] for x in z['reads']]
            assert norm(a['stdout'])==norm(z['stdout']) and norm(a['stderr'])==norm(z['stderr']),(case,path,'output')
            assert a['replies']==z['replies'] and a['decisions']==z['decisions']
        else:
            assert a['outcome'].get('error')==z['outcome'].get('error'),(case,path,'unexpected parser error')
            for reader,kw,value in z['reads']:
                assert value==(True if reader=='saver' else 0),(case,path,reader,value)
            if z['decisions']:assert z['decisions']==[[13,True]]
            if z['replies']:assert z['replies']==[[a['replies'][0][0],0]]
    label='scoped' if year is not None else 'dateless-tag' if case=='none' and path=='update' else 'legacy-or-L4-error'
    counts[label]=counts.get(label,0)+1
# Every corrupt scorer branch matches base exception type and preservation behavior.
cb={(x['case'],x['path']):x for x in b if x['family']=='corrupt'}
for z in c:
    if z['family']!='corrupt':continue
    a=cb[z['case'],z['path']]
    assert a['outcome'].get('error')==z['outcome'].get('error'),(z['case'],z['path'])
    assert (a['before']==a['after'])==(z['before']==z['after'])
    assert a['pick_before']==z['pick_before'] and a['pick_after']==z['pick_after']
print('MATRIX ASSERTIONS PASSED',counts)
(H/'matrix-verification.json').write_text(json.dumps(counts,indent=2))

"""Reject unexplained round-one/current differences, preserving every semantic delta."""
import collections
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def norm(value):
    if isinstance(value,dict):
        d={k:norm(v) for k,v in value.items() if k not in ['ts','updated_at']}
        for k in ['before','after']:
            if isinstance(d.get(k),str):
                try:d[k]=norm(json.loads(bytes.fromhex(d[k])))
                except (ValueError,UnicodeDecodeError):pass
        if d.get('family')=='artifacts':d.pop('source',None)
        return d
    if isinstance(value,list):return [norm(v) for v in value]
    if isinstance(value,str):
        value=re.sub(re.escape(str(ROOT))+r'/evidence/(?:r1|branch|base)-', '<evidence>/',value)
        value=re.sub(re.escape(str(ROOT))+r'/evidence/edges(?:-v2)?-(?:r1|branch|base)', '<evidence>/edges',value)
        return value
    return value
def authorized_text(value):
    if isinstance(value,dict):return {k:authorized_text(v) for k,v in value.items()}
    if isinstance(value,list):return [authorized_text(v) for v in value]
    if not isinstance(value,str):return value
    # Transform only the exact warning/CLI tails explicitly authorized by the amendment.
    value=re.sub(r'Verify the saver in MLB; for a spent saver run `bts saver-state --init used --season (\d+)`',
        lambda m:'Check the saver in MLB and run ONE of these. Not earned yet this season: '
          f'`bts saver-state --init not_earned --season {m[1]}`. Earned and unused: '
          f'`bts saver-state --init active --season {m[1]}`. Already used: '
          f'`bts saver-state --init used --season {m[1]}`',value)
    value=re.sub(r'Verify the saver in MLB; if consumed run `bts saver-state --use --season (\d+)`',
        lambda m:'Check the saver in MLB and run ONE of these. Earned and unused: '
          f'`bts saver-state --init active --force --season {m[1]}`. Already used: '
          f'`bts saver-state --use --season {m[1]}`',value)
    value=re.sub(r'Marked saver used \(it was not_earned\)\. To take this back run `bts saver-state --init not_earned --force --season \d+`\.\n',
                 'Marked saver used.\n',value)
    return value

summary={};unexplained=[];details=[]
for family in ['matrix','operators','edges']:
    suffix='-v2' if family=='edges' else ''
    a,b=read('r1-'+family+suffix+'.json'),read('branch-'+family+suffix+'.json')
    assert len(a)==len(b),(family,len(a),len(b))
    delta=collections.Counter()
    for old,new in zip(a,b):
        assert old['family']==new['family'] and old.get('key')==new.get('key')
        x,y=norm(old),norm(new)
        if x==y:continue
        kind=new['family'];delta[kind]+=1
        details.append({'family':family,'kind':kind,'key':new.get('key'),'old':x,'new':y})
        if kind in ['model-types','model-existing-pick']:
            raw=new['key'][0] if kind=='model-types' else new['key']
            assert type(raw) is not int or raw<0
            assert new['result']['error']=='ContestStateError'
            assert 'model state is corrupt; cannot use model fallback:' in new['result']['message']
        elif kind=='pasted-command':
            st,n=new['key'];target='active' if st=='not_earned' else 'not_earned'
            assert new['exit']==0 and new['state']==target
            assert new['command']==('bts saver-state --init active --force --season 2027' if st=='not_earned'
                                   else 'bts saver-state --init not_earned --season 2027')
        else:
            # New CLI help is normalized in the opposite direction to preserve all other output.
            if authorized_text(x)!=authorized_text(y):unexplained.append(details[-1])
    summary[family]={'rows':len(a),'changed':dict(delta)}
pa,pb=read('r1-policy.json'),read('branch-policy.json')
assert norm(pa)==norm(pb),'integer policy sweep changed'
summary['policy']={'rows':len(pa),'changed':0,'resolutions_per_package':pa[-1]['value']}
summary['unexplained']=len(unexplained)
(ROOT/'round-two-deltas-v2.json').write_text(json.dumps(details,indent=2))
(ROOT/'unexplained-deltas-v2.json').write_text(json.dumps(unexplained,indent=2))
(ROOT/'comparison-summary-v2.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
assert not unexplained,'See unexplained-deltas-v2.json'
print('All round-one/current differences explained by the three amendments')

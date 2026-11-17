"""Fill the valid-current-season model coverage gap; forbid actual local-pick/flag/auto reads."""
import itertools as it
import json
from pathlib import Path
import sys
from unittest.mock import patch
from matrix_probe import HERE,ROOT,NOW,changed,put
sys.path.insert(0,str(ROOT/'src'))
import bts.contest_state as cs
from bts import strategy
p=HERE/'isolation_files'; state=p/'account_state'
put(state/'contest_streak.json',dict(active_streak=58))
put(state/'saver_state.json',b'{"season":'+b'9'*5000+b'}')
put(p/'2027-07-16.json',dict(result='hit')); put(p/'2027-07-17.json',dict(result='miss'))
models=[None,b'{',dict(season=2027,streak=57,saver_available=True),
        dict(season=2027,streak=0,saver_available=False),
        dict(season=2027,streak={'bad':'value'},saver_available=['bad']),
        dict(season=2026,streak=57,saver_available=True)]
original=Path.open; total=policies=0; mdp=strategy._load_mdp()
def opened(path,mode='r',*a,**kw):
    if 'r' in mode and (path in (state/'contest_streak.json',state/'saver_state.json')
                       or path.parent==p and path.name!='streak.json'):
        raise AssertionError('authority leak: '+str(path))
    return original(path,mode,*a,**kw)
for streak,saver,required,model in it.product(range(58),('not_earned','active','used'),(False,True),models):
    put(state/'contest_streak.manual.json',changed(active_streak=streak,best_streak=streak,saver_state=saver))
    put(p/'streak.json',model)
    with patch.object(Path,'open',opened):
        d=cs.load_decision_streak_state(p,now=NOW,require_contest_state=required)
    expected=saver=='active' or saver=='not_earned' and streak<10
    assert (d.streak,d.saver_available,d.status,d.best_status)==(streak,expected,'fresh','trusted')
    total+=1
    for prob in range(70,91):
        ctx=strategy.DecisionContext(prob/100,(prob-1)/100,True,'2027-07-15',True,mdp,streak,'trusted')
        assert strategy.resolve_policy_decision(ctx,d.streak,d.saver_available)==strategy.resolve_policy_decision(ctx,streak,expected)
        policies+=1
(HERE/'model_isolation_summary.json').write_text(json.dumps(dict(decisions=total,comparisons=policies),indent=2))
print('MODEL ISOLATION',total,'decisions;',policies,'policy comparisons; no forbidden actual reads.')

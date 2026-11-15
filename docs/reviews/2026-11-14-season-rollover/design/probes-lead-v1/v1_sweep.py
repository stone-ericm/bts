import numpy as np, hashlib, collections
from datetime import date, timedelta
from pathlib import Path
from bts.strategy import DecisionContext, resolve_policy_decision
p=Path("data/models/mdp_policy.npz")
print(hashlib.sha256(p.read_bytes()).hexdigest()[:8])
z=np.load(p, allow_pickle=False)
mdp={"policy_table": z["policy_table"], "boundaries": z["boundaries"].tolist(), "season_length": int(z["season_length"])}
for s,day,pr in [(38,"2027-08-23",.82),(4,"2027-08-29",.80),(16,"2027-08-23",.82)]:
    ctx=DecisionContext(pr,pr-.01,True,day,True,mdp,s,"trusted")
    d=[resolve_policy_decision(ctx,s,sv) for sv in (False,True)]
    print(s,day,pr,"spent:",d[0].action,"available:",d[1].action,d[0].objective,d[0].days_effective)
# sweep: every 2027 date, streak 0..56, primary p on a realistic grid (rank-1 ~0.70..0.90)
band=collections.Counter(); tot=collections.Counter(); pairs=collections.Counter(); bymonth=collections.Counter()
d0=date(2027,3,24)
for k in range((date(2027,9,26)-d0).days+1):
    day=(d0+timedelta(k)).isoformat()
    for s in range(57):
        for pr in np.arange(0.70,0.901,0.01):
            ctx=DecisionContext(float(pr),float(pr)-.01,True,day,True,mdp,s,"trusted")
            a=resolve_policy_decision(ctx,s,False); b=resolve_policy_decision(ctx,s,True)
            if a.objective!="reach57": continue
            g="0-4" if s<5 else "5-15" if s<16 else "16-56"
            tot[g]+=1
            if a.action!=b.action:
                band[g]+=1; pairs[(g,a.action,b.action)]+=1; bymonth[(g,day[5:7])]+=1
for g in tot: print(g, band[g], "of", tot[g], f"{band[g]/tot[g]:.2%}")
print(sorted(pairs.items())); print(sorted(bymonth.items()))

"""Legacy and automatic parser byte comparisons, plus literal decoding edge inputs."""
import itertools as it
import json
from pathlib import Path
import subprocess
import sys
from matrix_probe import HERE,ROOT,NOW,obs,changed,put,call

def worker(package,label):
    sys.path.insert(0,str(package))
    import bts.contest_state as cs
    p=HERE/'parser_files'; path=p/'account_state'; results={}
    values=[None,True,False,1,1.0,-0.0,-1,57,58,'',{},[],float('nan'),
        '2027-7-5','20270705','2027-W27-1','2027-07-05T12:00:00Z','2027-07-16T03:59:59.999999Z']
    for filename,schema,field,value,required in it.product(('contest_streak.manual.json','contest_streak.json'),
        ('MISSING',None,'unknown','bts_contest_streak_manual_v1','bts_contest_streak_manual_v2','bts_contest_streak_manual_v3'),
        ('effective_date','season','expires_at','active_streak','best_streak','saver_state','recorded_at','source','reason','source_date','saver_available','override_expires_at'),
        values,(False,True)):
        # Exact-v3 manual is the intentional new branch; checked separately by matrix_probe.
        if filename=='contest_streak.manual.json' and schema=='bts_contest_streak_manual_v3': continue
        b=changed(**{field:value})
        if schema=='MISSING': b.pop('schema_version')
        else: b['schema_version']=schema
        f=path/filename; put(f,b)
        key=repr((filename,schema,field,value,required))
        results[key]=call(lambda:cs._parse_state_file(f,require_dated=required))
    for filename in ('contest_streak.manual.json','contest_streak.json'):
        for name,raw in [('deep',b'{"schema_version":"bts_contest_streak_manual_v2","x":'+b'['*1500+b'0'+b']'*1500+b'}'),
                         ('huge',b'{"schema_version":"bts_contest_streak_manual_v3","x":'+b'9'*5000+b'}'),
                         ('utf8',b'\xff'),('nonobject',b'[]')]:
            f=path/filename; put(f,raw)
            results[filename+name]=call(lambda:cs._parse_state_file(f))
    (HERE/f'parser_{label}.json').write_text(json.dumps(results,indent=1)); print('PARSER',label,len(results),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1: worker(Path(sys.argv[1]),sys.argv[2]); sys.exit()
    for package,label in ((HERE/'base_src/src','base'),(ROOT/'src','branch')):
        subprocess.run([sys.executable,__file__,str(package),label],check=True,cwd=ROOT)
    a=json.loads((HERE/'parser_base.json').read_text()); b=json.loads((HERE/'parser_branch.json').read_text())
    assert a==b,[k for k in a if a[k]!=b[k]]
    sys.path.insert(0,str(ROOT/'src'))
    import bts.contest_state as cs
    from command_probe import clock,invoke
    import datetime as dt
    from zoneinfo import ZoneInfo
    p=HERE/'literal_files'; manual=p/'account_state/contest_streak.manual.json'
    july5=dt.datetime(2027,7,5,12,tzinfo=ZoneInfo('America/New_York'))
    with clock(july5): r=invoke(p,day='2027-7-5')
    assert r.exit_code==0 and json.loads(manual.read_text())['effective_date']=='2027-07-05'
    for value in ('2027-7-5','20270705','2027-W27-1','2027-07-05T00:00:00'):
        data=obs('2027-07-05'); data['effective_date']=value; put(manual,data)
        got=call(lambda:cs.load_contest_streak_state(p,now=july5))
        assert got['error']=='ContestStateError' and 'check 2,' in got['message'],got
    for raw in ('-0','57','58','true','1.0','NaN'):
        data=obs(); data['best_streak']=57
        text=json.dumps(data).replace('"active_streak": 12','"active_streak": '+raw)
        manual.write_text(text); got=call(lambda:cs.load_contest_streak_state(p,now=NOW))
        assert ('ok' in got)==(raw in ('-0','57')),(raw,got)
    # Actual duplicate keys use the decoder's last value; they are not independently prohibited.
    manual.write_text(json.dumps(obs())[:-1]+',"saver_state":"active"}')
    assert cs.load_decision_streak_state(p,now=NOW).saver_flag_state=='active'
    print('LITERAL date/numeric/duplicate inputs passed; preservation cases',len(a),flush=True)


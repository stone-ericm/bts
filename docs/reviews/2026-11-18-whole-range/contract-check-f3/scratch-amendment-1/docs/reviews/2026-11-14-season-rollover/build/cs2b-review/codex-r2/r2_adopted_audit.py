"""Observe the adopted lifecycle test's actual interleaving and complete its unused corruption cases."""
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
H=Path(__file__).resolve().parent;R=H.parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('no network'))
socket.socket.connect_ex=socket.socket.connect
import tests.test_streak_season_cs2b_review as adopted
from click.testing import CliRunner
import bts.picks as p

metrics={'peer_calls':0,'nested_peer_calls':0,'max_nested_peer_calls':0,'cli_invocations':0,'resolver_error_outputs':0}
def profile(frame,event,arg):
    if frame.f_code.co_name=='resolve_with_peer_interleave' and frame.f_code.co_filename.endswith('test_streak_season_cs2b_review.py'):
        if event=='call':
            metrics['peer_calls']+=1;metrics['nested_peer_calls']+=1
            metrics['max_nested_peer_calls']=max(metrics['max_nested_peer_calls'],metrics['nested_peer_calls'])
        elif event=='return':metrics['nested_peer_calls']-=1
invoke=CliRunner.invoke
def observe(*a,**kw):
    metrics['cli_invocations']+=1
    result=invoke(*a,**kw)
    if 'ERROR: Failed to check game result' in result.output:metrics['resolver_error_outputs']+=1
    return result
d=H/'r2-adopted-lifecycle-profile';d.mkdir(exist_ok=True)
sys.setprofile(profile)
try:
    with patch.object(CliRunner,'invoke',side_effect=observe,autospec=True):
        adopted.test_full_rollover_lifecycle_void_hit_double_race_crash_and_reconcile(d)
    metrics['test']='passed'
except BaseException as e:metrics['test']='failed';metrics['error']=type(e).__name__+': '+str(e)
finally:sys.setprofile(None)
(H/'r2-adopted-profile.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics))


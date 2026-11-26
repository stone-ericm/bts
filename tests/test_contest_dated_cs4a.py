"""CS4a contract: real files/readers/CLI, pinned clocks, and failure boundaries."""
import builtins
import datetime as dt
import json
import shlex
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts.contest_state as cs
from bts.cli import cli
from bts.health.contest_state import check
import bts.health.contest_state as health
import bts.saver_state as flags
import bts.util as util

ET = ZoneInfo('America/New_York')
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
SCHEMA = 'bts_contest_streak_manual_v3'


def observation(day='2027-07-15', **changes):
    date = dt.date.fromisoformat(day)
    data = dict(schema_version=SCHEMA, effective_date=day, season=date.year,
        expires_at=dt.datetime.combine(date, dt.time.max, ET).isoformat(),
        active_streak=12, best_streak=14, saver_state='not_earned',
        recorded_at='2027-07-15T16:00:00Z', source='manual_cli')
    data.update(changes)
    return data


def write(picks, body, name='contest_streak.manual.json'):
    path = picks/'account_state'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def clock(monkeypatch, *times):
    class Clock(dt.datetime):
        calls = 0
        @classmethod
        def now(cls, tz=None):
            value = times[min(cls.calls, len(times)-1)]
            cls.calls += 1
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    monkeypatch.setattr(dt, 'datetime', Clock)
    monkeypatch.setattr(cs, 'datetime', Clock)
    monkeypatch.setattr(health, 'datetime', Clock)
    monkeypatch.setattr(flags, 'datetime', Clock)
    return Clock


def command(picks, *extras, streak=12, best=14, saver='not_earned', day='2027-07-15'):
    return ['set-contest-streak','--picks-dir',str(picks),'--date',day,
            '--streak',str(streak),'--best-streak',str(best),'--saver',saver,*extras]


def invoke(picks, *extras, **kwargs):
    return CliRunner().invoke(cli, command(picks, *extras, **kwargs))


def snapshot(picks):
    return {str(p.relative_to(picks)):p.read_bytes() for p in picks.rglob('*') if p.is_file()}


@pytest.mark.parametrize('reason', [None, '', 'MLB checked'])
def test_write_and_noop_order_output_and_isolation(tmp_path, monkeypatch, reason):
    tick = clock(monkeypatch, NOW)
    for name in ['saver_state.json','contest_streak.json','contest_fetch_status.json']:
        write(tmp_path, {'sentinel':name}, name)
    before = snapshot(tmp_path)
    fetch_status = tmp_path.parent/'health_state/contest_streak_fetch_status.json'
    fetch_status.parent.mkdir(exist_ok=True)
    fetch_status.write_bytes(b'fetch status sentinel')
    events = []
    original_open, original_atomic = builtins.open, util.atomic_write_text
    def opened(path, mode='r', *args, **kwargs):
        if str(path).endswith('contest_manual_requests.jsonl') and mode == 'a':
            events.append('append')
        return original_open(path, mode, *args, **kwargs)
    def atomic(path, text):
        events.append('replace')
        original_atomic(path,text)
    monkeypatch.setattr(builtins,'open',opened)
    monkeypatch.setattr(util,'atomic_write_text',atomic)
    extra = [] if reason is None else ['--reason', reason]
    result = invoke(tmp_path,*extra)
    assert result.exit_code == 0, result.output
    assert events == ['append','replace']
    assert tick.calls == 1
    path=tmp_path/'account_state/contest_streak.manual.json'
    body=json.loads(path.read_text())
    assert body == observation(**({} if reason is None else {'reason':reason}))
    log=tmp_path/'account_state/contest_manual_requests.jsonl'
    assert json.loads(log.read_text()) == dict(body, requested_at=body['recorded_at'])
    last='contest_streak: 12 (best 14, saver not_earned) for 2027-07-15, in effect today'
    assert result.output.splitlines()[-1] == last
    assert ('reason' in result.output.lower()) and (repr(reason) in result.output or reason is None)
    saved=snapshot(tmp_path); events.clear()
    result=invoke(tmp_path,*extra)
    assert result.exit_code == 0 and 'unchanged' in result.output
    assert result.output.splitlines()[-1] == last
    assert events == [] and snapshot(tmp_path) == saved
    assert all(snapshot(tmp_path)[key] == value for key,value in before.items())
    assert fetch_status.read_bytes() == b'fetch status sentinel'


REJECTIONS = [
    (['--streak','1','--saver','active'], '--saver'),
    (['--date','2027-07-15','--streak','1','--saver','active'], '--best-streak'),
    (['--date','2027-07-15','--streak','1','--best-streak','1'], '--saver'),
]
@pytest.mark.parametrize('extra,option', [
    (['--source-date','2027-07-15'],'--source-date'),(['--source','manual_cli'],'--source'),
    (['--username','x'],'--username'),(['--ttl-hours','24'],'--ttl-hours'),
    (['--override-expires-at','2099-01-01T00:00:00Z'],'--override-expires-at'),
    (['--saver-available'],'--saver'),(['--saver-unavailable'],'--saver'),
    (['--date','2027-07-14'],'--date'),(['--date','2027-07-16'],'--date'),
    (['--streak','-1'],'--streak'),(['--best-streak','11'],'--best-streak'),
    (['--best-streak','401'],'--best-streak'),
])
@pytest.mark.parametrize('existing',[False,True])
def test_rejections_before_reads_and_writes(tmp_path, monkeypatch, extra, option,existing):
    clock(monkeypatch,NOW)
    if existing: write(tmp_path,observation())
    before=snapshot(tmp_path)
    reads=[]
    original=Path.read_text
    original_path_open=Path.open
    original_open=builtins.open
    def read(path,*a,**kw):
        reads.append(path)
        return original(path,*a,**kw)
    def path_open(path,mode='r',*a,**kw):
        if 'r' in mode: reads.append(path)
        return original_path_open(path,mode,*a,**kw)
    def opened(path,mode='r',*a,**kw):
        if 'r' in mode and isinstance(path,(str,Path)):
            candidate=Path(path)
            if candidate.is_relative_to(tmp_path): reads.append(candidate)
        return original_open(path,mode,*a,**kw)
    with monkeypatch.context() as m:
        m.setattr(Path,'read_text',read)
        m.setattr(Path,'open',path_open)
        m.setattr(builtins,'open',opened)
        result=invoke(tmp_path,*extra)
    assert result.exit_code == 2, result.output
    assert option in result.output
    if option=='--date': assert '2027-07-15' in result.output
    assert reads == [] and snapshot(tmp_path)==before
    if not existing: assert not (tmp_path/'account_state').exists()


@pytest.mark.parametrize('args,option',REJECTIONS)
def test_missing_options_controls(tmp_path,monkeypatch,args,option):
    clock(monkeypatch,NOW)
    reader=Mock(side_effect=AssertionError('premature read'))
    monkeypatch.setattr(Path,'read_text',reader)
    result=CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(tmp_path),*args])
    assert result.exit_code==2 and option in result.output
    reader.assert_not_called()
    assert not (tmp_path/'account_state').exists()


def test_unpadded_date_and_reason_only_correction(tmp_path,monkeypatch):
    clock(monkeypatch,NOW)
    assert invoke(tmp_path,day='2027-7-15').exit_code==0
    assert invoke(tmp_path,'--reason','').exit_code==0
    assert invoke(tmp_path,'--reason','corrected').exit_code==0
    lines=(tmp_path/'account_state/contest_manual_requests.jsonl').read_text().splitlines()
    assert len(lines)==3 and json.loads(lines[0])['effective_date']=='2027-07-15'
    clock(monkeypatch,NOW+dt.timedelta(days=1))
    result=invoke(tmp_path,'--reason','corrected')
    assert result.exit_code==2 and '2027-07-16' in result.output
    assert len(lines)==len((tmp_path/'account_state/contest_manual_requests.jsonl').read_text().splitlines())


INVALID = [(2,'effective_date','20270715'),(2,'effective_date',None),(3,'season',True),
    (3,'season',2026),(4,'expires_at','2027-07-15T23:59:59.999999'),
    (4,'expires_at','2027-07-16T03:59:59Z'),(5,'active_streak',True),(5,'active_streak',401),
    (5,'active_streak',-1),(6,'best_streak',11),(6,'best_streak',True),(6,'best_streak',401),
    (7,'saver_state',[]),(7,'saver_state','uninitialized'),(8,'recorded_at','2027-07-15'),
    (9,'source',None),(10,'reason',None)]
@pytest.mark.parametrize('number,field,value',INVALID)
@pytest.mark.parametrize('delta',[0,1])
def test_ordered_invalid_dated_observation(tmp_path,monkeypatch,number,field,value,delta):
    path=write(tmp_path,observation(**{field:value}))
    automatic=write(tmp_path,dict(active_streak=3,source_date='2027-07-15'),'contest_streak.json')
    original=Path.read_text
    def read(candidate,*args,**kwargs):
        if candidate==automatic: raise AssertionError('invalid manual must be parsed first')
        return original(candidate,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(cs.ContestStateError) as err:
        cs.load_contest_streak_state(tmp_path,now=NOW+dt.timedelta(days=delta))
    assert str(path) in str(err.value) and field in str(err.value)
    assert f'check {number}' in str(err.value)


@pytest.mark.parametrize('field,number', [('effective_date',2),('season',3),('expires_at',4),
    ('active_streak',5),('best_streak',6),('saver_state',7),('recorded_at',8),('source',9)])
@pytest.mark.parametrize('delta',[0,1])
def test_missing_required_properties(tmp_path,field,number,delta):
    body=observation(); del body[field]
    write(tmp_path,body)
    with pytest.raises(cs.ContestStateError,match=f'check {number}'):
        cs.load_contest_streak_state(tmp_path,now=NOW+dt.timedelta(days=delta))


def test_first_failure_and_ignored_keys_and_offset(tmp_path):
    write(tmp_path,observation(season=False,active_streak=58))
    with pytest.raises(cs.ContestStateError,match='check 3'):
        cs.load_contest_streak_state(tmp_path,now=NOW)
    write(tmp_path,observation(source='',expires_at='2027-07-16T03:59:59.999999Z',
        override_expires_at=[],source_date={},streak='bad',saver_available=4,username=[]))
    state=cs.load_contest_streak_state(tmp_path,now=NOW)
    assert (state.streak,state.best_streak,state.source,state.source_date,state.saver_available)==(12,14,'',NOW.date(),None)
    assert state.effective_date==NOW.date() and state.saver_state=='not_earned'
    assert state.override_expires_at==dt.datetime(2027,7,16,3,59,59,999999,tzinfo=dt.timezone.utc)


@pytest.mark.parametrize('body',['{',b'\xff',[], 'directory', 'unreadable'])
def test_readability_errors_and_command_correction(tmp_path,monkeypatch,body):
    clock(monkeypatch,NOW)
    path=tmp_path/'account_state/contest_streak.manual.json'; path.parent.mkdir()
    if body=='directory': path.mkdir()
    elif isinstance(body,bytes): path.write_bytes(body)
    elif body=='unreadable': path.write_text(json.dumps(observation()))
    else: path.write_text(body if isinstance(body,str) else json.dumps(body))
    real=Path.read_text
    def read(p,*a,**kw):
        if p==path: raise PermissionError('unreadable')
        return real(p,*a,**kw)
    with monkeypatch.context() as m:
        if body=='unreadable': m.setattr(Path,'read_text',read)
        with pytest.raises(cs.ContestStateError): cs.load_contest_streak_state(tmp_path,now=NOW)
        result=invoke(tmp_path)
    assert result.exit_code==(1 if body=='directory' else 0),result.output
    if body=='directory': assert 'NOT saved' in result.output and 'appended' in result.output
    else: assert cs.load_decision_streak_state(tmp_path,now=NOW).streak==12


@pytest.mark.parametrize('day',['2027-07-14','2027-07-16'])
@pytest.mark.parametrize('auto',[False,True])
def test_out_of_date_is_absent_even_with_legacy_expiry(tmp_path,day,auto):
    write(tmp_path,observation(day,override_expires_at='2099-01-01T00:00:00Z'))
    if auto: write(tmp_path,dict(active_streak=3,source_date='2027-07-15'),'contest_streak.json')
    state=cs.load_contest_streak_state(tmp_path,now=NOW)
    assert (state.streak if state else None)==(3 if auto else None)
    assert not any('legacy/expired' in a.message for a in check(tmp_path,now=NOW))


@pytest.mark.parametrize('required',[False,True])
@pytest.mark.parametrize('auto',[None,{},dict(active_streak=401),'{'])
@pytest.mark.parametrize('saver',['not_earned','active','used'])
@pytest.mark.parametrize('streak',[0,9,10,15,16])
@pytest.mark.parametrize('flag',[None,'active','used','corrupt'])
def test_decision_observation_is_authority_and_isolated(tmp_path,monkeypatch,required,auto,saver,streak,flag):
    write(tmp_path,observation(active_streak=streak,best_streak=max(14,streak),saver_state=saver))
    if auto is not None:
        path=write(tmp_path,auto,'contest_streak.json')
        if auto=='{': path.write_text('{')
    if flag:
        path=write(tmp_path,dict(state=flag,season=2027),'saver_state.json')
        if flag=='corrupt': path.write_text('{')
    (tmp_path/'streak.json').write_text('{')
    (tmp_path/'2027-07-17.json').write_text('{"result":"miss"}')
    monkeypatch.setattr(cs,'load_saver_state',Mock(side_effect=AssertionError('flag read')))
    monkeypatch.setattr(cs,'contest_state_is_fresh',Mock(side_effect=AssertionError('local pick read')))
    state=cs.load_decision_streak_state(tmp_path,now=NOW,require_contest_state=required)
    assert (state.streak,state.contest_streak,state.source,state.status,state.best_status)==(streak,streak,'contest','fresh','trusted')
    assert state.contest_source_date==NOW.date() and state.saver_flag_state==saver
    assert state.saver_available==(saver=='active' or saver=='not_earned' and streak<10)


@pytest.mark.parametrize('day',['2027-07-15','2027-03-14','2027-11-07','2027-12-31','2028-01-01'])
def test_midnight_dst_year_boundary_and_clock_rollback(tmp_path,day):
    write(tmp_path,observation(day,saver_state='active'))
    date=dt.date.fromisoformat(day)
    start=dt.datetime.combine(date,dt.time.min,ET)
    end=dt.datetime.combine(date,dt.time.max,ET)
    for now,active in [(start,True),(end,True),(end+dt.timedelta(microseconds=1),False),
                       (start-dt.timedelta(microseconds=1),False),(start,True)]:
        state=cs.load_decision_streak_state(tmp_path,now=now)
        assert (state.source=='contest')==active
        if active:
            assert state.best_status=='trusted' and state.saver_flag_state=='active'
            assert state.status=='fresh' and state.saver_available


@pytest.mark.parametrize('operation',['reader','decision','health','command'])
def test_one_captured_clock_across_midnight(tmp_path,monkeypatch,operation):
    end=dt.datetime(2027,12,31,23,59,59,999999,tzinfo=ET)
    write(tmp_path,observation('2027-12-31',saver_state='active'))
    tick=clock(monkeypatch,end,end+dt.timedelta(microseconds=1))
    if operation=='reader': assert cs.load_contest_streak_state(tmp_path).effective_date==end.date()
    elif operation=='decision':
        state=cs.load_decision_streak_state(tmp_path)
        assert state.source=='contest' and state.best_status=='trusted' and state.saver_available
    elif operation=='health': assert check(tmp_path,expected=True)==[]
    else:
        result=invoke(tmp_path,day='2027-12-31',saver='used')
        assert result.exit_code==0 and 'in effect today' in result.output
        body=json.loads((tmp_path/'account_state/contest_streak.manual.json').read_text())
        assert body['season']==2027 and body['recorded_at']=='2028-01-01T04:59:59.999999Z'
    assert tick.calls==1


@pytest.mark.parametrize('expected',[False,True])
@pytest.mark.parametrize('saver',['not_earned','active','used'])
@pytest.mark.parametrize('streak',[0,9,10,12,15,16,57])
@pytest.mark.parametrize('flag',[None,'corrupt'])
def test_health_observation_saver_without_flag_read(tmp_path,monkeypatch,expected,saver,streak,flag):
    write(tmp_path,observation(active_streak=streak,best_streak=max(streak,14),saver_state=saver))
    if flag: (tmp_path/'account_state/saver_state.json').write_text('{')
    monkeypatch.setattr(flags,'load_saver_state',Mock(side_effect=AssertionError('flag read')))
    if saver=='used':
        bounded_read=Mock(wraps=flags._read_saver_state)
        monkeypatch.setattr(flags,'_read_saver_state',bounded_read)
    alerts=check(tmp_path,now=NOW,expected=expected)
    if saver=='used':
        bounded_read.assert_called_once_with(tmp_path,season=2027,historical=True)
    warnings=[a for a in alerts if a.source=='saver_state']
    assert len(warnings)==int(saver=='not_earned' and 10<=streak<=15)
    assert not any('legacy/expired' in a.message for a in alerts)
    if warnings:
        msg=warnings[0].message
        assert warnings[0].level=='WARN'
        for fragment in ['dated observation','2027',str(streak),'counting the saver as spent',
                         'ONE','changing the saver flag does not correct this observation']:
            assert fragment.lower() in msg.lower()


def test_warning_commands_pasted_into_real_cli(tmp_path,monkeypatch):
    tmp_path=tmp_path/'picks with spaces'
    clock(monkeypatch,NOW)
    write(tmp_path,observation())
    msg=next(a.message for a in check(tmp_path,now=NOW) if a.source=='saver_state')
    commands=[s for s in msg.split('`')[1::2] if s.startswith('bts set-contest-streak')]
    assert len(commands)==3
    for command_text,saver in zip(commands[:2],['active','used']):
        assert '--picks-dir' in command_text and str(tmp_path) in command_text
        result=CliRunner().invoke(cli,shlex.split(command_text)[1:])
        assert result.exit_code==0,result.output
        state=cs.load_decision_streak_state(tmp_path,now=NOW)
        assert state.saver_flag_state==saver and state.saver_available==(saver=='active')
        assert not any(a.source=='saver_state' for a in check(tmp_path,now=NOW))
    assert '--saver not_earned' in commands[2] and '--picks-dir' in commands[2]


@pytest.mark.parametrize('failure',['open','write','flush','close','replace'])
def test_failed_append_or_replacement_preserves_old_authority_and_retry(tmp_path,monkeypatch,failure):
    clock(monkeypatch,NOW)
    path=write(tmp_path,observation(saver_state='active'))
    for name in ['saver_state.json','contest_streak.json','contest_fetch_status.json']:
        write(tmp_path,dict(sentinel=name),name)
    before=snapshot(tmp_path)
    fetch_status=tmp_path.parent/'health_state/contest_streak_fetch_status.json'
    fetch_status.parent.mkdir(exist_ok=True)
    fetch_status.write_bytes(b'fetch status sentinel')
    original_open=builtins.open
    atomic=Mock(wraps=util.atomic_write_text)
    class FailingAppend:
        def __init__(self,path): self.file=original_open(path,'a',encoding='utf-8')
        def __enter__(self): return self
        def write(self,text):
            self.file.write(text[:20] if failure=='write' else text)
            if failure=='write': raise OSError('partial append')
        def flush(self):
            self.file.flush()
            if failure=='flush': raise OSError('flush failed')
        def __exit__(self,*args):
            self.file.close()
            if failure=='close': raise OSError('close failed')
    def opened(p,mode='r',*a,**kw):
        if str(p).endswith('contest_manual_requests.jsonl') and mode=='a':
            if failure=='open': raise OSError('open failed')
            return FailingAppend(p)
        return original_open(p,mode,*a,**kw)
    with monkeypatch.context() as m:
        m.setattr(util,'atomic_write_text',atomic)
        if failure=='replace': atomic.side_effect=OSError('replace failed')
        else: m.setattr(builtins,'open',opened)
        result=invoke(tmp_path,saver='used')
    assert result.exit_code==1,result.output
    if failure=='replace':
        assert 'NOT saved' in result.output and 'appended' in result.output
        assert atomic.call_count==1
    else:
        assert 'not written' in result.output and 'partial' in result.output
        atomic.assert_not_called()
    assert all(snapshot(tmp_path)[key]==value for key,value in before.items())
    assert fetch_status.read_bytes()==b'fetch status sentinel'
    assert cs.load_decision_streak_state(tmp_path,now=NOW).saver_flag_state=='active'
    assert invoke(tmp_path,saver='used').exit_code==0
    saved=snapshot(tmp_path)
    assert invoke(tmp_path,saver='used').exit_code==0 and snapshot(tmp_path)==saved
    assert json.loads(path.read_text())['saver_state']=='used'


@pytest.mark.parametrize('old',['legacy','invalid','yesterday','tomorrow'])
def test_corrections_replace_existing_kind(tmp_path,monkeypatch,old):
    clock(monkeypatch,NOW)
    body={'active_streak':12} if old=='legacy' else observation(
        '2027-07-14' if old=='yesterday' else '2027-07-16' if old=='tomorrow' else '2027-07-15')
    if old=='invalid': body['reason']=None
    write(tmp_path,body)
    assert invoke(tmp_path).exit_code==0
    assert json.loads((tmp_path/'account_state/contest_streak.manual.json').read_text())==observation()


def test_three_day_intermediate_sequence(tmp_path,monkeypatch):
    write(tmp_path,dict(active_streak=22,best_streak=30,source_date='2026-09-30'),'contest_streak.json')
    write(tmp_path,dict(state='active',season=2026),'saver_state.json')
    for day,streak in [('2027-03-24',0),('2027-03-25',None),('2027-03-26',2)]:
        now=dt.datetime.fromisoformat(day+'T20:00:00-04:00'); clock(monkeypatch,now)
        if streak is not None:
            result=invoke(tmp_path,day=day,streak=streak,best=streak)
            assert result.exit_code==0,result.output
        if streak is None:
            with pytest.raises(cs.ContestStateError,match='absent'):
                cs.load_decision_streak_state(tmp_path,now=now,require_contest_state=True)
            # Optional mode keeps the intermediate CS4a acceptance control.
            state=cs.load_decision_streak_state(tmp_path,now=now,require_contest_state=False)
        else:
            state=cs.load_decision_streak_state(tmp_path,now=now,require_contest_state=True)
        assert state.streak==(22 if streak is None else streak)
        assert state.saver_flag_state==('uninitialized' if streak is None else 'not_earned')
        assert state.saver_available==(streak is not None)


@pytest.mark.parametrize('point',['read','append','replace','output','noop-output'])
def test_deterministic_two_writer_interleavings(tmp_path,monkeypatch,point):
    """Step B through A's read/append/replace/output with nested real CLI calls."""
    import click
    clock(monkeypatch,NOW)
    path=write(tmp_path,observation(saver_state='active'))
    old_read,old_open,old_atomic,old_echo=Path.read_text,builtins.open,util.atomic_write_text,click.echo
    entered=False
    log=tmp_path/'account_state/contest_manual_requests.jsonl'
    def run_b():
        nonlocal entered
        if entered: return
        entered=True
        result=invoke(tmp_path,saver='used')
        assert result.exit_code==0,result.output
    def read(p,*a,**kw):
        saved=old_read(p,*a,**kw)
        if p==path and point=='read': run_b()
        return saved
    class Append:
        def __init__(self): self.file=old_open(log,'a',encoding='utf-8')
        def __enter__(self): return self
        def write(self,text): self.file.write(text)
        def flush(self): self.file.flush()
        def __exit__(self,*args):
            self.file.close()
            if point=='append': run_b()
    def opened(p,mode='r',*a,**kw):
        if str(p)==str(log) and mode=='a': return Append()
        return old_open(p,mode,*a,**kw)
    def atomic(p,text):
        old_atomic(p,text)
        if point=='replace': run_b()
    def echo(*a,**kw):
        if point in ['output','noop-output']: run_b()
        old_echo(*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(builtins,'open',opened)
    monkeypatch.setattr(util,'atomic_write_text',atomic)
    monkeypatch.setattr(click,'echo',echo)
    a_saver='active' if point=='noop-output' else 'not_earned'
    result=invoke(tmp_path,saver=a_saver)
    assert result.exit_code==0,result.output
    assert result.output.splitlines()[-1]==f'contest_streak: 12 (best 14, saver {a_saver}) for 2027-07-15, in effect today'
    assert entered
    final=json.loads(old_read(path))
    assert final['saver_state']==('not_earned' if point in ['read','append'] else 'used')
    entries=[json.loads(s)['saver_state'] for s in old_read(log).splitlines()]
    assert entries==(['used','not_earned'] if point=='read' else
                     ['not_earned','used'] if point!='noop-output' else ['used'])


@pytest.mark.parametrize('point',['before-replace','after-replace'])
def test_interruption_retry_boundaries(tmp_path,monkeypatch,point):
    import click
    clock(monkeypatch,NOW)
    path=write(tmp_path,observation(saver_state='active'))
    def interrupted(path,text): raise KeyboardInterrupt()
    def output(*args,**kwargs): raise KeyboardInterrupt()
    with monkeypatch.context() as m:
        if point=='before-replace': m.setattr(util,'atomic_write_text',interrupted)
        else: m.setattr(click,'echo',output)
        result=invoke(tmp_path,saver='used')
    assert result.exit_code!=0
    assert json.loads(path.read_text())['saver_state']==('active' if point=='before-replace' else 'used')
    result=invoke(tmp_path,saver='used')
    assert result.exit_code==0
    assert ('unchanged' in result.output)==(point=='after-replace')
    lines=(tmp_path/'account_state/contest_manual_requests.jsonl').read_text().splitlines()
    assert len(lines)==(2 if point=='before-replace' else 1)


def test_legacy_fixed_clock_bytes_and_output_control(tmp_path,monkeypatch):
    clock(monkeypatch,NOW)
    result=CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(tmp_path),
        '--streak','7','--best-streak','9','--saver-unavailable','--reason',''])
    assert result.exit_code==0,result.output
    expected=dict(schema_version='bts_contest_streak_manual_v2',active_streak=7,best_streak=9,
        source='manual_cli',source_date='2027-07-15',recorded_at='2027-07-15T16:00:00Z',
        override_expires_at='2027-07-16T16:00:00Z')
    path=tmp_path/'account_state/contest_streak.manual.json'
    assert path.read_text()==json.dumps(expected,indent=2,sort_keys=True)+'\n'
    assert result.output==('Note: --saver-available/--saver-unavailable is deprecated and no longer '
        'affects the live saver (use `bts saver-state`); ignoring it.\n'
        f'Wrote {path}: active_streak=7 source_date=2027-07-15 override_expires_at=2027-07-16T16:00:00Z\n')
    saver_path=tmp_path/'account_state/saver_state.json'
    assert saver_path.read_text()==json.dumps(dict(season=2027,state='not_earned',
        updated_at='2027-07-15T16:00:00+00:00',source='auto_earn'))
    assert not (tmp_path/'account_state/contest_manual_requests.jsonl').exists()


@pytest.mark.parametrize('schema',[None,'unknown','bts_contest_streak_manual_v2'])
def test_wrong_schema_remains_lax_legacy_control(tmp_path,schema):
    write(tmp_path,observation(schema_version=schema,reason=None,season=False,
        expires_at=[],saver_state=[],override_expires_at='2099-01-01T00:00:00Z'))
    state=cs.load_contest_streak_state(tmp_path,now=NOW)
    assert state.streak==12 and state.source_date is None
    assert state.override_expires_at.year==2099
    assert getattr(state,'effective_date',None) is None
    assert getattr(state,'saver_state',None) is None


def test_automatic_v3_schema_still_uses_automatic_parser_control(tmp_path):
    write(tmp_path,observation(), 'contest_streak.json')
    state=cs.load_contest_streak_state(tmp_path,now=NOW)
    assert state.streak==12 and state.source_date is None
    assert getattr(state,'effective_date',None) is None
    assert getattr(state,'saver_state',None) is None
    assert cs.load_contest_streak_state(tmp_path,now=NOW,require_dated_auto=True)==state
    body=observation()
    del body['recorded_at']
    write(tmp_path,body,'contest_streak.json')
    with pytest.raises(cs.ContestStateError,match='undated'):
        cs.load_contest_streak_state(tmp_path,now=NOW,require_dated_auto=True)


def test_noop_compares_valid_raw_snapshot_identity_only(tmp_path,monkeypatch):
    clock(monkeypatch,NOW)
    write(tmp_path,observation(source='',recorded_at='2027-01-01T00:00:00+03:00',
        expires_at='2027-07-16T03:59:59.999999Z',override_expires_at=[]))
    before=snapshot(tmp_path)
    reader=Mock(side_effect=AssertionError('must read raw JSON, not selection'))
    monkeypatch.setattr(cs,'load_contest_streak_state',reader)
    atomic=Mock(side_effect=AssertionError('no-op must not write'))
    monkeypatch.setattr(util,'atomic_write_text',atomic)
    result=invoke(tmp_path)
    assert result.exit_code==0 and 'unchanged' in result.output,result.output
    assert snapshot(tmp_path)==before
    reader.assert_not_called(); atomic.assert_not_called()


def test_legacy_option_defaults_are_not_explicit_commandline_options(tmp_path,monkeypatch):
    clock(monkeypatch,NOW)
    result=CliRunner().invoke(cli,command(tmp_path),default_map={'set-contest-streak':{
        'source':'from defaults','ttl_hours':36,'source_date':'2020-01-01',
        'username':'default name','saver_available':True,
        'override_expires_at':'2099-01-01T00:00:00Z'}})
    assert result.exit_code==0,result.output
    assert json.loads((tmp_path/'account_state/contest_streak.manual.json').read_text())==observation()


def test_shared_date_predicate_is_used_by_all_four_operations(tmp_path,monkeypatch):
    clock(monkeypatch,NOW)
    write(tmp_path,observation())
    predicate=Mock(wraps=cs.dated_observation_in_effect)
    monkeypatch.setattr(cs,'dated_observation_in_effect',predicate)
    monkeypatch.setattr(health,'dated_observation_in_effect',predicate)
    for operation in [lambda: cs.load_contest_streak_state(tmp_path,now=NOW),
                      lambda: cs.load_decision_streak_state(tmp_path,now=NOW),
                      lambda: check(tmp_path,now=NOW),lambda: invoke(tmp_path)]:
        predicate.reset_mock()
        operation()
        assert predicate.called
        assert all(call.args[1]==NOW for call in predicate.call_args_list)

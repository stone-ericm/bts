"""Real strategy/orchestrator/scheduler and delivery against each refusal class."""
import contextlib
import datetime as dt
import io
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import review_support as audit
from matrix import OUT, ROOT, NOW, ID, contract, automatic, dated, install

variant=sys.argv[1]
sys.path.insert(0,str(ROOT/"src" if variant=="branch" else OUT/"base/src"))
audit.freeze(NOW)
import pandas as pd
import bts.cli as cli
import bts.contest_state as cs
import bts.orchestrator as orch
import bts.scheduler as sch
import bts.picks as picks
import bts.strategy as strategy
from bts.daily_decision import write_decision
from click.testing import CliRunner
logging.disable(logging.CRITICAL)
OUTCOME={"source":cs.__file__,"cases":[],"races":[],"later_reads":[],"grading":[]}
DAY="2027-07-15"
FRAME=pd.DataFrame([dict(batter_name="Offline",batter_id=1,team="NYM",pitcher_name="P",pitcher_id=2,
                         p_game_hit=.999,p_hit_pa=.3,lineup=1,game_pk=999,game_time="2027-07-15T23:05:00Z",flags="")])
DETAILED={999:dict(abstract="P",detailed="Scheduled",code="S")}
BASE=OUT/("callers-"+variant+"-files")
BASE.mkdir(exist_ok=True)


def daily():
    return picks.DailyPick(date=DAY,run_time=NOW.isoformat(),pick=picks.pick_from_row(FRAME.iloc[0]),double_down=None,runner_up=None)


def state():
    return sch.SchedulerState(date=DAY,schedule_fetched_at="t",games=[],confirmed_game_pks=[],runs_completed=[],
                              pick_locked=False,pick_locked_at=None,result_status=None,next_wakeup=None)


def setup(label,sw,manual,auto,mode="dm",gate="scheduler"):
    p=BASE/label
    (p/"account_state").mkdir(parents=True,exist_ok=True)
    for name,value in [("season_contract.json",sw),("contest_streak.manual.json",manual),("contest_streak.json",auto)]:
        install(p/"account_state"/name,value)
    config=dict(orchestrator=dict(picks_dir=str(p),heartbeat_path=str(p/"heartbeat"),models_dir=str(p/"no-models")),
                scheduler=dict(pick_delivery=mode,contest_state_required=gate=="scheduler",live_forward_capture_on_lock=False),
                health_checks=dict(contest_state_expected=gate=="health",enabled=False),
                bluesky=dict(dm_recipient="offline"),tiers=[])
    return p,config


def attempt(fn):
    try:
        result=fn()
        return dict(ok=True,value=str(result))
    except Exception as exc:
        return dict(ok=False,error=type(exc).__name__,message=str(exc))


CASES={
    "covered":(contract(),None,automatic(),True),
    "dated-corrupt-switch":([],dated(),automatic(),True),
    "absent":(None,None,automatic(),False),
    "revoked":(contract(status="revoked",revoked_at=NOW.isoformat()),None,automatic(),False),
    "other-season":(contract(season=2026),None,automatic(),False),
    "corrupt":([],None,automatic(),False),
    "legacy":(contract(),dict(active_streak=5,override_expires_at="2027-07-16T16:00:00Z"),automatic(),False),
    "legacy-auto-schema":(contract(),automatic(override_expires_at="2027-07-16T16:00:00Z"),automatic(),False),
    "missing":(contract(),None,None,False),
    "v1":(contract(),None,automatic(schema_version="bts_contest_streak_auto_v1"),False),
    "wrong-season":(contract(),None,automatic(season=2026),False),
    "wrong-id":(contract(),None,automatic(activation_id="b"*32),False),
    "undated":(contract(),None,automatic(source_date=None,recorded_at=None),False),
    "invalid":(contract(),None,automatic(active_streak=58,best_streak=58),False),
}


def run():
    posts,dms,alerts=[],[],[]
    with contextlib.ExitStack() as stack:
        for target,value in [
            ("bts.orchestrator.run_cascade",lambda *a:(FRAME,"captured-predictions")),
            ("bts.picks.get_game_statuses_detailed",lambda *a:DETAILED),
            ("bts.strategy.get_game_statuses_detailed",lambda *a:DETAILED),
            ("bts.picks.get_game_statuses",lambda *a:{999:"P"}),
            ("bts.strategy.get_game_statuses",lambda *a:{999:"P"}),
            ("bts.scheduler.count_new_confirmations",lambda *a:0),
            ("bts.scheduler.heartbeat_watchdog",lambda *a,**kw:contextlib.nullcontext()),
            ("bts.scheduler._now_et",lambda:NOW),
            ("bts.cli._now_et",lambda:NOW),
            ("bts.scheduler._void_committed_games",lambda *a:{}),
            ("bts.scheduler._trigger_live_forward_capture_on_lock",lambda *a:None),
            ("bts.scheduler._alert_contest_state_failure",lambda *a:alerts.append(str(a[-1]))),
            ("bts.dm.send_dm",lambda *a,**kw:dms.append(a) or "captured"),
            ("bts.health.alert.send_dm",lambda *a,**kw:dms.append(a) or "captured"),
            ("bts.posting.post_to_bluesky",lambda *a,**kw:posts.append(a) or "at://captured"),
            ("bts.posting.should_post_now",lambda *a:True),
        ]:
            stack.enter_context(patch(target,value))
        for name,(sw,manual,auto,accepted) in CASES.items():
            expected=accepted if variant=="branch" else name not in ["missing","undated","invalid"]
            for gate in ["scheduler","health"]:
                for path in ["run_and_pick","orchestrate","single_check","fallback","dm","public"]:
                    p,config=setup(name+"-"+gate+"-"+path,sw,manual,auto,mode="public" if path=="public" else "dm",gate=gate)
                    posts.clear();dms.clear();alerts.clear()
                    st=state();d=daily()
                    with patch.object(orch,"load_config",lambda *a:config):
                        if path=="run_and_pick":
                            result=attempt(lambda:orch.run_and_pick(config,DAY))
                            actual=result["ok"]
                        elif path=="orchestrate":
                            result=attempt(lambda:orch.orchestrate(p/"unused",DAY))
                            actual=bool(posts)
                        elif path=="single_check":
                            raw=sch.run_single_check(DAY,[],set(),config,.03)
                            result=dict(pick=raw["pick_result"] is not None)
                            actual=result["pick"]
                        elif path=="fallback":
                            result=attempt(lambda:sch._refresh_pick_at_fallback_decision(config,DAY,d,.03))
                            actual=result["ok"]
                        else:
                            result=dict(returned=sch._deliver_and_lock_pick(d,config,p,st,DAY,path))
                            actual=result["returned"]
                    assert actual==expected,(variant,name,gate,path,result)
                    if not expected:
                        assert not posts and not st.pick_locked and not d.delivery_attempted
                        assert not (p/DAY/"decision.json").exists()
                        assert not (p/(DAY+".json")).exists()
                        assert not dms or (path=="orchestrate" and all("No pick made" in args[1] for args in dms))
                    OUTCOME["cases"].append(dict(name=name,gate=gate,path=path,result=result,
                                                post_count=len(posts),dms=dms.copy(),alerts=alerts.copy(),locked=st.pick_locked))
        # Refusal after a successful selection, and after the delivery's own snapshot.
        for when in ["before_delivery","after_delivery_read","already_delivered","private","crash"]:
            p,config=setup("race-"+when,contract(),None,automatic(),mode="private" if when=="private" else "dm")
            d=daily();st=state();posts.clear();dms.clear();alerts.clear()
            if when=="already_delivered":
                d.notification_sent=True;d.notification_channel="bluesky_dm";d.notification_id="old";d.delivered_at=NOW.isoformat()
            if when=="crash": d.delivery_attempted=True
            if when!="after_delivery_read": install(p/"account_state/season_contract.json",contract(status="revoked",revoked_at=NOW.isoformat()))
            original=cs.load_decision_streak_state
            reads=[]
            def read(*a,**kw):
                reads.append(kw)
                result=original(*a,**kw)
                install(p/"account_state/season_contract.json",contract(status="revoked",revoked_at=NOW.isoformat()))
                return result
            with patch.object(cs,"load_decision_streak_state",read):
                result=sch._deliver_and_lock_pick(d,config,p,st,DAY,when)
            if variant=="branch" and when=="before_delivery": assert not result and not st.pick_locked and not dms
            if when=="after_delivery_read": assert result and d.notification_sent
            if when in ["already_delivered","private","crash"]: assert not reads and st.pick_locked
            OUTCOME["races"].append(dict(when=when,result=result,locked=st.pick_locked,reads=reads,dms=dms.copy()))
        # Later orchestrator reads propagate the REAL error; the new-pick branch
        # has already saved a preview, as the row explicitly preserves.
        for kind in ["skip","locked","new"]:
            p,config=setup("later-"+kind,contract(status="revoked",revoked_at=NOW.isoformat()),None,automatic())
            d=daily();posts.clear();dms.clear()
            sel=SimpleNamespace(pick_result=None if kind=="skip" else strategy.PickResult(d,locked=kind=="locked"))
            with patch.object(orch,"load_config",lambda *a:config),patch.object(orch,"run_and_pick",lambda *a:(FRAME,sel,"captured")):
                result=attempt(lambda:orch.orchestrate(p/"unused",DAY))
            if variant=="branch":
                assert result.get("error")=="ContestStateError" and not posts and not dms
            OUTCOME["later_reads"].append(dict(kind=kind,result=result,preview=(p/(DAY+".json")).exists(),posts=posts.copy()))
        # The state loader is deliberately unavailable, even for optional reads:
        # result grading, streak updates, captured skips and day-outcome do not use it.
        with patch.object(cs,"load_decision_streak_state",side_effect=AssertionError("grading read live state")):
            p,config=setup("grading",None,None,None)
            d=daily();d.notification_sent=True;d.notification_channel="bluesky_dm";d.notification_id="old";d.delivered_at=NOW.isoformat()
            picks.save_pick(d,p)
            write_decision(DAY,p,action="single",source="mdp",streak=7,saver_available=False,
                           state_source="contest",state_status="fresh",delivery_status="delivered",scoreable=True)
            with patch.object(picks,"resolve_daily_slot_results",lambda *a:{"primary":"hit"}):
                r=CliRunner().invoke(cli.cli,["check-results","--date",DAY,"--picks-dir",str(p)])
            assert r.exit_code==0,(r.output,r.exception)
            assert json.loads((p/"streak.json").read_text())["streak"]==1
            skiproot,config=setup("skip-outcome",None,None,None,mode="private")
            st=state();st.final_skip_candidate=dict(streak=7,saver_available=False,state_source="contest",state_status="fresh")
            sch._write_endofday_skip(skiproot,DAY,st)
            assert json.loads((skiproot/DAY/"decision.json").read_text())["action"]=="skip"
            config["scheduler"]["entry_intent"]="research"
            with patch.object(orch,"load_config",lambda *a:config),patch.object(sch,"fetch_schedule",lambda *a:[dict(gamePk=999,status={})]):
                r=CliRunner().invoke(cli.cli,["day-outcome","--date",DAY,"--config",str(skiproot/"unused"),"--picks-dir",str(skiproot)])
            assert r.exit_code==0,(r.output,r.exception)
            OUTCOME["grading"].append(dict(streak=1,outcome_exit=r.exit_code,output=r.output))


try:
    with (OUT/("callers-"+variant+".log")).open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        run()
finally:
    (OUT/("callers-"+variant+".json")).write_text(json.dumps(OUTCOME,indent=2,default=str))
print(variant,{k:len(v) for k,v in OUTCOME.items() if isinstance(v,list)})

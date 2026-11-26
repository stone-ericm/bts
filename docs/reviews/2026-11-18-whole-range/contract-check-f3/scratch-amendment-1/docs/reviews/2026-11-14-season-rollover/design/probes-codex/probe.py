"""Offline design attacks. All contest values/outcomes are synthetic.

The proposal classifier below is a transcription of design-carbon.md:83-92,
NOT proposed production code and NOT an implementation test of the design.
"""
import datetime as dt
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts.cli as cli
import bts.contest_fetch as fetch
import bts.contest_state as state
import bts.leaderboard.auth as auth
import bts.orchestrator as orchestrator
import bts.scheduler as scheduler
import bts.strategy as strategy
from bts.health.contest_state import check
from bts.picks import load_pick
from bts.saver_state import load_saver_state, transition_saver_state, maybe_auto_earn_saver

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
out = {}


def clock(iso):
    return dt.datetime.fromisoformat(iso).replace(tzinfo=ET)


def capture(fn):
    try:
        value = fn()
        return vars(value) if hasattr(value, "__dataclass_fields__") else value
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def work(name):
    folder = HERE / "cases"
    folder.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=name+"-", dir=folder))


def fetch_cli(picks, now, active, best, predictions, rounds):
    with pytest.MonkeyPatch.context() as mp:
        class Clock(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return now.astimezone(tz) if tz else now.replace(tzinfo=None)
        mp.setattr(dt, "datetime", Clock)
        mp.setattr(auth, "load_session_cookies", lambda: {})
        mp.setattr(auth, "extract_uid", lambda _: "fixture")
        mp.setattr(auth, "fetch_login_session", lambda **kw: auth.AuthSession(xsid="fixture", user_id=1, username="fixture"))
        payload = {"activeStreak": active, "seasonBestStreak": best}
        if predictions is not None:
            payload["predictions"] = predictions
        mp.setattr(fetch, "fetch_profile", lambda *a: payload)
        mp.setattr(cli, "_fetch_rounds", lambda: rounds)
        r = CliRunner().invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
        assert r.exit_code == 0, (r.output, r.exception)
    return json.loads((picks / "account_state/contest_streak.json").read_text())


def proposal(obs, predictions, rounds, now, prior=None):
    y, today = now.year, now.date()
    prior = prior or {}
    settled = [p for p in predictions if p.get("result") in fetch.RESOLVED]
    mapped = [rounds[int(p["roundId"])] for p in settled if int(p["roundId"]) in rounds]
    years = {d.year for d in rounds.values()}
    earliest = min((d for d in rounds.values() if d.year == y), default=None)
    source = fetch.derive_source_date(predictions, rounds)
    lpsd = prior.get("last_proven_source_date")
    if lpsd and dt.date.fromisoformat(lpsd).year != y:
        lpsd = None
    if any(d.year == y and d <= today + dt.timedelta(days=1) for d in mapped):
        kind, proven, lpsd = "current_settled", y, source.isoformat()
    elif not settled and y in years and obs["active_streak"] == obs["best_streak"] == 0:
        kind, proven = "empty_reset", y
    elif not settled and y in years and (obs["active_streak"] > 0 or obs["best_streak"] > 0):
        kind = "counter_only_unsettled"
        proven = y if prior.get("proven_season") == y and earliest and today >= earliest else None
    elif settled and (not mapped or all(d > today + dt.timedelta(days=1) for d in mapped)):
        kind, proven = "unmapped_settled", y if lpsd else None
    elif (mapped and source.year < y) or y not in years:
        kind, proven = "prior_settled_only", source.year if source else max(years)
    else:
        kind, proven = "NO_ROW", None
    return {"provenance": kind, "proven_season": proven, "last_proven_source_date": lpsd,
            "streak": obs["active_streak"], "best": obs["best_streak"]}


def action(s, saver=True, day="2027-03-28", p=.75):
    ctx = strategy.DecisionContext(p, p-.01, True, day, True, strategy._load_mdp())
    return strategy.resolve_policy_decision(ctx, s, saver).action


# No accidental socket access if a transport patch is missed.
with patch("socket.socket.connect", side_effect=AssertionError("NETWORK FORBIDDEN")), patch("bts.dm.send_dm", side_effect=AssertionError("DM FORBIDDEN")):
    p = work("reset_then_old_counter")
    rounds = {9001: dt.date(2027, 3, 24)}
    first = fetch_cli(p, clock("2027-03-22T10:30"), 0, 0, [], rounds)
    anchor = proposal(first, [], rounds, clock("2027-03-22T10:30"))
    later = fetch_cli(p, clock("2027-03-28T13:30"), 7, 18, [], rounds)
    promoted = proposal(later, [], rounds, clock("2027-03-28T13:30"), anchor)
    assert promoted["proven_season"] == 2027
    out["empty_reset_launders_old_counter"] = {"first": anchor, "next": promoted,
        "action_at_stale_7": action(7), "action_at_true_5": action(5)}

    p = work("mixed_season_fields")
    preds = [{"roundId": 9004, "result": "hit"}]
    mixed_rounds = {9001:dt.date(2027,3,24),9004:dt.date(2027,3,27)}
    obs = fetch_cli(p, clock("2027-03-28T10:30"), 7, 7, preds, mixed_rounds)
    mixed = proposal(obs, preds, mixed_rounds, clock("2027-03-28T10:30"))
    assert mixed["provenance"] == "current_settled"
    out["current_row_is_not_counter_season_proof"] = mixed

    p = work("reused_round_id")
    preds = [{"roundId": 823, "result": "hit"}]
    reuse = {823: dt.date(2027, 3, 24)}
    obs = fetch_cli(p, clock("2027-03-28T13:30"), 7, 7, preds, reuse)
    got = proposal(obs, preds, reuse, clock("2027-03-28T13:30"))
    assert got["provenance"] == "current_settled"
    out["same_or_past_date_id_collision"] = got

    p = work("newer_weak_fetch")
    prior = {"proven_season": 2027, "last_proven_source_date": "2027-05-31"}
    obs = fetch_cli(p, clock("2027-06-01T13:30"), 14, 18,
        [{"roundId": 1007, "result": "hit"}], {9001: dt.date(2027, 3, 24)})
    got = proposal(obs, [{"roundId": 1007, "result": "hit"}], {9001: dt.date(2027, 3, 24)}, clock("2027-06-01T13:30"), prior)
    assert got["proven_season"] == 2027 and got["streak"] == 14
    out["old_settlement_proof_launders_weaker_snapshot"] = got

    p = work("saver_restore")
    transition_saver_state(p, expected_prior="uninitialized", new_state="not_earned", season=2027, source="fixture")
    backup = (p / "account_state/saver_state.json").read_bytes()
    maybe_auto_earn_saver(p, best_streak=12, season=2027)
    transition_saver_state(p, expected_prior="active", new_state="used", season=2027, source="fixture_save")
    before = load_saver_state(p, season=2027).state
    (p / "account_state/saver_state.json").write_bytes(backup)
    after_restore = load_saver_state(p, season=2027).state
    maybe_auto_earn_saver(p, best_streak=12, season=2027)  # design permits this on current_settled
    after_fetch = load_saver_state(p, season=2027).state
    assert (before, after_restore, after_fetch) == ("used", "not_earned", "active")
    out["saver_same_season_restore"] = {"before": before, "restored": after_restore, "after_current_fetch": after_fetch,
        "true_saver": False, "policy_saver": True, "policy_date":"2027-06-01", "action_true": action(13, False, day="2027-06-01", p=.80),
        "action_restored": action(13, True, day="2027-06-01", p=.80)}

    p = work("corrupt_model_blocks_manual")
    fetch_cli(p,clock("2027-03-24T10:30"),0,0,[],rounds)
    result = CliRunner().invoke(cli.cli, ["set-contest-streak", "--streak", "0", "--best-streak", "0", "--source-date", "2027-03-24",
        "--override-expires-at", "2027-03-25T23:00:00Z", "--picks-dir", str(p)])
    assert result.exit_code == 0
    (p / "streak.json").write_bytes(b"\xff")
    got = capture(lambda: state.load_decision_streak_state(p, require_contest_state=True, now=clock("2027-03-24T12:00")))
    assert got["error"] == "UnicodeDecodeError"
    out["manual_not_universal_recovery"] = {"command_exit": result.exit_code, "decision": got}

    p = work("stale_current_year_restore")
    fetch_cli(p, clock("2027-03-26T10:30"), 3, 3,
        [{"roundId": 9002, "result": "hit"}], {9002: dt.date(2027, 3, 25)})
    got = state.load_decision_streak_state(p, require_contest_state=True, now=clock("2027-06-01T13:00"))
    alerts = check(p, expected=True, now=clock("2027-06-01T13:00"))
    assert got.status == "fresh" and not alerts
    out["old_same_year_backup_is_fresh_with_no_history"] = {"decision": vars(got), "health": [vars(a) for a in alerts]}

    # No-predictions return comes before the loader and its refusal DM.
    p = work("no_predictions_masks_refusal")
    calls = []
    with patch("bts.orchestrator.run_cascade", return_value=(None, None)), patch("bts.contest_state.load_decision_streak_state", side_effect=lambda *a,**k: calls.append("loader")):
        result = orchestrator.run_and_pick({"orchestrator":{"picks_dir":str(p)}, "tiers":[], "scheduler":{"contest_state_required":True}}, "2027-03-24")
    assert not calls and result == (None, None, None)
    out["no_predictions_no_contest_state_check"] = {"loader_calls": calls, "run_and_pick": result}

    # Real main day loop; virtual clock, no predictions, fake EOD health and transports.
    # The EOD check emits the actual contest-state finding, then an override arrives during idle.
    p = work("override_after_last_check")
    game = {"gamePk":9001,"gameDate":"2027-03-24T23:10:00Z", "status":{"abstractGameCode":"P","detailedState":"Scheduled"},
            "teams":{"away":{"team":{"name":"A"}},"home":{"team":{"name":"B"}}}}
    current = [clock("2027-03-24T10:00")]
    cascades, loader_calls, health_times, dm_attempts = [], [], [], []
    def cascade(*a, **k):
        cascades.append(current[0].isoformat())
        return None, None
    def sleep(seconds):
        current[0] += dt.timedelta(seconds=seconds)
    def health(**kwargs):
        health_times.append({"at":current[0].isoformat(), "alerts":[vars(a) for a in check(p, expected=True, now=current[0])]})
    def idle(*a):
        current[0] = clock("2027-03-24T18:40")
        r = CliRunner().invoke(cli.cli, ["set-contest-streak", "--streak","0","--best-streak","0", "--source-date","2027-03-24",
            "--override-expires-at","2027-03-25T23:00:00Z", "--picks-dir",str(p)])
        assert r.exit_code == 0
    config = {"orchestrator":{"picks_dir":str(p),"heartbeat_path":str(p/"heartbeat")}, "tiers":[],
        "scheduler":{"contest_state_required":True,"lineup_check_offset_min":60,"pick_delivery":"dm"},
        "bluesky":{"dm_recipient":"fixture"}, "health_checks":{"enabled":True}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scheduler,"_now_et",lambda:current[0])
        mp.setattr(scheduler,"fetch_schedule",lambda day:[game] if day=="2027-03-24" else [])
        mp.setattr(scheduler,"count_new_confirmations",lambda *a:0)
        mp.setattr(scheduler,"_watchdog_ping_sleep",sleep)
        mp.setattr(scheduler,"heartbeat_watchdog",lambda *a,**k:nullcontext())
        mp.setattr(scheduler,"write_heartbeat",lambda *a,**k:None)
        mp.setattr(scheduler,"notify_ready",lambda:None)
        mp.setattr(scheduler,"notify_watchdog",lambda:None)
        mp.setattr(scheduler,"_idle_until_next_wakeup",idle)
        mp.setattr(orchestrator,"run_cascade",cascade)
        mp.setattr(state,"load_decision_streak_state",lambda *a,**k:loader_calls.append(True))
        mp.setattr("bts.health.runner.run_all_checks",health)
        mp.setattr("bts.health.runner.read_systemd_nrestarts",lambda:0)
        mp.setattr("bts.health.alert.dispatch_dm_for_health_alerts",lambda *a,**k:dm_attempts.append(True))
        scheduler.run_day("2027-03-24",config)
    assert len(cascades)==3 and not loader_calls and not dm_attempts and load_pick("2027-03-24",p) is None
    out["override_does_not_wake_main_scheduler"]={"cascades":cascades,"loader_calls":loader_calls,"immediate_dispatch_calls":dm_attempts,
        "eod_health":health_times,"override_at":current[0].isoformat(),"pick_file":None}

    p = work("legacy_date_free_auto")
    (p/"account_state").mkdir()
    (p/"account_state/contest_streak.json").write_text('{"active_streak":14,"best_streak":18}')
    got = state.load_decision_streak_state(p,require_contest_state=True,now=clock("2027-03-24T12:00"))
    assert got.streak==14
    out["legacy_fixture_exception_is_live"] = vars(got)

    # This recovery command is syntactically accepted twice and extends a frozen zero.
    p = work("repeated_zero_override")
    invocations = []
    for day in ("2027-03-24", "2027-03-28", "2027-04-03"):
        result = CliRunner().invoke(cli.cli, ["set-contest-streak", "--streak", "0", "--best-streak", "0", "--source-date", day,
            "--override-expires-at", day+"T23:00:00Z", "--picks-dir", str(p)])
        assert result.exit_code == 0
        invocations.append({"day":day,"exit":result.exit_code})
    out["zero_override_renewal"] = {"commands": invocations,"s0":action(0),"s7":action(7),"s9":action(9)}

    # Demonstrate that effective live requirement is a config option, not a caller invariant.
    out["live_requirement_default"] = {"orchestrator":orchestrator._contest_state_required({}),"scheduler":scheduler._contest_state_required({})}

    # The proposal's year assignment can point forward solely from the round calendar.
    out["future_calendar_before_year_boundary"] = proposal({"active_streak":14,"best_streak":18}, [],
        {9001:dt.date(2027,3,24)},clock("2026-12-31T23:30"))

    p = work("u1_corrupt_but_valid_json")
    fetch_cli(p,clock("2027-06-01T10:30"),13,20,[{"roundId":9001,"result":"hit"}],{9001:dt.date(2027,5,31)})
    path = p/"account_state/contest_streak.json"
    data = path.read_text()
    assert '"best_streak": 20' in data
    path.write_text(data.replace('"best_streak": 20','"best_streak": 10'))
    got = state.load_decision_streak_state(p,require_contest_state=True,now=clock("2027-06-01T13:00"))
    assert got.streak==13 and got.best_streak==10 and got.best_status=="untrusted"
    out["u1_bad_best_reachable_after_corruption"] = vars(got)

    p = work("absent_predictions")
    obs = fetch_cli(p,clock("2027-03-22T10:30"),0,0,None,{9001:dt.date(2027,3,24)})
    out["missing_array_is_silently_empty"] = proposal(obs,[],{9001:dt.date(2027,3,24)},clock("2027-03-22T10:30"))

    from bts.health.alert import Alert, dispatch_dm_for_health_alerts
    from bts.health.attention import apply_warn_attention_policy
    p = work("warnings_and_dedup")
    warn = Alert("WARN","contest_state","override disagrees / saver needs init")
    with patch("bts.health.alert.send_dm") as sender:
        policy_alerts, attention = apply_warn_attention_policy([warn],state_path=p/"warnings.json",today=dt.date(2027,3,24))
        warned = dispatch_dm_for_health_alerts([warn]+policy_alerts,"fixture",warn_attention=attention,status_path=p/"delivery.json",now_et_date=dt.date(2027,3,24))
        assert not warned and sender.call_count==0
        for message in ("preseason no source date", "live selection refused"):
            dispatch_dm_for_health_alerts([Alert("CRITICAL","contest_state",message)],"fixture",status_path=p/"delivery.json",now_et_date=dt.date(2027,3,24))
        assert sender.call_count==1
        out["existing_alert_transport"]={"warn_dm":warned,"critical_messages_attempted":2,"critical_messages_sent":sender.call_count}

(HERE / "results.json").write_text(json.dumps(out, indent=2, default=str)+"\n")
print(json.dumps(out, indent=2, default=str))

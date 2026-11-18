"""Adversarial regression tests for CS5a (chosen-on state guard on saved picks)."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, time, timezone
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from click.testing import CliRunner

from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime.fromisoformat("2027-07-15T18:35:00-04:00")


def _write_dated(
    picks_dir: Path,
    *,
    now: datetime = NOW,
    streak: int = 7,
    best: int = 9,
    saver: str = "used",
    reason: str | None = None,
    recorded_at: str | None = None,
) -> None:
    day = now.astimezone(ET).date()
    expires = datetime.combine(day, time.max, ET).isoformat()
    payload: dict[str, object] = {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": day.isoformat(),
        "season": day.year,
        "expires_at": expires,
        "active_streak": streak,
        "best_streak": best,
        "saver_state": saver,
        "recorded_at": recorded_at or now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "manual_cli",
    }
    if reason is not None:
        payload["reason"] = reason
    target = picks_dir / "account_state" / "contest_streak.manual.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))


def _sample_daily() -> picks.DailyPick:
    return picks.DailyPick(
        date=DAY,
        run_time=NOW.isoformat(),
        pick=picks.Pick(
            batter_name="Steven Kwan",
            batter_id=680757,
            team="CLE",
            lineup_position=1,
            pitcher_name="Gerrit Cole",
            pitcher_id=543037,
            p_game_hit=0.82,
            flags=[],
            projected_lineup=False,
            game_pk=101,
            game_time=f"{DAY}T23:10:00Z",
        ),
        double_down=None,
        runner_up=None,
    )


def _sched_state() -> scheduler.SchedulerState:
    return scheduler.SchedulerState(
        date=DAY,
        schedule_fetched_at=NOW.isoformat(),
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
    )


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from bts.model import predict

    clock = [NOW]

    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    statuses = {
        101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        102: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    }
    frame = pd.DataFrame([
        {
            "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
            "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
            "p_hit_pa": 0.24, "p_game_hit": 0.82, "flags": "", "projected_lineup": False,
            "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
        },
        {
            "batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
            "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
            "p_hit_pa": 0.23, "p_game_hit": 0.81, "flags": "", "projected_lineup": False,
            "game_pk": 102, "game_time": f"{DAY}T23:40:00Z",
        },
    ])
    config = {
        "orchestrator": {
            "picks_dir": str(picks_dir),
            "heartbeat_path": str(tmp_path / "heartbeat"),
            "models_dir": str(tmp_path / "models"),
        },
        "bluesky": {"dm_recipient": "offline.test"},
        "scheduler": {
            "contest_state_required": True,
            "pick_delivery": "dm",
            "live_forward_capture_on_lock": False,
        },
        "health_checks": {"enabled": False},
        "tiers": [],
    }
    monkeypatch.setattr(contest_state, "datetime", PinnedClock)
    monkeypatch.setattr(daily_decision, "datetime", PinnedClock)
    monkeypatch.setattr(alert, "datetime", PinnedClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: clock[0].astimezone(ET))
    monkeypatch.setattr(cli, "_now_et", lambda: clock[0].astimezone(ET))
    monkeypatch.setattr(predict, "run_pipeline", Mock(return_value=frame))
    monkeypatch.setattr(orchestrator, "run_cascade", Mock(return_value=(frame, "offline")))
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *_a: statuses)
    monkeypatch.setattr(strategy, "get_game_statuses_detailed", lambda *_a: statuses)
    monkeypatch.setattr(picks, "get_game_statuses", lambda *_a: {k: v["abstract"] for k, v in statuses.items()})
    monkeypatch.setattr(strategy, "get_game_statuses", lambda *_a: {k: v["abstract"] for k, v in statuses.items()})
    monkeypatch.setattr(scheduler, "fetch_schedule", Mock(return_value=[]))
    monkeypatch.setattr(scheduler, "check_confirmed_lineups", lambda *_a: {})
    monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(scheduler, "retry_urlopen", Mock(side_effect=AssertionError("unexpected network call")))
    monkeypatch.setattr(dm, "send_dm", Mock(return_value="dm://pick"))
    monkeypatch.setattr(alert, "send_dm", Mock(return_value="dm://health"))
    monkeypatch.setattr(posting, "post_to_bluesky", Mock(return_value="at://post/1"))
    monkeypatch.setattr(posting, "should_post_now", Mock(return_value=True))
    return config, picks_dir, clock, statuses, frame


@pytest.mark.parametrize(
    "case_name,payload",
    [
        (
            "auto_streak_bool_true",
            {
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "automatic",
                "streak": True,
                "saver_available": False,
                "dated": None,
            },
        ),
        (
            "auto_streak_bool_false",
            {
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "automatic",
                "streak": False,
                "saver_available": False,
                "dated": None,
            },
        ),
        (
            "dated_active_bool_equal_to_outer_int",
            {
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "dated",
                "streak": 1,
                "saver_available": False,
                "dated": {
                    "effective_date": DAY,
                    "active_streak": True,
                    "best_streak": 9,
                    "saver_state": "used",
                },
            },
        ),
        (
            "dated_outer_bool_equal_to_active_int",
            {
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "dated",
                "streak": True,
                "saver_available": False,
                "dated": {
                    "effective_date": DAY,
                    "active_streak": 1,
                    "best_streak": 9,
                    "saver_state": "used",
                },
            },
        ),
        (
            "dated_best_bool_true_ge_zero_active",
            {
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "dated",
                "streak": 0,
                "saver_available": False,
                "dated": {
                    "effective_date": DAY,
                    "active_streak": 0,
                    "best_streak": True,
                    "saver_state": "used",
                },
            },
        ),
        (
            "dated_best_bool_false_ge_zero_active",
            {
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "dated",
                "streak": 0,
                "saver_available": False,
                "dated": {
                    "effective_date": DAY,
                    "active_streak": 0,
                    "best_streak": False,
                    "saver_state": "used",
                },
            },
        ),
    ],
)
def test_validator_rejects_unmasked_bool_for_int_in_every_count_field(
    tmp_path: Path, case_name: str, payload: dict[str, object]
) -> None:
    """Ensure each type(x) is int check fails independently when x is a bool,
    without being masked by active != streak or best < active."""
    daily = _sample_daily()
    raw = json.dumps({**asdict(daily), "chosen_on": payload})
    (tmp_path / f"{DAY}.json").write_text(raw)
    (tmp_path / f"{DAY}.shadow.json").write_text(raw)

    assert picks.load_pick(DAY, tmp_path).chosen_on is None, case_name
    assert picks.load_shadow_pick(DAY, tmp_path).chosen_on is None, case_name

    _write_dated(tmp_path, streak=0, best=0, saver="used")
    state = contest_state.load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
    daily.chosen_on = copy.deepcopy(payload)
    refusal = picks.saved_pick_refusal(daily, state)
    assert refusal is not None and "no valid record" in refusal, case_name


@pytest.mark.parametrize("streak", [0, 9, 10, 15, 16, 57])
@pytest.mark.parametrize("saver_state", ["not_earned", "active", "used"])
@pytest.mark.parametrize("saver_available", [True, False])
def test_dated_saver_consistency_matrix_across_streak_boundaries(
    tmp_path: Path, streak: int, saver_state: str, saver_available: bool
) -> None:
    best = max(streak, 16)
    candidate = {
        "schema": "bts_pick_chosen_on_v1",
        "required": True,
        "kind": "dated",
        "streak": streak,
        "saver_available": saver_available,
        "dated": {
            "effective_date": DAY,
            "active_streak": streak,
            "best_streak": best,
            "saver_state": saver_state,
        },
    }
    expected_avail = (saver_state == "active") or (saver_state == "not_earned" and streak < 10)
    expected_valid = saver_available is expected_avail

    daily = _sample_daily()
    raw = json.dumps({**asdict(daily), "chosen_on": candidate})
    (tmp_path / f"{DAY}.json").write_text(raw)
    (tmp_path / f"{DAY}.shadow.json").write_text(raw)

    assert picks.load_pick(DAY, tmp_path).chosen_on == (candidate if expected_valid else None)
    assert picks.load_shadow_pick(DAY, tmp_path).chosen_on == (candidate if expected_valid else None)

    _write_dated(tmp_path, streak=streak, best=best, saver=saver_state)
    live_state = contest_state.load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
    daily.chosen_on = copy.deepcopy(candidate)
    refusal = picks.saved_pick_refusal(daily, live_state)
    if expected_valid:
        assert refusal is None
    else:
        assert refusal is not None and "no valid record" in refusal


def test_refusal_renders_all_four_chosen_dated_values_and_ignores_reason_and_recorded_at(
    isolated_env,
) -> None:
    config, picks_dir, _, _, _ = isolated_env
    _write_dated(
        picks_dir, streak=11, best=14, saver="active",
        reason="morning check", recorded_at="2027-07-15T11:00:00Z",
    )
    first_state = contest_state.load_decision_streak_state(picks_dir, now=NOW, require_contest_state=True)
    daily = _sample_daily()
    daily.chosen_on = picks.chosen_on_record(first_state, required=True)
    picks.save_pick(daily, picks_dir)

    # Updating only reason and recorded_at preserves dated_identity and delivers.
    _write_dated(
        picks_dir, streak=11, best=14, saver="active",
        reason="re-verified afternoon", recorded_at="2027-07-15T20:00:00Z",
    )
    second_state = contest_state.load_decision_streak_state(picks_dir, now=NOW, require_contest_state=True)
    assert picks.saved_pick_refusal(daily, second_state) is None
    st = _sched_state()
    assert scheduler._deliver_and_lock_pick(daily, config, picks_dir, st, DAY, "fallback") is True
    assert dm.send_dm.call_count == 1 and st.pick_locked is True

    # Differing live dated state renders all 4 chosen values AND all 4 live values in order.
    _write_dated(picks_dir, streak=12, best=15, saver="used")
    third_state = contest_state.load_decision_streak_state(picks_dir, now=NOW, require_contest_state=True)
    msg = picks.saved_pick_refusal(daily, third_state)
    assert msg is not None
    chosen_clause = "it was chosen on dated (effective date 2027-07-15, active streak 11, best streak 14, saver active)"
    live_clause = "The live state is dated (effective date 2027-07-15, active streak 12, best streak 15, saver used)."
    assert msg.index(DAY) < msg.index(chosen_clause) < msg.index(live_clause)


@pytest.mark.parametrize("matches", [True, False])
def test_unlocked_unavailable_reuse_reaches_lineup_delivery_and_is_checked_as_saved(
    isolated_env, monkeypatch: pytest.MonkeyPatch, matches: bool
) -> None:
    config, picks_dir, _, _, _ = isolated_env
    _write_dated(picks_dir, streak=7, best=9, saver="used")
    live_state = contest_state.load_decision_streak_state(picks_dir, now=NOW, require_contest_state=True)
    seed = _sample_daily()
    seed.chosen_on = picks.chosen_on_record(live_state, required=matches)
    picks.save_pick(seed, picks_dir)

    warmup = {pk: {"abstract": "P", "detailed": "Warmup", "code": "PW"} for pk in (101, 102)}
    sched = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in (101, 102)}
    # 1) pre-cascade lock check -> Warmup (unlocked)
    # 2) run_and_pick -> Warmup -> select_pick returns unlocked existing pick (line 546, decision=None)
    # 3) _lock_decision_from_predictions -> Scheduled -> should_post=True
    seq = Mock(side_effect=[warmup, warmup, sched, sched])
    monkeypatch.setattr(picks, "get_game_statuses_detailed", seq)
    monkeypatch.setattr(strategy, "get_game_statuses_detailed", seq)

    chk = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
    assert chk["should_post"] is True
    assert chk["selection"].decision is None
    assert chk["selection"].pick_result.daily is chk["pick_result"].daily

    st = _sched_state()
    delivered = scheduler._deliver_and_lock_pick(
        chk["pick_result"].daily, config, picks_dir, st, DAY, "lineup", selection=chk["selection"]
    )
    assert delivered is matches
    assert dm.send_dm.called is matches and st.pick_locked is matches
    assert alert.send_dm.called is (not matches)


def test_scheduler_rejects_copied_object_and_earlier_cycle_selection_and_dm_catchup_mismatch(
    isolated_env, tmp_path: Path
) -> None:
    config, picks_dir, _, _, _ = isolated_env
    _write_dated(picks_dir, streak=7, best=9, saver="used")
    _, sel_cycle1, _ = orchestrator.run_and_pick(config, DAY)
    assert sel_cycle1.decision is not None
    orig_daily = sel_cycle1.pick_result.daily
    picks.save_pick(orig_daily, picks_dir)

    # Live dated observation changes before delivery.
    _write_dated(picks_dir, streak=7, best=11, saver="used")

    # Equal-valued shallow copy with genuine selection.decision must NOT pass identity check.
    copied = copy.copy(orig_daily)
    st1 = _sched_state()
    assert scheduler._deliver_and_lock_pick(
        copied, config, picks_dir, st1, DAY, "lineup", selection=sel_cycle1
    ) is False
    assert not dm.send_dm.called and not st1.pick_locked

    # Reloaded pick paired with cycle 1's selection must also be checked as saved and refused.
    reloaded = picks.load_pick(DAY, picks_dir)
    st2 = _sched_state()
    assert scheduler._deliver_and_lock_pick(
        reloaded, config, picks_dir, st2, DAY, "fallback", selection=sel_cycle1
    ) is False
    assert not dm.send_dm.called and not st2.pick_locked

    # A pick previously delivered via DM (notification_sent=True, bluesky_posted=False)
    # is refused by bts run and bts orchestrate catch-up when the live state no longer matches.
    orig_daily.notification_sent = True
    orig_daily.notification_channel = "bluesky_dm"
    orig_daily.notification_id = "dm://earlier"
    orig_daily.delivered_at = NOW.isoformat()
    picks.save_pick(orig_daily, picks_dir)

    runner = CliRunner()
    r_run = runner.invoke(cli.cli, [
        "run", "--date", DAY, "--picks-dir", str(picks_dir),
        "--models-dir", config["orchestrator"]["models_dir"],
    ])
    assert r_run.exit_code == 1 and "bts run: no pick: " in r_run.stderr
    assert not posting.post_to_bluesky.called

    cfg_path = tmp_path / "orch.toml"
    cfg_path.write_text(
        'tiers = []\n'
        f'[orchestrator]\npicks_dir = {json.dumps(str(picks_dir))}\n'
        '[bluesky]\ndm_recipient = "offline.test"\n'
        '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\n'
    )
    r_orch = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfg_path)])
    assert r_orch.exit_code == 1 and "bts orchestrate: no pick: " in r_orch.stderr
    assert not posting.post_to_bluesky.called
    assert picks.load_pick(DAY, picks_dir).bluesky_posted is False


@pytest.mark.parametrize("has_record", [True, False])
def test_reconcile_nonterminal_and_result_polling_saves_preserve_record_and_legacy_absence(
    isolated_env, monkeypatch: pytest.MonkeyPatch, has_record: bool
) -> None:
    _, picks_dir, _, _, _ = isolated_env
    _write_dated(picks_dir, streak=7, best=9, saver="used")
    live_state = contest_state.load_decision_streak_state(picks_dir, now=NOW, require_contest_state=True)
    expected = picks.chosen_on_record(live_state, required=True) if has_record else None

    daily = _sample_daily()
    daily.notification_sent = True
    daily.notification_channel = "bluesky_dm"
    daily.notification_id = "dm://1"
    daily.delivered_at = NOW.isoformat()
    daily.chosen_on = copy.deepcopy(expected)
    pfile = picks.save_pick(daily, picks_dir)
    if not has_record:
        legacy = {k: v for k, v in asdict(daily).items() if k != "chosen_on"}
        pfile.write_text(json.dumps(legacy, indent=2))

    scheduler.save_nonterminal_result(DAY, picks_dir, "unresolved")
    assert picks.load_pick(DAY, picks_dir).chosen_on == expected

    monkeypatch.setattr(scheduler, "poll_game_result", lambda *_a: "final")
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *_a: {"primary": "hit"})
    assert scheduler.run_result_polling(101, DAY, picks_dir) == "final"
    assert picks.load_pick(DAY, picks_dir).result == "hit"
    assert picks.load_pick(DAY, picks_dir).chosen_on == expected

    next_morning = datetime.fromisoformat("2027-07-16T02:00:00-04:00")
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *_a: {"primary": "miss"})
    corrections = picks.reconcile_results(picks_dir, lookback_days=2, clock=lambda: next_morning)
    assert len(corrections) == 1
    assert picks.load_pick(DAY, picks_dir).result == "miss"
    assert picks.load_pick(DAY, picks_dir).chosen_on == expected

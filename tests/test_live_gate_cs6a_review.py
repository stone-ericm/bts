"""Adversarial CS6a regression tests for adoption.

Covers:
1. `day_outcome.gather` and `bts day-outcome` opting out of `live_gate_problem` (`include_live_gate=False`)
   so live-gate-refused configs keep their declared-intent classifications instead of degrading to `intent_unknown`.
2. `bts orchestrate` single-snapshot guarantee when the config file on disk is mutated/corrupted after `load_config`
   and `builtins.open` / `Path.open` forbid any second read of the config path.
3. Outward configs accepted via non-boolean truthy flags (`"false"`, `1`) still enforcing `require_contest_state=True`
   across real `scheduler.run_single_check`, `scheduler._deliver_and_lock_pick` (cached pick), and `bts orchestrate`.
4. Accepted private configs (`private`+`research` and `private_mode="false"`) locking locally (`private_locked`)
   through real `bts schedule` (`run_day`) with zero DMs and zero public posts.
5. `bts run` edge branches: `locked` with `bluesky_posted=True` (Read 1 required, Read 2 skipped), Read 3 refusal
   even when `should_post_now` is False, and a dated observation in effect bypassing a corrupt `season_contract.json`.
"""
from __future__ import annotations

import builtins
import contextlib
import datetime as dt
from datetime import datetime
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd
import pytest

from bts import cli, contest_state, daily_decision, day_outcome, entry_intent, orchestrator, picks, scheduler, strategy
from bts.model import predict

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.replace(tzinfo=None) if tz is None else NOW.astimezone(tz)


@pytest.fixture(scope="module", autouse=True)
def _forbid_network():
    blocked = {
        "socket.connect", "socket.connect_ex", "socket.getaddrinfo",
        "socket.gethostbyname", "socket.gethostbyname_ex", "socket.getnameinfo",
        "urllib.Request",
    }

    # An audit hook cannot be removed and is process-wide, so it must switch itself
    # off when this module is done (lead, on adoption): left on, it broke every later
    # test in the same run that opens a real loopback socket (tests/test_web_saver.py).
    active = [True]

    def _hook(event: str, args: tuple) -> None:
        if active[0] and event in blocked:
            raise RuntimeError(f"Network forbidden in test: {event} {args!r}")

    sys.addaudithook(_hook)
    yield
    active[0] = False


def _write_covered_auto(picks_dir: Path, now: datetime, payload: dict) -> Path:
    year = now.astimezone(ET).year
    identifier = "c" * 32
    directory = picks_dir / "account_state"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": year,
        "status": "automatic",
        "activation_id": identifier,
        "switched_at": now.isoformat(),
        "revoked_at": None,
        "note": None,
    }))
    path = directory / "contest_streak.json"
    path.write_text(json.dumps(payload | {
        "schema_version": "bts_contest_streak_auto_v2",
        "season": year,
        "activation_id": identifier,
    }))
    return path


@pytest.fixture
def isolated_transports(monkeypatch):
    """Capture all outbound transports and pin clocks."""
    posts: list[str] = []
    dms: list[tuple[str, str]] = []
    alert_dms: list[tuple[str, str]] = []

    monkeypatch.setattr("datetime.datetime", FixedClock)
    for mod in (contest_state, picks, strategy, scheduler):
        monkeypatch.setattr(mod, "datetime", FixedClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: NOW)
    monkeypatch.setattr(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40)
    monkeypatch.setattr("bts.posting.post_to_bluesky", lambda txt: posts.append(txt) or "at://test/1")
    monkeypatch.setattr("bts.dm.send_dm", lambda r, t: dms.append((r, t)) or "dm-1")
    monkeypatch.setattr("bts.health.alert.send_dm", lambda r, t: alert_dms.append((r, t)) or "alert-1")
    return SimpleNamespace(posts=posts, dms=dms, alert_dms=alert_dms)


def _sample_df(game_time: str = f"{DAY}T23:10:00Z") -> pd.DataFrame:
    return pd.DataFrame([{
        "batter_name": "Jacob Wilson",
        "batter_id": 700363,
        "team": "ATH",
        "lineup": 1,
        "pitcher_name": "Jose Suarez",
        "pitcher_id": 660761,
        "p_hit_pa": 0.350,
        "p_game_hit": 0.900,
        "flags": "",
        "game_pk": 778899,
        "game_time": game_time,
    }])


def _write_uncovered_2026_auto(picks_dir: Path) -> None:
    acc = picks_dir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    (acc / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-28T16:00:00Z",
    }))


@pytest.mark.parametrize(
    "toml_body,decision_kwargs,expected_state,expected_exit",
    [
        (
            '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n',
            {"action": "skip", "source": "mdp", "delivery_status": None, "scoreable": False},
            "deliberate_skip",
            0,
        ),
        (
            '[scheduler]\nentry_intent = "enter"\npick_delivery = "public"\n',
            None,
            "no_decision",
            1,
        ),
        (
            'health_checks = 1\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
            {
                "action": "single",
                "source": "mdp",
                "primary": {"batter_id": 1, "batter_name": "Kwan", "p_game_hit": 0.8},
                "delivery_status": "private_locked",
                "scoreable": True,
            },
            "research_commit",
            0,
        ),
    ],
)
def test_day_outcome_ignores_live_gate_problem_in_gather_and_cli(
    tmp_path, isolated_transports, monkeypatch, toml_body, decision_kwargs, expected_state, expected_exit
):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    cfg_path = tmp_path / "orch.toml"
    cfg_path.write_text(toml_body)
    cfg = tomllib.loads(toml_body)

    # Confirm the default check_entry_intent DOES fail on the live gate for every case here
    assert not entry_intent.check_entry_intent(cfg).ok
    assert entry_intent.live_gate_problem(cfg) is not None

    if decision_kwargs is not None:
        daily_decision.write_decision(DAY, picks_dir, **decision_kwargs)

    fake_games = [{"gamePk": 778899, "status": {"detailedState": "Final"}}]
    monkeypatch.setattr(scheduler, "fetch_schedule", lambda _d: fake_games)

    direct = day_outcome.gather(DAY, picks_dir, cfg, expected_username="stonehengee", fetch_schedule=lambda _d: fake_games)
    assert direct.state == expected_state

    res = CliRunner().invoke(cli.cli, [
        "day-outcome", "--date", DAY, "--config", str(cfg_path),
        "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
        "--dm-recipient", "op.bsky.social",
    ])
    assert res.exit_code == expected_exit
    payload = json.loads(res.stdout)
    assert payload["state"] == expected_state


def test_orchestrate_cli_never_rereads_config_file_even_via_direct_open(
    tmp_path, isolated_transports, monkeypatch
):
    picks_dir = tmp_path / "picks"
    evil_dir = tmp_path / "evil_picks"
    _write_covered_auto(picks_dir, NOW, {
        "active_streak": 5,
        "source": "mlb_bts_profile",
        "source_date": DAY,
        "recorded_at": NOW.isoformat(),
    })
    cfg_path = tmp_path / "orch.toml"
    cfg_path.write_text(
        f'tiers = []\n[orchestrator]\npicks_dir = "{picks_dir}"\n'
        '[bluesky]\ndm_recipient = "op.bsky.social"\n'
        '[scheduler]\ncontest_state_required = true\n'
    )

    real_load = orchestrator.load_config
    config_read_complete = False

    def load_then_corrupt(path: Path) -> dict:
        nonlocal config_read_complete
        loaded = real_load(path)
        # Corrupt the file on disk immediately after the single CLI load!
        cfg_path.write_text(f'[corrupt_toml\npicks_dir = "{evil_dir}"\n')
        config_read_complete = True
        return loaded

    real_builtin_open = builtins.open
    real_path_open = Path.open

    def guarded_builtin_open(file, *args, **kwargs):
        if config_read_complete and not isinstance(file, int) and Path(file).resolve() == cfg_path.resolve():
            raise AssertionError("orchestrate re-opened config_path via builtins.open")
        return real_builtin_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        if config_read_complete and self.resolve() == cfg_path.resolve():
            raise AssertionError("orchestrate re-opened config_path via Path.open")
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "load_config", load_then_corrupt)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(orchestrator, "run_cascade", lambda *_a: (_sample_df(), "local"))
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *_a: {778899: {"abstract": "P", "detailed": "Pre-Game"}})
    monkeypatch.setattr("bts.strategy.get_game_statuses", lambda *_a: {778899: "P"})
    monkeypatch.setattr("bts.posting.should_post_now", lambda *_a: True)

    res = CliRunner().invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfg_path)])
    assert res.exit_code == 0, (res.stderr, res.exception)
    assert len(isolated_transports.posts) == 1
    assert (picks_dir / f"{DAY}.json").exists()
    assert not evil_dir.exists()


@pytest.mark.parametrize("toml_flags", [
    '[scheduler]\npick_delivery = "bluesky-dm"\ncontest_state_required = "false"\n',
    '[scheduler]\npick_delivery = "feed"\n[health_checks]\ncontest_state_expected = 1\n',
])
def test_accepted_outward_truthy_nonbool_configs_still_block_uncovered_state_in_workers(
    tmp_path, isolated_transports, monkeypatch, toml_flags
):
    picks_dir = tmp_path / "picks"
    _write_uncovered_2026_auto(picks_dir)
    cfg_path = tmp_path / "orch.toml"
    cfg_path.write_text(
        f'tiers = []\n[orchestrator]\npicks_dir = "{picks_dir}"\nheartbeat_path = "{tmp_path / "hb"}"\n'
        '[bluesky]\ndm_recipient = "op.bsky.social"\n' + toml_flags
    )
    cfg = tomllib.loads(cfg_path.read_text())
    assert entry_intent.live_gate_problem(cfg) is None

    monkeypatch.setattr(orchestrator, "run_cascade", lambda *_a: (_sample_df(), "local"))
    monkeypatch.setattr(scheduler, "check_confirmed_lineups", lambda *_a: {778899: {"away", "home"}})
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *_a: {778899: {"abstract": "P", "detailed": "Pre-Game"}})
    monkeypatch.setattr(scheduler, "_void_committed_games", lambda *_a: {})
    monkeypatch.setattr("bts.posting.should_post_now", lambda *_a: True)

    sc_out = scheduler.run_single_check(DAY, [778899], set(), cfg, 0.03)
    assert sc_out["pick_result"] is None and sc_out["should_post"] is False
    assert isolated_transports.posts == [] and isolated_transports.dms == []
    assert len(isolated_transports.alert_dms) == 1

    cached = picks.DailyPick(
        date=DAY,
        run_time=NOW.isoformat(),
        pick=picks.Pick(
            batter_name="Jacob Wilson", batter_id=700363, team="ATH",
            lineup_position=1, pitcher_name="Jose Suarez", pitcher_id=660761,
            p_game_hit=0.900, flags=[], projected_lineup=True,
            game_pk=778899, game_time=f"{DAY}T23:10:00Z",
        ),
        double_down=None,
        runner_up=None,
    )
    st = scheduler.SchedulerState(
        date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
        confirmed_game_pks=[], runs_completed=[], pick_locked=False,
        pick_locked_at=None, result_status=None, next_wakeup=None,
    )
    assert scheduler._deliver_and_lock_pick(cached, cfg, picks_dir, st, DAY, "fallback") is False
    assert st.pick_locked is False
    assert isolated_transports.posts == [] and isolated_transports.dms == []

    r_orch = CliRunner().invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfg_path)])
    assert r_orch.exit_code == 1
    assert isolated_transports.posts == []
    assert len(isolated_transports.dms) == 1 and "No pick made" in isolated_transports.dms[0][1]


def test_accepted_private_schedule_locks_locally_without_dm_or_post(tmp_path, isolated_transports, monkeypatch):
    picks_dir = tmp_path / "picks"
    _write_uncovered_2026_auto(picks_dir)
    cfg_path = tmp_path / "priv.toml"
    cfg_path.write_text(
        f'tiers = []\n[orchestrator]\npicks_dir = "{picks_dir}"\nheartbeat_path = "{tmp_path / "hb"}"\n'
        '[bluesky]\ndm_recipient = "op.bsky.social"\n'
        '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
        '[health_checks]\nenabled = false\n'
    )
    gtime = f"{DAY}T16:40:00Z"
    game = {
        "gamePk": 778899,
        "gameDate": gtime,
        "teams": {"away": {"team": {"name": "ATH"}}, "home": {"team": {"name": "DET"}}},
        "status": {"abstractGameState": "Preview", "detailedState": "Pre-Game"},
    }
    monkeypatch.setattr(scheduler, "fetch_schedule", lambda _d: [game])
    monkeypatch.setattr(scheduler, "notify_ready", lambda: None)
    monkeypatch.setattr(scheduler, "notify_watchdog", lambda: None)
    monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", lambda _s: None)
    monkeypatch.setattr(scheduler, "_idle_until_next_wakeup", lambda *_a, **_k: None)
    monkeypatch.setattr(scheduler, "run_result_polling", lambda *_a, **_k: "hit")
    monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_a, **_k: None)
    monkeypatch.setattr(scheduler, "check_confirmed_lineups", lambda *_a: {778899: {"away", "home"}})
    monkeypatch.setattr(scheduler, "_void_committed_games", lambda *_a: {})
    monkeypatch.setattr(orchestrator, "run_cascade", lambda *_a: (_sample_df(gtime), "local"))
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *_a: {778899: {"abstract": "P", "detailed": "Pre-Game"}})
    monkeypatch.setattr("bts.strategy.get_game_statuses", lambda *_a: {778899: "P"})

    res = CliRunner().invoke(cli.cli, ["schedule", "--config", str(cfg_path), "--date", DAY])
    assert res.exit_code == 0, (res.stderr, res.exception)
    assert isolated_transports.posts == [] and isolated_transports.dms == []
    dec = json.loads((picks_dir / DAY / "decision.json").read_text())
    assert dec["delivery_status"] == "private_locked"


def test_bts_run_locked_already_posted_and_dated_corrupt_switch_and_read3_outside_window(
    tmp_path, isolated_transports, monkeypatch
):
    monkeypatch.setattr(predict, "run_pipeline", lambda *_a, **_k: _sample_df())
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *_a: {})
    monkeypatch.setattr(picks, "attach_provenance", lambda *_a, **_k: None)

    primary = picks.Pick(
        batter_name="Jacob Wilson", batter_id=700363, team="ATH",
        lineup_position=1, pitcher_name="Jose Suarez", pitcher_id=660761,
        p_game_hit=0.900, flags=[], projected_lineup=False,
        game_pk=778899, game_time=f"{DAY}T23:10:00Z",
    )

    # 1) Dated observation in effect + corrupt season_contract.json + locked(bluesky_posted=True):
    # Read 1 succeeds without reading the corrupt switch; Read 2 is NOT called because bluesky_posted=True!
    pdir1 = tmp_path / "p1"
    acc1 = pdir1 / "account_state"
    acc1.mkdir(parents=True)
    eod = datetime.combine(dt.date(2027, 7, 15), dt.time.max, ET)
    (acc1 / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": contest_state.DATED_MANUAL_SCHEMA,
        "effective_date": DAY,
        "season": 2027,
        "active_streak": 7,
        "best_streak": 9,
        "saver_state": "used",
        "source": "manual_cli",
        "recorded_at": NOW.isoformat(),
        "expires_at": eod.isoformat(),
    }))
    (acc1 / "season_contract.json").write_text("{corrupt switch")

    read_calls = []
    real_loader = contest_state.load_decision_streak_state
    monkeypatch.setattr(
        contest_state,
        "load_decision_streak_state",
        lambda d, **kw: (read_calls.append(kw), real_loader(d, **kw))[1],
    )
    locked_posted = picks.DailyPick(
        date=DAY, run_time=NOW.isoformat(), pick=primary,
        double_down=None, runner_up=None, bluesky_posted=True, bluesky_uri="at://prior/1",
    )
    monkeypatch.setattr(
        strategy, "select_pick",
        lambda *_a, **_k: SimpleNamespace(pick_result=strategy.PickResult(daily=locked_posted, locked=True), decision=None),
    )
    r1 = CliRunner().invoke(cli.cli, ["run", "--date", DAY, "--picks-dir", str(pdir1), "--models-dir", str(tmp_path / "m1")])
    assert r1.exit_code == 0 and r1.stderr == ""
    assert "Pick locked: Jacob Wilson (already posted)" in r1.stdout
    assert read_calls == [{"require_contest_state": True}]
    assert isolated_transports.posts == []

    # 2) Fresh pick with should_post_now=False STILL performs Read 3 (require_contest_state=True) and refuses on revoke!
    pdir2 = tmp_path / "p2"
    _write_covered_auto(pdir2, NOW, {
        "active_streak": 7, "source": "mlb_bts_profile",
        "source_date": DAY, "recorded_at": NOW.isoformat(),
    })
    read_calls.clear()

    def unlink_on_second(d, **kw):
        read_calls.append(kw)
        if len(read_calls) == 2:
            (Path(d) / "account_state" / "season_contract.json").unlink()
        return real_loader(d, **kw)

    monkeypatch.setattr(contest_state, "load_decision_streak_state", unlink_on_second)
    monkeypatch.setattr("bts.posting.should_post_now", lambda *_a: False)
    fresh = picks.DailyPick(date=DAY, run_time=NOW.isoformat(), pick=primary, double_down=None, runner_up=None)
    monkeypatch.setattr(
        strategy, "select_pick",
        lambda *_a, **_k: SimpleNamespace(pick_result=strategy.PickResult(daily=fresh, locked=False), decision=None),
    )
    r2 = CliRunner().invoke(cli.cli, ["run", "--date", DAY, "--picks-dir", str(pdir2), "--models-dir", str(tmp_path / "m2")])
    assert r2.exit_code == 1
    assert r2.stderr.startswith("bts run: no pick: ")
    assert read_calls == [{"require_contest_state": True}, {"require_contest_state": True}]
    assert (pdir2 / f"{DAY}.json").exists()
    assert isolated_transports.posts == []

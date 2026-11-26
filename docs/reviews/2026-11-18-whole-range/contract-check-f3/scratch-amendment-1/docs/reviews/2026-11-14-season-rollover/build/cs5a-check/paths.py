"""Check live saved-pick bypasses and observation transitions on main, offline."""

from reproduce import (
    DAY, ET, NOW, PREDICTIONS, STATUS, TemporaryDirectory, Path, CliRunner,
    alert, cli, contest_state, dm, json, network_attempts, orchestrator, patch,
    picks, posting, predict, scheduler,
)
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta
import sys


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def dated(account, *, streak=56, day=DAY):
    write_json(account / "contest_streak.manual.json", {
        "schema_version": "bts_contest_streak_manual_v3", "effective_date": day,
        "expires_at": f"{day}T23:59:59.999999-04:00", "season": 2027,
        "active_streak": streak, "best_streak": streak, "saver_state": "used",
        "source": "manual_cli", "recorded_at": NOW.isoformat(),
    })


def automatic(account, *, streak=0, activation="a" * 32, revoked=False):
    write_json(account / "season_contract.json", {
        "schema_version": "bts_season_contract_v1", "season": 2027,
        "status": "revoked" if revoked else "automatic", "activation_id": activation,
        "switched_at": NOW.isoformat(), "revoked_at": NOW.isoformat() if revoked else None,
        "note": None,
    })
    write_json(account / "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v2", "season": 2027,
        "activation_id": activation, "active_streak": streak, "best_streak": max(streak, 12),
        "source": "mlb_bts_profile", "source_date": "2027-07-14", "recorded_at": NOW.isoformat(),
    })


def run(root):
    clock = NOW
    schedule_status = STATUS
    schedule_games = []
    posts = []
    dms = []
    prediction_result = (PREDICTIONS, "offline")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz) if tz else clock.replace(tzinfo=None)

    def captured_dm(recipient, text):
        dms.append(text)
        return "captured-dm"

    def run_cli(*args):
        result = CliRunner().invoke(cli.cli, list(args), catch_exceptions=False)
        assert result.exit_code == 0, result.output
        return result

    def advance(seconds):
        nonlocal clock
        clock += timedelta(seconds=seconds)

    def cfg(folder):
        return {
            "orchestrator": {"picks_dir": str(folder), "heartbeat_path": str(folder / "heartbeat")},
            "bluesky": {"dm_recipient": "local.invalid"},
            "scheduler": {"contest_state_required": True, "entry_intent": "enter", "pick_delivery": "dm",
                          "live_forward_capture_on_lock": False},
            "health_checks": {"enabled": False}, "tiers": [],
        }

    with ExitStack() as capture:
        for module, name, replacement in (
            (contest_state, "datetime", Clock), (alert, "datetime", Clock),
            (scheduler, "_now_et", lambda: clock),
            (scheduler, "fetch_schedule", lambda date: schedule_games if date == DAY else []),
            (scheduler, "check_confirmed_lineups", lambda _: {}),
            (scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
            (scheduler, "notify_ready", lambda: None), (scheduler, "notify_watchdog", lambda: None),
            (scheduler, "_watchdog_ping_sleep", advance), (scheduler, "_idle_until_next_wakeup", lambda *_: None),
            (scheduler, "run_result_polling", lambda *a, **kw: "captured"),
            (orchestrator, "run_cascade", lambda *_: prediction_result),
            (predict, "run_pipeline", lambda *a, **kw: PREDICTIONS),
            (picks, "get_game_statuses_detailed", lambda _: schedule_status),
            (picks, "get_game_statuses", lambda _: dict.fromkeys(STATUS, "P")),
            (posting, "_now_et", lambda: clock),
            (posting, "post_to_bluesky", lambda text: posts.append(text) or "captured-post"),
            (dm, "send_dm", captured_dm), (alert, "send_dm", captured_dm),
        ):
            capture.enter_context(patch.object(module, name, replacement))

        for label, status in (("unlocked-warmup", "Warmup"), ("locked-started", "In Progress")):
            for command in ("run", "orchestrate"):
                folder = root / f"{command}-{label}"
                account = folder / "account_state"
                schedule_status = STATUS
                automatic(account)
                run_cli("preview", "--date", DAY, "--picks-dir", str(folder), "--models-dir", str(root / "models"))
                original = picks.load_pick(DAY, folder)
                assert original.double_down is not None
                dated(account)
                schedule_status = dict.fromkeys(STATUS, {"abstract": "P" if status == "Warmup" else "L",
                                                         "detailed": status, "code": "W" if status == "Warmup" else "I"})
                if command == "run":
                    flags = ["--picks-dir", str(folder), "--models-dir", str(root / "models")]
                else:
                    config = folder / "worker.toml"
                    config.write_text(f'tiers = []\n[orchestrator]\npicks_dir = "{folder}"\n'
                                      '[bluesky]\ndm_recipient = "local.invalid"\n'
                                      '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\n')
                    flags = ["--config", str(config)]
                before = len(posts)
                result = run_cli(command, "--date", DAY, *flags)
                assert len(posts) == before + 1 and "Streak: 56" in posts[-1], result.output
                persisted = picks.load_pick(DAY, folder)
                assert persisted.run_time == original.run_time and persisted.double_down == original.double_down
                assert ("catch-up" in result.output) == (status == "In Progress"), result.output
                print(f"{command} {label}: posted preview's unchanged double under dated 56")

        folder = root / "ordinary"
        account = folder / "account_state"
        config = cfg(folder)
        clock = NOW.replace(hour=17, minute=30)
        schedule_status = STATUS
        automatic(account, streak=5)
        # The 0.82 confirmed primary has only a 0.01 edge over a projected contender.
        projected = PREDICTIONS.copy()
        projected.loc[1, "flags"] = "PROJECTED lineup"
        projected.loc[1, "projected_lineup"] = True
        prediction_result = (projected, "offline")
        selection = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
        assert selection["selection"].decision is not None and not selection["should_post"]
        earlier = picks.load_pick(DAY, folder)
        clock = NOW
        automatic(account, streak=6)
        prediction_result = (None, None)
        check = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
        assert check["pick_result"] is None and check["selection"] is None
        # Like run_day's post-loop fallback after restart, read the on-disk candidate.
        cached = picks.load_pick(DAY, folder)
        refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
        assert refresh.daily is cached and earlier == cached and refresh.selection is None
        schedule_games = [
            {"gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",
             "teams": {"away": {"team": {"name": "Cleveland"}}, "home": {"team": {"name": "New York"}}}},
            {"gamePk": 102, "gameDate": f"{DAY}T23:40:00Z",
             "teams": {"away": {"team": {"name": "New York"}}, "home": {"team": {"name": "Los Angeles"}}}},
        ]
        dms_before = len(dms)
        scheduler.run_day(DAY, config)
        assert len(dms) == dms_before + 1 and "Streak: 6" in dms[-1]
        assert picks.load_pick(DAY, folder).notification_sent
        clock = NOW
        print("ordinary restart: gap held a live streak-5 pick; streak-6 fetch, failed check and refresh keep it cached")

        def required_snapshot():
            return contest_state.load_decision_streak_state(folder, require_contest_state=True)

        def assert_refused():
            try:
                required_snapshot()
            except contest_state.ContestStateError:
                return
            raise AssertionError("expected required loader refusal")

        automatic(account, streak=6, revoked=True)
        assert_refused()
        automatic(account, streak=56, activation="b" * 32)
        write_json(account / "saver_state.json", {"state": "used", "season": 2027})
        assert required_snapshot().streak == 56
        write_json(account / "saver_state.json", {"state": "active", "season": 2027})
        assert required_snapshot().saver_available
        write_json(account / "saver_state.json", {"state": "used", "season": 2027})
        assert not required_snapshot().saver_available
        dated(account, streak=4)
        assert required_snapshot().streak == 4
        clock = NOW.replace(hour=0, minute=0) + timedelta(days=1)
        assert required_snapshot().streak == 56
        dated(account, streak=4, day=clock.date().isoformat())
        assert required_snapshot().contest_source_date.isoformat() == "2027-07-16"
        print("required loader: revoke refused; new activation accepted; saver edit changed policy; dated override lapsed at ET midnight")

        scheduler._alert_missed_pick(config, cached, 10)
        assert "Steven Kwan" in dms[-1] and "Deliver manually" in dms[-1]
        print("live missed-pick DM names cached batter and suggests manual delivery")

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts import live_forward_capture_once, live_forward_provenance_inventory
        old_pick = json.loads((folder / f"{DAY}.json").read_text())
        assert "chosen_on" not in old_pick
        for project in (live_forward_capture_once.decision_snapshot,
                        live_forward_provenance_inventory._decision_snapshot):
            assert project(old_pick) != project(dict(old_pick, chosen_on=None))
        print("both live-forward projections flag a prior snapshot as different after a null chosen_on insertion")
        assert not network_attempts, network_attempts


if __name__ == "__main__":
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        run(Path(directory))

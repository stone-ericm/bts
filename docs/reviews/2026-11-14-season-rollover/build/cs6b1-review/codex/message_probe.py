"""Verify the two alert lanes and practical season/restart scenarios offline."""

from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from day_probe import MORNING, STALE, sandbox
from fault_probe import advisory


def run_messages(scratch):
    from bts import contest_state, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST, START, refusal, warning
    from tests.test_pick_chosen_on_cs5a import set_automatic, set_dated
    from tests.test_stale_delivery_cs4c import seed_automatic

    for name in ("refused", "long refusal", "dated", "fresh", "old", "future", "unreadable"):
        for recipient in (True, False):
            root = scratch / f"{name}-{recipient}"
            with sandbox(root) as (patch, config, directory, now, _, _), redirect_stderr(StringIO()) as err:
                if not recipient:
                    config["bluesky"].clear()
                gate = Mock(wraps=alert.dispatch_dm_for_health_alerts)
                actual_read = contest_state.load_decision_streak_state
                read = Mock(wraps=actual_read)
                if name == "long refusal":
                    private = "LOCAL-ONLY\n" + "é\t雪\r" * 250
                    read.side_effect = contest_state.ContestStateError(private)
                if name == "dated":
                    set_dated(directory, now=START.replace(hour=7), streak=30, best=30)
                if name == "fresh":
                    set_automatic(directory, now=START, streak=30)
                if name in {"old", "future", "unreadable"}:
                    recorded = {"old": (START - timedelta(hours=20)).isoformat(),
                                "future": (START + timedelta(hours=1)).isoformat(),
                                "unreadable": "no timestamp"}[name]
                    seed_automatic(directory, recorded, now=START, streak=30)
                patch.setattr(alert, "dispatch_dm_for_health_alerts", gate)
                patch.setattr(contest_state, "load_decision_streak_state", read)
                scheduler.run_day(DAY, config)
                read.assert_called_once_with(directory, require_contest_state=True, now=START)
                sent = alert.send_dm.call_args.args if alert.send_dm.called else None
                path = None
                if name in {"refused", "long refusal"}:
                    if name == "refused":
                        try:
                            actual_read(directory, require_contest_state=True, now=START)
                        except contest_state.ContestStateError as error:
                            private = str(error)
                    piece = ascii(private)[:800]
                    assert advisory(err) == [f"  MORNING STATE CHECK — no pick can be made yet: {piece}"]
                    if name == "long refusal":
                        assert len(piece) == 800
                    if recipient:
                        assert sent == ("offline.test", refusal()) and "LOCAL-ONLY" not in sent[1]
                        path = directory.parent / "health_state" / MORNING
                        gate.assert_called_once_with(
                            [alert.Alert("CRITICAL", "contest_state", refusal().split("- [contest_state] ")[1],
                                         incident_key="contest_state:morning")],
                            "offline.test", status_path=path, now_et_date=START.date(),
                        )
                elif name in {"dated", "fresh"}:
                    kind = "automatic" if name == "fresh" else "dated"
                    assert advisory(err) == [f"  Morning state check: {kind} contest state accepted (streak 30)."]
                else:
                    detail = {"old": f"last accepted fetch {recorded}, about 20.00 hours ago",
                              "future": f"its recorded_at {recorded} is in the future",
                              "unreadable": "it has no usable recorded_at"}[name]
                    assert advisory(err) == [f"  MORNING STATE CHECK — stale contest reading: {detail}"]
                    if recipient:
                        assert sent == ("offline.test", warning(detail))
                        path = directory.parent / "health_state" / STALE
                        gate.assert_called_once_with(
                            [], "offline.test", warn_attention=[alert.Alert(
                                "WARN", "contest_state", warning(detail).split("- [contest_state] ")[1],
                                incident_key="contest_state:stale_reading")],
                            status_path=path, now_et_date=START.date(),
                        )
                if path is None:
                    gate.assert_not_called()
                    alert.send_dm.assert_not_called()
                    assert not (directory.parent / "health_state").exists()
                else:
                    assert path.exists() and alert.send_dm.call_count == 1
                print(json.dumps({"case": name, "has_recipient": recipient, "line": advisory(err),
                                  "DM": sent, "status": json.loads(path.read_bytes()) if path else None}))


def season_scenarios(scratch):
    import datetime as calendar
    from click.testing import CliRunner
    from bts import cli, contest_state, orchestrator, scheduler, util
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST, START
    from tests.test_pick_chosen_on_cs5a import set_automatic, set_dated

    for name in ("yesterday", "Opening Night", "after off day", "restored 2026 snapshot", "revoke in progress"):
        with sandbox(scratch / name) as (patch, config, directory, now, predictions, _), redirect_stderr(StringIO()) as err:
            if name == "yesterday":
                set_dated(directory, now=START - timedelta(days=1))
            if name == "restored 2026 snapshot":
                set_automatic(directory, now=START.replace(year=2026))
            selected = DAY
            if name in {"Opening Night", "after off day"}:
                if name == "Opening Night":
                    selected = "2027-03-24"
                    now[0] = START.replace(month=3, day=24)
                games = [{**game, "gameDate": game["gameDate"].replace(DAY, selected)}
                         for game in scheduler.fetch_schedule(DAY)]
                if name == "Opening Night":
                    games = games[:1]
                scheduler.fetch_schedule.side_effect = lambda day: games if day == selected else []
                if name == "after off day":
                    now[0] -= timedelta(days=1)
                    scheduler.run_day(now[0].date().isoformat(), config)
                    alert.send_dm.assert_not_called()
                    now[0] += timedelta(days=1)
                    set_automatic(directory, now=now[0], streak=30)
            if name == "revoke in progress":
                set_dated(directory, now=START)
                original_write = util.atomic_write_text

                def write(path, data):
                    original_write(path, data)
                    if path.name == "season_contract.json":
                        assert (directory / "account_state/contest_streak.manual.json").is_file()
                        scheduler._morning_state_check(config, directory, DAY, START, START + timedelta(hours=1))

                patch.setattr(util, "atomic_write_text", write)
                with patch.context() as clock:
                    clock.setattr(calendar, "datetime", contest_state.datetime)
                    outcome = CliRunner().invoke(cli.cli, ["season-contract", "revoke", "--season", "2027",
                        "--reason", "offline exercise", "--picks-dir", str(directory)], catch_exceptions=False)
                assert outcome.exit_code == 0, outcome.output
                assert "Morning state check: dated" in outcome.stderr
                alert.send_dm.assert_not_called()
                now[0] = FIRST
                orchestrator.run_cascade.return_value = predictions
                scheduler.run_single_check(DAY, [101, 102], set(), config, .03)
                assert alert.send_dm.called and "BTS health CRITICAL alert(s):" in alert.send_dm.call_args.args[1]
                print(f"{name}: accepted at switch write, real first check refused after manual rename")
                continue

            scheduler.run_day(selected, config)
            assert alert.send_dm.called == (name != "after off day")
            print(f"{name}: {advisory(err)!r}; DMs={alert.send_dm.call_count}")


def refusal_traceback(root):
    from bts import contest_state, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, refusal

    raw = "raw refusal region\n" + "é" * 1000 + "\nEND-OF-LOCAL-REFUSAL"
    with sandbox(root) as (patch, config, directory, now, _, events), redirect_stderr(StringIO()) as err:
        reader = Mock(side_effect=contest_state.ContestStateError(raw))
        patch.setattr(contest_state, "load_decision_streak_state", reader)
        alert.send_dm.side_effect = OSError("captured morning DM failure")
        scheduler.run_day(DAY, config)
        alert.send_dm.assert_called_once_with("offline.test", refusal())
        bounded = f"  MORNING STATE CHECK — no pick can be made yet: {ascii(raw)[:800]}"
        assert advisory(err) == [bounded]
        assert raw in err.getvalue(), err.getvalue()
        assert any(event[0] == "run_single_check" for event in events)
        print(json.dumps({"case": "raw refusal in chained traceback", "raw_length": len(raw),
                          "stderr": err.getvalue(), "captured DM": alert.send_dm.call_args.args[1]}))


def test_module_cache(root):
    from types import SimpleNamespace
    from bts import strategy
    from tests.test_morning_state_cs6b1 import test_accepted_state_logs_without_health_writes

    capture = StringIO()
    capsys = SimpleNamespace(readouterr=lambda: SimpleNamespace(err=capture.getvalue()))
    assert strategy._mdp_cache is None
    with sandbox(root) as (_, config, directory, now, predictions, _), redirect_stderr(capture):
        test_accepted_state_logs_without_health_writes((config, directory, now, predictions), capsys, "dated")
    assert strategy._mdp_cache is not None
    print(f"process-wide strategy._mdp_cache populated after scoped fixture teardown, keys={list(strategy._mdp_cache)}")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="cs6b1-message-") as scratch:
        root = Path(scratch)
        test_module_cache(root / "cache")
        run_messages(root)
        season_scenarios(root)
        refusal_traceback(root / "traceback")

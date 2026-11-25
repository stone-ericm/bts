"""Exercise current operator messages and commands on synthetic ET slates."""
import contextlib
import datetime as calendar
import html
import io
import json
import socket
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from pytest import MonkeyPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-e"),
]

from bts import contest_state, dm, orchestrator, picks, scheduler, strategy, web
from bts.health import alert
from probe_edges import first_boot
from probe_more import research
from probe_state import ET, invoke, observation, ops_cli, run_case, saver_matrix, sent_pick, state_at
from probe_text import offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_automatic
from tests.test_saved_pick_paths_cs5a import scheduler_state, world
from tests.test_stale_delivery_cs4c import seed_automatic

CONTRACT = ROOT / "docs/reviews/2026-11-18-whole-range/kickoff-fix-e.md"
ENVELOPE = "BTS health WARN attention:\n- [contest_state] "


def exact_body(label, **values):
    lines = CONTRACT.read_text().splitlines()
    raw, = [line.split("`")[1] for line in lines if line.startswith(f"- **{label}:** `")]
    return raw.format(**values)


def inspect_skip(root, mp):
    config, directory, clock, statuses = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    frame, tier = orchestrator.run_cascade.return_value
    scheduled = dict(statuses)
    cases = (
        ("K1", "2027-09-19", 0, 18, [.72, .72], "used"),
        ("K2", DAY, 10, 10, [.999, .75], "used"),
        ("K3", "2027-08-30", 56, 56, [.99, .99], "used"),
        ("K4", DAY, 56, 56, [.99, .99], "used"),
        ("K5", DAY, 10, 10, [.999, .75], "used"),
    )
    texts = {}
    for case, day, streak, best, probabilities, saver in cases:
        clock[0] = datetime.fromisoformat(f"{day}T18:35:00-04:00")
        statuses.update(scheduled)
        if case in ("K2", "K5"):
            statuses[101] = {"abstract": "F", "detailed": "Postponed", "code": "D"}
        if case == "K5":
            statuses[102] = statuses[101]
        frame_with_times = frame.assign(
            p_game_hit=probabilities, game_time=[f"{day}T23:10:00Z", f"{day}T23:40:00Z"],
        )
        orchestrator.run_cascade.return_value = (frame_with_times, tier)
        assert observation(day, directory, streak=streak, best=best, saver=saver).exit_code == 0
        journal = io.StringIO()
        with contextlib.redirect_stderr(journal):
            result = scheduler.run_single_check(date=day, all_game_pks=list(statuses),
                                                confirmed_sides=set(), config=config, early_lock_gap=.03)
        summ = result["skip_summary"]
        template_line, = [line for line in CONTRACT.read_text().splitlines()
                          if line.startswith(f"  - {case} `")]
        template = template_line.split("`")[1]
        assert summ is not None
        reason = summ.get("reason", "")
        if reason.endswith((".", "!", "?")):
            template = template.removesuffix(".")
        body = template.format(
            candidate="Candidate used in the decision", b=summ["best_batter"], t=summ["best_team"],
            p=f'{summ["best_p"]:.1%}', reason=reason,
            bar_text=f'the streak-{streak} bar ({summ["bar"]:.1%})' if "bar" in summ else "",
        ) + f" Streak used at this check: {streak}."
        dm_text = scheduler.format_skip_dm(day, summ)
        assert dm_text == f"BTS {day}: No pick at this check. {body}"
        assert [line for line in journal.getvalue().splitlines() if line.startswith("  SKIP — ")] == [
            "  SKIP — " + body
        ]
        assert web.render_skip_banner(summ) == (
            '<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>'
            + html.escape(body) + "</span></div>"
        )
        print(case, dm_text, flush=True)
        texts[case] = body

    assert "75.0%, below" in texts["K2"] and "99.9%" in texts["K5"]
    assert all(("below the" in body) == (case == "K2") for case, body in texts.items())


def warnings_and_refusals(root, mp):
    config, directory, clock, _ = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    clock[0] = datetime(2027, 12, 31, 23, 59, tzinfo=ET)
    prefix = "0002-01-01T00:00:00."
    stamp = prefix + "1" * (128 - len(prefix) - 6) + "+00:00"
    for route in ("After a delivery", "In the morning"):
        target = directory / route / "picks"
        set_automatic(target, now=clock[0])
        account = target / "account_state/contest_streak.json"
        account.write_text(json.dumps(dict(json.loads(account.read_text()), recorded_at=stamp)))
        reading = contest_state.load_decision_streak_state(target, require_contest_state=True, now=clock[0])
        record = contest_state.automatic_reading_record(reading, clock[0])
        detail = f"last accepted fetch {stamp}, about {record['age_hours']:.2f} hours ago"
        day = clock[0].date().isoformat()
        journal = io.StringIO()
        with contextlib.redirect_stderr(journal):
            if route == "After a delivery":
                scheduler._warn_stale_contest_reading(record, config, target)
            else:
                scheduler._morning_state_check(config, target, day, clock[0], clock[0] + timedelta(minutes=30))
        expected = exact_body(route, D=day, detail=detail)
        assert alert.send_dm.call_args.args[1] == ENVELOPE + expected
        if route == "After a delivery":
            assert journal.getvalue().splitlines().count("  STALE CONTEST READING — " + expected) == 1
        assert len(ENVELOPE + expected) < 900
        print(route, len(ENVELOPE + expected), repr(ENVELOPE + expected), repr(journal.getvalue()), flush=True)
    clock[0] += timedelta(minutes=6)
    scheduler._warn_stale_contest_reading(record, config, directory)
    print("delivery after midnight", repr(alert.send_dm.call_args.args[1]), flush=True)

    clock[0] = NOW
    for label in ("Late", "Postponed"):
        target = directory / label
        daily = daily_pick()
        picks.save_pick(daily, target)
        state = scheduler_state()
        with patch.object(alert, "dispatch_dm_for_health_alerts") as dispatched:
            if label == "Late":
                cutoff = NOW - timedelta(minutes=3)
                scheduler._refuse_late_delivery(daily, config, target, state, DAY, label, cutoff, NOW)
                names = daily.pick.batter_name
                if daily.double_down:
                    names += " + " + daily.double_down.batter_name
                expected = exact_body(label, names=names, m=3, cutoff=cutoff.strftime("%H:%M ET"))
            else:
                scheduler._refuse_void_game_delivery(daily, config, target, state, DAY, label, {101: "Postponed"})
                expected = exact_body(label, date=DAY, games="game 101 is Postponed")
            actual, = dispatched.call_args.args[0]
            print(label, repr(actual.message), actual.source, state.delivery_refusals, flush=True)
            assert actual.message == expected, (actual.message, expected)
            assert picks.load_pick(DAY, target) is None


def morning_settlement(root, mp):
    config, directory, clock, statuses = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    clock[0] = datetime(2027, 7, 15, 0, 20, tzinfo=ET)
    seed_automatic(directory, now=clock[0], recorded="2027-07-13T18:00:00-04:00", streak=13)
    prior = daily_pick()
    prior.date = "2027-07-14"
    prior.pick.game_time = "2027-07-15T02:10:00Z"
    prior.pick.game_pk = 9001
    picks.save_pick(prior, directory)
    assert picks.load_pick(prior.date, directory).pick.game_pk == 9001
    mp.setattr(picks, "get_game_statuses_detailed",
               lambda day: {9001: {"abstract": "L", "detailed": "In Progress"}}
               if day == prior.date else statuses)
    assert picks.get_game_statuses_detailed(prior.date)[9001]["abstract"] == "L"
    scheduler._morning_state_check(config, directory, DAY, clock[0], NOW)
    print("morning while prior entry is live", alert.send_dm.call_args.args[1], flush=True)
    auto_before = state_at(directory, clock[0])
    print(invoke("season-contract", "status", directory=directory).output.strip(), flush=True)
    filed = observation(DAY, directory, streak=14, best=15, saver="used")
    assert filed.exit_code == 0, filed.output
    # The prior entry settles as a miss, without a saver, bringing the streak to zero.
    set_automatic(directory, streak=0, now=clock[0] + timedelta(minutes=10))
    clock[0] = NOW
    control = root / "control"
    set_automatic(control, streak=0, now=NOW)
    for picks_path in (directory, control):
        account = picks_path / "account_state/contest_streak.json"
        account.write_text(json.dumps(dict(json.loads(account.read_text()), best_streak=15)))
    untreated = state_at(control, clock[0])
    treated = state_at(directory, clock[0])
    actions = []
    original, tier = orchestrator.run_cascade.return_value
    orchestrator.run_cascade.return_value = (original.assign(p_game_hit=[.82, .81]), tier)
    for chosen in [control, directory]:
        config["orchestrator"]["picks_dir"] = str(chosen)
        result = run_case(config)
        actions.append(result["selection"].decision.action)
    print("pre-settlement automatic", auto_before, "accepted dated", treated,
          "settled automatic", untreated, "settled vs dated actions", actions)
    assert treated[:2] == ("dated", 14)
    assert untreated[:2] == ("automatic", 0)


def saver_cases(root, mp):
    artifact = strategy._load_mdp()
    assert artifact is not None
    mp.setattr(strategy, "_load_mdp", lambda: artifact)
    saver_matrix(root, mp)


def main():
    probes = {
        "skip": inspect_skip, "warning": warnings_and_refusals, "saver": saver_cases,
        "repair": lambda root, mp: sent_pick(root, mp, True),
        "control": lambda root, mp: sent_pick(root, mp, False),
        "cli": ops_cli, "restore": first_boot, "research": research, "settlement": morning_settlement,
    }
    with tempfile.TemporaryDirectory(dir=HERE) as temp, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", offline)
        mp.setattr(socket, "create_connection", offline)
        probes[sys.argv[1]](Path(temp), mp)


if __name__ == "__main__":
    main()

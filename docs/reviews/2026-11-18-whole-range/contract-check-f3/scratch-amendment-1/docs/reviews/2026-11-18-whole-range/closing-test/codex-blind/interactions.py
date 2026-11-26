"""Exercise CLI delivery, refusals, midnight and replay through real entry points."""

import gzip
import json
from datetime import timedelta
from io import BytesIO

from bts import dm, entry_receipt, picks, scheduler, strategy
from bts.health import tail_policy

from midseason import automatic, network_fixture, REAL_GRADE
from offline import Interrupted, World, at
from opening import OPEN, DAY2


def cli_delivery():
    for kind in ("dated", "stale"):
        with World(OPEN) as w:
            w.config["health_checks"]["enabled"] = False
            if kind == "stale":
                w.now = at(OPEN, "00:10")
                automatic(w, 12, 15, False)
            else:
                w.dated()
            w.now = at(OPEN, "18:00")
            w.run()
            w.snapshot()
            w.now = at(OPEN, "18:50")
            w.invoke("run", "--date", OPEN, expect=0)
            w.now = at(OPEN, "21:00")
            w.invoke("orchestrate", "--date", OPEN, "--config", str(w.toml()))
            w.snapshot()
            w.dump(f"interactions-cli-{kind}")
    with World(OPEN) as w:
        w.now = at(OPEN, "00:10")
        automatic(w, 0, 0, False)
        w.now = at(OPEN, "17:00")
        w.invoke("run", "--date", OPEN, expect=0)
        w.snapshot()
        w.dump("interactions-cli-fresh-public-stale")

    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        w.dated()
        def accept_then_stop(recipient, body):
            w.on_send = None
            raise Interrupted()
        w.on_send = accept_then_stop
        try:
            w.run()
        except Interrupted:
            w.snapshot()
        w.now = at(OPEN, "19:09")
        w.invoke("run", "--date", OPEN, expect=0)
        w.run()
        w.snapshot()
        w.dump("interactions-cli-unknown-public")

    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        w.dated()
        def stop_before_grade():
            raise Interrupted()
        w.actions = [(at(OPEN, "18:50"), stop_before_grade)]
        try:
            w.run()
        except Interrupted:
            w.record("process_restart", "first DM already confirmed")
        w.invoke("orchestrate", "--date", OPEN, "--config", str(w.toml()), expect=0)
        w.snapshot()
        w.dump("interactions-cli-orchestrate-public")

    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        automatic(w, 12, 15, False)
        w.run()
        w.now = at(DAY2, "00:05")
        w.invoke("run", "--date", OPEN, expect=0)
        w.snapshot(OPEN)
        w.dump("interactions-cli-prior-slate")


def response_loss():
    real_send = dm.send_dm
    for failure in ("not_authenticated", "response_lost"):
        with World(OPEN) as w:
            from urllib.error import HTTPError

            count = 0
            auth_count = 0

            class LostResponse:
                def read(self):
                    raise TimeoutError("fixture: accepted DM, response body timed out")

            def transport(req, *, idempotent=True, **kwargs):
                nonlocal count, auth_count
                route = req.full_url.split("/")[-1]
                if route == "com.atproto.server.createSession":
                    auth_count += 1
                    if auth_count == 1 and failure == "not_authenticated":
                        raise HTTPError(req.full_url, 401, "fixture", {}, None)
                    payload = {"accessJwt": "offline"}
                elif route.startswith("com.atproto.identity.resolveHandle"):
                    payload = {"did": "did:plc:offline"}
                elif route.startswith("chat.bsky.convo.getConvoForMembers"):
                    payload = {"convo": {"id": "offline-cid"}}
                elif route == "chat.bsky.convo.sendMessage":
                    assert not idempotent
                    count += 1
                    w.record("remote_accept", json.loads(req.data)["message"]["text"])
                    if count == 1 and failure == "response_lost":
                        return LostResponse()
                    payload = {"id": f"offline-{count}"}
                else:
                    raise AssertionError(route)
                return BytesIO(json.dumps(payload).encode())

            w.config["health_checks"]["enabled"] = False
            w.dated()
            w.patch(dm, "send_dm", real_send)
            w.patch(dm, "get_dm_password", lambda: "offline")
            w.patch(dm, "retry_urlopen", transport)
            w.run()
            w.snapshot()
            w.dump(f"interactions-dm-{failure}")


def after_midnight_restart():
    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        w.games[OPEN] = []
        w.game("22:10")
        w.game("19:10", 102, day=DAY2)
        w.finish = at(DAY2, "00:35")
        w.dated()
        def stop_polling():
            raise Interrupted()
        w.actions = [(at(DAY2, "00:05"), stop_polling)]
        try:
            w.run()
        except Interrupted:
            w.record("process_restart", "prior slate still live")
        w.snapshot(OPEN)
        w.day = DAY2
        w.finish = at(DAY2, "22:00")
        w.invoke("schedule", "--config", str(w.toml()))
        w.snapshot(OPEN)
        w.dump("interactions-after-midnight-restart")

    with World(OPEN) as w:
        w.games[OPEN] = []
        w.game("22:10")
        w.finish = at(DAY2, "00:20")
        w.dated()
        w.actions = [(at(DAY2, "00:05"), lambda: w.dated(1))]
        w.run()
        w.switch("status")
        w.snapshot(OPEN)
        w.dump("interactions-historical-health")


def automatic_changes_after_selection():
    day = "2027-09-20"
    for variant in ("cached", "fresh"):
        with World(day) as w:
            w.config["health_checks"]["enabled"] = False
            w.game("19:12", 102, pair="B")
            w.now = at(day, "00:10")
            automatic(w, 45, 45, False)
            w.now = at(day)
            def correction():
                w.profile["activeStreak"] = 0
                w.fetch()
                w.invoke("saver-state", "--use", "--season", "2027", expect=0)
                w.snapshot()
            w.actions = [(at(day, "16:11"), correction)]
            def prediction():
                if w.now > at(day, "16:11"):
                    if variant == "cached":
                        return None, None
                    return w.frame((0.99, 0.85)), "offline"
                return w.frame((0.99, 0.85), projected=True), "offline"
            w.prediction = prediction
            w.run()
            w.snapshot()
            w.dump(f"interactions-automatic-correction-{variant}")


def entry_game_identity():
    for label, feed in (("right_game", 101), ("wrong_game", 102), ("unbound", None)):
        with World(OPEN) as w:
            w.game("20:10", 102)
            w.config["health_checks"]["enabled"] = False
            w.dated(30)
            w.prediction = lambda: (w.frame((0.99, 0.05)), "offline")
            def enter():
                w.pending = [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1}]
            w.actions = [(at(OPEN, "18:30"), w.watchdog), (at(OPEN, "18:44"), enter),
                         (at(OPEN, "18:45"), w.watchdog), (at(OPEN, "19:00"), w.watchdog)]
            units = entry_receipt.units_snapshot_dir(w.directory)
            units.mkdir(parents=True)
            payload = {"units": [{"id": 77, "roundId": 1, "feedId": feed}]}
            (units / "2027-03-24T210000Z.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode()))
            w.run()
            w.record("receipt_qualification",
                     [row.get("qualification") for row in entry_receipt.discover(w.directory, OPEN)])
            w.now = at(DAY2, "08:30")
            w.outcome()
            if label == "right_game":
                w.config["scheduler"].update(pick_delivery="private", entry_intent="research")
                w.outcome()
            w.snapshot()
            w.dump(f"interactions-entry-{label}")


def replay():
    for day, tomorrow, season in (("2027-03-24", DAY2, 2027), ("2027-12-31", "2028-01-01", 2027)):
        with World(day) as w:
            w.patch(picks, "resolve_daily_slot_results", REAL_GRADE)
            w.patch(picks, "_fetch_json", lambda url, **opts: network_fixture(w, url, **opts))
            primary = picks.pick_from_row(w.frame().iloc[0])
            daily = picks.DailyPick(day, w.now.isoformat(), primary, None, None,
                                    notification_sent=True, notification_id="offline-confirmed")
            picks.save_pick(daily, w.directory)
            picks.save_streak(42, w.directory, False, season=season - 1 if day == OPEN else season)
            w.now = at(tomorrow, "01:00")
            w.grade_command(expect=0)
            w.snapshot()
            w.results[101] = "walk"
            w.now = at(tomorrow, "02:00")
            w.invoke("reconcile", expect=0)
            w.snapshot()
            w.now = at(tomorrow, "07:40")
            w.invoke("reconcile", expect=0)
            w.snapshot()
            w.dump(f"interactions-replay-{day}")

    with World(OPEN) as w:
        w.patch(picks, "resolve_daily_slot_results", REAL_GRADE)
        w.patch(picks, "_fetch_json", lambda url, **opts: network_fixture(w, url, **opts))
        primary = picks.pick_from_row(w.frame().iloc[0])
        daily = picks.DailyPick(OPEN, w.now.isoformat(), primary, None, None,
                                notification_sent=True, notification_id="offline-confirmed")
        picks.save_pick(daily, w.directory)
        picks.save_streak(1, w.directory, True, season=2027, applied_date=OPEN)
        w.now = at(DAY2, "01:00")
        w.grade_command(expect=0)
        w.snapshot(OPEN)
        w.dump("interactions-score-retry")


def refusals():
    for case in ("old_auto", "auto_list", "model_list", "malformed_schedule",
                 "truncated_decision", "bad_decision", "no_recipient"):
        with World(OPEN) as w:
            account = w.directory / "account_state"
            account.mkdir()
            if case == "auto_list":
                w.switch("enable")
                (account / "contest_streak.json").write_text("[]")
            if case == "model_list":
                (w.directory / "streak.json").write_text("[]")
                w.dated()
                w.pending = [{"roundId": 1, "playerId": 1}]
            if case == "old_auto":
                w.switch("enable")
                (account / "contest_streak.json").write_text(json.dumps(
                    dict(active_streak=14, best_streak=14, source_date="2026-09-24", saver_available=False)))
            if case == "malformed_schedule":
                w.patch(scheduler, "retry_urlopen", lambda *args, **kwargs: BytesIO(b'{"dates":null}'))
            if case in {"truncated_decision", "bad_decision"}:
                w.dated()
                primary = picks.pick_from_row(w.frame().iloc[0])
                picks.save_pick(picks.DailyPick(OPEN, w.now.isoformat(), primary, None, None), w.directory)
                decision_dir = w.directory / OPEN
                decision_dir.mkdir()
                (decision_dir / "decision.json").write_text(
                    "{" if case == "truncated_decision" else
                    '{"schema_version":"bts_daily_decision_v3","action":[],"scoreable":true,"date":"2027-03-24"}')
            if case == "no_recipient":
                w.dated()
                w.config["bluesky"].clear()
            for attempt in range(2):
                w.now = at(OPEN, "10:00") + timedelta(minutes=attempt)
                if case == "model_list":
                    w.actions = [(at(OPEN, "18:45"), w.watchdog)]
                w.run(via_cli=True)
            w.fetch(expect=None)
            if case == "model_list":
                w.now = at(DAY2, "01:00")
                w.grade_command()
                w.now = at(DAY2, "02:00")
                w.invoke("reconcile")
            w.now = at(DAY2, "08:30")
            w.outcome(OPEN)
            w.snapshot(OPEN)
            w.dump(f"interactions-refusal-{case}")


def automatic_refusal_matrix():
    for case in ("bad_switch", "other_activation", "wrong_season", "legacy_manual", "invalid_dated",
                 "fetch_status_directory"):
        with World(OPEN) as w:
            w.config["health_checks"]["enabled"] = False
            automatic(w, 12, 15, False)
            account = w.directory / "account_state"
            automatic_file = account / "contest_streak.json"
            if case == "bad_switch":
                (account / "season_contract.json").write_text('{"status":"automatic"}')
            elif case in {"other_activation", "wrong_season"}:
                state = json.loads(automatic_file.read_bytes())
                field, value = (("activation_id", "0" * 32) if case == "other_activation" else ("season", 2026))
                state[field] = value
                automatic_file.write_text(json.dumps(state))
            elif case == "legacy_manual":
                w.invoke("set-contest-streak", "--streak", "4", expect=0)
            elif case == "invalid_dated":
                w.dated(4)
                manual = account / "contest_streak.manual.json"
                state = json.loads(manual.read_bytes())
                state["saver_state"] = []
                manual.write_text(json.dumps(state))
            elif case == "fetch_status_directory":
                status = w.directory.parent / "health_state" / "contest_streak_fetch_status.json"
                status.rename(status.with_suffix(".saved"))
                status.mkdir()
                w.profile["activeStreak"] = 13
                w.fetch(expect=2)
                w.switch("status")
                w.snapshot()
                w.dump(f"interactions-refusal-{case}")
                continue
            w.switch("status", expect=None)
            w.run(via_cli=True)
            w.snapshot()
            w.now = at(OPEN, "18:45")
            w.dated()
            repaired = w.switch("status", expect=None)
            assert "live decision now: accepted" in repaired.output, repaired.output
            w.snapshot()
            w.dump(f"interactions-refusal-{case}")


def clocks():
    with World(OPEN) as w:
        w.dated()
        w.now = at(OPEN, "23:59:59.999999")
        w.switch("status")
        for day in (OPEN, DAY2, "2027-09-26", "2027-09-27", "2027-12-31", "2028-01-01", "2028-03-24"):
            for when in ("00:00", "20:30"):
                w.now = at(day, when)
                w.record("clock", day, when, strategy._days_remaining(day), strategy.season_end_date(day),
                         [alarm.message for alarm in tail_policy.season_end_alerts(today=w.now.date())])
                w.switch("status")
                if day != OPEN:
                    w.dated(day=OPEN, expect=2)
        w.dump("interactions-clock")

    with World("2027-09-26") as w:
        automatic(w, 56, 56, False)
        w.now = at("2027-12-31", "23:59:59.999999")
        w.switch("status")
        w.now = at("2028-01-01", "00:00")
        w.switch("status")
        w.outcome("2027-09-27")
        w.outcome("2027-12-31")
        w.day = "2028-03-24"
        w.now = at(w.day)
        w.game("19:10")
        w.run(via_cli=True)
        w.snapshot()
        w.dump("interactions-clock-rollover")


if __name__ == "__main__":
    cli_delivery()
    response_loss()
    after_midnight_restart()
    automatic_changes_after_selection()
    entry_game_identity()
    replay()
    refusals()
    automatic_refusal_matrix()
    clocks()

"""Check R1 receipt/restart, untrusted reads, and all early shadow lock routes."""

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import review_world
from bts import daily_decision, picks, scheduler
from bts.health import alert as health_dm


review_world.fixture.HERE = Path(__file__).resolve().parent
DAY, at, offline = review_world.DAY, review_world.at, review_world.offline


def run(case):
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        review_world.pending_window(world, mp, scheduler)
        world.config["scheduler"]["shadow_model"] = True
        original_check = scheduler.run_single_check
        real_shadow = scheduler._trigger_shadow_prediction_on_lock
        original_alert = scheduler._alert_delivery_outcome_unknown
        regular = picks.read_regular_file_bytes
        decision_path = daily_decision.decision_path(DAY, world.picks)
        receipt = world.health / "delivery_outcome_unknown" / f"{DAY}.json"
        original_health_send = health_dm.send_dm
        attempts, starts, shadow, polls = [], [], [], []
        journal = io.StringIO()
        published = False
        original_record = None

        def publish():
            nonlocal published, original_record
            published = True
            current = picks.load_pick(DAY, world.picks)
            if case == "replacement-shadow":
                current.pick = replace(current.pick, batter_id=999, batter_name="Other Committed Batter")
            if case not in {"initial-scoreable", "late-delivered"}:
                current.delivery_attempted = True
            if case == "late-delivered":
                current.notification_id = "already-delivered"
                current.notification_sent = True
            if case != "initial-scoreable":
                picks.save_pick(current, world.picks)
            if case in {"receipt", "receipt-unacknowledged", "initial-scoreable", "replacement-shadow"}:
                assert daily_decision.write_decision(
                    DAY, world.picks, action="single", source="mdp", scoreable=True,
                    delivery_status="private_locked" if case == "initial-scoreable" else "locked_unconfirmed",
                    primary=scheduler._row_from_daily(current.pick),
                )
                original_record = decision_path.read_bytes()
            elif case == "attempt-fifo":
                os.mkfifo(decision_path)
            elif case == "attempt-unreadable":
                decision_path.write_text("{")

        def check(*args, **kwargs):
            result = original_check(*args, **kwargs)
            if not published and case in {"initial-scoreable", "initial-matching-attempt"}:
                publish()
                if case == "initial-matching-attempt":
                    # Exercise the branch with an explicitly injected matching result.
                    result["saved_pick_bytes"] = (world.picks / f"{DAY}.json").read_bytes()
            return result

        def confirmation(pks):
            if pks == [102] and starts and not published:
                publish()
            return {pk: {"away", "home"} for pk in pks
                    if pk == 101 or published and case == "attempt-deliver"}

        def forecast():
            starts.append(world.now.isoformat())
            if len(starts) == 1:
                world.advance(22 * 60)
                return world.frame((0.99, 0.40), (True, True))
            return None

        def record_read(file):
            if case == "attempt-unreadable" and published and file == decision_path:
                raise PermissionError("fixture denies the added decision read")
            return regular(file)

        def trigger(config, date, batter):
            shadow.append(batter)
            if case == "replacement-shadow":
                real_shadow(config, date, batter)

        def alert(config, directory, date):
            attempts.append(date)
            original_alert(config, directory, date)

        def acknowledgement(recipient, body):
            result = original_health_send(recipient, body)
            return None if len(attempts) == 1 else result

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(scheduler, "check_confirmed_lineups", confirmation)
        mp.setattr(scheduler, "_trigger_shadow_prediction_on_lock", trigger)
        mp.setattr(scheduler, "predict_local_shadow", lambda *a, **kw: world.frame((.99, .40), (False, True)))
        mp.setattr(scheduler, "_alert_delivery_outcome_unknown", alert)
        mp.setattr(picks, "read_regular_file_bytes", record_read)
        mp.setattr(scheduler, "run_result_polling", lambda pk, *a, **kw: polls.append(pk) or "fixture")
        if case == "receipt-unacknowledged":
            mp.setattr(health_dm, "send_dm", acknowledgement)
        with redirect_stderr(journal):
            scheduler.run_day(DAY, world.config)
            if case in {"receipt", "receipt-unacknowledged"}:
                first_receipt = receipt.read_bytes()
                scheduler.run_day(DAY, world.config)
                assert (receipt.read_bytes() == first_receipt) == (case == "receipt")
                assert json.loads(first_receipt)["status"] == ("sent" if case == "receipt" else "failed")
                assert json.loads(receipt.read_bytes())["status"] == "sent"
        decision = daily_decision.load_decision(DAY, world.picks)
        health = review_world.health(world)
        shadow_lines = [line.strip() for line in journal.getvalue().splitlines() if "[SHADOW MODEL]" in line]
        print("RECOVERY", json.dumps({"case": case, "cascades": starts, "alert_calls": attempts,
                                      "health": health, "shadow_names": shadow, "shadow_journal": shadow_lines,
                                      "scoreable": decision["scoreable"], "primary": decision["primary"],
                                      "result_polls": polls,
                                      "receipt": json.loads(receipt.read_bytes()) if receipt.is_file() else None}), flush=True)
        assert len(starts) == 1 and not world.sends("dm")
        expected_alerts = 2 if case == "receipt-unacknowledged" else int(case not in {"initial-scoreable", "late-delivered"})
        assert sum("OUTCOME UNKNOWN" in msg for msg in health) == expected_alerts
        if original_record:
            assert decision_path.read_bytes() == original_record
        if case in {"receipt", "receipt-unacknowledged"}:
            assert attempts == [DAY, DAY] and shadow == ["Synthetic Batter 101"] * 2
        else:
            assert shadow == ["Synthetic Batter 101"]
        if case == "replacement-shadow":
            assert decision["primary"]["batter_name"] == "Other Committed Batter"
            assert any("AGREES" in line for line in shadow_lines)


if __name__ == "__main__":
    for case in sys.argv[1:] or ["receipt", "initial-scoreable", "initial-matching-attempt", "attempt-deliver",
                                "attempt-fifo", "attempt-unreadable", "late-delivered", "replacement-shadow"]:
        run(case)

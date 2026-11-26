"""Exercise actual slate-based recovery alongside legacy shared writers."""

from datetime import timedelta
from contextlib import redirect_stderr
from io import StringIO
import json

from probe_support import (
    attempt_direct, day_pick, deadline, fixtures, prepare_schedules, raw, receipt,
    shared, status_body, world_at,
)


def interleave(root, patch, opening, order):
    from bts import dm, picks, scheduler
    from bts.health import alert, runner

    day = "2027-03-25"
    with world_at(root, patch, opening, opening.at(day, "18:56")) as (config, directory, now, _, scoped):
        recovery = day_pick(day)
        recovery.delivery_attempted = True
        cutoff_store = directory.with_name("cutoff")
        path = shared(directory)
        path.parent.mkdir()
        path.write_bytes(b"{}\n")
        own = receipt(directory, day)

        def recovery_encounter():
            before = raw(path)
            assert attempt_direct(config, directory, recovery) is False
            assert raw(path) == before

        def cutoff_encounter():
            before = raw(own)
            assert attempt_direct(config, cutoff_store, day_pick(day)) is False
            assert raw(own) == before

        encounters = {"unknown": recovery_encounter, "cutoff": cutoff_encounter}
        for route in order:
            encounters[route]()
        bodies = [call.args[1] for call in alert.send_dm.call_args_list]
        assert [body == status_body(day) for body in bodies] == [case == "unknown" for case in order]
        assert sum("LATE DELIVERY REFUSED" in body for body in bodies) == 1
        assert json.loads(path.read_bytes())["sent_sources"] == ["CRITICAL:late_delivery"]
        original = own.read_bytes()

        alert.send_dm.side_effect = [OSError("synthetic unrelated failure"), "different-slate-eod"]
        scheduler._alert_contest_state_failure(config, ValueError("unrelated contest observation"))
        assert json.loads(path.read_bytes())["status"] == "failed"
        assert own.read_bytes() == original
        recovery_encounter()
        assert own.read_bytes() == original and alert.send_dm.call_count == 3

        eod_store = directory.with_name("eod")
        eod_day = now[0].date() - timedelta(days=1)
        eod_pick = day_pick(eod_day.isoformat())
        eod_pick.delivery_attempted = True
        picks.save_pick(eod_pick, eod_store)
        scoped.setattr(runner, "_safe_run", lambda name, check: check() if name == "post_failure" else [])
        scoped.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        runner.run_all_checks(eod_store, root / "models", "offline.test", today=eod_day)
        payload = json.loads(path.read_bytes())
        assert payload["status"] == "sent" and payload["sent_day"] == eod_day.isoformat()
        assert payload["sent_sources"] == ["CRITICAL:pick_delivery"]
        assert own.read_bytes() == original
        recovery_encounter()
        assert own.read_bytes() == original and alert.send_dm.call_count == 4
        dm.send_dm.assert_not_called()
    print(f"{order}: two independent initial DMs, failed unrelated alert and other-slate EOD; "
          "all intervening receipt byte comparisons passed.")


def rollover(root, patch, opening, old, new):
    from bts import dm, picks, scheduler
    from bts.health import alert

    with world_at(root, patch, opening, opening.at(old, "18:20")) as (config, directory, now, _, _):
        days = [day_pick(day) for day in (old, new)]
        prepare_schedules(*days)
        for daily in days:
            daily.delivery_attempted = True
            picks.save_pick(daily, directory)
        path = shared(directory)
        path.parent.mkdir()
        path.write_bytes(b'{"status":"sent","sent_day":"' + old.encode() +
                         b'","sent_sources":["CRITICAL:late_delivery"]}\n')
        shared_before = path.read_bytes()
        old_receipt = receipt(directory, old)
        new_receipt = receipt(directory, new)

        assert scheduler.run_day(old, config) is None
        assert path.read_bytes() == shared_before
        original = old_receipt.read_bytes()
        assert not new_receipt.exists()
        assert json.loads(original)["sent_day"] == old
        alert.send_dm.assert_called_once_with("offline.test", status_body(old))

        now[0] = opening.at(new, "00:01")
        assert scheduler.run_day(old, config) is None
        assert old_receipt.read_bytes() == original and not new_receipt.exists()
        assert path.read_bytes() == shared_before
        alert.send_dm.assert_called_once()

        now[0] = opening.at(new, "18:20")
        assert scheduler.run_day(new, config) is None
        assert old_receipt.read_bytes() == original and json.loads(new_receipt.read_bytes())["sent_day"] == new
        assert path.read_bytes() == shared_before
        assert [item.args[1] for item in alert.send_dm.call_args_list] == [status_body(old), status_body(new)]
        dm.send_dm.assert_not_called()
    print(f"{old} -> {new}: same-slate dedup at 00:01 ET, distinct next slate receipt, shared bytes unchanged.")


def main():
    stderr = StringIO()
    with deadline(45), fixtures("isolation") as (root, patch, opening), redirect_stderr(stderr):
        for order in [("unknown", "cutoff"), ("cutoff", "unknown")]:
            interleave(root / order[0], patch, opening, order)
        for old, new in [("2027-03-25", "2027-03-26"), ("2027-12-31", "2028-01-01")]:
            rollover(root / old, patch, opening, old, new)


if __name__ == "__main__":
    main()

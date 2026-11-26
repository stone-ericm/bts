"""Check persistent faults and operator-visible signals through scheduler and CLI."""
import gzip
import io
import json
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness import Clock, World, scheduler, write_json
from interactions import auto
from opening import D1, D2
from season import outcome


def doctor(root, name, real_fetch):
    research = name == "bad_model_fallback"
    w = World(root, D1, "16:01", live=not research, required=not research)
    with w.switched_on():
        w.slate(D1)
        account = w.picks / "account_state"
        account.mkdir()
        switch = account / "season_contract.json"
        reading = account / "contest_streak.json"
        manual = account / "contest_streak.manual.json"
        if name == "corrupt_switch":
            switch.write_text("{")
        if name == "bad_loadable_dated":
            write_json(manual, {"schema_version": "bts_contest_streak_manual_v3", "effective_date": D1})
        if name == "bad_automatic_utf8":
            reading.write_bytes(b"\xff")
            w.fetch(0, 0)
        if name == "no_source":
            w.invoke("season-contract", "enable", "--season", 2027, "--picks-dir", w.picks)
            w.fetch(0, 0)
        if name == "overlong_saver":
            auto(w)
            (account / "saver_state.json").write_text('{"season":' + "9" * 5000 + "}")
        if name == "bad_model_fallback":
            write_json(w.picks / "streak.json", {"season": 2027, "streak": []})
        if name == "bad_decision":
            w.dated()
            w.run()
            (w.picks / D1 / "decision.json").write_bytes(b"\xff")
        if name == "wrong_account":
            w.dated()
            write_json(reading, {"username": "other-offline-account"})
            w.fetch(0, 0)
        invalid_schedules = {
            "schedule_top": b'{"dates": {}}',
            "schedule_missing_time": json.dumps({"dates": [{"games": [{
                "gameType": "R", "gamePk": 700, "teams": {
                    "away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}},
                },
            }]}]}).encode(),
        }
        for tick in ("16:01", "17:01"):
            w.set_day(D1, tick)
            if name in invalid_schedules:
                with (
                    patch.object(scheduler, "fetch_schedule", real_fetch),
                    patch.object(scheduler, "retry_urlopen", lambda *a, **kw: io.BytesIO(invalid_schedules[name])),
                ):
                    w.invoke("schedule", "--config", w.config_path)
            else:
                w.invoke("schedule", "--config", w.config_path)
        w.set_day(D2, "08:30")
        if name in invalid_schedules:
            with (
                patch.object(scheduler, "fetch_schedule", real_fetch),
                patch.object(scheduler, "retry_urlopen", lambda *a, **kw: io.BytesIO(invalid_schedules[name])),
            ):
                outcome(w, D1)
        else:
            outcome(w, D1)
    return {"case": name, "events": w.events, "snapshot": w.snapshot()}


def park_drag(root):
    from bts.features import park_drag_producer
    w = World(root, D1, "07:45", health=False)

    class OpeningDate(date):
        @classmethod
        def today(cls):
            return date.fromisoformat(D1)

    store = root / "drag/producer"
    store.mkdir(parents=True)
    (store / "games_meta.csv").write_text("game_pk\n1\n")
    (store / "game_level_static.csv").write_text("game_pk\n1\n")
    with w.switched_on(), patch.object(park_drag_producer, "date", OpeningDate):
        for body in ("game_pk,game_date\n1,2026-09-27\n", "game_pk,game_date\n"):
            (store / "pitches_current.csv.gz").write_bytes(gzip.compress(body.encode()))
            w.invoke("park-drag-refresh", "--root", store.parent)
    return {"case": "B7", "events": w.events, "snapshot": w.snapshot()}


if __name__ == "__main__":
    campaign = ("corrupt_switch", "bad_loadable_dated", "bad_automatic_utf8", "no_source",
                "overlong_saver", "bad_model_fallback", "bad_decision", "wrong_account",
                "schedule_top", "schedule_missing_time")
    rows = []
    fetch_schedule = scheduler.fetch_schedule
    for case in campaign:
        with TemporaryDirectory(prefix="wrblind-refusal-") as temp:
            rows.append(doctor(Path(temp), case, fetch_schedule))
    with TemporaryDirectory(prefix="wrblind-park-") as temp:
        rows.append(park_drag(Path(temp)))
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2))

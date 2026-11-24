"""Replay literal operator instructions against synthetic live state."""

from datetime import date

from bts import cli, scheduler
from bts.leaderboard import auth
from midseason import automatic
from offline import World, at
from opening import OPEN, DAY2


def decision_file_drill():
    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        w.dated()
        w.run()
        decision = w.directory / OPEN / "decision.json"
        backup = decision.with_name("decision.drill-aside.json")
        w.now = at(OPEN, "18:44")
        decision.rename(backup)
        w.outcome(OPEN)
        w.pending = [{"roundId": 1, "playerId": 1}]
        w.now = at(OPEN, "18:45")
        w.watchdog()
        backup.rename(decision)
        w.now = at(OPEN, "19:00")
        w.watchdog()
        w.now = at(DAY2, "08:30")
        w.outcome(OPEN)
        w.snapshot(OPEN)
        w.dump("operator-decision-drill")


def stale_warning_after_midnight():
    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        w.games[OPEN] = []
        w.game("22:10")
        w.finish = at(DAY2, "00:35")
        w.now = at(OPEN, "00:10")
        automatic(w, 5, 5, False)
        w.now = at(OPEN)

        def read_warning():
            w.record("fixture_account", "prior game still in progress; counter on screen is 6")
            w.dated(6, 6)

        w.actions.append((at(DAY2, "00:05"), read_warning))
        w.run()
        w.day = DAY2
        w.now = at(DAY2)
        w.game("19:10")
        w.finish = at(DAY2, "22:00")
        w.profile.update(activeStreak=7, seasonBestStreak=7)
        w.rounds = {1: at(OPEN).date()}
        w.fetch()
        w.record("fixture_account", "prior game settled; counter on screen and in refreshed automatic file is 7")
        w.run()
        w.snapshot(DAY2)
        w.dump("operator-stale-midnight")


def rollback_config():
    with World(OPEN) as w:
        w.config["scheduler"]["pick_delivery"] = "private"
        # A7 says to change delivery and comment the cron line; intent remains enter.
        w.invoke("entry-intent", "--config", str(w.toml()))
        w.run(via_cli=True)
        w.now = at(DAY2, "08:30")
        w.outcome(OPEN)
        w.snapshot(OPEN)
        w.dump("operator-a7")


def park_drag():
    from bts.features import park_drag_producer
    import pandas as pd

    with World(OPEN) as w:
        class Today(date):
            @classmethod
            def today(cls):
                return w.now.date()

        w.patch(park_drag_producer, "date", Today)
        producer = w.root / "data/external/park_drag/producer"
        producer.mkdir(parents=True)
        for filename in ("games_meta.csv", "game_level_static.csv"):
            (producer / filename).write_text("game_pk\n")
        store = producer / "pitches_current.csv.gz"
        pd.DataFrame({"game_date": ["2026-09-14"]}).to_csv(store, index=False)
        w.invoke("park-drag-refresh", expect=1)
        pd.DataFrame({"game_date": []}).to_csv(store, index=False)
        w.invoke("park-drag-refresh", expect=1)

        # One realistic-sized pitch group from the first game. Empty incremental
        # fetch simulates Savant indexing behind an already seeded opener.
        w.now = at(DAY2, "07:45")
        rows = [dict(game_pk=101, game_date=OPEN, home_team="NYY", release_speed=91.0,
                     vx0=0.0, vy0=-130.0, vz0=0.0, ax=0.0, ay=27.0, az=-32.0)] * 30
        pd.DataFrame(rows).to_csv(store, index=False)
        pd.DataFrame([dict(game_pk=101, temp_f=60.0, venue_id=1, venue="Fixture Park")]).to_csv(
            producer / "games_meta.csv", index=False)
        pd.DataFrame([dict(game_pk=900, season=2026, game_date="2026-09-14", venue_id=1,
                           venue="Fixture Park", n_pitch=30, cd_trim=0.40)]).to_csv(
            producer / "game_level_static.csv", index=False)
        w.patch(park_drag_producer, "fetch_savant_ff", lambda *args, **kwargs: pd.DataFrame())
        w.patch(park_drag_producer, "_jitter_sleep", lambda: None)
        w.invoke("park-drag-refresh", expect=0)
        w.dump("operator-park-drag")


def manual_after_refusal():
    with World(OPEN) as w:
        w.config["health_checks"]["enabled"] = False
        w.game("20:10", 102, pair="B")
        w.game("21:10", 103, pair="C")
        w.dated(30, 30)
        w.prediction = lambda: (w.frame((0.999, 0.98)), "offline")
        current = scheduler._void_committed_games

        def response_latency(daily, day):
            if w.now < at(OPEN, "19:05"):
                w.now = at(OPEN, "19:05:30")
            return current(daily, day)

        def follow_manual_instruction(recipient, body):
            if "Enter a pick manually" in body:
                w.pending = [{"roundId": 1, "playerId": 9}]
                w.record("fixture_account", "operator entered different batter from later game")

        w.patch(scheduler, "_void_committed_games", response_latency)
        w.patch(cli, "_fetch_bts_to_mlb", lambda: {9: 100009})
        w.on_send = follow_manual_instruction
        w.actions.append((at(OPEN, "19:45"), w.watchdog))
        w.run()
        w.snapshot()
        w.dump("operator-manual-after-refusal")


def auth_wording():
    for error in (auth.RateLimitedLoginError, auth.TransientAuthError, auth.AuthError):
        with World(OPEN) as w:
            def fail(**args):
                raise error("captured fixture classification")
            w.patch(auth, "fetch_login_session", fail)
            w.fetch(expect=2)
            w.snapshot()
            w.dump(f"operator-auth-{error.__name__}")


def saver_instruction():
    from bts import contest_state, web

    with World(OPEN) as w:
        w.invoke("saver-state", "--init", "used", "--season", "2027", expect=0)
        w.dated(12, 12, "active")
        context = web.saver_dashboard_context(w.directory, now=w.now)
        w.record("dashboard", web.render_saver_section(context, decision_streak=12))
        w.record("click_undo", web.saver_transition_response(
            w.directory, expected_prior="used", new_state="active", same_origin=True, now=w.now))
        w.record("dated_authority", contest_state.load_decision_streak_state(w.directory, require_contest_state=True).saver_available)
        w.dump("operator-saver-dashboard")

    with World("2027-12-31") as w:
        w.now = at("2027-12-31", "23:55")
        w.invoke("saver-state", "--use", expect=1)
        w.now = at("2028-01-01", "00:10")
        w.invoke("saver-state", "--init", "not_earned", "--season", "2028", expect=0)
        w.snapshot()
        # A delayed old-year init command would be authorized by CLI against the
        # same single flag path even while the newer season's flag is present.
        w.invoke("saver-state", "--init", "used", "--season", "2027", expect=0)
        w.invoke("saver-state", expect=0)
        w.snapshot()
        w.dump("operator-saver-year")


def artifact_manifest_instruction():
    from bts.data import sync
    from datetime import timedelta

    with World(OPEN) as w:
        key = sync.DEFAULT_MANIFEST_KEY
        objects = {f"objects/offline/pa_{year}.parquet" for year in (2024, 2025)}
        manifest = {key: {"files": {name: {"key": name} for name in sorted(objects)}}}

        class MemoryStore:
            def get_object_json(self, name):
                return manifest.get(name)

            def put_object_json(self, name, data):
                manifest[name] = data

            def copy_object(self, source, target):
                manifest[target] = manifest[source]

            def delete_object(self, name):
                if name in manifest:
                    del manifest[name]
                else:
                    objects.remove(name)
                w.record("fixture_object_delete", name)

            def list_objects(self, prefix):
                return [{"key": name, "last_modified": w.now - timedelta(days=8)}
                        for name in sorted(objects) if name.startswith(prefix)]

        store = MemoryStore()
        w.patch(sync.R2Client, "from_env", lambda: store)
        processed = w.root / "empty/raw"
        models = w.root / "empty/models"
        processed.mkdir(parents=True)
        models.mkdir()
        command = ("data", "sync-to-r2", "--processed-dir", str(processed), "--models-dir", str(models))
        result = w.invoke(*command, expect=1)
        assert "delete the R2 manifest manually first" in str(result.exception)
        w.record("fixture_state_before", sorted(manifest), sorted(objects))
        store.delete_object(key)
        w.invoke(*command, expect=0)
        w.record("fixture_state_after", sorted(manifest), sorted(objects))
        w.dump("operator-r2-manifest")


if __name__ == "__main__":
    decision_file_drill()
    stale_warning_after_midnight()
    rollback_config()
    park_drag()
    manual_after_refusal()
    auth_wording()
    saver_instruction()
    artifact_manifest_instruction()

"""Compare restored health checks and replay the earlier offline controls."""
import datetime as dt
import json
import subprocess
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from review_runner import REVIEW, TREES, environment, first_review, load
from capture_probe import capture, compare_capture

prior = load("confirmation_probes", REVIEW / "confirm-probes.py")
ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt.datetime


def contest(root, revision):
    from bts.health.contest_state import check
    from bts.picks import save_pick
    from tests.conftest import covered_automatic
    from tests.test_shadow_eval import _daily, _pick

    picks = root.parent / "contest-picks"
    picks.mkdir(exist_ok=True)
    save_pick(_daily("2027-07-15", _pick("One", 1, game_pk=1001), result="hit"), picks)
    results = {}
    for when in ("2027-07-15T22:15", "2027-07-16T00:01", "2027-07-17T00:00"):
        now = REAL_DATETIME.fromisoformat(when).replace(tzinfo=ET)
        covered_automatic.__wrapped__()(picks, now, {
            "active_streak": 1, "best_streak": 1, "source_date": None,
            "recorded_at": (now - dt.timedelta(minutes=15)).isoformat(),
        })
        alerts = check(picks, expected=True, now=now)
        coverage = [alert.level for alert in alerts if "no source_date" in alert.message]
        expected = "INFO" if revision == "before" and not when.startswith("2027-07-17") else "CRITICAL"
        assert coverage == [expected], alerts
        results[when] = [(alert.level, alert.message) for alert in alerts]
    prior.write(root / "summary.json", results)


def park(root, revision):
    from bts import strategy
    from bts.health import park_drag_freshness as health

    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    instant = None
    status = {"ok": True}
    prior.write(root / "producer_status.json", status)
    (root / "park_drag_export.csv").write_text("fixture\n")
    results = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(health, "datetime", Clock)
        assert 2028 not in strategy.SEASON_END_DATES
        if revision == "before":
            patch.setitem(strategy.SEASON_OPENER_DATES, 2029, "2029-03-24")
        for label, day, source in (
            ("threshold-at-close", "2026-09-28", "2026-09-22"),
            ("threshold-next-opening", "2027-03-26", "2026-09-22"),
            ("unconfigured-2028-close", "2029-03-26", "2028-09-23"),
        ):
            instant = REAL_DATETIME.fromisoformat(f"{day}T12:00").replace(tzinfo=ET)
            prior.write(root / "park_drag_manifest.json", {
                "max_source_game_date": source,
                "generated_at": instant.replace(tzinfo=None).isoformat(),
            })
            alerts = health.check(root, today=instant.date())
            expected = [] if revision == "before" and label != "threshold-at-close" else ["CRITICAL"]
            assert prior.levels(alerts) == expected, (label, alerts)
            results[label] = [(alert.level, alert.message) for alert in alerts]
    prior.write(root / "summary.json", results)


def producer(root, revision):
    from bts.features import park_drag_producer as producer
    from bts.health import park_drag_freshness as health
    from tests.test_park_drag_producer import TestRefresh, _FakeResp

    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    instant = REAL_DATETIME(2027, 7, 15, 12, tzinfo=ET)
    seed = TestRefresh()
    results = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(producer, "_jitter_sleep", lambda: None)
        patch.setattr(producer, "_now_iso", lambda: instant.replace(tzinfo=None).isoformat())
        patch.setattr(health, "datetime", Clock)

        def seed_empty(directory, *, empty_static=False):
            seed._seed(directory)
            store = directory / "producer/pitches_current.csv.gz"
            empty = pd.read_csv(store).iloc[:0]
            empty.to_csv(store, index=False, compression="gzip")
            if empty_static:
                static = directory / "producer/game_level_static.csv"
                pd.read_csv(static).iloc[:0].to_csv(static, index=False)
            return lambda *a, **k: _FakeResp(text=empty.to_csv(index=False))

        reseed = root / "first-game"
        empty_feed = seed_empty(reseed)
        results["pre-opener"] = producer.refresh(reseed, today=dt.date(2027, 3, 24), http_get=empty_feed)
        results["first-game"] = producer.refresh(
            reseed, today=dt.date(2027, 3, 25), http_get=seed._fake_http("2027-03-24"),
        )
        for key in ("pre-opener", "first-game"):
            assert results[key]["ok"] is (revision != "base"), results
        if revision != "base":
            assert results["first-game"]["current_season_games"] == 1
            assert results["first-game"]["max_source_game_date"] == "2027-03-24"
            assert len(pd.read_csv(reseed / "producer/pitches_current.csv.gz")) == 30

        for existing in (False, True):
            directory = root / f"both-empty-existing-{existing}"
            empty_feed = seed_empty(directory, empty_static=True)
            manifest = directory / "park_drag_manifest.json"
            table = directory / "park_drag_export.csv"
            if existing:
                prior.write(manifest, {"max_source_game_date": "2027-07-14",
                                       "generated_at": "2027-07-15T07:45:00"})
                table.write_text("venue_id,date,park_drag_delta\n3313,2027-07-15,0.01\n")
            status = producer.refresh(directory, today=instant.date(), http_get=empty_feed)
            alerts = health.check(directory, today=instant.date())
            expected = (["WARN"] if existing else ["WARN", "WARN"]) if revision == "base" else (
                [] if existing else ["WARN"])
            assert prior.levels(alerts) == expected, (revision, existing, status, alerts)
            assert manifest.exists() is existing and table.exists() is existing
            results[f"both-empty-existing-{existing}"] = {
                "status": status, "health": [(alert.level, alert.message) for alert in alerts],
                "manifest_kept": manifest.exists(), "table_kept": table.exists(),
            }
    prior.write(root / "summary.json", results)


def cron(root, revision):
    # Reuse the earlier real cron entry-point fixture with a stub crontab and local CLI.
    real_run = subprocess.run

    def guarded_run(*args, **kwargs):
        if "env" in kwargs:
            guard = environment(TREES[revision])
            kwargs["env"].update({key: guard[key] for key in ("TZ", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")})
        return real_run(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "run", guarded_run)
        first_review().cron(root, revision)


def compare_cron(parent):
    base = json.loads((parent / "base/show.json").read_text())
    head = json.loads((parent / "head/show.json").read_text())
    for output, directory in ((base, "base"), (head, "head")):
        for intent in output:
            output[intent] = output[intent].replace(str(parent / directory), "<fixture>")
    assert base["enter"] == head["enter"]
    assert base["research"].count("fetch-contest-streak") == head["research"].count("fetch-contest-streak") == 4
    stripped = "\n".join(
        line.replace(" --dm-recipient stonehengee.bsky.social", "")
        if "bts fetch-contest-streak" in line else line
        for line in base["research"].split("\n")
    )
    assert stripped == head["research"]


PROBES = {
    "contest": contest, "park": park, "capture": capture, "producer": producer, "cron": cron,
    "shadow": prior.shadow, "p03": prior.closeout,
    "pick": lambda root, revision: first_review().pick_path(root, revision),
}
SELECTED = dict.fromkeys(PROBES, ("base", "head"))
for family in ("contest", "park", "capture"):
    SELECTED[family] = ("base", "before", "head")
COMPARES = dict.fromkeys(PROBES, lambda _: None)
COMPARES.update(
    contest=lambda parent: prior.match(parent, ("summary.json",)),
    park=lambda parent: prior.match(parent, ("summary.json",)),
    capture=compare_capture,
    cron=compare_cron,
    shadow=prior.compare_shadow,
    p03=lambda parent: prior.match(parent, ("closeout.json", "closeout.stdout")),
    pick=lambda parent: prior.match(parent, ("pick.json", "decision.json", "score.json", "sends.json")),
)

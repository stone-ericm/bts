"""C1 rank 4a, T6: orchestration end to end on a fully synthetic data tree (no real slate, feed or calendar)."""
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.audit.c1_r4a import calendar as C
from scripts.audit.c1_r4a import run as RUN

ET = ZoneInfo("America/New_York")
DAYS = [date(2027, 4, 1) + timedelta(days=i) for i in range(95)]


def cal_raw():
    return json.dumps({"schema": "c1_r4a_calendar_v1", "season": 2027, "dates": [d.isoformat() for d in DAYS]}).encode()


def feed(pk, d, hits):
    """batter_id -> hit? (each batter one PA)."""
    plays = [{"about": {"atBatIndex": i, "isComplete": True, "startTime": f"{d}T23:30:00Z"},
              "matchup": {"batter": {"id": b}}, "result": {"eventType": "single" if h else "strikeout"}}
             for i, (b, h) in enumerate(hits.items())]
    return json.dumps({"gamePk": pk, "gameData": {"game": {"pk": pk}, "status": {"detailedState": "Final"},
                                                  "datetime": {"officialDate": d}},
                       "liveData": {"plays": {"allPlays": plays}}}).encode()


def world(tmp_path, dates, *, missing_feed=()):
    """One game per date with two batters; batter 1 (p .8) hits on even days, batter 2 (p .6) never."""
    data = tmp_path / "data"
    (data / "picks" / "slates").mkdir(parents=True)
    (data / "hetzner_results" / "c1").mkdir(parents=True)      # exists on the box (the C1 cycle root)
    (data / "raw" / "2027").mkdir(parents=True)
    for n, d in enumerate(dates):
        pk = 900000 + n
        rows = [{"batter_id": 1, "game_pk": pk, "p_game_hit": 0.8, "game_time": f"{d}T23:10:00Z", "status": "Scheduled",
                 "projected": False},
                {"batter_id": 2, "game_pk": pk, "p_game_hit": 0.6, "game_time": f"{d}T23:10:00Z", "status": "Scheduled",
                 "projected": True}]
        (data / "picks" / "slates" / f"{d}.json").write_text(json.dumps(
            {"schema_version": "bts_slate_v2", "date": d.isoformat(), "tier": "local",
             "written_at": f"{d}T16:00:00+00:00", "n_rows": 2, "rows": rows}))
        if pk not in missing_feed:
            (data / "raw" / "2027" / f"{pk}.json").write_bytes(feed(pk, d.isoformat(), {1: n % 2 == 0, 2: False}))
    return data


@pytest.fixture
def patched(monkeypatch, tmp_path):
    raw = cal_raw()
    pin = hashlib.sha256(raw).hexdigest()
    cal_path = tmp_path / "calendar.json"
    cal_path.write_bytes(raw)

    def apply(data, now):
        monkeypatch.setattr(RUN, "DATA", data)
        monkeypatch.setattr(RUN, "admission_gate", lambda: ("f" * 40, {"input_pins": {"calendar": pin}},
                                                            {"review_report": "r.md"}))
        monkeypatch.setattr(RUN, "REPO", tmp_path)
        monkeypatch.setattr(RUN, "CALENDAR_REL", "calendar.json")
        monkeypatch.setattr(RUN, "REGISTER_REL", "reg.md")
        (tmp_path / "reg.md").write_text("")
        monkeypatch.setattr(RUN, "now_et", lambda: now)
    return apply


FIT_OPENS = datetime(2027, 5, 11, 8, 0, tzinfo=ET)        # day after date 40 (2027-05-10)
EVAL_OPENS = datetime(2027, 7, 1, 8, 0, tzinfo=ET)        # day after date 90 (2027-06-29)


def runs(data, stage):
    return [d for d in sorted((data / "hetzner_results" / "c1" / "r4a" / stage).iterdir()) if d.is_dir()]


def test_a_stage_is_refused_before_it_opens(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS - timedelta(seconds=1))
    with pytest.raises(SystemExit, match="opens at"):
        RUN.main(["--stage", "fit"])


def test_the_fit_freezes_slates_before_the_claim_and_reads_feeds_after(tmp_path, patched, monkeypatch):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS)
    events, real_read, real_claim = [], Path.read_bytes, RUN.A.write_claim

    def read_bytes(self):
        if "slates" in self.parts:
            events.append("slate")
        elif "raw" in self.parts:
            events.append("feed")
        return real_read(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(RUN.A, "write_claim", lambda d, h: (events.append("claim"), real_claim(d, h))[1])
    assert RUN.main(["--stage", "fit"]) == 0
    k = events.index("claim")
    assert set(events[:k]) == {"slate"} and set(events[k + 1:]) == {"feed"}
    (d,) = runs(data, "fit")
    freeze, res = json.loads((d / "freeze.json").read_text()), json.loads((d / "results.json").read_text())
    assert freeze["scheduled_dates"] == 30 and all(s["present"] and len(s["sha256"]) == 64 for s in freeze["slates"])
    assert res["inconclusive"] is False and isinstance(res["a"], float)
    assert res["counts"]["scoreable_dates"] == 30 and res["counts"]["rank1_known_dates"] == 30


def test_each_stage_runs_once(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    with pytest.raises(SystemExit, match="earlier claimed"):
        RUN.main(["--stage", "fit"])


def test_insufficient_fit_support_is_inconclusive_and_missing_feeds_are_counted(tmp_path, patched):
    data = world(tmp_path, DAYS[:30], missing_feed={900000 + i for i in range(8)})
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (d,) = runs(data, "fit")
    res = json.loads((d / "results.json").read_text())
    assert res["inconclusive"] is True and res["reason"] == "insufficient support"
    assert res["counts"]["outcome_exclusions"]["unknown"] == 16 and res["counts"]["outcome_exclusions"]["rank1_unknown"] == 8


def test_evaluation_binds_the_one_accepted_fit(tmp_path, patched):
    data = world(tmp_path, DAYS[:90])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (fit_dir,) = runs(data, "fit")
    fit_sha = hashlib.sha256((fit_dir / "results.json").read_bytes()).hexdigest()
    patched(data, EVAL_OPENS)
    assert RUN.main(["--stage", "evaluate"]) == 0
    (ev,) = runs(data, "evaluate")
    freeze, res = json.loads((ev / "freeze.json").read_text()), json.loads((ev / "results.json").read_text())
    assert freeze["fit"]["results_sha256"] == fit_sha and freeze["scheduled_dates"] == 60
    assert res["disposition"] in ("positive", "negative", "inconclusive") and res["counts"]["scoreable_dates"] == 60
    assert "reliability_deciles" in res["secondary_descriptive"]


def test_evaluation_refuses_without_exactly_one_accepted_fit(tmp_path, patched):
    data = world(tmp_path, DAYS[:90])
    patched(data, EVAL_OPENS)
    with pytest.raises(SystemExit, match="exactly one accepted fit run"):
        RUN.main(["--stage", "evaluate"])


def test_the_calendar_must_match_the_admitted_pin(tmp_path, patched, monkeypatch):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS)
    monkeypatch.setattr(RUN, "admission_gate", lambda: ("f" * 40, {"input_pins": {"calendar": "0" * 64}}, {}))
    with pytest.raises(C.CalendarError, match="pin"):
        RUN.main(["--stage", "fit"])


def test_a_missing_cycle_root_refuses_without_creating_ancestors(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    import shutil
    shutil.rmtree(data / "hetzner_results")
    patched(data, FIT_OPENS)
    with pytest.raises(SystemExit, match="no recursive creation"):
        RUN.main(["--stage", "fit"])
    assert not (data / "hetzner_results").exists()


def test_evaluation_refuses_two_accepted_fits(tmp_path, patched):
    data = world(tmp_path, DAYS[:90])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (fit_dir,) = runs(data, "fit")
    twin = fit_dir.with_name(fit_dir.name + "-twin")             # a second claimed, completed, uninvalidated run
    twin.mkdir()
    for f in ("CLAIM.json", "results.json"):
        (twin / f).write_bytes((fit_dir / f).read_bytes())
    patched(data, EVAL_OPENS)
    with pytest.raises(SystemExit, match="found 2"):
        RUN.main(["--stage", "evaluate"])

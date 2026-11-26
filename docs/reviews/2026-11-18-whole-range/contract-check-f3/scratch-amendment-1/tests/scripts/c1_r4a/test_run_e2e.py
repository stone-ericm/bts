"""C1 rank 4a, T6: orchestration end to end on a fully synthetic data tree (no real slate, feed or calendar).
Review r1: R2 (serving witnesses), R3 (retained inputs and replay), R4 (the sealed fit), R5 (unsupported slates),
R6 (foreign imports), R7 (the calendar stop), R8 (gzip), R10 (outputs)."""
import gzip
import hashlib
import json
import shutil
import sys
import types
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.audit.c1_r4a import calendar as C
from scripts.audit.c1_r4a import provenance as PV
from scripts.audit.c1_r4a import run as RUN
from tests.scripts.c1_r4a.fixtures import contract, feed, witness

ET = ZoneInfo("America/New_York")
DAYS = [date(2027, 4, 1) + timedelta(days=i) for i in range(120)]
FIT_OPENS = datetime.combine(DAYS[39] + timedelta(days=1), time(8), ET)
EVAL_OPENS = datetime.combine(DAYS[89] + timedelta(days=1), time(8), ET)
CLOSES = datetime.combine(DAYS[-1] + timedelta(days=1), time(0), ET)


def cal_raw():
    return json.dumps({"schema": "c1_r4a_calendar_v1", "season": 2027, "dates": [d.isoformat() for d in DAYS],
                       "contest_end": DAYS[-1].isoformat()}).encode()


def _write(path: Path, raw: bytes, gz: bool):
    if gz:
        path.with_name(path.name + ".gz").write_bytes(gzip.compress(raw, mtime=0))
    else:
        path.write_bytes(raw)


def world(tmp_path, dates, *, missing_feed=(), gz=False, serving=witness, schema="bts_slate_v2"):
    """One game per date with two away batters; batter 1 (p .8) hits on even days, batter 2 (p .6) never."""
    data = tmp_path / "data"
    (data / "picks" / "slates").mkdir(parents=True, exist_ok=True)
    (data / "hetzner_results" / "c1").mkdir(parents=True, exist_ok=True)      # exists on the box (the C1 cycle root)
    (data / "raw" / "2027").mkdir(parents=True, exist_ok=True)
    for d in dates:
        n = DAYS.index(d)
        pk = 900000 + n
        rows = [{"batter_id": 1, "game_pk": pk, "p_game_hit": 0.8, "game_time": f"{d}T23:10:00Z", "status": "Scheduled",
                 "projected": False},
                {"batter_id": 2, "game_pk": pk, "p_game_hit": 0.6, "game_time": f"{d}T23:10:00Z", "status": "Scheduled",
                 "projected": True}]
        slate = {"schema_version": schema, "date": d.isoformat(), "tier": "local", "written_at": f"{d}T16:00:00+00:00",
                 "n_rows": 2, "rows": rows, "serving": serving(d.isoformat()) if serving else None}
        _write(data / "picks" / "slates" / f"{d}.json", json.dumps(slate).encode(), gz)
        if pk not in missing_feed:
            plays = [(1, "single" if n % 2 == 0 else "strikeout", f"{d}T23:30:00Z", True),
                     (2, "strikeout", f"{d}T23:40:00Z", True)]
            _write(data / "raw" / "2027" / f"{pk}.json", feed(pk, plays), gz)
    return data


@pytest.fixture
def patched(monkeypatch, tmp_path):
    raw, craw = cal_raw(), contract()
    pins = {"calendar": hashlib.sha256(raw).hexdigest(), "serving_contract": hashlib.sha256(craw).hexdigest()}
    (tmp_path / "calendar.json").write_bytes(raw)
    (tmp_path / "contract.json").write_bytes(craw)
    (tmp_path / "reg.md").write_text("")

    def apply(data, now, *, adm_pins=None):
        monkeypatch.setattr(RUN, "DATA", data)
        monkeypatch.setattr(RUN, "admission_gate", lambda: ("f" * 40, {"input_pins": adm_pins or pins},
                                                            {"review_report": "r.md"}))
        monkeypatch.setattr(RUN, "REPO", tmp_path)
        monkeypatch.setattr(RUN, "CALENDAR_REL", "calendar.json")
        monkeypatch.setattr(RUN, "CONTRACT_REL", "contract.json")
        monkeypatch.setattr(RUN, "REGISTER_REL", "reg.md")
        monkeypatch.setattr(RUN, "now_et", lambda: now)
    return apply


def runs(data, stage):
    root = data / "hetzner_results" / "c1" / "r4a" / stage
    return [d for d in sorted(root.iterdir()) if d.is_dir()] if root.exists() else []


def result(run_dir):
    return json.loads((run_dir / "results.json").read_text())


@pytest.fixture
def reads(monkeypatch):
    """The order of source reads (slates, feeds, the fit's results) around the claim."""
    events, real_read, real_claim = [], Path.read_bytes, RUN._claim

    def read_bytes(self):
        if "picks" in self.parts:
            events.append("slate")
        elif "raw" in self.parts:
            events.append("feed")
        elif self.name == "results.json" and "fit" in self.parts:
            events.append("fit_result")
        return real_read(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(RUN, "_claim", lambda *a: (events.append("claim"), real_claim(*a))[1])
    return events


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_a_stage_is_refused_before_it_opens(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS - timedelta(seconds=1))
    with pytest.raises(SystemExit, match="opens at"):
        RUN.main(["--stage", "fit"])


def test_the_fit_freezes_and_retains_slates_before_the_claim_and_feeds_after(tmp_path, patched, reads):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS)
    assert RUN.main(["--stage", "fit"]) == 0
    k = reads.index("claim")
    assert set(reads[:k]) == {"slate"} and set(reads[k + 1:]) == {"feed"}
    (d,) = runs(data, "fit")
    freeze, res, complete = (json.loads((d / f).read_text()) for f in ("freeze.json", "results.json", "COMPLETE.json"))
    claim = json.loads((d / "CLAIM.json").read_text())
    assert claim["freeze_sha256"] == sha_file(d / "freeze.json") == complete["freeze_sha256"]
    for f, k2 in (("CLAIM.json", "claim_sha256"), ("populations.json", "populations_sha256"),
                  ("outcomes.json", "outcomes_sha256"), ("joined.json", "joined_sha256"), ("results.json", "results_sha256")):
        assert complete[k2] == sha_file(d / f)
    assert freeze["populations_sha256"] == sha_file(d / "populations.json")
    first = DAYS[0].isoformat()
    assert (d / "slates" / f"{first}.json").read_bytes() == (data / "picks" / "slates" / f"{first}.json").read_bytes()
    assert (d / "feeds" / "900000.json").read_bytes() == (data / "raw" / "2027" / "900000.json").read_bytes()
    pops = json.loads((d / "populations.json").read_text())
    assert [r["batter_id"] for r in pops[first]["eligible"]] == [1, 2] and pops[first]["rank1_index"] == 0
    assert freeze["scheduled_dates"] == 30 and all(s["status"] == "ok" for s in freeze["slates"])
    assert freeze["serving_contract"]["sha256"] and freeze["readers"]["scripts/audit/c1_r4a/outcomes.py"]
    assert res["status"] == "fitted" and isinstance(res["a"], float)
    assert res["counts"]["primary_scoreable_dates"] == 30 and res["counts"]["rank1_known_dates"] == 30
    iv = res["interval_descriptive"]
    assert iv["bounds"][0] < res["a"] < iv["bounds"][1] and iv["level"] == 0.95 and "descriptive" in iv["label"]
    assert res["counts"]["eligible_by_state"] == {"confirmed": 30, "projected": 30}


def test_a_run_replays_from_its_retained_artifacts_after_the_sources_change(tmp_path, patched):
    """R3: delete every source archive; the run's own artifacts still reproduce its populations and results."""
    data = world(tmp_path, DAYS[:90])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    patched(data, EVAL_OPENS)
    RUN.main(["--stage", "evaluate"])
    (fit,), (ev,) = runs(data, "fit"), runs(data, "evaluate")
    shutil.rmtree(data / "picks")
    shutil.rmtree(data / "raw")
    assert RUN.replay(fit) == result(fit)
    assert RUN.replay(ev) == result(ev)
    retained = ev / "feeds" / f"{900000 + 40}.json"
    retained.write_bytes(retained.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="does not match its manifest"):
        RUN.replay(ev)


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
    res = result(d)
    assert res["status"] == "inconclusive" and res["reason"] == "insufficient support" and res["a"] is None
    ex = res["counts"]["outcome_exclusions"]
    assert ex["by_outcome"] == {"unknown": 16} and ex["rank1"] == {"unknown": 8}
    assert ex["by_state"] == {"unknown|confirmed": 8, "unknown|projected": 8}


def test_evaluation_binds_the_one_sealed_fit_and_reports_the_registered_outputs(tmp_path, patched, reads):
    data = world(tmp_path, DAYS[:90])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (fit,) = runs(data, "fit")
    reads.clear()
    patched(data, EVAL_OPENS)
    assert RUN.main(["--stage", "evaluate"]) == 0
    k = reads.index("claim")
    assert "fit_result" not in reads[:k] and reads[k + 1] == "fit_result"         # R4: read after the claim
    (ev,) = runs(data, "evaluate")
    freeze, res = json.loads((ev / "freeze.json").read_text()), result(ev)
    assert freeze["fit"]["run"] == fit.name and freeze["fit"]["results_sha256"] == sha_file(fit / "results.json")
    assert freeze["fit"]["complete_sha256"] == sha_file(fit / "COMPLETE.json") and "a" not in freeze["fit"]
    assert freeze["scheduled_dates"] == 60 and res["a"] == result(fit)["a"]
    assert res["disposition"] in ("positive", "negative", "inconclusive") and res["primary"]["dates"] == 60
    assert res["counts"]["primary_scoreable_dates"] == 60 and res["guardrail"]["dates"] == 60
    sec = res["secondary_descriptive"]
    assert {"brier", "stated_minus_realized", "rank1_stated_minus_realized", "reliability"} <= set(sec)
    assert "difference_map_minus_identity" in sec["brier"] and len(sec["reliability"]["edges"]) == 11


def test_a_changed_fit_result_refuses_the_evaluation_without_reading_test_outcomes(tmp_path, patched, reads):
    data = world(tmp_path, DAYS[:90])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (fit,) = runs(data, "fit")
    res = result(fit)
    (fit / "results.json").write_text(json.dumps({**res, "a": 0.123456}))
    reads.clear()
    patched(data, EVAL_OPENS)
    assert RUN.main(["--stage", "evaluate"]) == RUN.EXIT_REFUSED
    (ev,) = runs(data, "evaluate")
    assert result(ev)["status"] == "refused" and "changed" in result(ev)["reason"]
    assert "feed" not in reads


def test_a_fabricated_fit_without_a_completion_record_is_refused(tmp_path, patched):
    data = world(tmp_path, DAYS[:90])
    fab = data / "hetzner_results" / "c1" / "r4a" / "fit" / "fabricated"
    fab.mkdir(parents=True)
    (fab / "CLAIM.json").write_text("{}")
    (fab / "results.json").write_text(json.dumps({"stage": "not-fit", "a": 0.2, "status": "fitted"}))
    patched(data, EVAL_OPENS)
    with pytest.raises(SystemExit, match="no readable completion record"):
        RUN.main(["--stage", "evaluate"])
    assert runs(data, "evaluate") == []


def test_a_completion_record_must_bind_its_claim_freeze_and_pins(tmp_path, patched):
    data = world(tmp_path, DAYS[:90])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (fit,) = runs(data, "fit")
    rec = json.loads((fit / "COMPLETE.json").read_text())
    for bad in ({**rec, "freeze_sha256": "0" * 64}, {**rec, "stage": "evaluate"}, {**rec, "contract_sha256": "0" * 64},
                {**rec, "results_sha256": "nothex"}):
        (fit / "COMPLETE.json").write_text(json.dumps(bad))
        patched(data, EVAL_OPENS)
        with pytest.raises(SystemExit, match="does not bind"):
            RUN.main(["--stage", "evaluate"])


def test_evaluation_refuses_without_exactly_one_fit_claim(tmp_path, patched):
    data = world(tmp_path, DAYS[:90])
    patched(data, EVAL_OPENS)
    with pytest.raises(SystemExit, match="found 0"):
        RUN.main(["--stage", "evaluate"])
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (fit_dir,) = runs(data, "fit")
    twin = fit_dir.with_name(fit_dir.name + "-twin")             # a second claimed, uninvalidated run
    shutil.copytree(fit_dir, twin)
    patched(data, EVAL_OPENS)
    with pytest.raises(SystemExit, match="found 2"):
        RUN.main(["--stage", "evaluate"])


def test_an_inconclusive_fit_makes_the_evaluation_inconclusive_without_outcome_reads(tmp_path, patched, reads):
    data = world(tmp_path, DAYS[:90], missing_feed={900000 + i for i in range(8)})
    patched(data, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    reads.clear()
    patched(data, EVAL_OPENS)
    assert RUN.main(["--stage", "evaluate"]) == 0
    (ev,) = runs(data, "evaluate")
    assert result(ev)["disposition"] == "inconclusive" and result(ev)["reason"] == "no fitted map"
    assert "feed" not in reads


def test_one_unsupported_slate_refuses_the_stage(tmp_path, patched, reads):
    """R5 (review counterexample): 29 valid slates and one v1 slate is a refusal, not 29 dates of support."""
    data = world(tmp_path, DAYS[:29])
    world(tmp_path, DAYS[29:30], schema="bts_slate_v1")
    patched(data, FIT_OPENS)
    assert RUN.main(["--stage", "fit"]) == RUN.EXIT_REFUSED
    (d,) = runs(data, "fit")
    res = result(d)
    assert res["status"] == "refused" and res["a"] is None and res["refusals"][0]["date"] == DAYS[29].isoformat()
    assert res["counts"]["present_captured"] == 30 and res["counts"]["refused"] == 1
    assert "feed" not in reads and (d / "COMPLETE.json").exists()
    assert (d / "slates" / f"{DAYS[29]}.json").exists()                       # the refused input is retained too


def test_a_slate_without_its_serving_witness_refuses(tmp_path, patched):
    data = world(tmp_path, DAYS[:29])
    world(tmp_path, DAYS[29:30], serving=None)
    patched(data, FIT_OPENS)
    assert RUN.main(["--stage", "fit"]) == RUN.EXIT_REFUSED
    (d,) = runs(data, "fit")
    assert result(d)["refusals"][0]["reasons"] == ["no bts_serving_witness_v1 witness"]


def test_an_unregistered_recipe_change_makes_the_stage_inconclusive_without_outcome_reads(tmp_path, patched, reads):
    data = world(tmp_path, DAYS[:29])
    world(tmp_path, DAYS[29:30], serving=lambda d: witness(d, env={"BTS_LGBM_RANDOM_STATE": "7",
                                                                   "BTS_USE_CALIBRATION": None}))
    patched(data, FIT_OPENS)
    assert RUN.main(["--stage", "fit"]) == 0
    (d,) = runs(data, "fit")
    res = result(d)
    assert res["status"] == "inconclusive" and res["reason"] == "unregistered recipe change" and res["a"] is None
    assert res["changes"][0]["date"] == DAYS[29].isoformat() and "feed" not in reads


def test_a_foreign_module_refuses_after_the_real_gate_passes(tmp_path, patched, monkeypatch):
    """R6: the report, closure and exposure checks pass; a loaded bts module from elsewhere still refuses."""
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS)
    pins = json.loads(json.dumps({"calendar": hashlib.sha256(cal_raw()).hexdigest(),
                                  "serving_contract": hashlib.sha256(contract()).hexdigest()}))
    monkeypatch.undo()                                  # back to the real admission_gate
    monkeypatch.setattr(RUN, "load_admission", lambda: ({"input_pins": pins}, "0" * 64))
    monkeypatch.setattr(RUN.A, "admission_check", lambda *a, **k: ("f" * 40, []))
    monkeypatch.setattr(RUN.A, "accepted_identity", lambda *a: {})
    probe = types.ModuleType("bts._c1_foreign_probe")
    probe.__file__ = str(tmp_path / "elsewhere" / "probe.py")
    monkeypatch.setitem(sys.modules, "bts._c1_foreign_probe", probe)
    with pytest.raises(SystemExit, match="outside this checkout"):
        RUN.main(["--stage", "fit"])


def test_the_admission_must_pin_the_calendar_and_the_serving_contract(monkeypatch):
    for pins in ({"calendar": "a" * 64}, {"calendar": "a" * 64, "serving_contract": None},
                 {"calendar": "a" * 64, "serving_contract": "b" * 64, "extra": "c" * 64}):
        monkeypatch.setattr(RUN, "load_admission", lambda p=pins: ({"input_pins": p}, "0" * 64))
        with pytest.raises(SystemExit, match="input_pins"):
            RUN.admission_gate()


def test_the_committed_admission_keeps_execution_closed():
    adm = json.loads((RUN.REPO / RUN.ADMISSION_REL).read_text())
    assert adm == {"reviewed_commit": None, "review_report": None, "exposure_commit": None,
                   "input_pins": {"calendar": None, "serving_contract": None}}
    with pytest.raises(SystemExit, match="input_pins"):
        RUN.admission_gate()


def test_after_the_calendar_stop_an_unrun_stage_is_inconclusive_without_reads(tmp_path, patched, reads):
    """R7: the cycle-end stop takes precedence; nothing is fitted or evaluated after closure."""
    data = world(tmp_path, DAYS[:90])
    patched(data, CLOSES)
    assert RUN.main(["--stage", "fit"]) == 0
    assert RUN.main(["--stage", "evaluate"]) == 0
    for stage in ("fit", "evaluate"):
        (d,) = runs(data, stage)
        res = result(d)
        assert res["status"] == "inconclusive" and "calendar stop" in res["reason"] and res["a"] is None
    assert set(reads) == {"claim"}
    patched(data, CLOSES - timedelta(seconds=1))
    with pytest.raises(SystemExit, match="earlier claimed"):
        RUN.main(["--stage", "fit"])


def test_gzipped_archives_score_exactly_like_plain_ones(tmp_path, patched):
    """R8: compression changes neither the population nor the outcomes."""
    plain = world(tmp_path / "p", DAYS[:30])
    patched(plain, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    zipped = world(tmp_path / "z", DAYS[:30], gz=True)
    patched(zipped, FIT_OPENS)
    RUN.main(["--stage", "fit"])
    (p,), (z,) = runs(plain, "fit"), runs(zipped, "fit")
    assert result(p) == result(z) and result(z)["status"] == "fitted"
    assert json.loads((z / "freeze.json").read_text())["slates"][0]["stored"]["used"] == "json.gz"
    assert (z / "feeds" / "900000.json.gz").exists()


def test_a_conflicting_slate_pair_refuses_and_a_corrupt_gzip_feed_is_unknown(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    s = data / "picks" / "slates" / f"{DAYS[3]}.json"
    s.with_name(s.name + ".gz").write_bytes(gzip.compress(s.read_bytes().replace(b"0.8", b"0.9")))
    patched(data, FIT_OPENS)
    assert RUN.main(["--stage", "fit"]) == RUN.EXIT_REFUSED
    (d,) = runs(data, "fit")
    assert "different content" in result(d)["refusals"][0]["reasons"][0]

    data2 = world(tmp_path / "two", DAYS[:30])
    f = data2 / "raw" / "2027" / "900005.json"
    f.unlink()
    f.with_name(f.name + ".gz").write_bytes(b"\x1f\x8b broken")
    patched(data2, FIT_OPENS)
    assert RUN.main(["--stage", "fit"]) == 0
    (d2,) = runs(data2, "fit")
    assert result(d2)["counts"]["outcome_exclusions"]["by_outcome"] == {"unknown": 2}
    feeds = json.loads((d2 / "outcomes.json").read_text())["feeds"]
    assert next(x for x in feeds if x["game_pk"] == 900005)["status"] == "unreadable"


def test_the_calendar_and_contract_must_match_the_admitted_pins(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    patched(data, FIT_OPENS, adm_pins={"calendar": "0" * 64, "serving_contract": "0" * 64})
    with pytest.raises(C.CalendarError, match="pin"):
        RUN.main(["--stage", "fit"])
    patched(data, FIT_OPENS, adm_pins={"calendar": hashlib.sha256(cal_raw()).hexdigest(), "serving_contract": "0" * 64})
    with pytest.raises(PV.ContractError, match="pin"):
        RUN.main(["--stage", "fit"])


def test_a_missing_cycle_root_refuses_without_creating_ancestors(tmp_path, patched):
    data = world(tmp_path, DAYS[:30])
    shutil.rmtree(data / "hetzner_results")
    patched(data, FIT_OPENS)
    with pytest.raises(SystemExit, match="no recursive creation"):
        RUN.main(["--stage", "fit"])
    assert not (data / "hetzner_results").exists()


@pytest.mark.parametrize("res, ok", [
    ({"stage": "fit", "status": "fitted", "a": 0.25}, True),
    ({"stage": "fit", "status": "inconclusive", "a": None}, True),
    ({"stage": "fit", "status": "refused", "a": None}, True),
    ({"stage": "fit", "status": "fitted", "a": "0.25"}, False),
    ({"stage": "fit", "status": "fitted", "a": True}, False),
    ({"stage": "fit", "status": "fitted", "a": 1}, False),
    ({"stage": "fit", "status": "fitted", "a": float("nan")}, False),
    ({"stage": "fit", "status": "fitted", "a": None}, False),
    ({"stage": "fit", "status": "inconclusive", "a": 0.25}, False),
    ({"stage": "evaluate", "status": "fitted", "a": 0.25}, False),
    ([0.25], False),
])
def test_a_sealed_fit_result_must_itself_be_valid(tmp_path, res, ok):
    """R4: even a result whose digest matches its completion record is parsed and validated after the claim."""
    (tmp_path / "run").mkdir()
    raw = json.dumps(res).encode()
    (tmp_path / "run" / "results.json").write_bytes(raw)
    binding = {"run": "run", "results_sha256": hashlib.sha256(raw).hexdigest()}
    if ok:
        assert RUN.load_fit_result(tmp_path, binding) == res
    else:
        with pytest.raises(RUN.Refused, match="not a valid fit result"):
            RUN.load_fit_result(tmp_path, binding)

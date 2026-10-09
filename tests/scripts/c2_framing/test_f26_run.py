"""The 2026 framing test's run, admission, preparation, validation, aggregate and launch (design
`docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md`, frozen at 9b344d9), on a stubbed world: the admission,
the feature computation and the walk-forward are stubbed; the run's own gates, files and checks are real."""
import hashlib
import io
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c1 import ledger
from scripts.audit.c2_framing import f26 as F
from scripts.audit.c2_framing import screen as S

IDENT = {"review_report": "docs/audit/r.md", "review_report_sha256": "d" * 64, "reviewed_commit": "a" * 40,
         "exposure_commit": "b" * 40, "admission_sha256": "e" * 64}
HEADS = ("a" * 40, "c" * 40)
TEAMS = (11, 12, 13, 14)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _world():
    """2025 and 2026: four teams in two daily games; team t's catcher alternates between 100t and 100t+1."""
    rng = np.random.default_rng(5)
    rows, table = [], []
    for season, days in ((2025, 30), (2026, 20)):
        for d in range(days):
            date = f"{season}-05-{d + 1:02d}"
            for g, (away, home) in enumerate(((11, 12), (13, 14)) if d % 2 else ((11, 13), (12, 14))):
                pk = season * 10000 + d * 10 + g
                catchers = {t: 100 * t + (d % 3 == 0) for t in (away, home)}
                for side, team in (("away", away), ("home", home)):
                    table.append({"game_pk": pk, "season": season, "game_type": "R", "official_date": date,
                                  "game_number": 1, "game_number_fallback": False, "fielding_side": side,
                                  "team_id": team, "catcher_id": catchers[team] if (pk + (side == "home")) % 7 else None,
                                  "reason": "identified"})
                for is_home, bat, field in ((False, away, home), (True, home, away)):
                    for b in range(3):
                        batter = bat * 10 + b
                        for _ in range(2):
                            rows.append({"season": season, "date": date, "game_pk": pk, "is_home": is_home,
                                         "batter_id": batter, "pitcher_id": field * 10 + 9,
                                         "fielding_catcher_id": catchers[field], "lineup_position": b + 1,
                                         "pa_borderline_csr": float(rng.random()),
                                         "is_hit": int(rng.random() < 0.3 + 0.1 * b),
                                         "is_resumed_portion": bool(season == 2026 and d == 4 and b == 2)})
    df = S.framing_by(pd.DataFrame(rows), "pitcher_id", S.OLD_COL)
    for r in table:
        if r["catcher_id"] is None:
            r["reason"] = "no_candidate"
    return df, table


def _stub_walk_forward(calls):
    """Ranks each predicted day's batters by their mean `catcher_framing` after the arm's transform (NaN last), so the
    arms differ only through the hook; the baseline ranks by batter id."""
    def wf(df, season, retrain_every, blend_configs, game_probability_mode, predict_day_transform, strict_predict):
        calls.append({"season": season, "mode": game_probability_mode, "retrain": retrain_every,
                      "cols": tuple(blend_configs[0][1]), "transform": predict_day_transform, "strict": strict_predict})
        part = df[df["season"] == season].copy()
        part["date"] = pd.to_datetime(part["date"])
        rows = []
        for day, g in part.groupby("date"):
            if predict_day_transform is not None:
                g = predict_day_transform(g.copy(), day)
                score = g.groupby("batter_id")[F.NEW_COL].mean().fillna(-1)
            else:
                score = pd.Series({b: -b for b in g["batter_id"].unique()})
            for rank, b in enumerate(score.sort_values(ascending=False, kind="stable").index[:10], start=1):
                rows.append({"date": day.date(), "rank": rank, "batter_id": int(b),
                             "game_pk": int(g.loc[g["batter_id"] == b, "game_pk"].iloc[0]),
                             "p_game_hit": 0.9 - 0.01 * rank, "actual_hit": 1, "n_pas": 2})
        return pd.DataFrame(rows)
    return wf


class _Admission:
    def __init__(self, pins):
        self.head, self.pins = "a" * 40, pins

    def __call__(self, repo=None, *, require_inputs=True):
        return self.head, {"input_pins": self.pins}, dict(IDENT)


def _allow_row(cap=None, budget="12", stop="4.1", source="Eric 2026-10-09, typed in the lead's pane"):
    cap = f"{ledger.CAP_H:g}" if cap is None else cap
    return (f"| {F.ALLOWANCE_ROW} | the allowance | **RULED 2026-10-09 (Eric): ALLOW the catcher framing 2026 test; "
            f"shared C1/C2 compute cap {cap} CPU-hours; declared budget {budget} CPU-hours per seed; first walk-forward "
            f"stop {stop} CPU-hours** | {source} |\n")


@pytest.fixture
def world(monkeypatch, tmp_path):
    import bts.features.compute as FC
    import bts.features.park_drag as PD
    import bts.model.predict as PR
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    monkeypatch.setitem(PR.LGB_PARAMS, "deterministic", True)
    monkeypatch.setitem(PR.LGB_PARAMS, "force_row_wise", True)
    monkeypatch.setattr(FC, "_build_probable_pitcher_lookup", FC._build_probable_pitcher_lookup)
    monkeypatch.setattr(PD, "attach_park_drag", PD.attach_park_drag)
    df, table = _world()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = {F.FROZEN_PA: _parquet(df[df["season"] == 2026]), F.LOOKUP_NAME: b'{"1":{"away":5}}\n',
             F.TABLE_NAME: F.canonical(table), F.SOURCES_NAME: b"{}\n"}
    for name, b in files.items():
        (inputs / name).write_bytes(b)
    pins = {n: "b" * 64 for n in F.INPUT_NAMES}
    pins.update({name: _sha(b) for name, b in files.items()})
    adm = _Admission(pins)
    register = tmp_path / "register.md"
    register.write_text("| ID | a | b | c |\n" + _allow_row())
    monkeypatch.setattr(F, "REGISTER_REL", str(register))           # REPO / an absolute path is that path
    monkeypatch.setattr(F, "admission_gate", adm)
    monkeypatch.setattr(F, "load_inputs", lambda data_dir, inputs_dir, pins: df.copy())
    monkeypatch.setattr(FC, "compute_all_features", lambda d: d)
    monkeypatch.setattr(S, "SCORING", {"mc_trials": 200, "season_length": 180})
    monkeypatch.setattr(F, "head_admitted", lambda repo, identity, head: [] if identity == IDENT and head in HEADS
                        else [f"run HEAD {head[:7]} is not admitted"])
    monkeypatch.setattr(F, "guarded_unit_problem", lambda seed, budget, text, c1: None)
    monkeypatch.setattr(S, "ny_clock", lambda: (12, 0))
    out = tmp_path / "out"
    out.mkdir()
    return {"out": out, "inputs": inputs, "adm": adm, "register": register, "df": df, "table": table, "pins": pins}


def _parquet(frame):
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def _calendar(w):
    return F.expected_calendar(w["df"][w["df"]["season"] == 2026])


def _run(w, seed, calls=None, **k):
    return F.run(seed, w["out"], w["inputs"], walk_forward=_stub_walk_forward([] if calls is None else calls),
                 _test_out_root=w["out"], **k)


def _run_dir(w, seed):
    dirs = [p for p in (w["out"] / f"seed_{seed}").iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def _validate(w, seed, **k):
    return F.validate_run(_run_dir(w, seed), seed, out_root=w["out"], identity=IDENT, pins=w["pins"],
                          calendar=_calendar(w), **k)


# ---------------------------------------------------------------- the run

def test_run_end_to_end(world):
    calls = []
    seed = F.SEEDS[0]
    assert _run(world, seed, calls) == 0
    d = _run_dir(world, seed)
    man = json.loads((d / "manifest.json").read_text())
    assert man["test_season"] == 2026 and man["arms"] == list(F.ARMS) and man["calendar"] == _calendar(world)
    assert man["allowance"] == {"cap": ledger.CAP_H, "budget": 12.0, "first_unit_stop": 4.1}
    assert [c["cols"] == tuple(S.blend_configs("A")[0][1]) for c in calls] == [False, True, True]
    assert [c["transform"] is None for c in calls] == [True, False, False]
    assert all(c["strict"] is True and c["season"] == 2026 and c["retrain"] == 7 for c in calls)
    assert [c["transform"].arm for c in calls[1:]] == ["A_posted", "A_projected"]
    res = json.loads((d / "results.json").read_text())
    assert [u["arm"] for u in res["units"]] == list(F.ARMS)
    posted = res["units"][1]["catcher"]["counts"]
    assert posted["pa_rows"]["identified"] > 0 and posted["side_games"]["no_catcher"] > 0
    assert set(res["arms"]) == {"A_posted", "A_projected"}
    assert isinstance(res["arms"]["A_posted"]["p_at_1_delta"], float)       # season keys normalized
    assert len(res["rank1"]["baseline"]) == len(_calendar(world))
    v = _validate(world, seed)
    assert v["rank1"]["A_posted"] == res["rank1"]["A_posted"]


def test_the_arms_differ_only_through_the_prediction_day_override(world):
    seed = F.SEEDS[0]
    assert _run(world, seed) == 0
    res = json.loads((_run_dir(world, seed) / "results.json").read_text())
    assert res["rank1"]["A_posted"] != res["rank1"]["baseline"] or res["rank1"]["A_projected"] != res["rank1"]["baseline"]


def test_a_second_run_of_the_same_seed_is_refused(world):
    assert _run(world, F.SEEDS[0]) == 0
    with pytest.raises(SystemExit, match="claimed run"):
        _run(world, F.SEEDS[0])


def test_seeds_run_one_at_a_time_in_order(world):
    with pytest.raises(SystemExit, match="earlier seed 2273360 has 0 runs"):
        _run(world, F.SEEDS[1])
    assert _run(world, F.SEEDS[0]) == 0
    assert _run(world, F.SEEDS[1]) == 0
    with pytest.raises(SystemExit, match="earlier seed 1746737973"):
        _run(world, F.SEEDS[3])


def test_an_unregistered_seed_or_a_missing_flag_is_refused(world, monkeypatch):
    with pytest.raises(SystemExit, match="not a registered seed"):
        _run(world, 42)
    monkeypatch.delenv("BTS_LGBM_DETERMINISTIC")
    with pytest.raises(SystemExit, match="BTS_LGBM_DETERMINISTIC"):
        _run(world, F.SEEDS[0])


@pytest.mark.parametrize("row, match", [
    ("", "compute allowance"),
    (_allow_row(source="Ericsson, typed"), "compute allowance"),
    (_allow_row(cap="999"), "is not the launcher's cap"),
    (_allow_row(stop="12"), "compute allowance"),
    (_allow_row(budget="0"), "compute allowance"),
    (_allow_row().replace("ALLOW the", "ALLOWS the"), "compute allowance"),
])
def test_the_run_needs_erics_exact_allowance_with_the_launchers_cap(world, row, match):
    world["register"].write_text("| ID | a | b | c |\n" + row)
    with pytest.raises(SystemExit, match=match):
        _run(world, F.SEEDS[0])


def test_the_run_refuses_inside_the_window(world, monkeypatch):
    monkeypatch.setattr(S, "ny_clock", lambda: (1, 30))
    with pytest.raises(SystemExit, match="00:45 to 03:10"):
        _run(world, F.SEEDS[0])
    root = world["out"] / f"seed_{F.SEEDS[0]}"
    assert not root.exists() or not [p for p in root.iterdir() if p.is_dir()]        # no run directory, no claim


@pytest.mark.parametrize("hm, inside", [((0, 44), False), ((0, 45), True), ((3, 9), True), ((3, 10), False)])
def test_the_window_edges(hm, inside):
    assert (F.launch_window_problem(*hm) is not None) == inside


def test_the_self_check_stops_the_run_before_any_walk_forward(world, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "self_check", lambda df: {"rows": 1, "identical": False, "max_abs_diff": 1.0})
    assert _run(world, F.SEEDS[0], calls) == 4
    assert calls == [] and (_run_dir(world, F.SEEDS[0]) / "STOPPED.json").exists()


def test_an_expensive_baseline_walk_forward_stops_the_run(world, monkeypatch):
    clock = iter(range(0, 10**9, 5 * 3600))
    monkeypatch.setattr(S, "cpu_seconds", lambda: float(next(clock)))
    calls = []
    assert _run(world, F.SEEDS[0], calls) == 3
    stop = json.loads((_run_dir(world, F.SEEDS[0]) / "STOPPED.json").read_text())
    assert stop["reason"] == "first_unit_cpu" and stop["limit_cpu_h"] == 4.1 and len(calls) == 1


def test_a_changed_input_file_is_refused(world):
    (world["inputs"] / F.TABLE_NAME).write_bytes(b"[]\n")
    with pytest.raises(Exception, match="sha256|pinned|digest|hash"):
        _run(world, F.SEEDS[0])


def test_the_guarded_unit_check_is_called_with_eric_s_budget(world, monkeypatch):
    seen = []
    monkeypatch.setattr(F, "guarded_unit_problem", lambda seed, budget, text, c1: seen.append(budget) or "not a unit")
    with pytest.raises(SystemExit, match="not a unit"):
        _run(world, F.SEEDS[0])
    assert seen == [12.0]


def test_guarded_unit_problem(tmp_path):
    unit = "c1-c2-f26-seed1-20261010T120000Z-0123abcd"
    good = f"0::/user.slice/user-1000.slice/user@1000.service/app.slice/{unit}.service/payload\n"
    (tmp_path / f"PENDING_{unit}.json").write_text(json.dumps({"unit": unit, "declared_cpu_hours": 12.0}))
    assert F.guarded_unit_problem(F.SEEDS[0], 12.0, good, tmp_path) is None
    assert F.guarded_unit_problem(F.SEEDS[1], 12.0, good, tmp_path) is not None          # another seed's unit
    assert F.guarded_unit_problem(F.SEEDS[0], 13.0, good, tmp_path) is not None          # another budget
    assert F.guarded_unit_problem(F.SEEDS[0], 12.0, good.replace("/payload", "/guard"), tmp_path) is not None
    assert F.guarded_unit_problem(F.SEEDS[0], 12.0, good.replace("c2-f26", "c2-framing"), tmp_path) is not None
    assert F.guarded_unit_problem(F.SEEDS[0], 12.0, "", tmp_path) is not None


def test_arm_summary_reads_int_and_str_season_keys():
    for key in (2026, "2026"):
        diff = {"p_at_1_by_season": {key: {"delta": 0.01}}, "p_57_exact": 1e-9,
                "streak_metrics": {"mean_max_streak": -0.5, "longest_replay_streak": 2}}
        assert F.arm_summary(diff) == {"p_at_1_delta": 0.01, "p_57_exact": 1e-9, "mean_max_streak": -0.5,
                                       "longest_replay_streak": 2}


# ---------------------------------------------------------------- validation

def _rewrite_profile(d, arm, fn):
    p = d / f"profiles_{arm}_2026.parquet"
    p.write_bytes(_parquet(fn(pd.read_parquet(p))))


def _drop_day(frame):
    first = sorted(frame["date"].unique())[0]
    return frame[frame["date"] != first]


@pytest.mark.parametrize("damage, match", [
    (lambda d: (d / "STOPPED.json").write_text("{}"), "stopped"),
    (lambda d: _rewrite_profile(d, "A_posted", _drop_day), "calendar|scorecard"),
    (lambda d: [_rewrite_profile(d, a, _drop_day) for a in F.ARMS], "calendar|scorecard"),
    (lambda d: _rewrite_profile(d, "baseline", lambda f: pd.concat([f, f[f["rank"] == 1]])), "ranks|calendar"),
    (lambda d: _rewrite_profile(d, "A_posted", lambda f: f.assign(actual_hit=1 - f["actual_hit"])), "scorecard"),
])
def test_validate_run_refuses_damaged_evidence(world, damage, match):
    seed = F.SEEDS[0]
    assert _run(world, seed) == 0
    damage(_run_dir(world, seed))
    with pytest.raises(F.RunInvalid, match=match):
        _validate(world, seed)


def _edit_json(d, name, fn):
    p = d / name
    obj = json.loads(p.read_text())
    fn(obj)
    p.write_text(json.dumps(obj))


@pytest.mark.parametrize("name, fn, match", [
    ("results.json", lambda r: r["rank1"]["A_posted"].reverse(), "rank-1 vector"),
    ("results.json", lambda r: r["units"][1]["catcher"]["counts"]["pa_rows"].update(identified=0), "no identified"),
    ("results.json", lambda r: r["arms"]["A_posted"].update(p_at_1_delta=0.5), "summary"),
    ("manifest.json", lambda m: m["calendar"].pop(), "calendar"),
    ("manifest.json", lambda m: m.update(test_season=2025), "season"),
    ("manifest.json", lambda m: m["input_pins"].update({F.TABLE_NAME: "f" * 64}), "pins"),
])
def test_validate_run_refuses_inconsistent_records(world, name, fn, match):
    seed = F.SEEDS[0]
    assert _run(world, seed) == 0
    _edit_json(_run_dir(world, seed), name, fn)
    with pytest.raises(F.RunInvalid, match=match):
        _validate(world, seed)


def test_validate_run_refuses_another_calendar(world):
    seed = F.SEEDS[0]
    assert _run(world, seed) == 0
    with pytest.raises(F.RunInvalid, match="calendar"):
        F.validate_run(_run_dir(world, seed), seed, out_root=world["out"], identity=IDENT, pins=world["pins"],
                       calendar=_calendar(world)[1:])


def test_validate_run_refuses_an_unadmitted_head(world):
    seed = F.SEEDS[0]
    world["adm"].head = "9" * 40
    assert _run(world, seed) == 0
    with pytest.raises(F.RunInvalid, match="not admitted"):
        _validate(world, seed)


# ---------------------------------------------------------------- the aggregate

@pytest.fixture
def ten(world):
    for s in F.SEEDS:
        assert _run(world, s) == 0
    return [_run_dir(world, s) for s in F.SEEDS]


def test_aggregate_the_ten_registered_seeds(world, ten):
    out = F.aggregate(ten, world["inputs"], _test_out_root=world["out"])
    assert out["seeds"] == list(F.SEEDS) and out["calendar_days"] == len(_calendar(world))
    assert out["A_posted"]["decides"] is True and out["A_projected"]["decides"] is False
    assert out["A_posted"]["disposition"] in ("positive", "negative", "inconclusive")
    rows = []
    for d, s in zip(ten, F.SEEDS):
        r = F.validate_run(d, s, out_root=world["out"], identity=IDENT, pins=world["pins"], calendar=_calendar(world))
        rows.append([a - b for a, b in zip(r["rank1"]["A_posted"], r["rank1"]["baseline"])])
    x = np.array(rows, dtype=float)
    assert out["A_posted"]["m"] == pytest.approx(x.mean())


@pytest.mark.parametrize("pick", [lambda t: t[:9], lambda t: t + t[:1], lambda t: t[:9] + t[:1]])
def test_aggregate_refuses_partial_duplicate_or_extra_runs(world, ten, pick):
    with pytest.raises(F.RunInvalid):
        F.aggregate(pick(ten), world["inputs"], _test_out_root=world["out"])


def test_aggregate_rederives_the_calendar_from_the_pinned_rows(world, ten):
    df = world["df"]
    shorter = df[(df["season"] == 2026) & (df["date"] != sorted(df["date"].unique())[-1])]
    b = _parquet(shorter)
    (world["inputs"] / F.FROZEN_PA).write_bytes(b)
    world["adm"].pins[F.FROZEN_PA] = _sha(b)                       # a different pinned file, a different calendar
    with pytest.raises(F.RunInvalid, match="calendar"):
        F.aggregate(ten, world["inputs"], _test_out_root=world["out"])


# ---------------------------------------------------------------- the inputs row and the allowance

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, text):
    (repo / "docs" / "audit").mkdir(parents=True, exist_ok=True)
    (repo / F.REGISTER_REL).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
    return _git(repo, "rev-parse", "HEAD")


def _inputs_row(digest):
    return f"| {F.INPUTS_ROW} | pins | **PINNED 2026-10-10: catcher framing 2026 test inputs `{digest}`** | the lead |\n"


def test_the_inputs_row_binds_the_pins_after_the_exposure_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    pins = {n: "1" * 64 for n in F.INPUT_NAMES}
    head = "| ID | a | b | c |\n"
    x = _commit(repo, head)
    assert F.inputs_row_problem(repo, head + _inputs_row(S.pins_digest(pins)), x, pins) is None
    assert "does not bind" in F.inputs_row_problem(repo, head + _inputs_row("2" * 64), x, pins)
    assert "no structured" in F.inputs_row_problem(repo, head, x, pins)
    early = _commit(repo, head + _inputs_row(S.pins_digest(pins)))
    assert "already existed" in F.inputs_row_problem(repo, head + _inputs_row(S.pins_digest(pins)), early, pins)


def test_pins_shape():
    assert F.pins_shape_problem({n: "1" * 64 for n in F.INPUT_NAMES}) is None
    assert F.pins_shape_problem({n: "1" * 64 for n in F.INPUT_NAMES[:-1]}) is not None
    assert F.pins_shape_problem({**{n: "1" * 64 for n in F.INPUT_NAMES}, F.TABLE_NAME: "X" * 64}) is not None


# ---------------------------------------------------------------- preparation

def _feed(pk, season, date, away=11, home=12, catcher_away=1101, catcher_home=1201):
    def side(base, catcher):
        players = {f"ID{base + k}": {"person": {"id": base + k}, "battingOrder": f"{k}00",
                                     "allPositions": [{"code": "8"}]} for k in range(1, 10)}
        players[f"ID{base + 2}"] = {"person": {"id": catcher}, "battingOrder": "200", "allPositions": [{"code": "2"}]}
        return {"players": players}
    return {"gameData": {"game": {"pk": pk, "season": str(season), "type": "R", "gameNumber": 1},
                         "datetime": {"officialDate": date},
                         "teams": {"away": {"id": away}, "home": {"id": home}},
                         "probablePitchers": {"away": {"id": 7001}, "home": {"id": 7002}}},
            "liveData": {"boxscore": {"teams": {"away": side(3000, catcher_away), "home": side(4000, catcher_home)}}}}


@pytest.fixture
def prep(tmp_path):
    data, raw, screen_inputs = tmp_path / "processed", tmp_path / "raw", tmp_path / "screen_inputs"
    for p in (data, raw / "2025", raw / "2026", screen_inputs):
        p.mkdir(parents=True)
    games = {2025: [(250001, "2025-09-27")], 2026: [(260001, "2026-03-26"), (260002, "2026-03-27")]}
    screen_pins = {f"pa_{s}.parquet": "0" * 64 for s in range(2017, 2025)}
    for s, gs in games.items():
        b = _parquet(pd.DataFrame({"game_pk": [g for g, _ in gs], "season": s}))
        (data / f"pa_{s}.parquet").write_bytes(b)
        if s == 2025:
            screen_pins["pa_2025.parquet"] = _sha(b)
        for pk, date in gs:
            (raw / str(s) / f"{pk}.json").write_text(json.dumps(_feed(pk, s, date)))
    lookup = b'{"250001":{"away":1,"home":2,"away_tid":11,"home_tid":12}}\n'
    (screen_inputs / S.LOOKUP_NAME).write_bytes(lookup)
    screen_pins[S.LOOKUP_NAME] = _sha(lookup)
    return {"data": data, "raw": raw, "screen_inputs": screen_inputs, "screen_pins": screen_pins,
            "out": tmp_path / "inputs"}


def _prepare(p):
    return F.prepare(p["data"], p["raw"], p["screen_inputs"], p["screen_pins"], p["out"])


def test_prepare_builds_the_inputs_from_exactly_the_pinned_games(prep):
    got = _prepare(prep)
    assert set(got["pins"]) == set(F.INPUT_NAMES)
    for name in (F.FROZEN_PA, F.LOOKUP_NAME, F.TABLE_NAME, F.SOURCES_NAME):
        assert got["pins"][name] == _sha((prep["out"] / name).read_bytes())
    assert got["pins_digest"] == S.pins_digest(got["pins"])
    lookup = json.loads((prep["out"] / F.LOOKUP_NAME).read_text())
    assert set(lookup) == {"250001", "260001", "260002"}
    assert lookup["260001"] == {"away": 7001, "home": 7002, "away_tid": 11, "home_tid": 12}
    table = json.loads((prep["out"] / F.TABLE_NAME).read_text())
    assert len(table) == 6 and {r["catcher_id"] for r in table} == {1101, 1201}
    sources = json.loads((prep["out"] / F.SOURCES_NAME).read_text())
    assert [s["path"] for s in sources["sources"]] == ["2025/250001.json", "2026/260001.json", "2026/260002.json"]
    assert got["games"] == {"2025": 1, "2026": 2} and got["proxy_reasons"]["2026"] == {"identified": 4}


def test_prepare_reads_no_unpinned_feed(prep):
    (prep["raw"] / "2026" / "269999.json").write_text("not json")      # an unrelated feed is never opened
    _prepare(prep)


def test_prepare_refuses_a_missing_feed(prep):
    (prep["raw"] / "2026" / "260002.json").unlink()
    with pytest.raises(F.InputRefused, match="no raw feed"):
        _prepare(prep)


def test_prepare_refuses_a_changed_2025_parquet(prep):
    (prep["data"] / "pa_2025.parquet").write_bytes(_parquet(pd.DataFrame({"game_pk": [1], "season": [2025]})))
    with pytest.raises(Exception, match="sha256|pinned|digest|hash"):
        _prepare(prep)


def test_prepare_refuses_an_existing_output_directory(prep):
    prep["out"].mkdir()
    with pytest.raises(FileExistsError):
        _prepare(prep)


# ---------------------------------------------------------------- launch

def test_launch_command():
    cmd = F.launch_command(F.SEEDS[2], 12.0, "/d", "/i")
    assert cmd[:6] == [".venv/bin/python", "-m", "scripts.audit.c1.launch", "run", "--name", "c2-f26-seed3"]
    assert cmd[cmd.index("--cpu-hours") + 1] == "12" and cmd[cmd.index("--max-hours") + 1] == "16"
    assert cmd[cmd.index("--") + 1:][:3] == ["env", "BTS_LGBM_DETERMINISTIC=1", "TZ=America/New_York"]
    assert cmd[-6:] == ["--seed", str(F.SEEDS[2]), "--data-dir", "/d", "--inputs-dir", "/i"]


def test_launch_runs_the_launcher_with_erics_budget_and_refuses_in_the_window(world, monkeypatch):
    seen = []
    execute = lambda argv, cwd: seen.append(argv) or subprocess.CompletedProcess(argv, 0)   # noqa: E731
    assert F.launch(F.SEEDS[0], world["out"], world["inputs"], execute=execute, _test_out_root=world["out"]) == 0
    assert seen[0][seen[0].index("--cpu-hours") + 1] == "12"
    monkeypatch.setattr(S, "ny_clock", lambda: (2, 0))
    with pytest.raises(SystemExit, match="00:45 to 03:10"):
        F.launch(F.SEEDS[0], world["out"], world["inputs"], execute=execute, _test_out_root=world["out"])
    assert len(seen) == 1

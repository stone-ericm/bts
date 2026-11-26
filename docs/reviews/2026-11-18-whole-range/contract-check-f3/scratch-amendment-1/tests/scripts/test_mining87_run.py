"""#87 adapter: frozen registration, X-22 gate, drift refusal, disclosure discipline and the end-to-end driver
(review r1 findings 8, 9; required changes 1, 10, 11)."""
from __future__ import annotations

import copy
import json
import subprocess

import pandas as pd
import pytest

from scripts.audit.mining87 import registration as reg
from scripts.audit.mining87 import run
from scripts.audit.season_ledger.compile import CONTEST_SCHEMA, LEDGER_SCHEMA
from scripts.audit.season_ledger.io import build_table, write_table

FP = "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f"
DATES = [f"2026-06-{d:02d}" for d in range(8, 30)]          # 22 dates; slates exist from 6/11
COHORT = [f"user {i}" for i in range(6)]                     # sanitized to user_<i>
OTHERS = [f"other{i}" for i in range(4)]


# --- registration ----------------------------------------------------------------------------------------------------

def test_registration_fingerprint_is_pinned_and_any_drift_is_refused(monkeypatch):
    assert reg.registration_fingerprint() == reg.REGISTRATION_FINGERPRINT
    reg.check_registration()
    drifted = copy.deepcopy(reg.REGISTRATION)
    drifted["bootstrap"]["seed"] = 1
    monkeypatch.setattr(reg, "REGISTRATION", drifted)
    with pytest.raises(reg.RegistrationError, match="fingerprint"):
        reg.check_registration()


def test_registration_freezes_the_prescribed_values():
    r = reg.REGISTRATION
    assert r["window"] == {"start": "2026-03-26", "end": "2026-07-03"}
    assert r["cohort"] == {"snapshot_file": "2026-07-04.parquet", "tab": "active_streak"}
    assert r["public_capture_end_exclusive_utc_naive"] == "2026-07-05T04:00:00"      # end of 7/04 ET, naive UTC
    assert r["bootstrap"]["expected_block_length"] == 7 and r["bootstrap"]["n_bootstrap"] == 2000
    assert r["bootstrap"]["seed"] == 20260510 and r["top_k"] == [1, 2, 5, 10]
    assert r["fdr"]["min_resolved_disagreement_units"] == 15 and r["fdr"]["q_threshold"] == 0.10
    assert r["nomination"] == {"min_resolved_disagreement_units": 30, "min_absolute_lift": 0.05}
    assert r["streams"] == ["primary", "tie_excluded_sensitivity"] and r["nominating_stream"] == "primary"
    from scripts.leaderboard_mechanism_mining import DECOMPOSITION_VARIABLES
    assert r["decomposition_variables"] == DECOMPOSITION_VARIABLES        # the protocol's ordered 14 axes


@pytest.mark.parametrize("override", [{"seed": 1}, {"window_end": "2026-07-10"}, {"window_start": "2026-04-01"},
                                      {"snapshot_file": "2026-07-05.parquet"}, {"n_bootstrap": 100},
                                      {"expected_block_length": 3}])
def test_registered_mode_refuses_parameter_drift_and_exploratory_mode_cannot_nominate(override):
    with pytest.raises(reg.RegistrationError, match="drift"):
        reg.resolve_params(override, exploratory=False)
    params = reg.resolve_params(override, exploratory=True)
    assert params.mode == "exploratory" and params.can_nominate is False and params.overrides == override
    registered = reg.resolve_params({}, exploratory=False)
    assert registered.mode == "registered" and registered.can_nominate is True and registered.seed == 20260510


def test_unknown_override_keys_are_rejected():
    with pytest.raises(reg.RegistrationError, match="unknown"):
        reg.resolve_params({"fdr_min_n": 5}, exploratory=True)


def test_gate_refuses_while_x22_commit_is_unset_and_without_the_register_row(monkeypatch):
    monkeypatch.setattr(reg, "X22_COMMIT", None)
    with pytest.raises(SystemExit, match="X22_COMMIT is unset"):
        reg.x22_gate()
    head = subprocess.run(["git", "-C", str(reg.REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    monkeypatch.setattr(reg, "X22_COMMIT", head)
    register = (reg.REPO / reg.REGISTER_PATH).read_text()
    if "| X-22 |" not in register:
        with pytest.raises(SystemExit, match="no X-22 row"):
            reg.x22_gate()


# --- synthetic production data ---------------------------------------------------------------------------------------

def _p(i):
    return round(0.70 + (i % 9) * 0.015, 3)


def _write_inputs(root, *, snapshot_name="2026-07-04.parquet", witness_dates=(), null_game_dates=(),
                  write_slates=True):
    data, ledger_dir = root / "data", root / "ledger"
    rows, contest, slates = [], [], {}
    for i, d in enumerate(DATES):
        picks = [("primary", 100 + i, 9000 + i)] + ([("double_down", 200 + i, 9500 + i)] if i % 2 == 0 else [])
        for slot, b, g in picks:
            null_game = slot == "primary" and d in null_game_dates     # committed, but no recorded game (R2 #3)
            g = None if null_game else g
            sid = f"{d}|{slot}|{b}|{g}"
            grade = "void" if i == 3 and slot == "primary" else ("hit" if (i + len(slot)) % 3 else "not_hit")
            status = "unknown" if (i == 5 or null_game) else "graded"
            unlinked = i == 5 or null_game
            row = {"row_id": sid, "row_kind": "selection", "date": d, "slot": slot, "selection_id": sid,
                   "batter_id": b, "batter_name": f"P{b}", "game_pk": g, "p_stated": _p(i),
                   "projected_lineup": i % 4 == 0, "finalization": "decision", "commit_status": "committed_evidenced",
                   "predicted_at": f"{d}T15:00:00.000000Z", "locked_at": f"{d}T17:00:00.000000Z",
                   "entry_status": "unknown" if unlinked else "confirmed",
                   "match": None if unlinked else "evidenced", "match_reason": None if unlinked else "unit_capture",
                   "round_id": 1000 + i, "unit_id": None if unlinked else g * 10, "player_id": b + 5000,
                   "bts_outcome": None if unlinked else grade, "bts_outcome_status": status,
                   "game_eligibility": "unknown"}
            rows.append(row)
            if not unlinked:
                contest.append({"round_id": 1000 + i, "unit_id": g * 10, "player_id": b + 5000, "date": d,
                                "batter_id": b, "game_pk": g, "selection_id": sid, "match": "evidenced",
                                "match_reason": "unit_capture", "slot_result": grade})
        if i % 2 == 1:                                   # an undelivered DD preview: never a locked unit
            rows.append({"row_id": f"{d}|preview", "row_kind": "selection", "date": d, "slot": "double_down",
                         "selection_id": f"{d}|preview", "batter_id": 1, "game_pk": 1, "finalization": "decision",
                         "commit_status": "unconfirmed", "entry_status": "unknown",
                         "bts_outcome_status": "unknown"})
        if write_slates and d >= "2026-06-11":
            slate_rows = [{"batter_id": 100 + i, "batter_name": "x", "game_pk": 9000 + i, "p_game_hit": _p(i)},
                          {"batter_id": 300 + i, "batter_name": "y", "game_pk": 9300 + i, "p_game_hit": 0.69},
                          {"batter_id": 400 + i, "batter_name": "z", "game_pk": 9400 + i, "p_game_hit": 0.6}]
            slates[d] = json.dumps({"schema_version": "bts_slate_v1", "date": d, "tier": "t",
                                    "written_at": f"{d}T15:00:00+00:00", "n_rows": 3, "rows": slate_rows})
    ledger_dir.mkdir(parents=True)
    write_table(build_table(rows, LEDGER_SCHEMA, sort_keys=["row_id"], name="l"), ledger_dir / "season_2026_ledger.parquet")
    write_table(build_table(contest, CONTEST_SCHEMA, sort_keys=["round_id", "unit_id", "player_id"], name="c"),
                ledger_dir / "season_2026_ledger_contest_slots.parquet")
    (ledger_dir / "ACCEPTED.json").write_text(json.dumps(
        {"run": "synthetic", "rules_fingerprint": FP,
         "files": ["season_2026_ledger.parquet", "season_2026_ledger_contest_slots.parquet"]}))

    picks_dir = data / "leaderboard" / "user_picks"
    picks_dir.mkdir(parents=True)
    for k, user in enumerate(COHORT + OTHERS):
        recs = []
        for i, d in enumerate(DATES):
            primary = 300 + i if (k + i) % 5 else 100 + i                    # mostly the consensus batter
            if user in OTHERS:
                primary = 400 + i
            for slot, b in ((1, primary), (2, 500 + i)):
                res = "hit" if (b + slot) % 4 else "not_hit"
                recs.append({"captured_at": pd.Timestamp(f"{d}T23:00:00"), "round_id": 1000 + i,
                             "pick_date": pd.Timestamp(d).date(), "pick_number": slot, "unit_id": b * 7,
                             "bts_player_id": b, "result": res, "at_bats": 4, "hits": int(res == "hit"),
                             "streak_after": 1, "batter_id": b, "batter_name": f"N{b}", "batter_team": "AAA",
                             "opponent_team": "BBB", "home_or_away": "home"})
        pd.DataFrame(recs).to_parquet(picks_dir / f"{user.replace(' ', '_')}.parquet", index=False)
    snaps = data / "leaderboard" / "leaderboard_snapshots"
    snaps.mkdir(parents=True)
    pd.DataFrame([{"captured_at": pd.Timestamp("2026-07-04T12:00:00"), "tab": "active_streak", "rank": i + 1,
                   "username": u, "streak": 20 - i, "hits_today": 0} for i, u in enumerate(COHORT)]).to_parquet(
        snaps / snapshot_name, index=False)
    slate_dir = data / "picks" / "slates"
    slate_dir.mkdir(parents=True)
    for d, body in slates.items():
        (slate_dir / f"{d}.json").write_text(body)
    witness = None
    if witness_dates:
        import hashlib
        witness = root / "witness.json"
        witness.write_text(json.dumps({"schema": "mining87_surface_witness_v1", "witnesses": [
            {"date": d, "surface_sha256": hashlib.sha256(slates[d].encode()).hexdigest(),
             "independent_of_served_slate": True, "source": "synthetic independent log",
             "candidate_universe": "ref:u", "lineup_assumptions": "ref:l", "feature_computation": "ref:f",
             "prediction_timestamp_utc": f"{d}T15:00:00+00:00"} for d in witness_dates]}))
    return data, ledger_dir, witness


def _main(root, data, ledger_dir, *extra):
    return run.main(["--data-root", str(data), "--ledger-dir", str(ledger_dir), "--out", str(root / "out"), *extra])


@pytest.fixture
def gated(monkeypatch):
    """A mock exposure gate, and the real relevant-code identity marked clean: this checkout may be dirty while the
    tests run; the dirty/unverifiable refusal has its own tests."""
    monkeypatch.setattr(reg, "x22_gate", lambda: "f" * 40)
    monkeypatch.setattr(run, "x22_gate", lambda: "f" * 40)
    clean = {**run.code_identity(), "worktree_dirty": False}
    monkeypatch.setattr(run, "code_identity", lambda: copy.deepcopy(clean))
    return clean


def _attempts(root):
    return sorted(p for p in (root / "out").iterdir() if p.is_dir()) if (root / "out").exists() else []


def _completed(root):
    return [p for p in _attempts(root) if (p / "COMPLETE.json").exists()]


def _fail_first_loader(monkeypatch):
    def fail(_source):
        raise RuntimeError("synthetic technical failure at the first loader")
    monkeypatch.setattr(run, "load_accepted_ledger", fail)


def _sha(raw: bytes) -> str:
    import hashlib
    return hashlib.sha256(raw).hexdigest()


FORBIDDEN_IN_STREAMS = ("hit_rate", "mean_delta", "q_BH", "p_one_sided", "nominat", "actionable", "delta", "lift")


def test_main_runs_end_to_end_on_synthetic_inputs(tmp_path, gated, capsys):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    assert _main(tmp_path, data, ledger_dir) == 0
    captured = capsys.readouterr()
    for word in FORBIDDEN_IN_STREAMS:
        assert word not in captured.out and word not in captured.err
    out_line = json.loads(captured.out.strip().splitlines()[-1])
    run_dir = tmp_path / "out" / out_line["run_dir"].split("/")[-1]
    assert out_line["complete"] is True and run_dir.is_dir() and not list((tmp_path / "out").glob("*.partial"))

    complete = json.loads((run_dir / "COMPLETE.json").read_text())
    report = json.loads((run_dir / "report.json").read_text())
    manifest = json.loads((run_dir / "input_manifest.json").read_text())
    import hashlib
    for name, sha in complete["outputs"].items():
        assert hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == sha
    assert complete["input_manifest_sha256"] == hashlib.sha256((run_dir / "input_manifest.json").read_bytes()).hexdigest()
    assert len(manifest["inputs"]["user_picks"]) == len(COHORT + OTHERS)
    optional = manifest["inputs"]["optional_inputs"]                  # absence is frozen, never assumed
    assert optional["surface_witness"] == {"status": "absent"} and optional["mechanism_records"] == {"status": "absent"}
    assert optional["production_context"] == {"status": "absent"} and optional["ledger_build_manifest"] == {
        "status": "absent"}
    assert optional["unit_captures"] == {"status": "absent", "reason": "no surface witness supplied"}
    assert set(manifest["inputs"]["ledger"]) == {"ACCEPTED.json", "season_2026_ledger.parquet",
                                                 "season_2026_ledger_contest_slots.parquet"}
    assert set(manifest["documents"]) >= {"protocol", "amendment", "review_r1", "exposure_register"}

    assert report["research_only"] is True and report["production_deploy_claim"] is False
    assert report["no_policy_edit_supported"] is True
    sections = list(report)
    assert sections.index("coverage_and_denominators") < sections.index("streams")
    assert report["run"]["run_mode"] == "registered" and report["run"]["registration_fingerprint"] == \
        reg.REGISTRATION_FINGERPRINT
    pre = report["pre_registration"]
    assert pre["untouched_holdout"] is False and "X-10" in pre["prior_overlapping_exposures"]
    assert pre["exposure_row"] == "X-22"

    cov = report["coverage_and_denominators"]
    prod = cov["production"]
    assert prod["locked_units"] == len(DATES) + len(DATES[::2])
    assert prod["lock_unsupported_selections"] == {"commit_unconfirmed": len(DATES[1::2])}
    assert set(prod["settlement_status"]) <= {"resolved", "void", "unknown", "ungraded"}
    assert cov["surfaces"]["admitted_dates"] == 0
    assert cov["surfaces"]["decisions"]["no_independent_witness"] == len([d for d in DATES if d >= "2026-06-11"])
    assert cov["cohorts"]["fixed_cohort"]["n_units"] == prod["locked_units"]

    top = report["primary_estimand_top_n"]["fixed_cohort"]
    assert top["available"] is False and top["reason"] == "no_admitted_surface"
    assert report["same_slot_agreement_fallback"]["fixed_cohort"]["denominator"] > 0
    paired = report["secondary_paired_outcomes"]["fixed_cohort"]
    assert paired["all_joint_resolved"]["bootstrap"]["seed"] == 20260510
    assert paired["resolved_disagreement"]["bootstrap"]["seed"] == 20260510
    assert report["secondary_conditional_miscalibration"]["fixed_cohort"]["available"] is False

    streams = report["streams"]
    assert set(streams) == {"primary", "tie_excluded_sensitivity"}
    assert streams["tie_excluded_sensitivity"]["summary"]["can_nominate"] is False
    assert all(c["consensus_model_rank_bin"] == "missing_surface" for c in streams["primary"]["cells"])
    assert report["nomination"]["nominated_cells"] == []
    assert report["served_slate_diagnostic"]["outside_registered_family"] is True

    units = pd.read_parquet(run_dir / "units.parquet")
    assert len(units) == 2 * prod["locked_units"] and set(units["cohort"]) == {"fixed_cohort", "all_tracked"}
    assert units["consensus_model_rank"].isna().all()


def _null_game_units(run_dir, dates):
    units = pd.read_parquet(run_dir / "units.parquet")
    return units[units["date"].isin(dates) & (units["pick_number"] == 1)]


def test_a_committed_null_game_primary_completes_without_slates_or_witnesses_and_keeps_its_unit(tmp_path, gated):
    """R2 finding 3: a compiler-valid committed primary with no recorded game must reach units.parquet."""
    dates = ("2026-06-08", "2026-06-15")
    data, ledger_dir, _ = _write_inputs(tmp_path, null_game_dates=dates, write_slates=False)
    assert _main(tmp_path, data, ledger_dir) == 0
    run_dir = next(p for p in (tmp_path / "out").iterdir() if (p / "COMPLETE.json").exists())
    kept = _null_game_units(run_dir, dates)
    assert len(kept) == 4 and kept["production_game_pk"].isna().all()          # 2 dates x 2 cohorts
    assert not kept["production_resolved"].any() and set(kept["production_settlement"]) == {"unknown"}
    assert set(kept["consensus_model_rank_bin"]) == {"missing_surface"} and not kept["surface_admitted"].any()
    report = json.loads((run_dir / "report.json").read_text())
    per_date = {a["date"]: a for a in report["coverage_and_denominators"]["surfaces"]["per_date"]}
    assert all(per_date[d]["reason"] == "no_served_slate" and not per_date[d]["admitted"] for d in dates)
    assert report["coverage_and_denominators"]["production"]["locked_units"] == len(DATES) + len(DATES[::2])


def test_a_committed_null_game_primary_with_a_candidate_slate_and_witness_is_never_admitted(tmp_path, gated):
    data, ledger_dir, witness = _write_inputs(tmp_path, null_game_dates=("2026-06-15",),
                                              witness_dates=("2026-06-12", "2026-06-15"))
    assert _main(tmp_path, data, ledger_dir, "--surface-witness", str(witness)) == 0
    run_dir = next(p for p in (tmp_path / "out").iterdir() if (p / "COMPLETE.json").exists())
    report = json.loads((run_dir / "report.json").read_text())
    per_date = {a["date"]: a for a in report["coverage_and_denominators"]["surfaces"]["per_date"]}
    assert per_date["2026-06-15"]["admitted"] is False
    assert per_date["2026-06-15"]["reason"] == "incomplete_production_selection_identity"
    assert per_date["2026-06-15"]["selection_consistency"] == "incomplete_selection_identity"
    assert per_date["2026-06-12"]["admitted"] is True                              # the other witnessed date still is
    kept = _null_game_units(run_dir, ("2026-06-15",))
    assert len(kept) == 2 and not kept["surface_admitted"].any() and kept["consensus_model_rank"].isna().all()


def test_witnessed_dates_are_admitted_and_an_exploratory_run_is_labelled_and_cannot_nominate(tmp_path, gated):
    data, ledger_dir, witness = _write_inputs(tmp_path, witness_dates=("2026-06-12", "2026-06-14"))
    assert _main(tmp_path, data, ledger_dir, "--surface-witness", str(witness), "--exploratory",
                 "--seed", "7") == 0
    run_dir = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    assert "exploratory" in run_dir.name
    report = json.loads((run_dir / "report.json").read_text())
    assert report["run"]["run_mode"] == "exploratory" and report["run"]["overrides"] == {"seed": 7}
    assert report["coverage_and_denominators"]["surfaces"]["admitted_dates"] == 2
    assert report["primary_estimand_top_n"]["fixed_cohort"]["available"] is True
    assert report["streams"]["primary"]["summary"]["can_nominate"] is False
    assert report["nomination"]["outcome_state"] == "diagnostic_cannot_nominate"
    assert report["secondary_paired_outcomes"]["fixed_cohort"]["all_joint_resolved"]["bootstrap"]["seed"] == 7


def test_context_and_mechanism_inputs_are_hashed_and_only_named_context_columns_are_read(tmp_path, gated):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    ctx = pd.DataFrame([{"date": d, "slot": "primary", "batter_id": 100 + i, "game_pk": 9000 + i,
                         "batter_skill_prior_pa": 50 * i, "batter_skill_quartile": 1 + i % 4,
                         "pick_weather_temp": 60.0 + i, "pick_is_indoor": False, "is_park_driven": False,
                         "actual_hit": bool(i % 2)} for i, d in enumerate(DATES)])
    ctx_path = tmp_path / "context.parquet"
    ctx.to_parquet(ctx_path, index=False)
    mech_path = tmp_path / "mech.json"
    mech_path.write_text(json.dumps({"schema": "mining87_mechanism_records_v1", "records": []}))
    assert _main(tmp_path, data, ledger_dir, "--production-context", str(ctx_path), "--mechanism-records",
                 str(mech_path)) == 0
    run_dir = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    report = json.loads((run_dir / "report.json").read_text())
    ctx_meta = report["coverage_and_denominators"]["production_context"]
    assert ctx_meta["rows_matched"] == len(DATES) and "actual_hit" not in ctx_meta["columns"]
    assert report["run"]["mechanism_records"]["n_records"] == 0
    manifest = json.loads((run_dir / "input_manifest.json").read_text())
    import hashlib
    assert manifest["inputs"]["optional_inputs"]["production_context"] == {
        "status": "supplied", "sha256": hashlib.sha256(ctx_path.read_bytes()).hexdigest()}
    assert manifest["inputs"]["optional_inputs"]["mechanism_records"]["status"] == "supplied"
    units = pd.read_parquet(run_dir / "units.parquet")
    primaries = units[units["pick_number"] == 1]
    assert (primaries["production_batter_skill_prior_pa_bin"] != "missing").all()
    assert "actual_hit" not in units.columns


def test_registered_run_refuses_overrides_before_reading_anything(tmp_path, gated, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an input was read")
    monkeypatch.setattr(run, "load_public_picks", boom)
    monkeypatch.setattr(run, "load_accepted_ledger", boom)
    with pytest.raises(SystemExit, match="drift"):
        _main(tmp_path, tmp_path / "data", tmp_path / "ledger", "--seed", "7")
    assert not (tmp_path / "out").exists()


def test_the_gate_runs_before_any_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "X22_COMMIT", None)
    def boom(*a, **k):
        raise AssertionError("an input was read")
    monkeypatch.setattr(run, "load_public_picks", boom)
    monkeypatch.setattr(run, "load_accepted_ledger", boom)
    with pytest.raises(SystemExit, match="X22_COMMIT is unset"):
        _main(tmp_path, tmp_path / "data", tmp_path / "ledger")
    assert not (tmp_path / "out").exists()


def test_the_cohort_snapshot_is_pinned_never_the_latest_file(tmp_path, gated):
    data, ledger_dir, _ = _write_inputs(tmp_path, snapshot_name="2026-07-05.parquet")
    with pytest.raises(SystemExit, match="2026-07-04.parquet"):
        _main(tmp_path, data, ledger_dir)
    assert not list(tmp_path.glob("out/*/COMPLETE.json"))


def test_pins_are_persisted_before_loaders_and_a_failed_attempt_keeps_them_for_its_retry(tmp_path, gated,
                                                                                         monkeypatch):
    """R2 finding 5: a technical failure at the first loader leaves the attempt's manifest; the retry must use it."""
    data, ledger_dir, _ = _write_inputs(tmp_path)
    original = run.load_accepted_ledger
    _fail_first_loader(monkeypatch)
    with pytest.raises(RuntimeError, match="synthetic technical failure"):
        _main(tmp_path, data, ledger_dir)
    [partial] = _attempts(tmp_path)
    pins = partial / "input_manifest.json"
    assert partial.name.endswith(".partial") and pins.exists() and not (partial / "COMPLETE.json").exists()
    frozen = pins.read_bytes()
    assert json.loads(frozen)["inputs"]["optional_inputs"]["surface_witness"] == {"status": "absent"}
    monkeypatch.setattr(run, "load_accepted_ledger", original)
    with pytest.raises(SystemExit, match="prior registered attempt"):
        _main(tmp_path, data, ledger_dir)                                  # a fresh attempt may not skip the pins
    outside = tmp_path / "copy_of_pins.json"
    outside.write_bytes(frozen.replace(b'"run_mode"', b' "run_mode"'))
    with pytest.raises(SystemExit, match="not the manifest of a prior registered attempt"):
        _main(tmp_path, data, ledger_dir, "--expect-inputs", str(outside))
    assert _main(tmp_path, data, ledger_dir, "--expect-inputs", str(pins)) == 0
    assert pins.read_bytes() == frozen                                     # the failed attempt's pins are unchanged
    [done] = _completed(tmp_path)
    retry = json.loads((done / "input_manifest.json").read_text())["retry_of"]
    assert retry == {"pinned_manifest_sha256": _sha(frozen), "code_identical": True}


def test_a_completed_registered_run_is_not_a_technical_failure_eligible_for_retry(tmp_path, gated):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    assert _main(tmp_path, data, ledger_dir) == 0
    [first] = _completed(tmp_path)
    n = len(_attempts(tmp_path))
    with pytest.raises(SystemExit, match="completed registered run"):
        _main(tmp_path, data, ledger_dir)
    with pytest.raises(SystemExit, match="completed registered run"):
        _main(tmp_path, data, ledger_dir, "--expect-inputs", str(first / "input_manifest.json"))
    assert len(_attempts(tmp_path)) == n and _completed(tmp_path) == [first]
    assert _main(tmp_path, data, ledger_dir, "--exploratory") == 0            # labelled exploration stays possible


def test_a_retry_with_changed_inputs_documents_code_or_mode_is_refused(tmp_path, gated, monkeypatch):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    original = run.load_accepted_ledger
    _fail_first_loader(monkeypatch)
    with pytest.raises(RuntimeError):
        _main(tmp_path, data, ledger_dir)
    [partial] = _attempts(tmp_path)
    pins = str(partial / "input_manifest.json")
    monkeypatch.setattr(run, "load_accepted_ledger", original)

    real_docs = run.document_hashes
    monkeypatch.setattr(run, "document_hashes",
                        lambda: {**real_docs(), "amendment": {"path": "x", "sha256": "0" * 64}})
    with pytest.raises(SystemExit, match="documents"):
        _main(tmp_path, data, ledger_dir, "--expect-inputs", pins)
    monkeypatch.setattr(run, "document_hashes", real_docs)

    changed = copy.deepcopy(gated)
    changed["files"]["scripts/audit/mining87/inference.py"] = "0" * 64
    monkeypatch.setattr(run, "code_identity", lambda: copy.deepcopy(changed))
    with pytest.raises(SystemExit, match="code_identity"):
        _main(tmp_path, data, ledger_dir, "--expect-inputs", pins)
    monkeypatch.setattr(run, "code_identity", lambda: copy.deepcopy(gated))

    with pytest.raises(SystemExit, match="run_mode"):
        _main(tmp_path, data, ledger_dir, "--expect-inputs", pins, "--exploratory")

    victim = data / "leaderboard" / "user_picks" / "other0.parquet"
    frame = pd.read_parquet(victim)
    frame.loc[0, "result"] = "not_hit" if frame.loc[0, "result"] == "hit" else "hit"
    frame.to_parquet(victim, index=False)
    with pytest.raises(SystemExit, match="user_picks"):
        _main(tmp_path, data, ledger_dir, "--expect-inputs", pins)
    assert _attempts(tmp_path) == [partial] and not _completed(tmp_path)


def test_registered_mode_refuses_dirty_or_unverifiable_relevant_code(tmp_path, gated, monkeypatch):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    for dirty in (True, None):
        monkeypatch.setattr(run, "code_identity", lambda d=dirty: {**copy.deepcopy(gated), "worktree_dirty": d})
        with pytest.raises(SystemExit, match="dirty or unverifiable"):
            _main(tmp_path, data, ledger_dir)
    assert _attempts(tmp_path) == []
    assert _main(tmp_path, data, ledger_dir, "--exploratory") == 0            # exploratory: recorded only
    [done] = _completed(tmp_path)
    assert json.loads((done / "input_manifest.json").read_text())["code_identity"]["worktree_dirty"] is None


def test_an_exploratory_retry_records_a_code_change_as_diagnostic_metadata(tmp_path, gated, monkeypatch):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    original = run.load_accepted_ledger
    _fail_first_loader(monkeypatch)
    with pytest.raises(RuntimeError):
        _main(tmp_path, data, ledger_dir, "--exploratory")
    [partial] = _attempts(tmp_path)
    monkeypatch.setattr(run, "load_accepted_ledger", original)
    changed = copy.deepcopy(gated)
    changed["files"]["scripts/audit/mining87/inference.py"] = "0" * 64
    monkeypatch.setattr(run, "code_identity", lambda: copy.deepcopy(changed))
    assert _main(tmp_path, data, ledger_dir, "--exploratory", "--expect-inputs",
                 str(partial / "input_manifest.json")) == 0
    [done] = _completed(tmp_path)
    assert json.loads((done / "input_manifest.json").read_text())["retry_of"]["code_identical"] is False


def test_the_analysis_uses_the_pinned_bytes_even_if_an_input_file_is_replaced_mid_run(tmp_path, gated, monkeypatch):
    """R2 finding 4: replacing other0.parquet after pinning must not reach the analysis under the old hash."""
    data, ledger_dir, _ = _write_inputs(tmp_path)
    victim = data / "leaderboard" / "user_picks" / "other0.parquet"
    old_sha = _sha(victim.read_bytes())
    original = run.load_accepted_ledger

    def replace_then_load(source):
        frame = pd.read_parquet(victim)
        frame.loc[0, "batter_id"] = 987654
        frame.to_parquet(victim, index=False)
        return original(source)

    monkeypatch.setattr(run, "load_accepted_ledger", replace_then_load)
    assert _main(tmp_path, data, ledger_dir) == 0
    [done] = _completed(tmp_path)
    manifest = json.loads((done / "input_manifest.json").read_text())
    votes = pd.read_parquet(done / "consensus_votes.parquet")
    assert manifest["inputs"]["user_picks"][victim.name] == old_sha
    assert _sha(victim.read_bytes()) != old_sha                              # the file did change on disk ...
    assert not (votes["batter_id"] == 987654).any()                         # ... the analysis used the pinned bytes


def test_completion_is_refused_when_the_bytes_a_loader_consumed_differ_from_the_manifest(tmp_path, gated,
                                                                                        monkeypatch):
    data, ledger_dir, _ = _write_inputs(tmp_path)
    original = run.load_public_picks

    def swap_then_load(files, **kw):
        files = dict(files)
        assert files["other0.parquet"] != files["user_0.parquet"]
        files["other0.parquet"] = files["user_0.parquet"]                   # a parseable file with other bytes
        return original(files, **kw)

    monkeypatch.setattr(run, "load_public_picks", swap_then_load)
    with pytest.raises(SystemExit, match="consumed"):
        _main(tmp_path, data, ledger_dir)
    [partial] = _attempts(tmp_path)
    assert partial.name.endswith(".partial") and not _completed(tmp_path)
    assert (partial / "input_manifest.json").exists()

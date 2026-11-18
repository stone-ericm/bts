"""Feed legacy, null, and valid records through raw pick readers and re-saves."""

import contextlib
import gzip
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from offline import forbid_network

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
forbid_network()

import pandas as pd
from click.testing import CliRunner

from bts import cli, contest_state, picks, scorecard, shadow_eval, skip_policy_shadow, web
from bts.entry_receipt import EntryReceipt
from bts.experiment.artifacts import load_production_pick_snapshot
from bts.health import (calibration, mdp_policy_alignment, pick_entry, pitcher_sparsity, post_failure,
                        predicted_vs_realized, projected_lineup, realized_calibration, same_team_corr)
from bts.model.calibrate import _resolve_pick_outcomes
from bts.simulate.mdp import DEFAULT_POLICY_PATH
from bts.state import export, regenerate, verify
from scripts import (audit_mdp_live_aggressiveness, canonicalize_realized_picks, evaluate_mdp_dd_guardrail,
                     gate_b_production_metric_rebaseline, live_forward_capture_once, live_forward_provenance_inventory,
                     measure_raw_rebin_gate, production_live_boundary_feasibility)
from scripts.audit import boundary_shadow_census, p05_tail_policy_audit
from scripts.audit.benchmark_bridge import run as bridge
from scripts.audit.season_ledger.sources.pick_files import parse_pick_file
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, record


def script_fixtures(root):
    snapshot = root / "p05"
    pick_dir = snapshot / "data" / "picks"
    (pick_dir / "account_state").mkdir(parents=True)
    (pick_dir / "account_state" / "contest_ledger.jsonl").write_text('{"predictions": []}\n')
    model_dir = snapshot / "data" / "models"
    model_dir.mkdir()
    for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        shutil.copyfile(ROOT / "data" / "models" / name, model_dir / name)
    rounds_dir = snapshot / "data" / "leaderboard" / "static_snapshots" / "rounds"
    rounds_dir.mkdir(parents=True)
    with gzip.open(rounds_dir / "rounds.json.gz", "wt") as rounds:
        json.dump({"rounds": []}, rounds)
    (snapshot / "journal_bts-scheduler_retained.txt").touch()
    (root / "p05.sha256").touch()
    processed = snapshot / "data" / "processed"
    processed.mkdir()
    pd.DataFrame([{"date": p05_tail_policy_audit.WINDOW[0], "batter_id": 680757,
                   "game_pk": 101, "is_hit": 1, "lineup_position": 1}]).to_parquet(processed / "pa_2026.parquet")
    return snapshot, pick_dir


def standalone_readers(root, fixture, body):
    snapshot, pick_dir = fixture
    historical_day = p05_tail_policy_audit.WINDOW[0]
    (pick_dir / f"{historical_day}.json").write_text(json.dumps(dict(body, date=historical_day)))
    output = root / "p05-out.json"
    csv = root / "slots.csv"
    slots_source = (ROOT / "scripts" / "audit" / "build_slot_dataset.py").read_text()
    slots_source = slots_source.replace('Path("/home/bts/projects/bts")', f"Path({str(snapshot)!r})")
    slots_source = slots_source.replace('Path("/tmp/slot_dataset_2026.csv")', f"Path({str(csv)!r})")
    slate_dir = pick_dir / "slates"
    slate_dir.mkdir(exist_ok=True)
    (slate_dir / f"{historical_day}.json").write_text("{}")
    with contextlib.redirect_stdout(io.StringIO()):
        p05_tail_policy_audit.main(["--snapshot", str(snapshot), "--out", str(output)])
        original_path = list(sys.path)
        try:
            exec(compile(slots_source, "build_slot_dataset.py", "exec"), {})
        finally:
            sys.path[:] = original_path
    loaded, = bridge.load_inputs(snapshot / "data", [historical_day])["days"]
    assert loaded["pick"] == dict(body, date=historical_day)
    return json.loads(output.read_text())["rows"], csv.read_text(), bridge.primary_of(loaded["pick"])


def dashboard_routes(path):
    messages = []
    handler = SimpleNamespace(_json_response=messages.append, _html_response=messages.append)
    with patch.object(web, "PICKS_DIR", path), patch.object(scorecard, "fetch_live_scorecard", return_value=None):
        web.Handler._handle_api_live(handler, {"date": [DAY]})
        web.Handler._handle_api_live_html(handler, {"date": [DAY]})
    return messages


def exercise(path):
    daily = daily_pick()
    daily.double_down = replace(daily.pick, batter_id=665742, batter_name="Juan Soto", game_pk=102)
    daily.result = "hit"
    daily.slot_results = {"primary": "hit", "double_down": "hit"}
    daily.policy_npz_sha256 = "policy"
    daily.feature_env_hash = "environment"
    original = asdict(daily)
    del original["chosen_on"]
    variants = (original, dict(original, chosen_on=None), dict(original, chosen_on=record()),
                dict(original, chosen_on=record(required=False)))
    today = NOW.date()
    pick_file = path / f"{DAY}.json"
    historic = path / "regenerated"
    historic.mkdir()
    (historic / f"{DAY}.json").write_text(json.dumps(original))
    manifest = path / "manifest.json"
    baseline_raw = json.dumps(original, indent=2).encode()
    snapshot = {"present": True, "source_sha256": sha256(baseline_raw).hexdigest(), "production_pick_json": original}
    manifest.write_text(json.dumps({"production_pick_snapshot": snapshot}))
    raw_pa = pd.DataFrame([{"date": DAY, "batter_id": slot.batter_id, "is_hit": 1} for slot in (daily.pick, daily.double_down)])
    lookup = {(slot.batter_id, today): 1 for slot in (daily.pick, daily.double_down)}
    phase2 = importlib.util.spec_from_file_location("phase2_diff", ROOT / "scripts/cutover/phase2-shadow-diff.py")
    phase2_driver = importlib.util.module_from_spec(phase2)
    phase2.loader.exec_module(phase2_driver)
    fixture = script_fixtures(path)
    projections = {}
    for number, body in enumerate(variants):
        raw = json.dumps(body, indent=2).encode()
        pick_file.write_bytes(raw)
        (path / f"{DAY}.shadow.json").write_text(json.dumps(dict(body, shadow_model_version=shadow_eval.SHADOW_MODEL_NAME)))
        assert picks.load_pick_bytes(DAY, path)[1] == raw
        assert picks.load_pick(DAY, path).chosen_on == body.get("chosen_on")
        with patch.object(web, "PICKS_DIR", path):
            assert next(value for value in web.load_all_picks() if value.get("date") == DAY) == body
        snapshot_read = load_production_pick_snapshot(pick_file, expected_date=DAY)
        assert snapshot_read["production_pick_json"] == body and snapshot_read["source_sha256"] == sha256(raw).hexdigest()
        rows = parse_pick_file(pick_file.name, raw).rows
        assert len(rows) == 2
        for ledger in rows:
            assert json.loads(ledger["record_raw_json"]) == body and ledger["content_sha256"] == sha256(raw).hexdigest()
        receipt = EntryReceipt(et_date=DAY, started_at=NOW, clock=lambda: NOW, expected_username=None, picks_dir=path)
        receipt.selection(picks.load_pick(DAY, path), raw, None, None)
        assert receipt.record()["selection"]["pick_file_sha256"] == sha256(raw).hexdigest()
        capture = live_forward_capture_once.existing_snapshot_state(manifest_path=manifest, pick_path=pick_file, expected_date=DAY)
        inventory = live_forward_provenance_inventory._current_pick_snapshot_summary(snapshot=snapshot, date_key=DAY, picks_dir=path)
        assert capture["snapshot_decision_matches_current_pick"] is inventory["decision_matches_current_pick"] is (number < 2)
        assert capture["snapshot_matches_current_pick"] is inventory["matches_current_pick"] is (number == 0)
        assert capture["stale_pick_snapshot"] is (number >= 2)
        assert capture["current_pick_sha256"] == inventory["current_pick_sha256"] == sha256(raw).hexdigest()
        history = export._pick_to_historical(body)
        shadow_result = CliRunner().invoke(cli.cli, ["shadow-report", "--picks-dir", str(path)], catch_exceptions=False)
        assert shadow_result.exit_code == 0 and "Agreement rate: 1/1" in shadow_result.stdout
        (path / f"{DAY}.shadow.json").unlink()
        with patch.object(measure_raw_rebin_gate, "_day_hit_lookup", return_value=lookup):
            output = dict(
                dashboard=dashboard_routes(path),
                shadow_report=shadow_result.stdout,
                contest=contest_state.latest_resolved_pick_date(path),
                regime=skip_policy_shadow._read_regime(DAY, path),
                drift=calibration.compute_drift_metrics(path, today=today),
                mdp=mdp_policy_alignment.compute_metrics(path, DEFAULT_POLICY_PATH, today=today),
                entry=pick_entry._earliest_cutoff_utc(path, today),
                post_failure=post_failure.check(path, today=today, now=NOW),
                pitcher=pitcher_sparsity.check(path, today=today, thresholds={"min_examined": 1}),
                realized=predicted_vs_realized.compute_metrics(path, today=today),
                lineup=projected_lineup.check(path, today=today),
                calibration=realized_calibration.check(path, today=today),
                same_team=same_team_corr.compute_metrics(path, today=today),
                model_calibrate=_resolve_pick_outcomes(path, raw_pa, today, 2),
                history=history,
                regenerated=regenerate._hist_to_pick_file(history),
                verify=verify.diff_pick_files(path, historic).issues,
                production_slots=snapshot_read["slots"],
                canonical=canonicalize_realized_picks.extract_picks(json.loads(pick_file.read_text()), pick_file.name),
                guardrail=evaluate_mdp_dd_guardrail._load_production_p_both(path),
                placement=production_live_boundary_feasibility.load_pick_slot_rows(path, today=today),
                rebaseline=gate_b_production_metric_rebaseline._load_production_rows(path, today=today),
                pairs=measure_raw_rebin_gate.load_resolved_pair_rows(path, path / "unused.parquet", today),
                aggressiveness=audit_mdp_live_aggressiveness.load_live_pick_rows(path, today=today),
                boundary=boundary_shadow_census._flat_primary_ps(path),
                stamp=boundary_shadow_census._pick_policy_stamp(path, DAY),
                standalone=standalone_readers(path, fixture, body),
                cutover=phase2_driver.compare_picks(json.loads(pick_file.read_text()), original),
            )
        if number == 0:
            projections = output
        else:
            for key in output:
                assert output[key] == projections[key], (key, number, output[key], projections[key])

    assert live_forward_capture_once.decision_snapshot(variants[2]) != live_forward_capture_once.decision_snapshot(variants[3])
    assert live_forward_provenance_inventory._decision_snapshot(variants[2]) != live_forward_provenance_inventory._decision_snapshot(variants[3])
    pick_file.write_bytes(baseline_raw)
    old = picks.load_pick(DAY, path)
    picks.save_pick(old, path)
    new_raw = pick_file.read_bytes()
    assert new_raw == baseline_raw[:-2] + b',\n  "chosen_on": null\n}'
    for supplied in (None, record()):
        daily.chosen_on = supplied
        shadow = picks.save_shadow_pick(daily, path)
        shadow_data = json.loads(shadow.read_text())
        assert list(shadow_data)[-1] == "chosen_on" and shadow_data["chosen_on"] == supplied
        assert picks.load_shadow_pick(DAY, path).chosen_on == supplied
    daily.chosen_on = record()
    picks.attach_provenance(daily)
    picks.save_pick(daily, path)
    next_morning = (NOW + timedelta(days=1)).replace(hour=2)
    with patch.object(picks, "resolve_daily_slot_results", return_value={"primary": "miss", "double_down": "hit"}):
        corrections = picks.reconcile_results(path, clock=lambda: next_morning)
    assert corrections and corrections[0]["new_result"] == "miss"
    corrected = picks.load_pick(DAY, path)
    assert corrected.result == "miss" and corrected.chosen_on == record()
    print(f"{len(projections)} stable projections over four wire inputs, raw hashes, shadow save, and reconcile")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=REVIEW) as tmp:
        exercise(Path(tmp))

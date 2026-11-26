"""Probe the current capture, verifier, resolve and consumer paths offline."""

import dataclasses
import json
import socket
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch


def forbidden(*args, **kwargs):
    raise AssertionError(f"uncaptured transport: {args!r}")


for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
             "gethostbyaddr", "getnameinfo"):
    setattr(socket, name, forbidden)
for name in ("connect", "connect_ex", "sendto", "sendmsg"):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, forbidden)
subprocess.Popen = forbidden

import pandas as pd
from click.testing import CliRunner

from bts.cli import cli
from bts.experiment import artifacts
from bts.health import analytics_artifacts_missing as health
from bts.model import predict
from scripts import live_forward_capture_once as capture
from scripts import live_forward_provenance_inventory as inventory
from scripts import live_forward_resolve_once as resolve
from scripts import live_forward_surface_export as surface


DAY = "2027-03-24"
COMMIT = "offline-capture-sha"
CANDIDATE = "decision_weighted_lgbm_v0"
NOW = "2027-03-24T18:05:00+00:00"
VARIANTS = ("production", "candidate")
CALLS = []
SCORES = (9, 9)
NULLS = 0
DUPLICATE = False


def predictions(*args, **kwargs):
    variant = "candidate" if kwargs.get("blend_configs_override") is not None else "production"
    size = SCORES[VARIANTS.index(variant)]
    batters = list(range(101, 101 + size + NULLS))
    if DUPLICATE:
        batters[-1] = batters[0]
    CALLS.append(["predictions", variant, size, NULLS])
    return pd.DataFrame({
        "batter_id": batters,
        "game_pk": 900001,
        "p_game_hit": [0.9 - i * 0.02 for i in range(size)] + [float("nan")] * NULLS,
    })


def transport(config, args):
    CALLS.append(["cli", args[1], args])
    result = CliRunner().invoke(cli, args)
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return subprocess.CompletedProcess(args, result.exit_code, result.stdout, result.stderr)


def terminal_statuses(**kwargs):
    CALLS.append(["schedule", kwargs])
    return {}


def read(path):
    return json.loads(path.read_bytes())


def write(path, body):
    path.write_text(json.dumps(body, indent=2) + "\n")


def fixture(root, name, *, top_n=10):
    production = root / name
    picks = production / "picks"
    picks.mkdir(parents=True)
    pick = picks / f"{DAY}.json"
    write(pick, {
        "date": DAY, "run_time": NOW, "locked": True,
        "pick": {"batter_id": 101, "batter_name": "Fixture", "team": "AAA",
                 "game_pk": 900001, "p_game_hit": 0.9},
    })
    config = capture.CaptureConfig(
        date=DAY, production_root=production, live_forward_root=root,
        python=Path("/unused/python"), candidate=CANDIDATE,
        artifact_root=Path("live"), data_dir=Path("processed"), picks_dir=Path("picks"),
        top_n=top_n, overwrite=False, fail_on_pending=False,
        auto_recapture_on_snapshot_drift=False,
    )
    return config, pick, production / "live" / DAY


def add_outcomes(config):
    pa = config.production_root / config.data_dir / "pa_2027.parquet"
    pa.parent.mkdir(exist_ok=True)
    pd.DataFrame({"date": [DAY] * 10, "batter_id": list(range(101, 111)),
                  "game_pk": [900001] * 10, "is_hit": [1] * 10}).to_parquet(pa, index=False)


def failed_checks(directory):
    return [item["name"] for item in read(directory / "verification.json")["checks"]
            if item["status"] == "fail"]


def tick(config, pick, directory):
    offset = len(CALLS)
    rc, status = capture.capture_once(config)
    alerts = health.check(pick.parent, date.fromisoformat(DAY), capture_expected=True,
                          capture_artifact_root=directory.parent, capture_unit=None)
    assert [a.level for a in alerts] == [a.level for a in health._check_capture(
        pick.parent, DAY, {}, directory.parent, None,
    )], "fixture must exercise the public health check"
    return {"rc": rc, "status": status["status"], "alerts": [a.level for a in alerts],
            "failed_checks": failed_checks(directory) if (directory / "verification.json").exists() else [],
            "cli_calls": [c[1] for c in CALLS[offset:] if c[0] == "cli"]}


def counts(directory):
    manifest = read(directory / "manifest.json")
    return {v: len(pd.read_parquet(directory / manifest["profile_paths"][v][DAY])) for v in VARIANTS}


def rewrite(directory, size, *, update_manifest):
    path = directory / "manifest.json"
    manifest = read(path)
    for variant in VARIANTS:
        parquet = directory / manifest["profile_paths"][variant][DAY]
        pd.read_parquet(parquet).iloc[:size].to_parquet(parquet, index=False)
        manifest["row_counts"][variant][DAY] = size
    if update_manifest:
        manifest["effective_top_n"] = size
        write(path, manifest)


def consume(directory, pick, resolved_root, *, label):
    shared = dict(resolved_root=resolved_root, picks_dir=pick.parent,
                  expected_candidate=CANDIDATE, expected_top_n=10,
                  require_production_pick_snapshot=True)
    readiness = inventory.inspect_artifact_dir(directory, **shared)
    output = {}
    for official in (False, True):
        name = "official" if official else "research"
        try:
            report = surface.build_surface(
                artifact_root=directory.parent, variant="candidate", require_official_ready=official,
                output_path=directory.parent.parent / f"{label}-{name}.parquet",
                manifest_output_path=None, dates={DAY}, min_date=None, max_date=None, **shared,
            )
        except ValueError as exc:
            output[name] = str(exc)
        else:
            output[name] = report["rows"]
    return {k: readiness[k] for k in ("at_lock_ranked_surface_joinable", "official_fresh_target_ready",
                                      "resolved_outcome_joinable")} | {"surface": output}


def exercise(root):
    global SCORES, NULLS, DUPLICATE
    results = {}
    config, pick, thin = fixture(root, "thin")
    results["thin"] = [tick(config, pick, thin) for _ in range(3)]
    assert [item["status"] for item in results["thin"]] == [
        "failed_verify", "failed_verify_existing", "failed_verify_existing",
    ]
    assert all(item["rc"] == 1 and item["alerts"] == ["CRITICAL"] for item in results["thin"])
    assert results["thin"][0]["cli_calls"] == ["export-live-candidate-artifacts", "verify-candidate-artifacts"]
    assert results["thin"][1]["cli_calls"] == ["verify-candidate-artifacts"]
    results["thin_actual_rows"] = counts(thin)
    assert results["thin_actual_rows"] == dict.fromkeys(VARIANTS, 9)

    add_outcomes(config)
    resolver = resolve.ResolveConfig(
        production_root=config.production_root, python=config.python, candidate=CANDIDATE,
        preoutcome_root=config.artifact_root, resolved_root=Path("resolved"), status_root=Path("status"),
        data_dir=config.data_dir, top_n=10, dates=(DAY,), overwrite=False, fail_on_pending=False,
    )
    resolved = config.production_root / resolver.resolved_root / DAY
    results["resolve"] = []
    for _ in range(2):
        rc, payload = resolve.resolve_artifact_date(resolver, date=DAY, production_head=COMMIT)
        results["resolve"].append({"rc": rc, "status": payload["status"],
                                   "failed_checks": failed_checks(resolved), "rows": counts(resolved)})
    assert [item["status"] for item in results["resolve"]] == ["failed_verify", "failed_verify_existing"]
    assert all(len(item["failed_checks"]) == 2 and all("top_n_rows_per_date" in check
               for check in item["failed_checks"]) for item in results["resolve"])
    results["base_consumers"] = consume(thin, pick, resolved.parent, label="base")
    assert results["base_consumers"]["at_lock_ranked_surface_joinable"] is True
    assert results["base_consumers"]["resolved_outcome_joinable"] is True
    assert results["base_consumers"]["official_fresh_target_ready"] is False
    # The new verifier is unbuilt. Model only its successful persisted signals.
    verification = read(thin / "verification.json")
    status = read(thin / "capture_status.json")
    write(thin / "verification.json", verification | {"ok": True, "failure_count": 0})
    write(thin / "capture_status.json", status | {"status": "exported_verified"})
    results["consumers_with_pass_signals"] = consume(thin, pick, resolved.parent, label="pass")
    assert results["consumers_with_pass_signals"]["surface"] == {"official": 9, "research": 9}
    results["post_outcome_overwrite"] = tick(dataclasses.replace(config, overwrite=True), pick, thin)
    assert results["post_outcome_overwrite"]["status"] == "failed_export_post_outcomes"
    assert not results["post_outcome_overwrite"]["cli_calls"]

    late_cfg, late_pick, late_dir = fixture(root, "late")
    add_outcomes(late_cfg)
    results["late_wrapper"] = tick(late_cfg, late_pick, late_dir)
    assert results["late_wrapper"]["status"] == "failed_export_post_outcomes"
    direct = transport(late_cfg, ["experiment", "export-live-candidate-artifacts", "--date", DAY,
                                  "--candidate", CANDIDATE, "--output-dir", str(late_dir),
                                  "--data-dir", str(late_cfg.production_root / late_cfg.data_dir),
                                  "--top-n", "10", "--production-pick-file", str(late_pick)])
    assert direct.returncode == 0, direct.stderr
    results["late_direct_export"] = {"rc": direct.returncode, "rows": counts(late_dir),
                                     "wrapper_recheck": tick(late_cfg, late_pick, late_dir)}
    assert results["late_direct_export"]["wrapper_recheck"]["status"] == "failed_verify_existing"

    rec_cfg, rec_pick, rec_dir = fixture(root, "recapture")
    results["recapture"] = [tick(rec_cfg, rec_pick, rec_dir)]
    body = read(rec_pick)
    body["pick"]["batter_id"] = 102
    write(rec_pick, body)
    rec_cfg = dataclasses.replace(rec_cfg, auto_recapture_on_snapshot_drift=True)
    results["recapture"].append(tick(rec_cfg, rec_pick, rec_dir))
    add_outcomes(rec_cfg)
    results["recapture"].append(tick(rec_cfg, rec_pick, rec_dir))
    assert [item["status"] for item in results["recapture"]] == [
        "failed_verify", "failed_recapture_verify", "failed_recapture_post_outcomes",
    ]

    dup_cfg, dup_pick, dup_dir = fixture(root, "duplicate")
    DUPLICATE = True
    results["nine_rows_eight_batters"] = tick(dup_cfg, dup_pick, dup_dir)
    assert counts(dup_dir) == dict.fromkeys(VARIANTS, 9)
    dup_manifest = read(dup_dir / "manifest.json")
    dup_profile = pd.read_parquet(dup_dir / dup_manifest["profile_paths"]["production"][DAY])
    assert dup_profile["batter_id"].nunique() == 8
    DUPLICATE = False
    NULLS = 3
    null_cfg, null_pick, null_dir = fixture(root, "null")
    results["twelve_rows_nine_scores"] = tick(null_cfg, null_pick, null_dir)
    assert counts(null_dir) == dict.fromkeys(VARIANTS, 9)
    NULLS = 0

    SCORES = (10, 10)
    full_cfg, full_pick, full = fixture(root, "full")
    results["full"] = [tick(full_cfg, full_pick, full)]
    assert results["full"][0]["status"] == "exported_verified"
    rewrite(full, 9, update_manifest=False)
    results["full"].append(tick(full_cfg, full_pick, full))
    rewrite(full, 9, update_manifest=True)
    results["full"].append(tick(full_cfg, full_pick, full))
    assert len(results["full"][1]["failed_checks"]) == 4
    assert all("top_n_rows_per_date" in item for item in results["full"][2]["failed_checks"])
    report = artifacts.verify_candidate_artifact_pair(artifact_dir=full, require_live_preoutcome=True)
    assert not report["ok"], "implicit live check also uses manifest.top_n"
    path = full / "manifest.json"
    manifest = read(path)
    write(path, manifest | {"run_kind": ["live_forward_preoutcome"]})
    results["bad_run_kind"] = tick(full_cfg, full_pick, full)
    assert "live_run_kind" in results["bad_run_kind"]["failed_checks"]

    SCORES = (9, 8)
    skew_cfg, skew_pick, skew = fixture(root, "skew")
    results["unequal_scores"] = tick(skew_cfg, skew_pick, skew) | {"rows": counts(skew)}
    assert results["unequal_scores"]["rows"] == {"production": 9, "candidate": 8}
    return results


if __name__ == "__main__":
    with (patch.object(predict, "run_pipeline", predictions),
          patch.object(artifacts, "current_git_commit", lambda: COMMIT),
          patch.object(artifacts, "utc_timestamp", lambda: NOW),
          patch.object(artifacts, "_load_terminal_void_statuses", terminal_statuses),
          patch.object(capture, "utc_now_dt", lambda: datetime.fromisoformat(NOW)),
          patch.object(capture, "git_head", lambda _: COMMIT),
          patch.object(capture, "run_bts", transport),
          patch.object(resolve, "run_bts", transport)):
        with tempfile.TemporaryDirectory(prefix="capture-", dir=Path(__file__).resolve().parent) as temp:
            summary = exercise(Path(temp))
        write(Path(__file__).with_name("capture-transports.json"), CALLS)
        print(json.dumps(summary, indent=2))

"""Run the actual F7 exporter, verifier, wrapper, and capture alert offline."""
import json
from pathlib import Path
import subprocess

from click.testing import CliRunner
import pandas as pd
import pytest

DAY = "2027-03-24"
COMMIT = "offline-capture-sha"
VARIANTS = ("production", "candidate")


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def capture(root, revision):
    from bts.cli import cli
    from bts.experiment import artifacts
    from bts.experiment.models import DecisionWeightedLightGBMExperiment
    from bts.health.analytics_artifacts_missing import _check_capture
    from bts.model import predict
    from scripts import live_forward_capture_once as live
    from tests.experiment.test_artifacts import _write_pick_file

    base = revision == "base"
    head = revision == "head"
    original_convert = artifacts.predictions_to_ranked_profiles
    results = {}
    predictions = []
    exports = []
    scores = (10, 10)
    no_score = 0
    lost = ()
    missing_score = False

    def pipeline(*args, **kwargs):
        which = "candidate" if kwargs.get("blend_configs_override") is not None else "production"
        scored = scores[VARIANTS.index(which)]
        frame = pd.DataFrame({
            "batter_id": [11 * (i + 1) for i in range(scored + no_score)],
            "game_pk": list(range(1001, 1001 + scored + no_score)),
            "p_game_hit": [0.91 - 0.02 * i for i in range(scored)] + [float("nan")] * no_score,
        })
        if missing_score:
            frame = frame.drop(columns="p_game_hit")
        frame.attrs["variant"] = which
        predictions.append([which, len(frame), scored])
        return frame

    def convert(frame, **kwargs):
        profile = original_convert(frame, **kwargs)
        return profile.iloc[:-1].copy() if frame.attrs["variant"] in lost else profile

    def transport(config, args):
        if args[1] == "export-live-candidate-artifacts":
            exports.append(args)
        response = CliRunner().invoke(cli, args, catch_exceptions=True)
        stderr = response.stderr
        if response.exception and not isinstance(response.exception, SystemExit):
            stderr += f"{type(response.exception).__name__}: {response.exception}\n"
        return subprocess.CompletedProcess(args, response.exit_code, response.stdout, stderr)

    def fixture(label, top_n=10):
        production = root / label
        pick = production / "picks" / f"{DAY}.json"
        pick.parent.mkdir(parents=True)
        _write_pick_file(pick, date=DAY)
        cfg = live.CaptureConfig(
            date=DAY, production_root=production, live_forward_root=root,
            python=Path("/unused/python"), candidate="decision_weighted_lgbm_v0",
            artifact_root=Path("live"), data_dir=Path("processed"), picks_dir=Path("picks"),
            top_n=top_n, overwrite=False, fail_on_pending=False, auto_recapture_on_snapshot_drift=False,
        )
        return cfg, pick, production / cfg.artifact_root / DAY

    def tick(cfg, pick, arc):
        count = len(exports)
        rc, payload = live.capture_once(cfg)
        health = _check_capture(pick.parent, DAY, {}, arc.parent, None)
        return {"rc": rc, "status": payload["status"], "exports": len(exports) - count,
                "health": [item.level for item in health], "detail": payload["message"]}

    def verify(arc, top_n=10, *, resolved=False):
        kwargs = {"expected_top_n": top_n} if top_n is not None else {}
        report = artifacts.verify_candidate_artifact_pair(
            artifact_dir=arc, require_live_preoutcome=not resolved, expected_date=DAY,
            expected_git_commit=COMMIT, require_production_pick_snapshot=True,
            expected_run_kind="live_forward_resolved" if resolved else "live_forward_preoutcome", **kwargs,
        )
        return {"ok": report["ok"], "failed": [c["name"] for c in report["checks"] if c["status"] == "fail"]}

    def direct(label, top_n=10):
        cfg, pick, arc = fixture(label, top_n)
        try:
            manifest = artifacts.materialize_live_candidate_profile_pair(
                date=DAY, candidate=DecisionWeightedLightGBMExperiment(), output_dir=arc,
                top_n=top_n, production_pick_file=pick, git_commit=COMMIT,
                generated_at="2027-03-24T13:00:00+00:00",
            )
        except ValueError as exc:
            results[label] = {"exception": str(exc), "directory_exists": arc.is_dir(),
                              "files": sorted(str(path.relative_to(arc)) for path in arc.rglob("*"))}
            assert results[label]["files"] == [], results[label]
        else:
            results[label] = {"requested": top_n, "effective": manifest.get("effective_top_n"),
                              "rows": {v: manifest["row_counts"][v][DAY] for v in VARIANTS},
                              **verify(arc, top_n)}
        return cfg, pick, arc

    def shorten(arc, count, which=VARIANTS):
        manifest = json.loads((arc / "manifest.json").read_text())
        for variant in which:
            path = arc / manifest["profile_paths"][variant][DAY]
            pd.read_parquet(path).iloc[:count].to_parquet(path, index=False)
        return manifest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live, "git_head", lambda *_: COMMIT)
        patch.setattr(live, "run_bts", transport)
        patch.setattr(artifacts, "current_git_commit", lambda: COMMIT)
        patch.setattr(predict, "run_pipeline", pipeline)
        patch.setattr(artifacts, "predictions_to_ranked_profiles", convert)

        for which in (VARIANTS, ("production",), ("candidate",)):
            lost = which
            label = "conversion-lost-" + "-".join(which)
            direct(label)
            case = results[label]
            if head:
                assert "scored prediction rows" in case["exception"] and case["directory_exists"]
            else:
                assert case["ok"] is (not base and len(which) == 2), case
            assert predictions[-2:] == [[v, 10, 10] for v in VARIANTS]

        lost = VARIANTS
        cfg, pick, arc = fixture("wrapper-conversion-lost")
        failed = tick(cfg, pick, arc)
        expected = "failed_export" if head else "failed_verify" if base else "exported_verified"
        assert failed["status"] == expected and failed["exports"] == 1, failed
        assert failed["health"] == ([] if revision == "before" else ["CRITICAL"])
        failed["files"] = sorted(str(path.relative_to(arc)) for path in arc.rglob("*") if path.is_file())
        if head:
            assert failed["files"] == ["capture_status.json"], failed
            failed["next_tick"] = tick(cfg, pick, arc)
            assert failed["next_tick"]["status"] == "failed_partial_artifact_dir"
            assert failed["next_tick"]["health"] == ["CRITICAL"]
        elif revision == "before":
            failed["next_tick"] = tick(cfg, pick, arc)
            assert failed["next_tick"]["status"] == "existing_verified"
        results["wrapper-conversion-lost"] = failed

        lost = ()
        scores = (0, 0)
        cfg, pick, arc = fixture("wrapper-zero-scored")
        failed = tick(cfg, pick, arc)
        assert (failed["rc"], failed["status"], failed["health"]) == (1, "failed_export", ["CRITICAL"])
        failed["files"] = sorted(str(path.relative_to(arc)) for path in arc.rglob("*") if path.is_file())
        assert failed["files"] == ["capture_status.json"], failed
        results["wrapper-zero-scored"] = failed
        direct("zero-scored")
        assert "no scored rows" in results["zero-scored"]["exception"]
        scores = (10, 10)
        missing_score = True
        direct("missing-score-column")
        assert "missing columns: ['p_game_hit']" in results["missing-score-column"]["exception"]
        missing_score = False

        scores = (9, 9)
        _, _, thin = direct("nine-scored-cap-ten")
        assert results["nine-scored-cap-ten"]["ok"] is not base
        no_score = 3
        direct("twelve-rows-nine-scored")
        assert results["twelve-rows-nine-scored"]["ok"] is not base
        assert predictions[-2:] == [[v, 12, 9] for v in VARIANTS]
        no_score = 0
        direct("cap-smaller-than-slate", 5)
        assert results["cap-smaller-than-slate"]["ok"]
        assert results["cap-smaller-than-slate"]["rows"] == dict.fromkeys(VARIANTS, 5)
        if not base:
            assert results["cap-smaller-than-slate"]["effective"] == 5
        scores = (9, 8)
        cfg, pick, arc = direct("different-scored-counts")
        assert not results["different-scored-counts"]["ok"]
        assert results["different-scored-counts"]["rows"] == {"production": 9, "candidate": 8}
        if not base:
            assert results["different-scored-counts"]["effective"] == 9
            assert results["different-scored-counts"]["failed"] == [f"candidate_{DAY}_top_n_rows_per_date"]
        results["wrapper-different-counts"] = tick(cfg, pick, arc)
        assert results["wrapper-different-counts"]["health"] == ["CRITICAL"]

        if not base:
            data = root / "resolved-data"
            data.mkdir()
            pd.DataFrame({"date": [DAY] * 9, "batter_id": [11 * (i + 1) for i in range(9)],
                          "game_pk": list(range(1001, 1010)), "is_hit": [1] * 9}).to_parquet(
                              data / "pa_2027.parquet", index=False)
            resolved = root / "resolved"
            artifacts.resolve_live_candidate_artifact_pair(artifact_dir=thin, output_dir=resolved, data_dir=data)
            assert json.loads((resolved / "manifest.json").read_text())["effective_top_n"] == 9
            results["resolved"] = verify(resolved, resolved=True)
            assert results["resolved"]["ok"]

        scores = (10, 10)
        for count in (10, 9):
            scores = (count, count)
            _, _, arc = direct(f"legacy-input-{count}")
            manifest = json.loads((arc / "manifest.json").read_text())
            manifest.pop("effective_top_n", None)
            write(arc / "manifest.json", manifest)
            results[f"legacy-{count}"] = verify(arc)
            assert results[f"legacy-{count}"]["ok"] is (count == 10)

        scores = (10, 10)
        cfg, pick, arc = fixture("rewritten-full-slate")
        results["full-before-rewrite"] = tick(cfg, pick, arc)
        assert results["full-before-rewrite"]["status"] == "exported_verified"
        path = arc / "manifest.json"
        manifest = shorten(arc, 9)
        results["truncated-intact-manifest"] = verify(arc)
        assert not results["truncated-intact-manifest"]["ok"]
        manifest["effective_top_n"] = 9
        write(path, manifest)
        results["truncated-only-effective-edited"] = verify(arc)
        assert not results["truncated-only-effective-edited"]["ok"]
        for which in VARIANTS:
            manifest["row_counts"][which][DAY] = 9
        write(path, manifest)
        results["truncated-effective-and-counts-edited"] = verify(arc)
        assert results["truncated-effective-and-counts-edited"]["ok"] is not base
        results["wrapper-rewritten-slate"] = tick(cfg, pick, arc)
        status = "failed_verify_existing" if base else "existing_verified"
        assert results["wrapper-rewritten-slate"]["status"] == status
        assert results["wrapper-rewritten-slate"]["exports"] == 0
        assert results["wrapper-rewritten-slate"]["health"] == (["CRITICAL"] if base else [])
        archived_manifest = arc / "manifest-edited-evidence.json"
        archived_manifest.write_bytes(path.read_bytes())
        # Without a separately supplied expected top-N, both trust an edited request.
        manifest["top_n"] = 9
        write(path, manifest)
        results["truncated-request-also-edited-implicit-cap"] = verify(arc, top_n=None)
        results["truncated-request-also-edited-explicit-cap"] = verify(arc)
        assert results["truncated-request-also-edited-implicit-cap"]["ok"]
        assert not results["truncated-request-also-edited-explicit-cap"]["ok"]

        _, _, corrupt = direct("unreadable-input")
        body = json.loads((corrupt / "manifest.json").read_text())
        (corrupt / body["profile_paths"]["candidate"][DAY]).write_bytes(b"PAR1 truncated")
        results["unreadable-parquet"] = verify(corrupt)
        assert not results["unreadable-parquet"]["ok"]

        if head:
            for effective in (True, -1):
                _, _, arc = direct(f"invalid-input-{effective!r}", top_n=1)
                path = arc / "manifest.json"
                data = json.loads(path.read_text())
                write(path, data | {"effective_top_n": effective})
                result = verify(arc, top_n=1)
                assert result == {"ok": False, "failed": ["live_effective_top_n"]}, result
                results[f"invalid-effective-{effective!r}"] = result
            _, _, arc = direct("historical-kind-input")
            path = arc / "manifest.json"
            data = json.loads(path.read_text())
            kind = "historical_local_screen"
            for variant in VARIANTS:
                profile = arc / data["profile_paths"][variant][DAY]
                pd.read_parquet(profile).assign(run_kind=kind).to_parquet(profile, index=False)
            write(path, data | {"run_kind": kind, "effective_top_n": True})
            result = artifacts.verify_candidate_artifact_pair(
                artifact_dir=arc, expected_top_n=10, expected_run_kind=kind,
            )
            names = [c["name"] for c in result["checks"] if c["name"] == "live_effective_top_n"]
            assert result["ok"] and not names, result
            results["historical-effective-ignored"] = {"ok": result["ok"], "live_checks": names}

    write(root / "summary.json", results)
    write(root / "prediction-calls.json", predictions)


def compare_capture(parent):
    base = json.loads((parent / "base/summary.json").read_text())
    head = json.loads((parent / "head/summary.json").read_text())
    for case in ("legacy-10", "legacy-9", "zero-scored", "missing-score-column", "unreadable-parquet"):
        assert base[case] == head[case], case

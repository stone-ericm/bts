"""Exercise the shipped entry points against the three archived revisions."""
from contextlib import redirect_stdout
import datetime as dt
import io
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd
import pytest

from review_runner import first_review

ET = ZoneInfo("America/New_York")
DAY = "2027-03-24"
UTC = dt.timezone.utc
REAL_DATETIME = dt.datetime


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def levels(alerts):
    return [alert.level for alert in alerts]


def failures(report):
    return [item["name"] for item in report["checks"] if item["status"] == "fail"]


def contest(root, tree):
    from bts.health.contest_state import check
    from bts.picks import save_pick
    from tests.conftest import covered_automatic
    from tests.test_shadow_eval import _daily, _pick

    results = {}

    def case(label, when, dates, old, new, *, recorded=None, source=None, void=False, double=False):
        target = root / label
        now = REAL_DATETIME.fromisoformat(when).replace(tzinfo=ET)
        target.mkdir()
        for date in dates:
            save_pick(_daily(date, _pick("One", 1, game_pk=1001),
                             double_down=_pick("Two", 2, game_pk=1002) if double else None,
                             result="void" if void and date == DAY else "hit"), target)
        covered_automatic.__wrapped__()(target, now, {
            "active_streak": 1, "best_streak": 1, "source_date": source,
            "recorded_at": recorded or (now - dt.timedelta(minutes=15)).isoformat(),
        })
        alerts = check(target, expected=True, now=now)
        observed = [a.level for a in alerts if "no source_date" in a.message]
        expected = old if tree == "before" else new
        assert observed == ([expected] if expected else []), (label, alerts)
        results[label] = [(a.level, a.message) for a in alerts]

    for day in ("2027-01-02", DAY):
        case(f"july-old-{day}", "2027-07-15T22:15", [day], "INFO", "CRITICAL")
        case(f"july-old-and-today-{day}", "2027-07-15T22:15", [day, "2027-07-15"], "INFO", "CRITICAL")
    for year, month_day in ((2027, "03-24"), (2027, "04-20"), (2028, "04-03")):
        opener = dt.date.fromisoformat(f"{year}-{month_day}")
        next_day = (opener + dt.timedelta(days=1)).isoformat()
        label = str(opener)
        case(f"opener-{label}", f"{opener}T22:15", [label], "INFO", "INFO")
        case(f"next-{label}", f"{next_day}T22:15", [label, next_day], "INFO", "INFO")
    case("after-midnight-et", "2027-03-25T00:01", [DAY], "INFO", "INFO")
    case("opening-doubleheader", f"{DAY}T22:15", [DAY], "INFO", "INFO", double=True)
    case("opening-void", f"{DAY}T22:15", [DAY], "INFO", "INFO", void=True)
    case("void-then-real-pick", "2027-03-25T22:15", [DAY, "2027-03-25"], "INFO", "INFO", void=True)
    case("off-day", "2027-03-25T22:15", [DAY], "INFO", "INFO")
    case("day-three-boundary", "2027-03-26T00:00", [DAY], "INFO", "CRITICAL")
    case("second-game", "2027-03-26T22:15", [DAY, "2027-03-26"], "INFO", "CRITICAL")
    case("following-midnight", "2027-03-27T00:01", [DAY, "2027-03-26"], "INFO", "CRITICAL")
    case("coverage-catches-up", "2027-03-27T10:00", [DAY, "2027-03-26"], None, None, source=DAY)
    case("january-plus-opener", f"{DAY}T22:15", ["2027-01-02", DAY], "INFO", "CRITICAL")
    case("july-first-local-pick", "2027-07-15T22:15", ["2027-07-15"], "INFO", "INFO")
    case("july-first-local-pick-next-morning", "2027-07-16T00:01", ["2027-07-15"], "INFO", "INFO")
    case("july-first-local-pick-after-grace", "2027-07-17T00:00", ["2027-07-15"], "INFO", "CRITICAL")
    write(root / "summary.json", results)


def park(root, tree):
    from bts import strategy
    from bts.health import park_drag_freshness as freshness

    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    results = {}
    folder = root / "park-drag"
    folder.mkdir()
    (folder / "park_drag_export.csv").write_text("fixture\n")
    write(folder / "producer_status.json", {"ok": True})

    def case(name, day, source, expected, **thresholds):
        nonlocal instant
        instant = REAL_DATETIME.fromisoformat(f"{day}T12:00").replace(tzinfo=ET)
        write(folder / "park_drag_manifest.json", {
            "max_source_game_date": source, "generated_at": instant.replace(tzinfo=None).isoformat(),
        })
        alerts = freshness.check(folder, today=instant.date(), thresholds=thresholds)
        assert levels(alerts) == expected, (name, alerts, expected)
        results[name] = [(a.level, a.message) for a in alerts]

    instant = None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freshness, "datetime", Clock)
        hard = ["CRITICAL"]
        old = tree == "before"
        base = tree == "base"
        case("midseason-stop", "2027-03-26", "2026-06-15", [] if old else hard)
        case("two-seasons-old", "2027-03-26", "2025-09-28", [] if old else hard)
        case("normal-carryover", "2027-03-26", "2026-09-27", hard if base else [])
        case("normal-warn", "2027-03-27", "2026-09-27", hard if base else ["WARN"])
        case("normal-critical", "2027-03-30", "2026-09-27", hard)
        case("threshold-at-close", "2026-09-28", "2026-09-22", hard)
        case("threshold-next-opening", "2027-03-26", "2026-09-22", hard if base else [])
        case("threshold-next-warn", "2027-03-27", "2026-09-22", hard if base else ["WARN"])
        case("threshold-next-critical", "2027-03-30", "2026-09-22", hard)
        case("threshold-plus-one", "2027-03-26", "2026-09-21", [] if old else hard)
        if not base:
            patch.setitem(strategy.SEASON_OPENER_DATES, 2029, "2029-03-24")
            assert 2028 not in strategy.SEASON_END_DATES
            results["fallback-end"] = strategy.season_end_date("2028-09-23")
            assert results["fallback-end"] == "2028-09-28"
            case("fallback-admits-grace", "2029-03-26", "2028-09-23", [])
            patch.setitem(strategy.SEASON_END_DATES, 2028, "2028-10-01")
            case("configured-later-end", "2029-03-26", "2028-09-23", [] if old else hard)
    write(root / "summary.json", results)


def capture(root, tree):
    from bts.cli import cli
    from bts.experiment import artifacts
    from bts.health.analytics_artifacts_missing import _check_capture
    from bts.model import predict
    from scripts import live_forward_capture_once as live
    from tests.experiment.test_artifacts import _write_pick_file

    if tree != "head":
        original = root / "first-review"
        original.mkdir(exist_ok=True)
        first_review().capture(original, "head" if tree == "before" else "base")
    else:
        from tests.scripts.test_live_forward_capture_once import test_opening_capture_keeps_frozen_evidence
        for scenario in ("pa_at_export", "pa_at_swap", "verified_recheck"):
            with pytest.MonkeyPatch.context() as patch:
                test_opening_capture_keeps_frozen_evidence(root / scenario, patch, scenario)

    head = "offline-capture-sha"
    production = root / "fixture"
    pick = production / "picks" / f"{DAY}.json"
    pick.parent.mkdir(parents=True)
    _write_pick_file(pick, date=DAY)
    cfg = live.CaptureConfig(
        date=DAY, production_root=production, live_forward_root=root,
        python=Path("/unused/python"), candidate="decision_weighted_lgbm_v0",
        artifact_root=Path("live"), data_dir=Path("processed"), picks_dir=Path("picks"),
        top_n=10, overwrite=False, fail_on_pending=False, auto_recapture_on_snapshot_drift=False,
    )
    arc = production / cfg.artifact_root / DAY
    manifest_path = arc / "manifest.json"
    results = {}
    exports = []
    scored = []
    count = 9
    missed_row = False

    def pipeline(*args, **kwargs):
        scored.append(count)
        return pd.DataFrame({
            "batter_id": [11 * (i + 1) for i in range(count)],
            "game_pk": [1001 + i for i in range(count)],
            "p_game_hit": [0.91 - 0.02 * i for i in range(count)],
        })

    def transport(*args):
        commands = args[-1]
        if commands[1] == "export-live-candidate-artifacts":
            exports.append(commands)
        response = CliRunner().invoke(cli, commands, catch_exceptions=False)
        return subprocess.CompletedProcess(commands, response.exit_code, response.stdout, response.stderr)

    original_convert = artifacts.predictions_to_ranked_profiles

    def convert(*args, **kwargs):
        profile = original_convert(*args, **kwargs)
        return profile.iloc[:-1].copy() if missed_row else profile

    def verify(label, *, artifact=arc, resolved=False):
        report = artifacts.verify_candidate_artifact_pair(
            artifact_dir=artifact, expected_top_n=10, require_live_preoutcome=not resolved,
            expected_run_kind="live_forward_resolved" if resolved else None,
        )
        results[label] = [report["ok"], failures(report)]
        return report

    def tick(label):
        rc, payload = live.capture_once(cfg)
        health = _check_capture(pick.parent, DAY, {}, arc.parent, None)
        results[label] = [rc, payload["status"], len(exports), levels(health)]
        return rc

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live, "git_head", lambda *_: head)
        patch.setattr(live, "run_bts", transport)
        patch.setattr(artifacts, "current_git_commit", lambda: head)
        patch.setattr(predict, "run_pipeline", pipeline)
        patch.setattr(artifacts, "predictions_to_ranked_profiles", convert)
        base = tree == "base"
        assert tick("nine-scored-first-tick") == (1 if base else 0)
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        assert manifest["top_n"] == 10
        count = 18
        assert tick("nine-scored-next-tick") == (1 if base else 0)
        assert len(exports) == 1 and scored == [9, 9] and manifest_path.read_bytes() == original
        legacy = {key: value for key, value in manifest.items() if key != "effective_top_n"}
        write(manifest_path, legacy)
        assert not verify("legacy-nine")["ok"]
        if not base:
            assert manifest["effective_top_n"] == 9
            for effective in (2.0, True, 0, -1, 11, 8):
                write(manifest_path, manifest | {"effective_top_n": effective})
                assert not verify(f"invalid-effective-{effective!r}")["ok"]
            write(manifest_path, manifest)
            pa = production / cfg.data_dir / "pa_2027.parquet"
            pa.parent.mkdir()
            pd.DataFrame({"date": [DAY] * 9, "batter_id": [11 * (i + 1) for i in range(9)],
                          "game_pk": list(range(1001, 1010)), "is_hit": [1] * 9}).to_parquet(pa, index=False)
            resolved = root / "resolved"
            artifacts.resolve_live_candidate_artifact_pair(
                artifact_dir=arc, output_dir=resolved, data_dir=pa.parent,
            )
            assert json.loads((resolved / "manifest.json").read_text())["effective_top_n"] == 9
            assert verify("resolved-nine", artifact=resolved, resolved=True)["ok"]
            prod = arc / manifest["profile_paths"]["production"][DAY]
            frame = pd.read_parquet(prod)
            frame.iloc[:-1].to_parquet(prod, index=False)
            assert not verify("truncated-with-untouched-manifest")["ok"]
            manifest["row_counts"]["production"][DAY] = 8
            write(manifest_path, manifest)
            assert not verify("variants-disagree-with-updated-row-count")["ok"]
            prod.write_bytes(b"truncated parquet bytes")
            assert not verify("physically-truncated-parquet")["ok"]

        count = 10
        assert live.export_artifact(cfg, pick_path=pick, artifact_dir=arc).returncode == 0
        full = json.loads(manifest_path.read_text())
        write(manifest_path, {key: value for key, value in full.items() if key != "effective_top_n"})
        assert verify("legacy-ten")["ok"]
        missed_row = True
        assert live.export_artifact(cfg, pick_path=pick, artifact_dir=arc).returncode == 0
        shortened = json.loads(manifest_path.read_text())
        assert scored[-2:] == [10, 10]
        assert all(shortened["row_counts"][v][DAY] == 9 for v in ("production", "candidate"))
        results["dropped-row-manifest"] = {key: shortened.get(key) for key in ("top_n", "effective_top_n", "row_counts")}
        assert verify("ten-scored-nine-exported")["ok"] is not base
        assert tick("ten-scored-nine-exported-tick") == (1 if base else 0)
    write(root / "summary.json", results)


def shadow(root, tree):
    from bts import cli, daily_decision, shadow_eval, skip_policy_shadow as policy
    from tests.test_skip_policy_shadow import _cand

    now = REAL_DATETIME(2027, 3, 26, 18, tzinfo=UTC)

    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.astimezone(ET).replace(tzinfo=None)

    old_days = ["2026-09-25", "2026-09-26", "2026-09-27"]
    new_day = "2027-03-24"
    aged = old_days[-1]
    results = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dt, "datetime", Clock)
        patch.setattr(policy, "datetime", Clock)
        patch.setattr(shadow_eval, "_current_git_commit", lambda: "offline-git-sha")
        # The real default command reconciles no records when all outcomes are already terminal.
        patch.setattr(policy, "make_hit_checker", lambda: lambda *_: pytest.fail("unexpected outcome transport"))
        for sample, dates in (("mixed", [*old_days, new_day]), ("historical", old_days)):
            picks = root / sample
            for index, day in enumerate(dates):
                daily_decision.write_decision(
                    day, picks, action="skip", source="mdp", primary=_cand(),
                    now=now, streak=10, saver_available=False,
                    delivery_status="not_applicable", scoreable=False,
                )
                assert policy.record_skip_from_decision(day, picks, now=now)
                path = policy.decision_path(day, picks)
                write(path, json.loads(path.read_text()) | {"shadow_pick_result": "miss" if index == 1 else "hit"})
            daily_decision.write_decision(
                aged, picks, action="single", source="mdp", primary=_cand(),
                now=now, delivery_status="delivered", scoreable=True,
            )
            assert policy.find_aged_contradictions(picks, now=now) == [aged]
            commands = [("default", [])]
            if tree != "base":
                commands += [("2026", ["--season", "2026"])]
            for name, args in commands:
                out = root / f"{sample}-{name}.json"
                response = CliRunner().invoke(cli.cli, [
                    "skip-policy-shadow-update", "--picks-dir", str(picks),
                    "--status-output", str(out), *args,
                ], catch_exceptions=False)
                assert response.exit_code == 0, response.output
                (root / f"{sample}-{name}.stdout").write_text(response.output)
                status = json.loads(out.read_text())
                warning = [line for line in response.output.splitlines() if "AGED CONTRADICTIONS" in line]
                expected_dates = old_days[:2] if name == "2026" or tree == "base" else []
                if sample == "mixed" and name == "default":
                    expected_dates += [new_day]
                assert [row["date"] for row in status["rows"]] == expected_dates
                expected_warning = tree != "before" or name == "2026"
                assert status["counts"]["aged_superseded_records"] == ([aged] if expected_warning else [])
                assert bool(warning) is expected_warning
                results[f"{sample}-{name}"] = {"warning": warning, "dates": expected_dates}
    write(root / "summary.json", results)


def compare_shadow(parent):
    def normalized(path):
        status = json.loads(path.read_text())
        status.pop("picks_dir", None)
        return status

    base = parent / "base"
    head = parent / "head"
    assert normalized(base / "historical-default.json") == normalized(head / "mixed-2026.json")
    for name in ("mixed-default", "historical-default"):
        def warning(directory):
            return [line for line in (directory / f"{name}.stdout").read_text().splitlines()
                    if "AGED CONTRADICTIONS" in line]
        assert warning(base) == warning(head)


def closeout(root, tree):
    from bts.picks import save_pick, save_shadow_pick
    from scripts.audit import p03_context_shadow_closeout as script
    from tests.test_shadow_eval import _daily, _pick

    assert dt.date.today().year == dt.datetime.now().year == 2027
    common = root.parent
    snapshot = common / "final-20260928"
    picks = snapshot / "data" / "picks"
    picks.mkdir(parents=True, exist_ok=True)
    raw = common / "raw"
    for day, prod, model in (("2026-09-14", "miss", "hit"), ("2026-09-27", "hit", "miss")):
        save_pick(_daily(day, _pick("Prod", 1), result=prod), picks)
        save_shadow_pick(_daily(day, _pick("Shadow", 2), result=model), picks)
    for game in (800001, 800002):
        write(raw / "2026" / f"{game}.json", {"gameData": {"status": {"abstractGameCode": "I"}}})
    (common / "final-20260928.sha256").write_text("offline frozen fixture manifest\n")
    out = root / "closeout.json"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert script.main(["--snapshot", str(snapshot), "--raw-dir", str(raw), "--out", str(out)]) == 0
    (root / "closeout.stdout").write_text(stdout.getvalue())
    report = json.loads(out.read_text())
    assert report["recorded"]["n_evaluable_days"] == (0 if tree == "before" else 2)
    assert report["recompute_crosscheck"]["api_calls"] == 0


def match(parent, files):
    for name in files:
        assert (parent / "head" / name).read_bytes() == (parent / "base" / name).read_bytes(), name


PROBES = {
    "contest": contest, "park": park, "capture": capture, "shadow": shadow,
    "p03": closeout, "pick": lambda root, tree: first_review().pick_path(root, tree),
}
SELECTED = {name: ("base", "before", "head") for name in PROBES}
SELECTED.update(contest=("before", "head"), pick=("base", "head"))
COMPARES = dict.fromkeys(PROBES, lambda _: None)
COMPARES.update(
    shadow=compare_shadow,
    p03=lambda parent: match(parent, ("closeout.json", "closeout.stdout")),
    pick=lambda parent: match(parent, ("pick.json", "decision.json", "score.json", "sends.json")),
)

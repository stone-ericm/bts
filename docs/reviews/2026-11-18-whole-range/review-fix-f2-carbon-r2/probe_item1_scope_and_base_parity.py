"""Item 1 probe: contract file scope check + paired base (a4e8278d) vs candidate (3b7d5a18) parity grid."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas  # initialize before datetime patching
from click.testing import CliRunner
from pytest import MonkeyPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SRC = Path(__file__).resolve().parent / "base_archive" / "src"
CAND_SRC = REPO_ROOT / "src"

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 0, tzinfo=ET)
NEXT = REAL_DT(2027, 7, 16, 0, 5, tzinfo=ET)
ACTIVATION = "a" * 32


def check_git_scope() -> dict:
    """Verify changed files in 2e20fdfc and 3b7d5a18 against base a4e8278d."""
    def git_lines(*args: str) -> list[str]:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, check=True,
        )
        return [line for line in proc.stdout.splitlines() if line.strip()]

    r1_files = git_lines("diff", "--name-only", "a4e8278d", "2e20fdfc", "--", ":!docs/reviews")
    assert sorted(r1_files) == [
        "src/bts/health/contest_state.py",
        "src/bts/health/runner.py",
        "tests/test_contest_dated_cs4a.py",
        "tests/test_wr_f2_saver_flag.py",
    ], r1_files

    r2_files = git_lines("diff", "--name-only", "a4e8278d", "3b7d5a18", "--", ":!docs/reviews")
    assert sorted(r2_files) == [
        "docs/ops/2027-season-start.md",
        "src/bts/health/contest_state.py",
        "src/bts/health/runner.py",
        "tests/test_contest_dated_cs4a.py",
        "tests/test_f2_b1_stat_oserror_regression.py",
        "tests/test_wr_f2_saver_flag.py",
    ], r2_files

    # Verify tests/test_contest_dated_cs4a.py diff has 0 deletions and only the 5-line bounded_read spy for saver=='used':
    cs4a_diff = subprocess.run(
        ["git", "diff", "-U0", "a4e8278d", "3b7d5a18", "--", "tests/test_contest_dated_cs4a.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, check=True,
    ).stdout
    deleted = [ln for ln in cs4a_diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in cs4a_diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    assert deleted == [], deleted
    assert added == [
        "    if saver=='used':",
        "        bounded_read=Mock(wraps=flags._read_saver_state)",
        "        monkeypatch.setattr(flags,'_read_saver_state',bounded_read)",
        "    if saver=='used':",
        "        bounded_read.assert_called_once_with(tmp_path,season=2027,historical=True)",
    ], added

    # Verify docs/ops/2027-season-start.md diff changes ONLY row B8a with exact Amendment 2 item 3 strings:
    base_ops = subprocess.run(
        ["git", "show", "a4e8278d:docs/ops/2027-season-start.md"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, check=True,
    ).stdout
    cand_ops = (REPO_ROOT / "docs/ops/2027-season-start.md").read_text(encoding="utf-8")
    add_a = (
        " Coordinate saver corrections with anyone using the dashboard. "
        "Immediately before writing, refresh MLB and the scheduler's flag. "
        "If either changed since your verification, stop and reconcile the current state before retrying."
    )
    add_b = (
        "<li>**On the end-of-day warning that begins `Streak Saver on D ET. The dated observation says "
        "the saver was used.`** (change set F2): D is the ET date the warning prints. Run "
        "`bts season-contract status --picks-dir DIR` (DIR as above) and compare the dated row's saver "
        "and the saver flag with MLB. If MLB shows this season's saver used, follow **The saver** in the "
        "bullet above. If MLB shows it not used, the dated observation was the mistaken side and the flag "
        "is right: leave the flag as it is (a dated observation applies only on its own date).</li>"
    )
    reconstructed = base_ops.replace(
        "if initialization races, reread the flag.",
        "if initialization races, reread the flag." + add_a,
        1,
    ).replace(
        "check the scheduler host's clock.</li></ul> Do not run `bts schedule --dry-run`",
        "check the scheduler host's clock.</li>" + add_b + "</ul> Do not run `bts schedule --dry-run`",
        1,
    )
    assert cand_ops == reconstructed, "docs/ops/2027-season-start.md differs from exact Amendment 2 item 3 patch!"

    return {"r1_files": r1_files, "r2_files": r2_files, "cs4a_added_lines": len(added), "ops_exact_match": True}


def run_side_worker(work_root: Path, out_json: Path) -> None:
    import bts
    from bts import cli, contest_state, dm, orchestrator, saver_state, scheduler, season_contract, strategy
    from bts.health import alert, contest_state as health, runner
    from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

    print(f"bts.__file__ = {bts.__file__}", flush=True)

    def make_clock(fixed_now: dt.datetime):
        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
        return Clock

    def put_json(p: Path, obj: object) -> Path:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
        return p

    def write_dated(picks: Path, when: dt.datetime, saver: str, streak: int = 12) -> Path:
        clk = make_clock(when)
        with patch.object(dt, "datetime", clk), patch.object(cli, "datetime", clk):
            res = CliRunner().invoke(cli.cli, [
                "set-contest-streak", "--date", str(when.date()),
                "--streak", str(streak), "--best-streak", str(max(streak, 16)),
                "--saver", saver, "--picks-dir", str(picks),
            ])
        assert res.exit_code == 0, res.output
        return picks / "account_state" / "contest_streak.manual.json"

    def write_auto(picks: Path, when: dt.datetime, coverage: dt.date | None = None) -> None:
        acct = picks / "account_state"
        put_json(acct / "season_contract.json", {
            "schema_version": "bts_season_contract_v1",
            "season": when.year,
            "status": "automatic",
            "activation_id": ACTIVATION,
            "switched_at": (when - dt.timedelta(hours=1)).isoformat(),
            "revoked_at": None,
            "note": None,
        })
        put_json(acct / "contest_streak.json", {
            "schema_version": season_contract.AUTO_SCHEMA,
            "season": when.year,
            "activation_id": ACTIVATION,
            "active_streak": 12,
            "best_streak": 16,
            "source": "mlb_bts_profile",
            "source_date": str(coverage) if coverage is not None else None,
            "recorded_at": when.isoformat(),
        })

    def arrange_scenario(picks: Path, scenario: str) -> None:
        if scenario == "missing":
            return
        cov = DAY.date()
        if scenario == "coverage_lag_1":
            cov = DAY.date() - dt.timedelta(days=1)
        elif scenario == "coverage_lag_2":
            cov = DAY.date() - dt.timedelta(days=2)
        elif scenario == "future_source_date":
            cov = DAY.date() + dt.timedelta(days=4)
        elif scenario == "null_source_date":
            cov = None
        write_auto(picks, DAY, coverage=cov)

        if scenario == "dated_active":
            write_dated(picks, DAY, "active", 12)
        elif scenario == "dated_not_earned_9":
            write_dated(picks, DAY, "not_earned", 9)
        elif scenario == "dated_not_earned_12":
            write_dated(picks, DAY, "not_earned", 12)
        elif scenario == "dated_not_earned_16":
            write_dated(picks, DAY, "not_earned", 16)
        elif scenario == "expired_dated_used":
            write_dated(picks, DAY - dt.timedelta(days=1), "used", 12)
        elif scenario == "future_dated_used":
            write_dated(picks, DAY + dt.timedelta(days=1), "used", 12)
        elif scenario == "invalid_v3":
            p = write_dated(picks, DAY, "used", 12)
            bad = json.loads(p.read_text())
            bad["saver_state"] = "bogus"
            p.write_text(json.dumps(bad))
        elif scenario in ("legacy_active", "legacy_expired"):
            exp = DAY + dt.timedelta(hours=2 if scenario == "legacy_active" else -2)
            put_json(picks / "account_state" / "contest_streak.manual.json", {
                "active_streak": 8, "best_streak": 12, "source_date": str(DAY.date()),
                "override_expires_at": exp.isoformat(), "recorded_at": DAY.isoformat(),
                "source": "manual_cli",
            })
        elif scenario == "coverage_lag_1":
            put_json(picks / f"{DAY.date()}.json", {"result": "hit"})
        elif scenario == "coverage_lag_2":
            put_json(picks / f"{DAY.date() - dt.timedelta(days=1)}.json", {"result": "hit"})
            put_json(picks / f"{DAY.date()}.json", {"result": "hit"})
        elif scenario == "refusal_absent_switch":
            (picks / "account_state" / "season_contract.json").unlink()

    def arrange_flag(picks: Path, flag_kind: str) -> None:
        acct = picks / "account_state"
        acct.mkdir(parents=True, exist_ok=True)
        target = acct / "saver_state.json"
        if flag_kind == "absent":
            return
        elif flag_kind in ("active", "not_earned", "used"):
            put_json(target, {"season": DAY.year, "state": flag_kind})
        elif flag_kind == "malformed":
            target.write_text("{bad", encoding="utf-8")
        elif flag_kind == "enametoolong_symlink":
            os.symlink("z" * 300, target)

    def read_health_files(picks: Path) -> dict[str, str]:
        hs = picks.parent / "health_state"
        if not hs.exists():
            return {}
        return {
            str(p.relative_to(hs)): p.read_text(encoding="utf-8")
            for p in sorted(hs.rglob("*")) if p.is_file()
        }

    scenarios = (
        "missing", "automatic", "dated_active", "dated_not_earned_9",
        "dated_not_earned_12", "dated_not_earned_16", "expired_dated_used",
        "future_dated_used", "legacy_active", "legacy_expired", "invalid_v3",
        "future_source_date", "null_source_date", "coverage_lag_1",
        "coverage_lag_2", "refusal_absent_switch",
    )
    flag_kinds = ("active", "not_earned", "used", "absent", "malformed", "enametoolong_symlink")
    results = {}

    orig_safe_run = runner._safe_run
    for scenario in scenarios:
        for flag_kind in flag_kinds:
            for historical, wall in ((False, DAY), (True, NEXT)):
                for recipient in ("ops.bsky.social", None):
                    key = f"{scenario}|{flag_kind}|hist={historical}|recip={recipient}"
                    picks = work_root / key.replace("|", "__") / "picks"
                    picks.mkdir(parents=True, exist_ok=True)
                    arrange_scenario(picks, scenario)
                    arrange_flag(picks, flag_kind)

                    sends = []
                    clk = make_clock(wall)
                    existing = [
                        alert.Alert("INFO", "parity_probe", "pre-check info"),
                        alert.Alert("CRITICAL", "parity_probe", "pre-check critical"),
                    ]

                    def run_check(name, fn):
                        if name == "contest_state":
                            return orig_safe_run(name, fn)
                        return list(existing) if name == "calibration" else []

                    with patch.object(dt, "datetime", clk), \
                         patch.object(contest_state, "datetime", clk), \
                         patch.object(saver_state, "datetime", clk), \
                         patch.object(health, "datetime", clk), \
                         patch.object(runner, "datetime", clk), \
                         patch.object(alert, "datetime", clk), \
                         patch.object(runner, "_safe_run", run_check), \
                         patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None), \
                         patch.object(alert, "send_dm", side_effect=lambda r, b: sends.append([r, b]) or "ack-id"):
                        raw = runner.run_all_checks(
                            picks, picks.parent / "models", recipient,
                            today=DAY.date(), contest_state_expected=True,
                        )
                    files = read_health_files(picks)
                    assert not any(k.endswith(".dated_used_mismatch.json") for k in files), files
                    results[key] = {
                        "alerts": [asdict(a) for a in raw],
                        "sends": sends,
                        "files": files,
                    }

    # Also test expected=False across key scenarios:
    for scenario in ("missing", "automatic", "dated_active", "expired_dated_used"):
        for historical, wall in ((False, DAY), (True, NEXT)):
            key = f"expected_false|{scenario}|hist={historical}"
            picks = work_root / key.replace("|", "__") / "picks"
            picks.mkdir(parents=True, exist_ok=True)
            arrange_scenario(picks, scenario)
            arrange_flag(picks, "active")
            sends = []
            clk = make_clock(wall)
            with patch.object(dt, "datetime", clk), \
                 patch.object(contest_state, "datetime", clk), \
                 patch.object(saver_state, "datetime", clk), \
                 patch.object(health, "datetime", clk), \
                 patch.object(runner, "datetime", clk), \
                 patch.object(alert, "datetime", clk), \
                 patch.object(runner, "_safe_run", lambda n, fn: orig_safe_run(n, fn) if n == "contest_state" else []), \
                 patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None), \
                 patch.object(alert, "send_dm", side_effect=lambda r, b: sends.append([r, b]) or "ack-id"):
                raw = runner.run_all_checks(
                    picks, picks.parent / "models", "ops.bsky.social",
                    today=DAY.date(), contest_state_expected=False,
                )
            results[key] = {
                "alerts": [asdict(a) for a in raw],
                "sends": sends,
                "files": read_health_files(picks),
            }

    # Also 8 real scheduler.run_day runs across enabled={True,False}, historical={False,True}, recipient={set,None}:
    for historical, wall in ((False, DAY), (True, NEXT)):
        for enabled in (False, True):
            for recipient in ("ops.bsky.social", None):
                key = f"scheduler|hist={historical}|enabled={enabled}|recip={recipient}"
                sub = work_root / key.replace("|", "__")
                with MonkeyPatch.context() as mp:
                    config, picks, _, _ = world.__wrapped__(sub, mp)
                    prepare_run_day(mp)
                    mp.setattr(strategy, "_mdp_cache", None)
                    mp.setattr(scheduler, "notify_ready", Mock())
                    mp.setattr(scheduler, "notify_watchdog", Mock())
                    mp.setattr(runner, "get_self_pid", lambda: None)
                    mp.setattr(runner, "read_systemd_nrestarts", lambda: None)
                    mp.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
                    config["health_checks"]["enabled"] = enabled
                    config["bluesky"]["dm_recipient"] = recipient
                    orchestrator.run_cascade.return_value = (None, None)
                    write_dated(picks, DAY, "not_earned", 12)
                    arrange_flag(picks, "active")

                    captured_raw = []
                    real_run_all = runner.run_all_checks

                    def spy_run_all(*a, **kw):
                        out = real_run_all(*a, **kw)
                        captured_raw.extend(asdict(x) for x in out)
                        return out

                    mp.setattr(runner, "run_all_checks", spy_run_all)
                    clk = make_clock(wall)
                    with patch.object(dt, "datetime", clk), \
                         patch.object(cli, "datetime", clk), \
                         patch.object(contest_state, "datetime", clk), \
                         patch.object(saver_state, "datetime", clk), \
                         patch.object(health, "datetime", clk), \
                         patch.object(runner, "datetime", clk), \
                         patch.object(alert, "datetime", clk):
                        scheduler.run_day(str(DAY.date()), config)

                    results[key] = {
                        "raw": captured_raw,
                        "health_sends": [list(c.args) for c in alert.send_dm.call_args_list],
                        "pick_sends": [list(c.args) for c in dm.send_dm.call_args_list],
                        "files": read_health_files(picks),
                    }

    out_json.write_text(json.dumps(results, sort_keys=True, indent=2), encoding="utf-8")


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_side_worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return 0

    scope = check_git_scope()
    print("Git scope verified:", json.dumps(scope))

    with TemporaryDirectory(prefix="f2_item1_") as tmp:
        tmp_dir = Path(tmp)
        for label, src_path in (("base", BASE_SRC), ("candidate", CAND_SRC)):
            shim = tmp_dir / f"{label}_shim"
            shim.mkdir()
            (shim / "sitecustomize.py").write_text(
                f"import sys\n"
                f"from pathlib import Path\n"
                f"cand = Path({str(CAND_SRC)!r}).resolve()\n"
                f"sys.path = [p for p in sys.path if Path(p or '.').resolve() != cand]\n"
                f"sys.meta_path = [f for f in sys.meta_path if 'editable' not in type(f).__name__.lower()]\n"
                f"sys.path.insert(0, {str(src_path.resolve())!r})\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PYTHONPATH": f"{shim}:{src_path.resolve()}:{REPO_ROOT}",
                "TZ": "America/New_York",
            }
            work_sub = tmp_dir / "shared_work"
            out_file = tmp_dir / f"{label}.json"
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", str(work_sub), str(out_file)],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=45, check=True,
            )
            print(f"[{label}] {proc.stdout.strip()}")
            work_sub.rename(tmp_dir / f"done_{label}")

        base_data = json.loads((tmp_dir / "base.json").read_text(encoding="utf-8"))
        cand_data = json.loads((tmp_dir / "candidate.json").read_text(encoding="utf-8"))
        assert base_data.keys() == cand_data.keys()
        diffs = [k for k in base_data if base_data[k] != cand_data[k]]
        for k in diffs[:5]:
            print("DIFF:", k, "\n  base:", base_data[k], "\n  cand:", cand_data[k])
        assert not diffs, f"{len(diffs)} parity differences found!"
        print(f"PASS: {len(base_data)} paired base-vs-candidate cases (including 8 scheduler.run_day runs) matched byte-for-byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

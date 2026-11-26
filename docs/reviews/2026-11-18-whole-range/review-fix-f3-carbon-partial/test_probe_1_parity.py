"""Item 1 regression probe: diff scope and byte-for-byte parity with base 5e0167b5 on all non-deliberate_skip paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_BASE = Path(
    "/Users/eric/.gemini/antigravity-cli/brain/874253bb-d296-4a20-9309-5c6b286c2052/scratch/base_5e0167b5"
)


def _run_worker(mode: str, work_dir: Path) -> None:
    import bts
    from click.testing import CliRunner
    from bts import cli, daily_decision, day_outcome, dm, entry_receipt, scheduler

    day = "2027-04-02"
    sha = "d" * 64
    pick_1 = {"batter_id": 1, "batter_name": "Kwan", "team": "CLE", "game_pk": 900, "p_game_hit": 0.78}
    pick_2 = {"batter_id": 2, "batter_name": "Ramirez", "team": "CLE", "game_pk": 901, "p_game_hit": 0.75}

    def mk_dec(action="single", source="mdp", *, scoreable=True, delivery="delivered", double=None, extra=None):
        rec = {
            "schema_version": "bts_daily_decision_v3",
            "date": day,
            "action": action,
            "source": source,
            "primary": pick_1 if action != "skip" else None,
            "double_down": double,
            "streak": 30,
            "saver_available": False,
            "state_source": "contest",
            "delivery_status": delivery,
            "scoreable": scoreable,
        }
        if extra:
            rec.update(extra)
        return rec

    def mk_rec(outcome="observed", *, confirmed=True, before_cutoff=True, reason="match",
               rec_sha=sha, username="stonehengee", et_date=day, degraded=(), slots=None):
        rec = {
            "schema": "bts_pick_entry_receipt_v1",
            "outcome": outcome,
            "et_date": et_date,
            "season": int(et_date[:4]),
            "selection": {"commit": {"decision_sha256": rec_sha}},
            "degraded": list(degraded),
        }
        if outcome == "observed":
            qual = {"all_confirmed": confirmed}
            if slots is not None:
                qual["slots"] = slots
            rec.update({
                "account": {"username": username},
                "observation": {
                    "before_cutoff": before_cutoff,
                    "response_completed_at": f"{day}T17:31:00-04:00",
                },
                "verifier": {"reason": reason},
                "qualification": qual,
            })
        return rec

    skip_mdp = mk_dec("skip", "mdp", scoreable=False, delivery="not_applicable")
    cases = [
        ("entered_single", mk_dec(double=pick_2), [{}], "enter", "dm", [mk_rec()]),
        ("entered_double", mk_dec("double", double=pick_2), [{}], "enter", "dm", [mk_rec()]),
        ("entered_game_unverified", mk_dec(), [{}], "enter", "dm",
         [mk_rec(confirmed=False, slots=[{"state": "unit_unverified"}])]),
        ("entered_late", mk_dec(), [{}], "enter", "dm", [mk_rec(before_cutoff=False)]),
        ("not_entered_no_pick", mk_dec(), [{}], "enter", "dm", [mk_rec(confirmed=False, reason="no_pick")]),
        ("not_entered_mismatch", mk_dec(), [{}], "enter", "dm", [mk_rec(confirmed=False, reason="mismatch")]),
        ("entry_unverifiable_empty", mk_dec(), [{}], "enter", "dm", []),
        ("entry_unverifiable_failed", mk_dec(), [{}], "enter", "dm",
         [mk_rec("fetch_failed"), mk_rec("outside_window")]),
        ("entry_unverifiable_unbound", mk_dec(), [{}], "enter", "dm", [mk_rec(rec_sha="e" * 64)]),
        ("no_decision_absent", None, [{}, {}], "enter", "dm", []),
        ("no_decision_bad_json", "{broken", [{}], "enter", "dm", []),
        ("no_decision_scoreable_int", mk_dec("skip", scoreable=0), [{}], "enter", "dm", []),
        ("no_games_zero_with_skip", skip_mdp, [], "enter", "dm", []),
        ("no_games_zero_no_dec", None, [], "enter", "dm", []),
        ("no_games_all_void_skip", skip_mdp,
         [{"status": {"detailedState": "Postponed"}}, {"status": {"detailedState": "Cancelled"}}],
         "enter", "dm", []),
        ("no_games_all_void_commit", mk_dec(),
         [{"status": {"detailedState": "Canceled"}}], "enter", "dm", []),
        ("contradiction_single", mk_dec(), [], "enter", "dm", []),
        ("contradiction_double", mk_dec("double", double=pick_2), [], "enter", "dm", []),
        ("commit_not_scoreable_single", mk_dec(scoreable=False, delivery="failed"), [{}], "enter", "dm", []),
        ("commit_not_scoreable_double", mk_dec("double", scoreable=False, delivery="failed", double=pick_2),
         [{}], "enter", "dm", []),
        ("research_commit", mk_dec(delivery="private_locked"), [{}], "research", "private", []),
        ("delivered_under_research", mk_dec(delivery="delivered"), [{}], "research", "private", []),
        ("intent_unknown_missing", skip_mdp, [{}], None, "dm", []),
        ("intent_unknown_invalid", skip_mdp, [{}], "invalid_intent", "dm", []),
        ("intent_unknown_mismatch", skip_mdp, [{}], "enter", "private", []),
        ("schedule_unverifiable", skip_mdp, OSError("mlb offline"), "enter", "dm", []),
        ("skip_not_deliberate_heuristic", mk_dec("skip", "heuristic", scoreable=False), [{}], "enter", "dm", []),
        ("skip_not_deliberate_none", mk_dec("skip", None, scoreable=False), [{}], "enter", "dm", []),
    ]

    results = {}
    runner = CliRunner()
    for name, dec, sched, intent, delivery, receipts in cases:
        case_dir = work_dir / name
        picks_dir = case_dir / "picks"
        account = picks_dir / "account_state"
        account.mkdir(parents=True)
        (account / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3",
            "season": 2027,
            "effective_date": day,
            "expires_at": f"{day}T23:59:59.999999-04:00",
            "active_streak": 1,
            "best_streak": 30,
            "saver_state": "not_earned",
            "recorded_at": f"{day}T20:30:00Z",
            "source": "manual_cli",
        }))
        (account / "contest_streak.manual.revoked-20270402T203100000000Z.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3",
            "season": 2027,
            "effective_date": day,
            "expires_at": f"{day}T23:59:59.999999-04:00",
            "active_streak": 2,
            "best_streak": 30,
            "saver_state": "active",
            "recorded_at": f"{day}T20:31:00Z",
            "source": "manual_cli",
        }))
        (account / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "revoked",
            "activation_id": "a" * 32,
            "switched_at": None,
            "revoked_at": f"{day}T20:35:00Z",
            "note": "revoked",
        }))
        (account / "contest_manual_requests.jsonl").write_text(
            json.dumps({"effective_date": day, "active_streak": 3, "saver_state": "active"}) + "\n"
        )

        dec_path = daily_decision.decision_path(day, picks_dir)
        dec_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(dec, str):
            dec_path.write_text(dec)
        elif isinstance(dec, dict):
            dec_path.write_text(json.dumps(dec))

        cfg_path = case_dir / "config.toml"
        cfg_lines = ["[scheduler]", f'pick_delivery = "{delivery}"']
        config_dict: dict = {"scheduler": {"pick_delivery": delivery}}
        if intent is not None:
            cfg_lines.append(f'entry_intent = "{intent}"')
            config_dict["scheduler"]["entry_intent"] = intent
        cfg_path.write_text("\n".join(cfg_lines) + "\n")

        forbidden = {
            str(account / "contest_streak.manual.json"),
            str(account / "contest_streak.manual.revoked-20270402T203100000000Z.json"),
            str(account / "season_contract.json"),
            str(account / "contest_manual_requests.jsonl"),
            str(account),
        }
        real_open = os.open
        real_scandir = os.scandir
        touched: list[str] = []

        def guarded_open(p, *a, **kw):
            s = os.fspath(p)
            if s in forbidden:
                touched.append(f"open:{s}")
            return real_open(p, *a, **kw)

        def guarded_scandir(p=None):
            s = os.fspath(p) if p is not None else ""
            if s in forbidden:
                touched.append(f"scandir:{s}")
            return real_scandir(p)

        def fake_schedule(_d):
            if isinstance(sched, Exception):
                raise sched
            return list(sched)

        sends: list[tuple[str, str]] = []

        def fake_send(who, msg):
            sends.append((who, msg))
            return "ok"

        before_files = {
            str(p.relative_to(picks_dir)): p.read_bytes()
            for p in sorted(picks_dir.rglob("*"))
            if stat.S_ISREG(p.lstat().st_mode)
        }

        with patch.object(os, "open", side_effect=guarded_open), \
             patch.object(os, "scandir", side_effect=guarded_scandir), \
             patch.object(scheduler, "fetch_schedule", side_effect=fake_schedule), \
             patch.object(entry_receipt, "_sha", side_effect=lambda raw: sha if raw is not None else None), \
             patch.object(entry_receipt, "discover", return_value=receipts), \
             patch.object(dm, "send_dm", side_effect=fake_send):
            g_out = day_outcome.gather(day, picks_dir, config_dict, expected_username="stonehengee")
            sends.clear()
            cli_res = runner.invoke(cli.cli, [
                "day-outcome", "--date", day, "--config", str(cfg_path),
                "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
                "--dm-recipient", "operator.bsky.social",
            ])

        after_files = {
            str(p.relative_to(picks_dir)): p.read_bytes()
            for p in sorted(picks_dir.rglob("*"))
            if stat.S_ISREG(p.lstat().st_mode)
        }
        assert before_files == after_files, f"{mode} mutated files in {name}"
        assert touched == [], f"{mode} touched evidence in {name}: {touched}"
        assert g_out.state != "deliberate_skip", f"unexpected deliberate_skip in {name}"

        results[name] = {
            "gather_dict": g_out.as_dict(),
            "gather_ok": g_out.ok,
            "gather_msg": g_out.message(),
            "cli_exit": cli_res.exit_code,
            "cli_stdout": cli_res.stdout,
            "cli_stderr": cli_res.stderr,
            "sends": sends,
        }

    for err_name, toml_text in [("exit2_bad_toml", "[broken"), ("exit2_missing_cfg", None)]:
        cdir = work_dir / err_name
        pdir = cdir / "picks"
        pdir.mkdir(parents=True)
        cpath = cdir / "config.toml"
        if toml_text is not None:
            cpath.write_text(toml_text)
        sends = []
        with patch.object(dm, "send_dm", side_effect=lambda w, m: sends.append((w, m)) or "ok"):
            cli_res = runner.invoke(cli.cli, [
                "day-outcome", "--date", day, "--config", str(cpath),
                "--picks-dir", str(pdir), "--dm-recipient", "operator.bsky.social",
            ])
        results[err_name] = {
            "cli_exit": cli_res.exit_code,
            "cli_stdout": cli_res.stdout,
            "cli_stderr": cli_res.stderr,
            "sends": sends,
        }

    print(json.dumps({
        "mode": mode,
        "bts_file": str(Path(bts.__file__).resolve()),
        "day_outcome_file": str(Path(day_outcome.__file__).resolve()),
        "results": results,
    }))


def test_item1_diff_and_base_parity(tmp_path: Path) -> None:
    show = subprocess.run(
        ["git", "show", "--name-status", "--format=", "dee14603"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=15,
    )
    dee_files = [line.split("\t", 1) for line in show.stdout.strip().splitlines() if line.strip()]
    expected_dee = [
        ["M", "CLAUDE.md"],
        ["M", "src/bts/day_outcome.py"],
        ["A", "tests/test_wr_f3_skip_state_review.py"],
    ]
    assert dee_files == expected_dee

    diff_range = subprocess.run(
        ["git", "diff", "--name-status", "5e0167b5", "dee14603"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=15,
    )
    non_review = [
        line.split("\t", 1) for line in diff_range.stdout.strip().splitlines()
        if not line.split("\t", 1)[1].startswith("docs/reviews/2026-11-18-whole-range/")
    ]
    assert non_review == expected_dee

    base_root = tmp_path / "base_archive_src"
    base_root.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "5e0167b5", "src"],
        cwd=REPO_ROOT, capture_output=True, check=True, timeout=20,
    )
    subprocess.run(["tar", "-x", "-C", str(base_root)], input=archive.stdout, check=True, timeout=20)

    py = str(REPO_ROOT / ".venv/bin/python")
    shared_work = tmp_path / "shared_run_dir"
    shared_work.mkdir()
    env_base = dict(os.environ, PYTHONPATH=str(base_root / "src"), TZ="America/New_York")
    env_head = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), TZ="America/New_York")

    res_base = subprocess.run(
        [py, str(Path(__file__).resolve()), "--worker", "base", str(shared_work)],
        cwd=REPO_ROOT, env=env_base, capture_output=True, text=True, check=True, timeout=45,
    )
    shutil.rmtree(shared_work)
    shared_work.mkdir()
    res_head = subprocess.run(
        [py, str(Path(__file__).resolve()), "--worker", "head", str(shared_work)],
        cwd=REPO_ROOT, env=env_head, capture_output=True, text=True, check=True, timeout=45,
    )

    base_data = json.loads(res_base.stdout)
    head_data = json.loads(res_head.stdout)
    print(f"base bts.__file__:        {base_data['bts_file']}")
    print(f"base day_outcome.__file__: {base_data['day_outcome_file']}")
    print(f"head bts.__file__:        {head_data['bts_file']}")
    print(f"head day_outcome.__file__: {head_data['day_outcome_file']}")
    assert str(base_root / "src") in base_data["bts_file"]
    assert str(REPO_ROOT / "src") in head_data["bts_file"]
    assert base_data["bts_file"] != head_data["bts_file"]
    assert base_data["results"] == head_data["results"]
    assert len(base_data["results"]) == 30


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], Path(sys.argv[3]))
    else:
        raise SystemExit(pytest.main([__file__, "-s", "-v"]))

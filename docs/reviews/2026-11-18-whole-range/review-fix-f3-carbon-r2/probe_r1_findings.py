"""Check round-1 findings B1, E1, E2, bytes-read scratch check, and round-1/2 mutations."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
SCRATCH_ROOT = Path("/tmp/wr-f3-r2-scratch").resolve()
BASE_SRC = SCRATCH_ROOT / "base" / "src"


def snapshot_bytes(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        if stat.S_ISREG(st.st_mode):
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def run_b1_child(mode: str) -> None:
    import bts
    from click.testing import CliRunner
    from pytest import MonkeyPatch
    from bts import cli, contest_state, daily_decision, day_outcome, orchestrator, scheduler
    from tests.test_contest_fetch import _mock_fetch

    src_expected = BASE_SRC if mode == "base" else (REPO / "src")
    assert Path(bts.__file__).resolve().is_relative_to(src_expected.resolve()), bts.__file__
    harness_path = REPO / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3/test_slice3_interactions.py"
    spec = importlib.util.spec_from_file_location("b1_harness", harness_path)
    carbon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(carbon)
    carbon.REPO_ROOT = REPO

    start = dt.datetime(2027, 3, 24, 10, tzinfo=ET)
    day = "2027-03-24"
    results = {}

    for scenario in ("K2e_b1_sequence", "K2c_reenable_no_archive"):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with carbon._harness(root, start) as env:
                clock, picks_dir = env["clock"], env["picks_dir"]
                account = picks_dir / "account_state"
                manual = account / "contest_streak.manual.json"
                runner = CliRunner()
                env["confirmed"][101] = {"home", "away"}

                def invoke(*args):
                    with patch("datetime.datetime", carbon._make_Clock(clock)):
                        res = runner.invoke(cli.cli, [*args, "--picks-dir", str(picks_dir)])
                    assert res.exit_code == 0, (args, res.exit_code, res.output, res.exception)
                    return res

                carbon._write_auto_obs(
                    picks_dir, start - dt.timedelta(days=1), 30,
                    source_date=day, saver_flag="used", recorded_at=start.isoformat(),
                )
                clock[0] = start.replace(hour=9, minute=0)
                if scenario == "K2e_b1_sequence":
                    invoke("set-contest-streak", "--date", day, "--streak", "1",
                           "--best-streak", "30", "--saver", "not_earned")
                clock[0] = start.replace(hour=9, minute=1)
                invoke("season-contract", "revoke", "--season", "2027", "--reason", "pre-check revoke")
                clock[0] = start.replace(hour=9, minute=2)
                invoke("season-contract", "enable", "--season", "2027")
                with MonkeyPatch.context() as mp:
                    _mock_fetch(
                        mp,
                        {"activeStreak": 30, "seasonBestStreak": 30,
                         "predictions": [{"roundId": 1, "result": "hit"}]},
                        {1: start.date()},
                    )
                    invoke("fetch-contest-streak", "--expected-username", "stonehengee")

                assert not manual.exists()
                archives = list(account.glob("contest_streak.manual.revoked-*.json"))
                assert len(archives) == (1 if scenario == "K2e_b1_sequence" else 0)

                clock[0] = start
                checks = []
                seen_states = []
                real_load = contest_state.load_decision_streak_state

                def record_load(*args, **kwargs):
                    st = real_load(*args, **kwargs)
                    seen_states.append((clock[0].strftime("%H:%M"), st.observation_kind, st.streak, st.saver_available))
                    return st

                def cascade(_tiers, slate_day):
                    checks.append(clock[0].strftime("%H:%M"))
                    if len(checks) == 1:
                        return carbon._pred_frame(slate_day, p1=0.72, p2=None), "local"
                    return None, None

                def sleep(secs, *_a, **_kw):
                    env["advance"](max(0, secs))

                with patch.object(orchestrator, "run_cascade", side_effect=cascade), \
                     patch.object(contest_state, "load_decision_streak_state", side_effect=record_load), \
                     patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep):
                    scheduler.run_day(day, env["config"])

                dec = daily_decision.load_decision(day, picks_dir)
                assert (dec["action"], dec["source"], dec["streak"], dec["saver_available"]) == (
                    "skip", "mdp", 30, False
                )
                assert ("16:10", "automatic", 30, False) in seen_states
                assert json.loads((account / "contest_streak.json").read_text())["active_streak"] == 30

                cfg = root / "morning.toml"
                cfg.write_text('[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
                config_dict = {"scheduler": {"pick_delivery": "dm", "entry_intent": "enter"}}
                clock[0] = dt.datetime(2027, 3, 25, 8, 30, tzinfo=ET)
                before = snapshot_bytes(picks_dir)
                n_before = len(env["pick_dms"])
                gathered = day_outcome.gather(day, picks_dir, config_dict, expected_username="operator")
                with patch("datetime.datetime", carbon._make_Clock(clock)):
                    cli_res = runner.invoke(
                        cli.cli,
                        ["day-outcome", "--date", day, "--config", str(cfg),
                         "--picks-dir", str(picks_dir), "--expected-username", "operator",
                         "--dm-recipient", "operator.bsky.social"],
                    )
                dms = env["pick_dms"][n_before:]
                assert snapshot_bytes(picks_dir) == before
                results[scenario] = {
                    "bts_file": bts.__file__,
                    "exit": cli_res.exit_code,
                    "stdout": json.loads(cli_res.stdout),
                    "stderr": cli_res.stderr,
                    "gathered": gathered.as_dict(),
                    "dms": dms,
                }
    print(json.dumps(results))


def check_b1() -> None:
    print("=== 1A. B1 (K2e sequence and K2c consistency) ===", flush=True)
    outputs = {}
    for mode, py_path in [("base", f"{BASE_SRC}:{REPO}"), ("f3", f"{REPO / 'src'}:{REPO}")]:
        env = os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": py_path}
        proc = subprocess.run(
            [sys.executable, __file__, f"--b1-child={mode}"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=30, check=True,
        )
        outputs[mode] = json.loads(proc.stdout)

    base_k2e = outputs["base"]["K2e_b1_sequence"]
    f3_k2e = outputs["f3"]["K2e_b1_sequence"]
    f3_k2c = outputs["f3"]["K2c_reenable_no_archive"]

    assert base_k2e["exit"] == 0
    assert base_k2e["stdout"]["state"] == "deliberate_skip"
    assert base_k2e["dms"] == []

    expected_detail = (
        "The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. "
        "An archived dated observation for 2027-03-24 records streak 1 and policy saver availability available. "
        "Check MLB for unsettled entries and verify the current streak, season best and saver."
    )
    assert f3_k2e["exit"] == 1
    assert f3_k2e["stdout"] == {
        "date": "2027-03-24",
        "state": "skip_state_review",
        "ok": False,
        "detail": expected_detail,
        "pick": None,
    }
    assert f3_k2e["gathered"] == f3_k2e["stdout"]
    assert "revocation" not in f3_k2e["stdout"]["detail"]
    assert f3_k2e["dms"] == [[
        "operator.bsky.social",
        f"BTS day check 2027-03-24: SKIP STATE REVIEW. {expected_detail}",
    ]]

    assert f3_k2c["exit"] == 0
    assert f3_k2c["stdout"]["state"] == "deliberate_skip"
    assert f3_k2c["dms"] == []

    kickoff = (REPO / "docs/reviews/2026-11-18-whole-range/kickoff-fix-f3.md").read_text()
    assert ("A pre-check enable and successful fetch leave no revocation reason. A differing retained dated "
            "archive can still prompt review, including one set aside before the skip.") in kickoff
    assert "New table row **K2e**" in kickoff
    print(f"base bts.__file__={base_k2e['bts_file']}")
    print(f"f3   bts.__file__={f3_k2e['bts_file']}")
    print("B1/K2e outcome:", json.dumps(f3_k2e["stdout"]))
    print("K2c re-enable without archive:", json.dumps(f3_k2c["stdout"]))


def check_e1() -> None:
    print("=== 1B. E1 (ET overflow on dated timestamps and mixed archives) ===", flush=True)
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, day_outcome, dm, scheduler
    from tests.test_day_outcome import DATE, _decision

    skip_dec = _decision("skip", scoreable=False) | {"streak": 30, "saver_available": False}
    end_et = dt.datetime.combine(dt.date.fromisoformat(DATE), dt.time.max, ET).isoformat()

    def make_dated(streak: int, recorded_at: str, saver: str = "used") -> dict:
        return {
            "schema_version": contest_state.DATED_MANUAL_SCHEMA,
            "season": int(DATE[:4]),
            "effective_date": DATE,
            "expires_at": end_et,
            "active_streak": streak,
            "best_streak": max(streak, 30),
            "saver_state": saver,
            "recorded_at": recorded_at,
            "source": "manual_cli",
        }

    def make_switch(revoked_at: str = f"{DATE}T20:00:00Z") -> dict:
        return {
            "schema_version": "bts_season_contract_v1",
            "season": int(DATE[:4]),
            "status": "revoked",
            "activation_id": "a" * 32,
            "switched_at": None,
            "revoked_at": revoked_at,
        }

    def run_report(root: Path, files: dict[str, dict]) -> tuple[dict, int, list]:
        picks = root / "picks"
        account = picks / "account_state"
        account.mkdir(parents=True, exist_ok=True)
        dec_path = daily_decision.decision_path(DATE, picks)
        dec_path.parent.mkdir(parents=True, exist_ok=True)
        dec_path.write_text(json.dumps(skip_dec))
        for name, body in files.items():
            (account / name).write_text(json.dumps(body))
        cfg = root / "config.toml"
        cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        sends = []
        with patch.object(scheduler, "fetch_schedule", return_value=[{}]), \
             patch.object(dm, "send_dm", side_effect=lambda r, m: sends.append((r, m)) or "ok"):
            res = CliRunner().invoke(
                cli.cli,
                ["day-outcome", "--date", DATE, "--config", str(cfg),
                 "--picks-dir", str(picks), "--expected-username", "stonehengee",
                 "--dm-recipient", "operator.bsky.social"],
            )
            out = day_outcome.gather(
                DATE, picks, {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}},
                expected_username="stonehengee",
            )
        assert json.loads(res.stdout) == out.as_dict()
        return out.as_dict(), res.exit_code, sends

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        overflow_stamps = [
            "0001-01-01T00:00:00Z",
            "9999-12-31T23:59:59-12:00",
            "0001-01-01T04:56:01Z",       # 1 second before ET LMT MINYEAR
            "9999-12-31T19:00:00-10:00",  # exact 10000-01-01T00:00:00 EST overflow
        ]
        for idx, stamp in enumerate(overflow_stamps):
            for fname in ("contest_streak.manual.json", "contest_streak.manual.revoked-a.json"):
                body, code, sends = run_report(root / f"ov_{idx}_{fname}", {fname: make_dated(1, stamp)})
                assert code == 0 and body["state"] == "deliberate_skip" and sends == [], (stamp, fname, body)
                print(f"overflow discarded: {fname} @ {stamp} -> {body['state']} (exit {code})")

        body, code, sends = run_report(root / "mixed_no_revoke", {
            "contest_streak.manual.revoked-valid.json": make_dated(2, f"{DATE}T14:00:00Z"),
            "contest_streak.manual.revoked-overflow.json": make_dated(30, "9999-12-31T23:59:59-12:00"),
        })
        assert code == 1 and body["state"] == "skip_state_review" and len(sends) == 1
        assert "records streak 2" in body["detail"]
        print("mixed archive (no revoke):", json.dumps(body))

        body, code, sends = run_report(root / "mixed_with_revoke", {
            "contest_streak.manual.revoked-valid.json": make_dated(1, f"{DATE}T14:00:00Z"),
            "contest_streak.manual.revoked-ov1.json": make_dated(30, "0001-01-01T00:00:00Z"),
            "contest_streak.manual.revoked-ov2.json": make_dated(30, "9999-12-31T23:59:59-12:00"),
            "season_contract.json": make_switch(),
        })
        assert code == 1 and body["state"] == "skip_state_review" and len(sends) == 1
        assert "records streak 1" in body["detail"] and "records a revocation" in body["detail"]
        print("mixed archive (with revoke):", json.dumps(body))

        body_min, code_min, _ = run_report(root / "valid_min_et", {
            "contest_streak.manual.revoked-min.json": make_dated(4, "0001-01-01T04:56:02Z"),
        })
        assert code_min == 1 and "records streak 4" in body_min["detail"]
        body_max, code_max, _ = run_report(root / "valid_max_et", {
            "contest_streak.manual.revoked-min.json": make_dated(4, "0001-01-01T04:56:02Z"),
            "contest_streak.manual.revoked-max.json": make_dated(5, "9999-12-31T18:59:59-10:00"),
        })
        assert code_max == 1 and "records streak 5" in body_max["detail"]
        print("valid extreme ET boundaries ranked cleanly: streak 4 vs streak 5 -> streak 5")


def check_e2() -> None:
    print("=== 1C. E2 (exact documentation wording) ===", flush=True)
    from bts import day_outcome

    req_skip = (
        "A deliberate MDP skip can prompt `skip_state_review` if the selected dated observation for that day "
        "records a different streak or policy saver availability. The report uses a valid installed observation "
        "when present and otherwise checks retained archives. In enter mode it also prompts review if the switch "
        "remains revoked for that season, the revocation was on the skip date in ET, and no valid retained dated "
        "observation matches both inputs."
    )
    req_missing = (
        "Missing or unusable optional contest evidence can leave a deliberate skip clean. The base outcome checks "
        "report missing schedule, decision or entry evidence where required."
    )
    claude_line_66 = (REPO / "CLAUDE.md").read_text().splitlines()[65]
    doc_norm = " ".join((day_outcome.__doc__ or "").split())

    assert req_skip in claude_line_66, claude_line_66
    assert req_missing in claude_line_66, claude_line_66
    assert "Missing evidence reads as unverifiable, never as a pass." not in claude_line_66

    assert req_skip in doc_norm, doc_norm
    assert req_missing in doc_norm, doc_norm
    assert "Missing or failed evidence is reported as unverifiable" not in doc_norm
    print("CLAUDE.md:66 and day_outcome.__doc__ match Amendment 3 item 3 verbatim.")


def replace_once(text: str, before: str, after: str) -> str:
    assert text.count(before) == 1, f"expected 1 occurrence of {before!r}, got {text.count(before)}"
    return text.replace(before, after, 1)


MUTATIONS = {
    "control": [],
    "bound_after_read_eof": [
        (
            "            while len(raw) <= EVIDENCE_MAX_BYTES:\n"
            "                chunk = os.read(fd, EVIDENCE_MAX_BYTES + 1 - len(raw))\n"
            "                if not chunk:\n"
            "                    return json.loads(raw.decode(\"utf-8\"))\n"
            "                raw.extend(chunk)",
            "            while True:\n"
            "                chunk = os.read(fd, 65536)\n"
            "                if not chunk:\n"
            "                    return json.loads(raw.decode(\"utf-8\")) if len(raw) <= EVIDENCE_MAX_BYTES else None\n"
            "                raw.extend(chunk)",
        ),
    ],
    "read_regular_file_text_eof": [
        (
            "def _read_evidence(path: Path) -> object:",
            "def _read_evidence(path: Path) -> object:\n"
            "    from bts.util import read_regular_file_text\n"
            "    try:\n"
            "        txt = read_regular_file_text(path)\n"
            "        return json.loads(txt) if len(txt.encode('utf-8')) <= EVIDENCE_MAX_BYTES else None\n"
            "    except (OSError, ValueError, RecursionError):\n"
            "        return None\n"
            "def _unused_read_evidence(path: Path) -> object:",
        ),
    ],
    "drop_dated_overflow_handling": [
        (
            "    try:\n        recorded_at.astimezone(ET)\n    except OverflowError:\n        return None\n",
            "",
        ),
    ],
    "read_on_all_outcomes": [
        (
            "    outcome = classify_day(",
            "    _read_evidence(Path(picks_dir) / 'account_state' / 'contest_streak.manual.json')\n"
            "    outcome = classify_day(",
        ),
    ],
    "drop_archives": [("def _archives(account: Path):\n", "def _archives(account: Path):\n    return\n")],
    "drop_revoke_suppression": [("    if revoked is not None and not matching_dated:", "    if revoked is not None:")],
    "compare_streak_only": [
        ("current[1] == skip_inputs", "current[1][0] == streak"),
        ("matching_dated |= inputs == skip_inputs", "matching_dated |= inputs[0] == streak"),
        ("dated_inputs != skip_inputs", "dated_inputs[0] != streak"),
    ],
    "drop_byte_bound": [
        (
            "            while len(raw) <= EVIDENCE_MAX_BYTES:\n"
            "                chunk = os.read(fd, EVIDENCE_MAX_BYTES + 1 - len(raw))",
            "            while True:\n                chunk = os.read(fd, 4096)",
        ),
    ],
    "drop_switch_overflow_handling": [("except (InvalidContract, OverflowError):", "except InvalidContract:")],
    "drop_enter_condition": [
        ('_revocation_time(account, day) if intent == "enter" else None', "_revocation_time(account, day)"),
    ],
    "keep_deliberate_skip_text": [
        (
            'return DayOutcome(date=date, state="skip_state_review", detail=detail)',
            'return DayOutcome(date=date, state="skip_state_review", detail="The policy skipped the day.")',
        ),
    ],
}


def run_mutations(selected: list[str]) -> None:
    work = SCRATCH_ROOT / "mutations"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(REPO / "src", work / "src")
    target = work / "src/bts/day_outcome.py"
    source = (REPO / "src/bts/day_outcome.py").read_text()
    assert target.read_text() == source

    env = os.environ | {
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{work / 'src'}:{REPO}",
    }
    entry = (
        "import bts, pytest, sys; "
        "print('bts.__file__=' + bts.__file__, flush=True); "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    for name in selected:
        edits = MUTATIONS[name]
        mutated = source
        for before, after in edits:
            mutated = replace_once(mutated, before, after)
        try:
            target.write_text(mutated)
            res = subprocess.run(
                [sys.executable, "-c", entry, "-p", "no:cacheprovider", "-q",
                 "tests/test_wr_f3_skip_state_review.py"],
                cwd=REPO, env=env, capture_output=True, text=True, timeout=25, check=False,
            )
        finally:
            target.write_text(source)
        out = res.stdout + res.stderr
        first_line = out.splitlines()[0]
        assert Path(first_line.split("=", 1)[1]).resolve() == (work / "src/bts/__init__.py").resolve(), first_line
        assert "collected 92 items" in out, out[:500]
        if name == "control":
            assert res.returncode == 0, out[-500:]
        else:
            assert res.returncode == 1, (name, res.returncode, out[-500:])
        if name in {"bound_after_read_eof", "read_regular_file_text_eof"}:
            assert "test_evidence_read_byte_limit" in out, (name, out)
        summary = [ln for ln in out.splitlines() if "===" in ln and " in " in ln][-1]
        print(f"mutation {name}: exit {res.returncode} | {summary}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].startswith("--b1-child="):
        run_b1_child(sys.argv[1].split("=", 1)[1])
    elif len(sys.argv) >= 2 and sys.argv[1] == "b1_e1_e2":
        check_b1()
        check_e1()
        check_e2()
    elif len(sys.argv) >= 3 and sys.argv[1] == "mutations":
        run_mutations(sys.argv[2:])
    else:
        raise SystemExit(f"unknown args: {sys.argv[1:]}")

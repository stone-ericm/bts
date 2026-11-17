"""Clean verification probe for PEP-578 audit events + Path.lstat tracing across Items 3, 7, 8, and 9."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

AUDIT = {
    "active": False,
    "denied_net": [],
    "events": [],
}

_NET_EVENTS = frozenset({
    "socket.getaddrinfo",
    "socket.getnameinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.connect",
    "socket.connect_ex",
    "socket.sendto",
})


def _hook(event: str, args: tuple) -> None:
    if event in _NET_EVENTS:
        AUDIT["denied_net"].append((event, repr(args)))
        raise RuntimeError(f"network forbidden by audit hook: {event}")
    if not AUDIT["active"]:
        return
    if event == "open" and args:
        mode = str(args[1]) if len(args) > 1 else "r"
        AUDIT["events"].append(("open", str(args[0]), mode))
    elif event == "os.rename" and len(args) >= 2:
        # PEP 578 emits "os.rename" for BOTH os.rename and os.replace!
        AUDIT["events"].append(("os.rename", str(args[0]), str(args[1])))
    elif event in ("os.mkdir", "os.unlink", "os.rmdir", "os.remove") and args:
        AUDIT["events"].append((event, str(args[0])))


sys.addaudithook(_hook)

from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas  # noqa: F401
import httpx
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import bts.cli as cli
import bts.contest_fetch as cf
import bts.contest_state as cs
import bts.dm as dm_mod
import bts.health.contest_state as hc
import bts.leaderboard.auth as auth
import bts.posting as posting_mod
import bts.saver_state as ss
import bts.season_contract as sc

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32


@contextmanager
def traced():
    """Record both PEP-578 audit events and Path.lstat calls in unified chronological order."""
    AUDIT["events"].clear()
    real_lstat = Path.lstat

    def traced_lstat(self, *a, **kw):
        if AUDIT["active"]:
            AUDIT["events"].append(("lstat", str(self)))
        return real_lstat(self, *a, **kw)

    with patch.object(Path, "lstat", traced_lstat):
        AUDIT["active"] = True
        try:
            yield AUDIT["events"]
        finally:
            AUDIT["active"] = False


def _put_json(picks: Path, name: str, obj) -> Path:
    p = picks / "account_state" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, bytes):
        p.write_bytes(obj)
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _sw(**kw):
    return {
        "schema_version": sc.SCHEMA,
        "season": 2027,
        "status": "automatic",
        "activation_id": ID,
        "switched_at": "2027-07-15T16:00:00Z",
        "revoked_at": None,
        "note": None,
    } | kw


def _auto(**kw):
    return {
        "schema_version": sc.AUTO_SCHEMA,
        "season": 2027,
        "activation_id": ID,
        "active_streak": 12,
        "best_streak": 15,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "recorded_at": "2027-07-15T15:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
    } | kw


def _dated(day: str = "2027-07-15", **kw):
    d = dt.date.fromisoformat(day)
    return {
        "schema_version": cs.DATED_MANUAL_SCHEMA,
        "effective_date": day,
        "season": d.year,
        "expires_at": REAL_DT.combine(d, dt.time.max, ET).isoformat(),
        "active_streak": 12,
        "best_streak": 15,
        "saver_state": "used",
        "recorded_at": "2027-07-15T15:00:00Z",
        "source": "manual_cli",
    } | kw


@contextmanager
def captured_env(now_box: list[dt.datetime]):
    class FakeDT(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            cur = now_box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    with (
        patch.object(dt, "datetime", FakeDT),
        patch.object(httpx, "get", side_effect=AssertionError("uncaptured httpx")),
        patch.object(auth, "load_session_cookies", return_value={}),
        patch.object(auth, "extract_uid", return_value="u1"),
        patch.object(auth, "fetch_login_session", side_effect=lambda **_: SimpleNamespace(user_id=50311, username="stonehengee", xsid="x")),
        patch.object(cf, "fetch_profile", side_effect=lambda *_: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]}),
        patch.object(cli, "_fetch_rounds", side_effect=lambda: {1: now_box[0].astimezone(ET).date() - dt.timedelta(days=1)}),
        patch.object(dm_mod, "send_dm", return_value="dm-1"),
        patch.object(posting_mod, "post_to_bluesky", return_value="uri-1"),
    ):
        yield


def verify_lstat_non_touch_and_revoke() -> dict:
    runner = CliRunner()
    now_box = [NOW]
    report = {}

    # Verify zero lstat AND zero open on season_contract.json in non-live, expected=False, and dated-in-effect
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        _put_json(picks, "season_contract.json", b"\xff\xfe")  # corrupt switch on disk!
        _put_json(picks, "contest_streak.json", _auto(schema_version="bts_contest_streak_auto_v1"))
        with traced() as evs:
            cs.load_decision_streak_state(picks, require_contest_state=False, now=NOW)
            hc.check(picks, expected=False, now=NOW)
            cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=False)
            cs.load_contest_streak_state(picks, now=NOW, require_dated_auto=True)
        assert not any("season_contract.json" in str(e) for e in evs), evs

        _put_json(picks, "contest_streak.manual.json", _dated())
        with traced() as evs2:
            cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
            hc.check(picks, expected=True, now=NOW)
        assert not any("season_contract.json" in str(e) for e in evs2), evs2
    report["zero_lstat_and_open_on_switch_verified"] = True

    with captured_env(now_box):
        # 1. Usage errors: zero lstat and zero open on account_state / health_state
        for bad_args in [
            [],
            ["--season", "2026", "--reason", "test"],
            ["--season", "notint", "--reason", "test"],
            ["--season", "2027"],
            ["--season", "2027", "--reason", ""],
            ["--season", "2027", "--reason", "  \t\n  "],
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                _put_json(picks, "season_contract.json", _sw())
                with traced() as evs:
                    res = runner.invoke(cli.cli, ["season-contract", "revoke", *bad_args, "--picks-dir", str(picks)])
                assert res.exit_code == 2
                touches = [e for e in evs if any("account_state" in str(x) or "health_state" in str(x) for x in e[1:])]
                assert touches == [], (bad_args, touches)
        report["revoke_usage_zero_lstat_and_open"] = True

        # 2. Exact 3-step chronological order via PEP-578 audit hook + lstat trace
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "season_contract.json", _sw())
            _put_json(picks, "contest_streak.manual.json", _dated())
            _put_json(picks, "saver_state.json", {"season": 2027, "state": "used"})
            with traced() as evs:
                res = runner.invoke(
                    cli.cli,
                    ["season-contract", "revoke", "--season", "2027", "--reason", " audit ", "--picks-dir", str(picks)],
                )
            assert res.exit_code == 0, res.output
            steps = []
            for e in evs:
                if "account_state" not in " ".join(str(x) for x in e):
                    continue
                if e[0] == "lstat" and e[1].endswith("season_contract.json"):
                    steps.append("lstat_prior_switch")
                elif e[0] == "open" and e[1].endswith("season_contract.json") and "r" in e[2]:
                    steps.append("read_prior_switch")
                elif e[0] == "os.rename" and e[2].endswith("season_contract.json"):
                    steps.append("replace_revoked_switch")
                elif e[0] == "lstat" and e[1].endswith("contest_streak.manual.json"):
                    steps.append("lstat_manual_src")
                elif e[0] == "lstat" and "contest_streak.manual.revoked-" in e[1]:
                    steps.append("lstat_manual_dst")
                elif e[0] == "os.rename" and e[1].endswith("contest_streak.manual.json"):
                    steps.append("rename_manual")
                elif e[0] == "lstat" and e[1].endswith("saver_state.json"):
                    steps.append("lstat_saver_src")
                elif e[0] == "lstat" and "saver_state.revoked-" in e[1]:
                    steps.append("lstat_saver_dst")
                elif e[0] == "os.rename" and e[1].endswith("saver_state.json"):
                    steps.append("rename_saver")
            expected = [
                "lstat_prior_switch",
                "read_prior_switch",
                "replace_revoked_switch",
                "lstat_manual_src",
                "lstat_manual_dst",
                "rename_manual",
                "lstat_saver_src",
                "lstat_saver_dst",
                "rename_saver",
            ]
            assert steps == expected, steps
            report["revoke_exact_step_order"] = steps

        # 3. Remaining Item 7 checks: collisions (all 4 types), readonly step a, dangling sources,
        #    second revoke, Dec 31 midnight crossing, carry rules, non-re-coverability
        stamp = "20270715T160000000000Z"
        for target_kind in ("regular", "directory", "valid_symlink", "dangling_symlink"):
            for stage in ("manual", "saver"):
                for manual_present in ([True] if stage == "manual" else [True, False]):
                    with tempfile.TemporaryDirectory() as tmp:
                        picks = Path(tmp) / "picks"
                        _put_json(picks, "season_contract.json", _sw())
                        auto_p = _put_json(picks, "contest_streak.json", _auto())
                        ledger_p = _put_json(picks, "contest_ledger.jsonl", {"row": 1})
                        fstat_p = picks.parent / "health_state/contest_streak_fetch_status.json"
                        fstat_p.parent.mkdir(parents=True, exist_ok=True)
                        fstat_p.write_text('{"ok": true}', encoding="utf-8")
                        untouched_before = {p: p.read_bytes() for p in (auto_p, ledger_p, fstat_p)}
                        if manual_present:
                            _put_json(picks, "contest_streak.manual.json", _dated())
                        _put_json(picks, "saver_state.json", {"season": 2027, "state": "used"})
                        stem = "contest_streak.manual" if stage == "manual" else "saver_state"
                        tgt = picks / "account_state" / f"{stem}.revoked-{stamp}.json"
                        if target_kind == "regular":
                            tgt.write_text("keep_me", encoding="utf-8")
                        elif target_kind == "directory":
                            tgt.mkdir()
                        elif target_kind == "valid_symlink":
                            real_f = picks / "account_state/other_real.txt"
                            real_f.write_text("real", encoding="utf-8")
                            tgt.symlink_to(real_f)
                        else:
                            tgt.symlink_to(picks / "account_state/does_not_exist")
                        with traced() as evs:
                            res = runner.invoke(
                                cli.cli,
                                ["season-contract", "revoke", "--season", "2027", "--reason", "fail", "--picks-dir", str(picks)],
                            )
                        assert res.exit_code == 1
                        assert sc.switch_state(picks, NOW).state == "revoked"
                        if stage == "manual":
                            assert not any("saver_state" in str(e) for e in evs)
                            assert "saver flag was not attempted" in res.output
                        else:
                            assert ("renamed to" if manual_present else "absent") in res.output
                            assert "saver flag was not set aside" in res.output
                        for p, b in untouched_before.items():
                            assert p.read_bytes() == b
        report["collisions_all_4_types_verified"] = True

        # Dangling sources + second revoke + Dec 31 midnight crossing + carry + non-re-coverability
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            acct = picks / "account_state"
            acct.mkdir(parents=True)
            (acct / "contest_streak.manual.json").symlink_to(acct / "gone1")
            (acct / "saver_state.json").symlink_to(acct / "gone2")
            assert runner.invoke(
                cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "d1", "--picks-dir", str(picks)]
            ).exit_code == 0
            assert (acct / f"contest_streak.manual.revoked-{stamp}.json").is_symlink()
            assert (acct / f"saver_state.revoked-{stamp}.json").is_symlink()
            _put_json(picks, "contest_streak.manual.json", _dated())
            assert runner.invoke(
                cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "same_us", "--picks-dir", str(picks)]
            ).exit_code == 1
            now_box[0] = NOW + dt.timedelta(microseconds=1)
            assert runner.invoke(
                cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "next_us", "--picks-dir", str(picks)]
            ).exit_code == 0
            assert (acct / "contest_streak.manual.revoked-20270715T160000000001Z.json").exists()
            now_box[0] = NOW
        report["dangling_and_second_revoke_verified"] = True

        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            dec31 = REAL_DT(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
            _put_json(picks, "season_contract.json", _sw(season=2027))
            _put_json(picks, "contest_streak.manual.json", _dated("2027-12-31"))
            now_box[0] = dec31
            orig_sw = sc.switch_state

            def tick(*a, **kw):
                now_box[0] = dec31 + dt.timedelta(microseconds=2)
                return orig_sw(*a, **kw)

            with patch.object(sc, "switch_state", side_effect=tick):
                r_ny = runner.invoke(cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "nye", "--picks-dir", str(picks)])
            assert r_ny.exit_code == 0
            w = json.loads((picks / "account_state/season_contract.json").read_text(encoding="utf-8"))
            assert w["season"] == 2027 and w["revoked_at"] == "2028-01-01T04:59:59.999999Z"
            assert "--date 2027-12-31" in r_ny.output
            now_box[0] = NOW
        report["dec31_midnight_crossing_verified"] = True

        # Carry rules across all 13 prior states + executing templates from revoke output
        prior_cases = [
            ("absent", None, False, None),
            ("corrupt_1", [], False, None),
            ("corrupt_2", _sw(schema_version="v0"), False, None),
            ("corrupt_3", _sw(season=True), False, None),
            ("corrupt_4", _sw(status="x"), False, None),
            ("corrupt_5", _sw(activation_id="123"), False, None),
            ("corrupt_6", _sw(switched_at=None), False, None),
            ("corrupt_7", _sw(revoked_at="bad"), False, None),
            ("corrupt_8", _sw(note={}), False, None),
            ("other_season", _sw(season=2026), False, None),
            ("automatic_y", _sw(), True, "2027-07-15T16:00:00Z"),
            ("revoked_null_sw", _sw(status="revoked", activation_id="d" * 32, switched_at=None, revoked_at="2027-07-14T00:00:00Z"), True, None),
            ("revoked_with_sw", _sw(status="revoked", activation_id="e" * 32, switched_at="2027-03-25T14:00:00Z", revoked_at="2027-07-14T00:00:00Z"), True, "2027-03-25T14:00:00Z"),
        ]
        for label, prior_val, expect_carry, expect_sw_at in prior_cases:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "space ' \" & picks"
                picks.mkdir()
                if prior_val is not None:
                    _put_json(picks, "season_contract.json", prior_val)
                r = runner.invoke(cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "carry", "--picks-dir", str(picks)])
                assert r.exit_code == 0, (label, r.output)
                snap = sc.switch_state(picks, NOW).snapshot
                assert (snap["activation_id"] == prior_val["activation_id"] if expect_carry else sc.valid_activation_id(snap["activation_id"]))
                if not expect_carry and isinstance(prior_val, dict) and "activation_id" in prior_val:
                    assert snap["activation_id"] != prior_val["activation_id"]
                assert snap["switched_at"] == expect_sw_at
                raw_cmds = re.findall(r"`(bts [^`]+)`", r.output)
                assert len(raw_cmds) == 3
                saver_argv = shlex.split(raw_cmds[1].replace("<state read from MLB>", "not_earned"))
                assert runner.invoke(cli.cli, saver_argv[1:]).exit_code == 0
                assert ss.load_saver_state(picks, season=2027).state == "not_earned"
                assert runner.invoke(cli.cli, shlex.split(raw_cmds[2])[1:]).exit_code == 0
        report["carry_and_templates_verified"] = len(prior_cases)

        # Pre-revoke observation never covered again without restoring season_contract.json
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            _put_json(picks, "season_contract.json", _sw())
            auto_p = _put_json(picks, "contest_streak.json", _auto())
            pre_bytes = auto_p.read_bytes()
            assert runner.invoke(cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "r1", "--picks-dir", str(picks)]).exit_code == 0
            for act in [
                lambda: runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]),
                lambda: auto_p.write_bytes(pre_bytes),
                lambda: runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]),
                lambda: auto_p.write_bytes(pre_bytes),
                lambda: runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks)]),
                lambda: runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)]),
                lambda: auto_p.write_bytes(pre_bytes),
            ]:
                act()
                if auto_p.read_bytes() == pre_bytes:
                    try:
                        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
                        raise AssertionError("pre-revoke auto became covered!")
                    except cs.ContestStateError:
                        pass
        report["pre_revoke_never_covered_again"] = True

    return report


def main() -> None:
    from probe_cli_sequences_v2 import probe_item8_sequences
    work = Path(__file__).resolve().parent
    repo = REPO_ROOT

    item7 = verify_lstat_non_touch_and_revoke()
    item8 = probe_item8_sequences()

    # Run Item 9 caller comparison via probe_6_7_8_9.py's --caller-worker mode
    p6789 = work / "probe_6_7_8_9.py"
    base_callers_path = work / "callers_base.json"
    branch_callers_path = work / "callers_branch.json"
    for src, out_p in [(work / "base/src", base_callers_path), (repo / "src", branch_callers_path)]:
        proc = subprocess.run(
            [sys.executable, str(p6789), "--caller-worker", "--src", str(src), "--out", str(out_p)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"caller worker failed ({src}):\n{proc.stdout}\n{proc.stderr}")

    base_c = json.loads(base_callers_path.read_text(encoding="utf-8"))
    branch_c = json.loads(branch_callers_path.read_text(encoding="utf-8"))
    assert base_c["denied_network"] == []
    assert branch_c["denied_network"] == []
    assert AUDIT["denied_net"] == []

    assert base_c["run_and_pick_uncovered"]["accepted"] is True
    assert branch_c["run_and_pick_uncovered"]["accepted"] is False
    assert base_c["orchestrate_initial_uncovered"] == {"returned": True, "pick_saved": True, "dms_count": 0, "posts_count": 1}
    assert branch_c["orchestrate_initial_uncovered"] == {"returned": False, "pick_saved": False, "dms_count": 1, "posts_count": 0}
    assert branch_c["orchestrate_mid_revoke_post_save"]["raised"] is True
    assert branch_c["orchestrate_mid_revoke_post_save"]["posts"] == 0
    assert base_c["scheduler_run_single_check_uncovered"]["pick_file_exists"] is True
    assert branch_c["scheduler_run_single_check_uncovered"] == {
        "should_post": False,
        "pick_result_none": True,
        "selection_none": True,
        "pick_file_exists": False,
        "decision_exists": False,
        "alert_dms": 1,
    }
    assert base_c["deliver_and_lock_dm"]["returned"] is True and base_c["deliver_and_lock_dm"]["pick_dms"] == 1
    assert branch_c["deliver_and_lock_dm"]["returned"] is False and branch_c["deliver_and_lock_dm"]["pick_dms"] == 0
    assert base_c["deliver_and_lock_public"]["returned"] is True and base_c["deliver_and_lock_public"]["posts"] == 1
    assert branch_c["deliver_and_lock_public"]["returned"] is False and branch_c["deliver_and_lock_public"]["posts"] == 0
    assert branch_c["deliver_and_lock_private"]["returned"] is True
    assert branch_c["deliver_and_lock_delivered"]["returned"] is True
    assert branch_c["deliver_and_lock_crash"]["returned"] is False and branch_c["deliver_and_lock_crash"]["pick_locked"] is True
    assert base_c["fallback_refresh_uncovered"]["raised"] is False
    assert branch_c["fallback_refresh_uncovered"]["raised"] is True
    assert branch_c["non_blocked_grading_skip_day_outcome"]["check_results_exit"] == 0
    assert branch_c["non_blocked_grading_skip_day_outcome"]["streak_after_check_results"] == 1
    assert branch_c["non_blocked_grading_skip_day_outcome"]["eod_skip_written"] is True
    assert branch_c["non_blocked_grading_skip_day_outcome"]["day_outcome_exit"] == 0

    combined = {
        "item7": item7,
        "item8": item8,
        "callers_base": base_c,
        "callers_branch": branch_c,
    }
    (work / "report_7_8_9.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps({
        "item7": item7,
        "item8_sequences": list(item8.keys()),
        "callers_branch": branch_c,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Item 4 & Whole-Range Adversarial Pass probe across B1, B2, B3, B5, B6 against base 9e0ec28a."""
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import signal
import stat
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import httpx
from click.testing import CliRunner

REVIEW = Path(__file__).resolve().parent
REV = sys.argv[1]
assert REV in ("base", "head", "compare"), REV

if REV == "compare":
    b = json.loads((REVIEW / "whole-base.json").read_text())
    h = json.loads((REVIEW / "whole-head.json").read_text())

    # 1. B2 delivered/private_locked/legacy DMs, console, markers, exits must match base byte-for-byte
    for mode in ("delivered", "private_locked", "legacy"):
        for reason in ("no_pick", "mismatch"):
            key = f"{mode}:{reason}"
            assert b["b2"][key] == h["b2"][key], (key, b["b2"][key], h["b2"][key])

    # 2. B2 locked_unconfirmed must emit verification copy at all 3 tiers on HEAD (vs Fix it now! on base)
    for reason in ("no_pick", "mismatch"):
        key = f"locked_unconfirmed:{reason}"
        assert "Fix it now!" in b["b2"][key][0]["dm"][0]
        for idx, minutes in ((0, 45), (2, 25), (3, 10)):
            dm = h["b2"][key][idx]["dm"][0]
            assert "Delivery of this pick was NOT confirmed." in dm
            assert "Aaron Judge + DD Second Batter is the pick the scheduler tried to send." in dm
            assert ("No entry found in MLB." if reason == "no_pick" else "The MLB entry differs.") in dm
            assert f"({minutes} min to submit)" in dm
            assert "Check your DMs and MLB entry before entering anything" in dm
            assert "bts season-contract status --picks-dir " in dm
            assert "Fix it now!" not in dm and "STILL NOT entered" not in dm
        # All-clear at step 5 byte-identical to base
        assert h["b2"][key][5]["dm"] == b["b2"][key][5]["dm"] == ["✅ BTS pick entry confirmed for 2027-07-15 (match)."]
        assert [x["exit"] for x in h["b2"][key]] == [x["exit"] for x in b["b2"][key]]

    # 3. B3 day-outcome controls and fixes
    assert h["b3"]["double"] == b["b3"]["double"]
    for state_key in ("no_games", "deliberate_skip", "schedule_unverifiable", "no_decision"):
        assert h["b3"][state_key] == b["b3"][state_key], state_key
    for cls_key in ("single_unverifiable", "single_entered", "single_game_unverified",
                    "single_late", "single_not_entered", "single_contradiction",
                    "single_not_scoreable", "single_research_ok", "single_research_bad"):
        assert h["b3"][cls_key]["pick"] == "Batter A", (cls_key, h["b3"][cls_key])
        assert b["b3"][cls_key]["pick"] == "Batter A + Batter B", cls_key
        assert h["b3"][cls_key]["state"] == b["b3"][cls_key]["state"], cls_key
        assert h["b3"][cls_key]["exit"] == b["b3"][cls_key]["exit"], cls_key

    # 4. B5 conforming status byte-for-byte parity
    assert h["b5"]["conforming_sequence"] == b["b5"]["conforming_sequence"]

    # 5. B6 checklist: only B8a changed in 2027-season-start.md
    assert h["b6"]["non_b8a_sha256"] == b["b6"]["non_b8a_sha256"]
    assert h["b6"]["b8a_gates_present"] is True and b["b6"]["b8a_gates_present"] is False

    print("Whole-range B1/B2/B3/B5/B6 comparison against base 9e0ec28a: ALL ASSERTIONS PASSED")
    raise SystemExit(0)

ROOT = REVIEW.parents[1] if REV == "head" else REVIEW / "base"
sys.path.insert(0, str(ROOT / "src"))

import pandas  # noqa: F401
from bts import (
    cli, contest_fetch as cf, contest_state as cs, daily_decision as dd,
    day_outcome, entry_receipt, picks as pick_io, receipt_io, saver_state as saver,
    scheduler, season_contract as sc,
)
from bts.health import alert as health_alert, contest_state as health_cs
from bts.leaderboard import auth

expected_src = (ROOT / "src").resolve()
for mod in (cli, cf, cs, dd, day_outcome, sc, health_cs):
    assert Path(mod.__file__).resolve().is_relative_to(expected_src), (REV, mod.__file__)

# Also import test helpers from workspace tests
sys.path.insert(1, str(REVIEW.parents[1]))
from tests.test_day_outcome import _receipt
from tests.test_day_outcome_entry_sequence import _setup_committed_day

REAL_DT = dt.datetime
ET = ZoneInfo("America/New_York")
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
DAY = NOW.date().isoformat()
ID = "a" * 32


class BlockedError(BaseException):
    pass


@contextmanager
def bounded(seconds=0.1):
    def handler(*_):
        raise BlockedError("blocked")
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@contextmanager
def pinned_offline(now=NOW):
    class PinnedDT(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    send_mock = Mock(return_value="dm-ack")
    profile_mock = Mock(return_value={
        "activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}],
    })
    with (
        patch.object(dt, "datetime", PinnedDT),
        patch.object(cs, "datetime", PinnedDT),
        patch.object(dd, "datetime", PinnedDT),
        patch.object(saver, "datetime", PinnedDT),
        patch.object(sc, "datetime", PinnedDT),
        patch.object(health_cs, "datetime", PinnedDT),
        patch.object(httpx, "get", Mock(side_effect=AssertionError("network forbidden"))),
        patch.object(auth, "load_session_cookies", lambda: {}),
        patch.object(auth, "extract_uid", lambda _: "u"),
        patch.object(auth, "fetch_login_session", lambda **_: SimpleNamespace(
            user_id=1, username="offline", xsid="x")),
        patch.object(cf, "fetch_profile", profile_mock),
        patch.object(cli, "_fetch_rounds", lambda: {2: NOW.date(), 1: NOW.date() - dt.timedelta(days=1)}),
        patch("bts.dm.send_dm", send_mock),
    ):
        yield send_mock, profile_mock


def make_switch(now=NOW):
    return {
        "schema_version": "bts_season_contract_v1", "season": now.astimezone(ET).year,
        "status": "automatic", "activation_id": ID,
        "switched_at": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "revoked_at": None, "note": None,
    }


def make_auto(now=NOW, **ov):
    return {
        "schema_version": "bts_contest_streak_auto_v2", "active_streak": 12, "best_streak": 15,
        "source": "mlb_bts_profile",
        "source_date": (now.astimezone(ET).date() - dt.timedelta(days=1)).isoformat(),
        "recorded_at": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "season": now.astimezone(ET).year, "activation_id": ID,
    } | ov


report = {"b1": {}, "b2": {}, "b3": {}, "b5": {}, "b6": {}}

with TemporaryDirectory(dir=REVIEW, prefix=f"whole-{REV}-") as tmp:
    root = Path(tmp)

    # --- B1: Saver warning repair command across path shapes & New Year ---
    prev_cwd = Path.cwd()
    os.chdir(root)
    try:
        for idx, p in enumerate((root / "dir with spaces", root / "quote's \"picks\"", Path("rel picks"))):
            (p / "account_state").mkdir(parents=True)
            (p / "account_state/season_contract.json").write_text(json.dumps(make_switch(NOW)))
            (p / "account_state/contest_streak.json").write_text(json.dumps(make_auto(NOW, active_streak=11)))
            flag = p / "account_state/saver_state.json"
            flag.write_text(json.dumps({"season": NOW.year, "state": "not_earned"}))
            before_flag = (flag.read_bytes(), flag.stat().st_mtime_ns)
            warn = next(a for a in health_cs.check(p, expected=True, now=NOW)
                        if a.source == "saver_state" and "flag not_earned" in a.message)
            m = re.search(r"`(bts set-contest-streak [^`]+)`", warn.message)
            with pinned_offline(NOW):
                if m:
                    args = [{"N": "0", "B": "0", "S": "not_earned"}.get(tok, tok) for tok in shlex.split(m[1])[1:]]
                    res = CliRunner().invoke(cli.cli, args)
                else:
                    res = CliRunner().invoke(cli.cli, ["set-contest-streak"])
                live = cs.load_decision_streak_state(p, require_contest_state=True, now=NOW)
            assert (flag.read_bytes(), flag.stat().st_mtime_ns) == before_flag
            if REV == "head":
                assert (res.exit_code, live.observation_kind, live.streak) == (0, "dated", 0)
            else:
                assert (res.exit_code, live.observation_kind, live.streak) == (2, "automatic", 11)
            report["b1"][f"path_{idx}"] = {"exit": res.exit_code, "kind": live.observation_kind, "streak": live.streak}
    finally:
        os.chdir(prev_cwd)

    # --- B2: Watchdog across tiers, modes, stale date, disagreement ---
    for mode in ("locked_unconfirmed", "delivered", "private_locked", "legacy", "stale_date"):
        for reason in ("no_pick", "mismatch"):
            picks = _setup_committed_day(root / f"b2-{mode}-{reason}", DAY, game_time_utc=f"{DAY}T23:10:00Z")
            dec_path = picks / DAY / "decision.json"
            rec = json.loads(dec_path.read_bytes())
            daily = pick_io.load_pick(DAY, picks)
            daily.double_down = replace(daily.pick, batter_id=123, batter_name="Second Batter")
            rec["action"] = "double"
            rec["double_down"] = {"batter_id": 123, "batter_name": "Second Batter"}
            if mode == "legacy":
                daily.notification_id = "ack"
                dec_path.unlink()
            else:
                daily.notification_id = "ack" if mode in ("delivered", "stale_date") else None
                daily.notification_sent = mode == "delivered"
                rec["delivery_status"] = "locked_unconfirmed" if mode == "stale_date" else mode
                if mode == "stale_date":
                    rec["date"] = "2027-07-14"
                dec_path.write_text(json.dumps(rec))
            pick_io.save_pick(daily, picks)
            marker_path = picks.parent / "health_state/pick_entry_check.json"
            seq = []
            with pinned_offline(NOW) as (send, _), patch.object(
                cli, "_fetch_bts_to_mlb", return_value={88: 592450, 89: 123, 98: 998, 99: 999}
            ):
                for clock, obs in (
                    ("18:20", reason), ("18:22", reason), ("18:40", reason),
                    ("18:55", reason), ("18:56", reason), ("18:57", "match"), ("18:58", "match"),
                ):
                    pids = [] if obs == "no_pick" else [98, 99] if obs == "mismatch" else [88, 89]
                    pending = [{"roundId": 2, "playerId": pid} for pid in pids]
                    send.reset_mock()
                    with patch.object(cf, "fetch_pending_predictions", return_value=pending):
                        r = CliRunner().invoke(cli.cli, [
                            "check-pick-entered", "--picks-dir", str(picks),
                            "--now-et", f"{DAY}T{clock}:00",
                            "--dm-recipient", "offline.test", "--expected-username", "offline",
                        ])
                    mk = json.loads(marker_path.read_text())
                    mk.pop("receipt")
                    seq.append({
                        "exit": r.exit_code,
                        "console": r.output.replace(str(picks), "<PICKS>"),
                        "marker": mk,
                        "dm": [c.args[1].replace(str(picks), "<PICKS>") for c in send.call_args_list],
                    })
            if REV == "head" and mode == "stale_date":
                assert "Fix it now!" in seq[0]["dm"][0], "stale embedded date must not trigger same-date branch"
            report["b2"][f"{mode}:{reason}"] = seq

    # --- B3: day-outcome across ALL pick-carrying and non-pick classifications ---
    cfg_enter = root / "enter.toml"
    cfg_enter.write_text('[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
    cfg_res = root / "research.toml"
    cfg_res.write_text('[scheduler]\npick_delivery = "none"\nentry_intent = "research"\n')

    def run_outcome(name, *, action="single", primary=True, runner=True, delivery="delivered",
                    scoreable=True, games=1, sched_err=False, cfg=cfg_enter, receipt_kind=None, no_dec=False):
        p = root / f"b3-{name}"
        p.mkdir(parents=True)
        raw_before = None
        if not no_dec:
            rec = dd.write_decision(
                DAY, p, action=action, source="mdp",
                primary={"batter_name": "Batter A"} if primary is True else primary,
                double_down={"batter_name": "Batter B"} if runner is True else None,
                delivery_status=delivery, scoreable=scoreable,
            )
            dpath = p / DAY / "decision.json"
            if runner is not True and runner is not None:
                dpath.write_text(json.dumps(json.loads(dpath.read_text()) | {"double_down": runner}))
            raw_before = dpath.read_bytes()
            if receipt_kind is not None:
                sha = hashlib.sha256(raw_before).hexdigest()
                rbody = _receipt(date=DAY, sha=sha)
                if receipt_kind == "game_unverified":
                    rbody["qualification"] = {"all_confirmed": False, "slots": [{"state": "unit_unverified"}]}
                elif receipt_kind == "late":
                    rbody["observation"]["before_cutoff"] = False
                    rbody["observation"]["response_completed_at"] = "2027-07-15T23:20:00Z"
                elif receipt_kind == "absent":
                    rbody["qualification"] = {"all_confirmed": False, "slots": []}
                    rbody["verifier"]["reason"] = "no_pick"
                receipt_io.publish(entry_receipt.receipts_root(p) / DAY / "r.json", json.dumps(rbody).encode())

        sched_fn = (Mock(side_effect=RuntimeError("boom")) if sched_err
                    else lambda _: [{"gamePk": i} for i in range(games)])
        with pinned_offline(NOW) as (send, _), patch.object(scheduler, "fetch_schedule", sched_fn):
            res = CliRunner().invoke(cli.cli, [
                "day-outcome", "--date", DAY, "--config", str(cfg), "--picks-dir", str(p),
                "--expected-username", "stonehengee", "--dm-recipient", "offline.test",
            ])
        if raw_before is not None:
            assert (p / DAY / "decision.json").read_bytes() == raw_before
        parsed = json.loads(res.stdout) if res.stdout.strip() else None
        return {
            "exit": res.exit_code,
            "state": parsed["state"] if parsed else None,
            "pick": parsed["pick"] if parsed else None,
            "dm": [c.args[1] for c in send.call_args_list],
            "exc": type(res.exception).__name__ if (res.exception and not isinstance(res.exception, SystemExit)) else None,
        }

    report["b3"]["single_unverifiable"] = run_outcome("s-unv")
    report["b3"]["single_entered"] = run_outcome("s-ent", receipt_kind="confirmed")
    report["b3"]["single_game_unverified"] = run_outcome("s-gunv", receipt_kind="game_unverified")
    report["b3"]["single_late"] = run_outcome("s-late", receipt_kind="late")
    report["b3"]["single_not_entered"] = run_outcome("s-not", receipt_kind="absent")
    report["b3"]["single_contradiction"] = run_outcome("s-contra", games=0)
    report["b3"]["single_not_scoreable"] = run_outcome("s-nosc", delivery="preview", scoreable=False)
    report["b3"]["single_research_ok"] = run_outcome("s-res-ok", delivery="private_locked", cfg=cfg_res)
    report["b3"]["single_research_bad"] = run_outcome("s-res-bad", delivery="delivered", cfg=cfg_res)
    report["b3"]["double"] = run_outcome("double", action="double")
    report["b3"]["single_bad_runner_list"] = run_outcome("s-list", runner=["bad"])
    report["b3"]["no_games"] = run_outcome("no-games", games=0, no_dec=True)
    report["b3"]["deliberate_skip"] = run_outcome("skip", action="skip", primary=None, runner=None, scoreable=False)
    report["b3"]["schedule_unverifiable"] = run_outcome("sched-err", sched_err=True)
    report["b3"]["no_decision"] = run_outcome("no-dec", no_dec=True)
    if REV == "head":
        assert report["b3"]["single_bad_runner_list"]["pick"] == "Batter A"
        assert report["b3"]["single_bad_runner_list"]["exit"] == 1

    # --- B5: Non-object / FIFO / directory status & conforming byte parity ---
    p_conf = root / "b5-conf" / "picks"
    (p_conf / "account_state").mkdir(parents=True)
    (p_conf / "account_state/season_contract.json").write_text(json.dumps(make_switch(NOW)))
    st_conf = p_conf.parent / "health_state/contest_streak_fetch_status.json"
    st_conf.parent.mkdir(parents=True)
    st_conf.write_text(json.dumps({
        "last_error": "old", "last_alert_at": "2027-07-15T15:00:00Z",
        "last_alert_at_by_category": {"actionable": "2027-07-15T15:00:00Z"},
    }))
    seq_b5 = []
    for step_name, dt_val, fail_flag in (
        ("throttled_fail", NOW, True),
        ("success_preserves", NOW + dt.timedelta(hours=1), False),
        ("still_throttled", NOW + dt.timedelta(hours=3), True),
        ("alerts_after_6h", NOW + dt.timedelta(hours=6), True),
    ):
        with pinned_offline(dt_val) as (send, prof):
            if fail_flag:
                prof.return_value = {"activeStreak": 7, "seasonBestStreak": 9, "predictions": {}}
            r = CliRunner().invoke(cli.cli, [
                "fetch-contest-streak", "--picks-dir", str(p_conf), "--dm-recipient", "offline.test",
            ])
            seq_b5.append({
                "step": step_name, "exit": r.exit_code, "dm": send.call_count,
                "status_bytes": st_conf.read_bytes().decode("utf-8"),
            })
    report["b5"]["conforming_sequence"] = seq_b5

    if REV == "head":
        for bad_label, bad_val in (("list", b"[]"), ("null", b"null"), ("fifo", "fifo"), ("dir", "dir")):
            for fail_lane in (False, True):
                p_bad = root / f"b5-{bad_label}-{fail_lane}" / "picks"
                (p_bad / "account_state").mkdir(parents=True)
                (p_bad / "account_state/season_contract.json").write_text(json.dumps(make_switch(NOW)))
                auto_f = p_bad / "account_state/contest_streak.json"
                auto_f.write_text(json.dumps(make_auto(NOW)))
                prior_auto = auto_f.read_bytes()
                st_f = p_bad.parent / "health_state/contest_streak_fetch_status.json"
                st_f.parent.mkdir(parents=True)
                if bad_val == "fifo":
                    os.mkfifo(st_f)
                elif bad_val == "dir":
                    st_f.mkdir()
                else:
                    st_f.write_bytes(bad_val)
                with pinned_offline(NOW) as (send, prof), bounded():
                    if fail_lane:
                        prof.return_value = {"activeStreak": 7, "seasonBestStreak": 9, "predictions": {}}
                    r = CliRunner().invoke(cli.cli, [
                        "fetch-contest-streak", "--picks-dir", str(p_bad), "--dm-recipient", "offline.test",
                    ])
                expected_exit = 2 if (fail_lane or bad_val == "dir") else 0
                assert r.exit_code == expected_exit, (bad_label, fail_lane, r.exit_code, r.output)
                assert send.call_count == int(fail_lane)
                assert (auto_f.read_bytes() == prior_auto) == fail_lane
                if bad_val != "dir":
                    assert stat.S_ISREG(st_f.lstat().st_mode)
                    if not fail_lane:
                        assert (p_bad / "account_state/contest_ledger.jsonl").is_file()
                        assert json.loads((p_bad / "account_state/saver_state.json").read_text())["state"] == "not_earned"

    # --- B6: Checklist doc & cutover rehearsal ---
    doc_path = ROOT / "docs/ops/2027-season-start.md"
    doc_lines = doc_path.read_text(encoding="utf-8").splitlines()
    b8a_line = next(ln for ln in doc_lines if ln.startswith("| B8a |"))
    non_b8a = "\n".join(ln for ln in doc_lines if not ln.startswith("| B8a |"))
    gates_ok = bool(re.search(
        r"fetch-contest-streak` succeeds.*automatic path alone: accepted.*"
        r"read `bts saver-state`.*Initialize only if it reports `uninitialized`",
        b8a_line,
    ))
    report["b6"] = {
        "non_b8a_sha256": hashlib.sha256(non_b8a.encode("utf-8")).hexdigest(),
        "b8a_gates_present": gates_ok,
    }

(REVIEW / f"whole-{REV}.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"{REV}: whole-range probe completed cleanly")

"""Item 6 adversarial probe: standing rows and rollback past H2 to base (36fd3e35)."""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_SRC = OUT_DIR / "base_tree" / "src"
ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt.datetime


def _refuse_net(*args, **kwargs):
    raise OSError("offline probe: network refused")


class _Clock(REAL_DATETIME):
    at: REAL_DATETIME = REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz) if tz else cls.at.astimezone(ET).replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.at.astimezone(dt.timezone.utc).replace(tzinfo=None)


@contextmanager
def pinned_offline(at: REAL_DATETIME):
    import bts.cli
    import bts.contest_fetch
    import bts.contest_state
    import bts.daily_decision
    import bts.health.contest_state
    import bts.orchestrator
    import bts.picks
    import bts.saver_state
    import bts.scheduler
    import bts.season_contract
    import bts.strategy

    _Clock.at = at
    with ExitStack() as stack:
        stack.enter_context(patch.object(dt, "datetime", _Clock))
        for name, mod in list(sys.modules.items()):
            if (name == "bts" or name.startswith("bts.")) and getattr(mod, "datetime", None) is REAL_DATETIME:
                stack.enter_context(patch.object(mod, "datetime", _Clock))
        for m in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, m, _refuse_net))
        stack.enter_context(patch.object(socket, "getaddrinfo", _refuse_net))
        yield


def main() -> int:
    from click.testing import CliRunner
    from bts import cli, contest_fetch, contest_state, daily_decision, orchestrator, picks, saver_state, scheduler
    from bts.health import contest_state as health_cs
    from bts.leaderboard import auth
    from scripts.audit.season_walk import walk

    problems: list[str] = []
    notes: list[str] = []
    runner = CliRunner()
    day = dt.date(2027, 7, 15)
    now = REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)

    def fail_if(cond: bool, msg: str):
        if cond:
            problems.append(msg)

    with pinned_offline(now), tempfile.TemporaryDirectory(prefix="h2c-item6-") as tmp:
        root = Path(tmp)

        # === 1. Interrupts between fetch's writes at 58/58 ===
        def seed_57(case_name: str) -> Path:
            p = root / case_name / "picks"
            p.mkdir(parents=True)
            _Clock.at = REAL_DATETIME(2027, 7, 15, 9, 0, tzinfo=ET)
            assert walk.install_state(p, day, walk.State("automatic", 57, 57, "not_earned", 0.96)) == []
            _Clock.at = now
            return p

        # 1a) Interrupt BEFORE observation replacement (_atomic_write_json raises KeyboardInterrupt on first call)
        p_int1 = seed_57("int_before_obs")
        obs_b = (p_int1 / "account_state/contest_streak.json").read_bytes()
        led_b = (p_int1 / "account_state/contest_ledger.jsonl").read_bytes()
        sav_b = (p_int1 / "account_state/saver_state.json").read_bytes()
        st_b = (p_int1.parent / "health_state/contest_streak_fetch_status.json").read_bytes()
        with walk._fetch_stubs(58, 58, day - dt.timedelta(days=1)), \
                patch.object(cli, "_atomic_write_json", side_effect=KeyboardInterrupt("mid-obs")):
            res = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(p_int1), "--expected-username", walk.ACCOUNT])
        fail_if(not isinstance(res.exception, KeyboardInterrupt), f"1a expected KeyboardInterrupt: {res.exception}")
        fail_if(
            (p_int1 / "account_state/contest_streak.json").read_bytes() != obs_b
            or (p_int1 / "account_state/contest_ledger.jsonl").read_bytes() != led_b
            or (p_int1 / "account_state/saver_state.json").read_bytes() != sav_b
            or (p_int1.parent / "health_state/contest_streak_fetch_status.json").read_bytes() != st_b,
            "1a interrupt before obs modified files!",
        )

        # 1b) Interrupt AFTER observation replacement, BEFORE status write
        p_int2 = seed_57("int_before_status")
        led_b2 = (p_int2 / "account_state/contest_ledger.jsonl").read_bytes()
        sav_b2 = (p_int2 / "account_state/saver_state.json").read_bytes()
        st_b2 = (p_int2.parent / "health_state/contest_streak_fetch_status.json").read_bytes()
        with walk._fetch_stubs(58, 58, day - dt.timedelta(days=1)), \
                patch.object(cli, "_write_contest_fetch_status", side_effect=KeyboardInterrupt("mid-status")):
            res2 = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(p_int2), "--expected-username", walk.ACCOUNT])
        fail_if(not isinstance(res2.exception, KeyboardInterrupt), f"1b expected KeyboardInterrupt: {res2.exception}")
        st_after2 = contest_state.load_decision_streak_state(p_int2, require_contest_state=True, now=now)
        fail_if((st_after2.streak, st_after2.best_streak) != (58, 58), f"1b new 58/58 obs not usable: {st_after2}")
        fail_if(
            (p_int2 / "account_state/contest_ledger.jsonl").read_bytes() != led_b2
            or (p_int2 / "account_state/saver_state.json").read_bytes() != sav_b2
            or (p_int2.parent / "health_state/contest_streak_fetch_status.json").read_bytes() != st_b2,
            "1b status/ledger/saver unexpectedly changed!",
        )

        # 1c) Interrupt AFTER ledger append, BEFORE saver auto-earn
        p_int3 = seed_57("int_before_saver")
        sav_b3 = (p_int3 / "account_state/saver_state.json").read_bytes()
        with walk._fetch_stubs(58, 58, day - dt.timedelta(days=1)), \
                patch("bts.saver_state.maybe_auto_earn_saver", side_effect=KeyboardInterrupt("mid-saver")):
            res3 = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(p_int3), "--expected-username", walk.ACCOUNT])
        fail_if(not isinstance(res3.exception, KeyboardInterrupt), f"1c expected KeyboardInterrupt: {res3.exception}")
        fail_if(len((p_int3 / "account_state/contest_ledger.jsonl").read_text().splitlines()) != 2, "1c ledger not appended")
        fail_if((p_int3 / "account_state/saver_state.json").read_bytes() != sav_b3, "1c saver changed")
        notes.append("1. Fetch interrupts verified (atomic obs; separate status, ledger, saver).")

        # === 2. Malformed and non-regular files at each state path ===
        p_mal = root / "malformed" / "picks"
        acct = p_mal / "account_state"
        acct.mkdir(parents=True)

        # Malformed bytes / non-object / directories at manual, auto, switch
        for fname in ("contest_streak.manual.json", "contest_streak.json", "season_contract.json"):
            for mode_name, content in (("bad_utf8", b"\xff"), ("trunc", b"{"), ("list", b"[]"), ("dir", None)):
                for x in acct.iterdir():
                    if x.is_dir():
                        x.rmdir()
                    else:
                        x.unlink()
                target = acct / fname
                if content is None:
                    target.mkdir()
                else:
                    target.write_bytes(content)
                try:
                    contest_state.load_decision_streak_state(p_mal, require_contest_state=True, now=now)
                    problems.append(f"2. live loader accepted {fname} {mode_name}")
                except contest_state.ContestStateError:
                    pass
                h_alerts = health_cs.check(p_mal, expected=True, now=dt.datetime.combine(day, dt.time.max, ET), historical_day=day)
                fail_if(not any(a.level == "CRITICAL" for a in h_alerts), f"2. hist health missed {fname} {mode_name}: {h_alerts}")
                if target.is_dir():
                    target.rmdir()
                else:
                    target.unlink()

        # Saver malformed / directory degrades to uninitialized
        for content in (b"\xff", b"{", b"[]", None):
            sp = acct / "saver_state.json"
            if sp.exists():
                sp.rmdir() if sp.is_dir() else sp.unlink()
            if content is None:
                sp.mkdir()
            else:
                sp.write_bytes(content)
            fail_if(saver_state.load_saver_state(p_mal, season=2027).state != "uninitialized", f"2. saver {content} did not degrade")
            sp.rmdir() if sp.is_dir() else sp.unlink()

        # Prior automatic file of non-object shape (b"[]") raises AttributeError in fetch identity read (base behavior)
        (acct / "contest_streak.json").write_bytes(b"[]")
        with walk._fetch_stubs(58, 58, day - dt.timedelta(days=1)):
            r_list = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(p_mal), "--expected-username", walk.ACCOUNT])
        fail_if(r_list.exit_code != 1 or not isinstance(r_list.exception, AttributeError),
                f"2. non-object prior auto unexpected: code={r_list.exit_code} exc={r_list.exception!r}")
        (acct / "contest_streak.json").unlink()

        # Named pipes (FIFOs): historical reader rejects immediately; live reader blocks (bounded subprocess timeout=1.5s)
        for fname in ("contest_streak.manual.json", "contest_streak.json", "season_contract.json", "saver_state.json"):
            for x in acct.iterdir():
                x.unlink()
            if fname == "saver_state.json":
                # Put valid switched 58/58 automatic file so historical check reaches saver_state.json!
                assert walk.install_state(p_mal, day, walk.State("automatic", 58, 58, "used", 0.96)) == []
                (acct / "saver_state.json").unlink()
            fifo_path = acct / fname
            os.mkfifo(fifo_path)
            try:
                h_fifo = health_cs.check(p_mal, expected=True, now=dt.datetime.combine(day, dt.time.max, ET), historical_day=day)
                if fname == "saver_state.json":
                    fail_if(not any(a.source == "saver_state" and a.level == "WARN" for a in h_fifo),
                            f"2. hist saver FIFO unexpected: {h_fifo}")
                else:
                    fail_if(not any(a.level == "CRITICAL" for a in h_fifo),
                            f"2. hist {fname} FIFO unexpected: {h_fifo}")
                if fname == "contest_streak.json":
                    # Confirm ordinary live read blocks on FIFO (bounded by 1.2s timeout)
                    blocked = False
                    try:
                        subprocess.run(
                            [str(REPO / ".venv/bin/python"), "-c",
                             f"from pathlib import Path; from bts.contest_state import load_decision_streak_state; "
                             f"load_decision_streak_state(Path({str(p_mal)!r}), require_contest_state=True)"],
                            env=dict(os.environ, PYTHONPATH=str(REPO / "src")),
                            timeout=1.2, capture_output=True,
                        )
                    except subprocess.TimeoutExpired:
                        blocked = True
                    fail_if(not blocked, "2. ordinary live read on FIFO did not block as documented")
            finally:
                for x in list(acct.iterdir()):
                    x.unlink()
        notes.append("2. Malformed, directory, and FIFO paths verified (historical non-blocking; live FIFO bounded).")

        # === 3. December 31 and January 1 with a 58 reading (automatic and dated) ===
        p_ny = root / "newyear" / "picks"
        p_ny.mkdir(parents=True)
        dec31_eve = REAL_DATETIME(2027, 12, 31, 20, 0, tzinfo=ET)
        _Clock.at = dec31_eve
        assert walk.install_state(p_ny, dt.date(2027, 12, 31), walk.State("automatic", 58, 60, "used", 0.96)) == []

        dec31_last = REAL_DATETIME(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
        jan1_first = REAL_DATETIME(2028, 1, 1, 0, 0, 0, tzinfo=ET)
        _Clock.at = dec31_last
        st_dec31 = contest_state.load_decision_streak_state(p_ny, require_contest_state=True, now=dec31_last)
        fail_if((st_dec31.streak, st_dec31.best_streak, st_dec31.best_status) != (58, 60, "trusted"),
                f"3. Dec 31 23:59:59.999999 auto 58/60 failed: {st_dec31}")

        _Clock.at = jan1_first
        try:
            contest_state.load_decision_streak_state(p_ny, require_contest_state=True, now=jan1_first)
            problems.append("3. Jan 1 00:00 accepted 2027 automatic switch!")
        except contest_state.ContestStateError as exc:
            fail_if("season 2028 switch is other_season (file season 2027)" not in str(exc), f"3. Jan 1 wrong refusal: {exc}")
        st_jan1_opt = contest_state.load_decision_streak_state(p_ny, require_contest_state=False, now=jan1_first)
        fail_if(st_jan1_opt.best_status != "untrusted", f"3. Jan 1 2027 best_streak was still trusted: {st_jan1_opt}")

        # Dated 58/60 written at 23:59 on 2027-12-31: wins at 23:59:59.999999, expires at 2028-01-01 00:00:00
        _Clock.at = REAL_DATETIME(2027, 12, 31, 23, 59, 0, tzinfo=ET)
        assert runner.invoke(cli.cli, [
            "set-contest-streak", "--picks-dir", str(p_ny),
            "--date", "2027-12-31", "--streak", "58", "--best-streak", "60", "--saver", "active",
        ]).exit_code == 0
        st_d_dec31 = contest_state.load_decision_streak_state(p_ny, require_contest_state=True, now=dec31_last)
        fail_if((st_d_dec31.observation_kind, st_d_dec31.streak, st_d_dec31.saver_available) != ("dated", 58, True),
                f"3. Dec 31 dated 58/60 failed: {st_d_dec31}")
        _Clock.at = jan1_first
        try:
            contest_state.load_decision_streak_state(p_ny, require_contest_state=True, now=jan1_first)
            problems.append("3. Jan 1 accepted expired Dec 31 dated 58/60!")
        except contest_state.ContestStateError as exc:
            fail_if("other_season" not in str(exc), f"3. Jan 1 after dated expiry wrong refusal: {exc}")
        _Clock.at = now
        notes.append("3. Dec 31 / Jan 1 year rollover at 58/60 verified (both automatic and dated).")

        # === 4. Failed send, unconfirmed send, and restart at 58 ===
        p_send = root / "send58" / "picks"
        p_send.mkdir(parents=True)
        assert walk.install_state(p_send, day, walk.State("dated", 58, 58, "used", 0.96)) == []
        frame = walk._predictions(day, 0.96).head(2)
        games = [int(g) for g in frame["game_pk"]]
        statuses = {g: {"abstract": "P", "detailed": "Scheduled"} for g in games}
        cfg = walk._config(p_send)
        cfg["orchestrator"]["heartbeat_path"] = str(root / "send58" / ".hb")
        cfg["bluesky"] = {"dm_recipient": "synthetic"}

        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            sel = scheduler.run_single_check(day.isoformat(), games, set(), cfg, 0.01)["selection"]

        saved = picks.load_pick(day.isoformat(), p_send)
        s_state = scheduler.SchedulerState(day.isoformat(), None, [], [], [], False, None, None, None)

        # 4a) Failed send clears delivery_attempted, no commit, retryable!
        with ExitStack() as st:
            st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games]))
            st.enter_context(patch("bts.dm.send_dm", side_effect=RuntimeError("transient dm failure")))
            ok_fail = scheduler._deliver_and_lock_pick(saved, cfg, p_send, s_state, day.isoformat(), "lineup", selection=sel)
        reloaded_after_fail = picks.load_pick(day.isoformat(), p_send)
        fail_if(
            ok_fail is not False
            or reloaded_after_fail.delivery_attempted is not False
            or s_state.pick_locked is not False
            or daily_decision.load_decision(day.isoformat(), p_send) is not None,
            "4a failed send did not clear attempt cleanly",
        )

        # 4b) Unconfirmed send (crash mid-send while delivery_attempted=True): restart locks locked_unconfirmed without resending pick, pages once!
        reloaded_after_fail.delivery_attempted = True
        picks.save_pick(reloaded_after_fail, p_send)
        unknown_dms = []
        s_restart = scheduler.SchedulerState(day.isoformat(), None, [], [], [], False, None, None, None)
        with ExitStack() as st:
            st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games]))
            st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (unknown_dms.append(m) or "dm-unk-1")))
            ok_unconf = scheduler._deliver_and_lock_pick(
                picks.load_pick(day.isoformat(), p_send), cfg, p_send, s_restart, day.isoformat(), "restart", selection=None,
            )
        rec_unconf = daily_decision.load_decision(day.isoformat(), p_send)
        fail_if(
            ok_unconf is not False
            or s_restart.pick_locked is not True
            or rec_unconf["delivery_status"] != "locked_unconfirmed"
            or rec_unconf["scoreable"] is not True
            or rec_unconf["objective"] != "longest_streak"
            or rec_unconf["best_streak"] != 58
            or len(unknown_dms) != 1
            or "DELIVERY OUTCOME UNKNOWN" not in unknown_dms[0],
            f"4b unconfirmed restart unexpected: ok={ok_unconf} rec={rec_unconf} dms={unknown_dms}",
        )
        notes.append("4. Failed send, unconfirmed send, and restart at 58 verified.")

        # === 5. Rollback to base (36fd3e35) with 58 in active/best, expired dated, pending chosen_on, and committed pick ===
        p_rb = root / "rollback" / "picks"
        p_rb.mkdir(parents=True)
        # Create a pending pick at dated 12/58 (active<=57, best=58!) and another dir for committed 58/60
        assert walk.install_state(p_rb, day, walk.State("dated", 12, 58, "used", 0.96)) == []
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            scheduler.run_single_check(day.isoformat(), games, set(), walk._config(p_rb), 0.01)

        # Also create an already-committed 58/60 pick on 2027-07-15 in p_rb_committed
        p_rb_comm = root / "rollback_comm" / "picks"
        p_rb_comm.mkdir(parents=True)
        assert walk.install_state(p_rb_comm, day, walk.State("dated", 58, 60, "used", 0.96)) == []
        cfg_comm = walk._config(p_rb_comm)
        cfg_comm["bluesky"] = {"dm_recipient": "synthetic"}
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
            st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games]))
            st.enter_context(patch("bts.dm.send_dm", return_value="dm-c"))
            sel_c = scheduler.run_single_check(day.isoformat(), games, set(), cfg_comm, 0.01)["selection"]
            sc_st = scheduler.SchedulerState(day.isoformat(), None, [], [], [], False, None, None, None)
            assert scheduler._deliver_and_lock_pick(
                picks.load_pick(day.isoformat(), p_rb_comm), cfg_comm, p_rb_comm, sc_st, day.isoformat(), "lineup", selection=sel_c
            )
        picks.save_streak(58, p_rb_comm, saver_available=False, season=2027)
        # Replace observation in p_rb_comm with valid 5/5? Even WITHOUT touching contest_streak.manual.json (58/60 on disk!),
        # check whether base's check-results STILL grades the already-committed pick!

        rb_script = f"""
import datetime as dt, json
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
import bts
from bts import cli, contest_state, daily_decision, picks

ET = ZoneInfo("America/New_York")
now = dt.datetime(2027, 7, 15, 12, 0, tzinfo=ET)
next_day = dt.datetime(2027, 7, 16, 12, 0, tzinfo=ET)
p_rb = Path({str(p_rb)!r})
p_comm = Path({str(p_rb_comm)!r})
out = {{"bts_file": bts.__file__}}

# (a) Active dated 12/58 (active<=57, best=58) refused by base check 6!
try:
    contest_state.load_decision_streak_state(p_rb, require_contest_state=True, now=now)
    out["dated_12_58"] = "accepted"
except contest_state.ContestStateError as e:
    out["dated_12_58"] = str(e)

# (b) Next day (2027-07-16): dated 12/58 is EXPIRED, put valid covered automatic 5/5 beside it:
sw = {{"schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
       "activation_id": "a"*32, "switched_at": "2027-07-01T00:00:00Z", "revoked_at": None, "note": None}}
(p_rb / "account_state/season_contract.json").write_text(json.dumps(sw))
auto5 = {{"schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "a"*32,
          "active_streak": 5, "best_streak": 5, "source": "mlb_bts_profile",
          "source_date": "2027-07-15", "recorded_at": "2027-07-16T14:00:00Z"}}
(p_rb / "account_state/contest_streak.json").write_text(json.dumps(auto5))
(p_rb / "account_state/saver_state.json").write_text(json.dumps({{"season": 2027, "state": "not_earned"}}))
try:
    contest_state.load_decision_streak_state(p_rb, require_contest_state=True, now=next_day)
    out["expired_dated_blocks_auto"] = "accepted"
except contest_state.ContestStateError as e:
    out["expired_dated_blocks_auto"] = str(e)

# (c) Archive incompatible manual file -> valid automatic 5/5 now works on base,
#     AND check pending pick chosen at 12/58: loads with chosen_on=None, refused by saved_pick_refusal!
(p_rb / "account_state/contest_streak.manual.json").unlink()
st_ok = contest_state.load_decision_streak_state(p_rb, require_contest_state=True, now=now)
out["after_archive_streak"] = st_ok.streak
loaded_pending = picks.load_pick("2027-07-15", p_rb)
out["pending_chosen_on"] = loaded_pending.chosen_on
out["pending_refusal"] = picks.saved_pick_refusal(loaded_pending, st_ok)

# (d) Already committed pick at 58/60 remains scoreable on base even with 58/60 still in manual file!
out["comm_scoreable"] = daily_decision.is_scoreable_commit("2027-07-15", p_comm, picks.load_pick("2027-07-15", p_comm))
with patch.object(cli, "_now_et", lambda: dt.datetime(2027, 7, 16, 1, 0, tzinfo=ET)), \\
     patch.object(picks, "get_game_statuses_detailed", lambda *a: {{70000: {{"abstract": "F", "detailed": "Final", "code": "F"}}, 70001: {{"abstract": "F", "detailed": "Final", "code": "F"}}}}), \\
     patch.object(picks, "check_hit", lambda *a, **kw: "hit"):
    r = CliRunner().invoke(cli.cli, ["check-results", "--date", "2027-07-15", "--picks-dir", str(p_comm)])
out["comm_grade_exit"] = r.exit_code
out["comm_new_streak"] = picks.load_streak(p_comm, season=2027)
print(json.dumps(out))
"""
        proc = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", rb_script],
            env=dict(os.environ, PYTHONPATH=str(BASE_SRC)),
            capture_output=True, text=True, check=True, timeout=20,
        )
        rb_out = json.loads(proc.stdout)
        fail_if("base_tree" not in rb_out["bts_file"], f"5. rollback did not run base_tree: {rb_out}")
        fail_if("check 6, best_streak: must be an integer from active_streak to 57" not in rb_out["dated_12_58"],
                f"5a base did not refuse 12/58 at check 6: {rb_out['dated_12_58']}")
        fail_if("check 6, best_streak: must be an integer from active_streak to 57" not in rb_out["expired_dated_blocks_auto"],
                f"5b expired dated 12/58 did not block valid auto 5/5: {rb_out['expired_dated_blocks_auto']}")
        fail_if(
            rb_out["after_archive_streak"] != 5
            or rb_out["pending_chosen_on"] is not None
            or "there is no valid record of what it was chosen on" not in (rb_out["pending_refusal"] or ""),
            f"5c pending pick after archive unexpected: {rb_out}",
        )
        fail_if(rb_out["comm_scoreable"] is not True or rb_out["comm_grade_exit"] != 0 or rb_out["comm_new_streak"] != 59,
                f"5d committed pick scoring on base failed: {rb_out}")
        notes.append("5. Rollback to base verified (12/58 best check, expired dated blocking auto, pending chosen_on=None refusal, committed pick still scoreable 58->59).")

    out_lines = [*notes, f"problems={len(problems)}"]
    for p in problems[:50]:
        out_lines.append(f"PROBLEM: {p}")
    text = "\n".join(out_lines) + "\n"
    (OUT_DIR / "item6_standing_rows.log").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

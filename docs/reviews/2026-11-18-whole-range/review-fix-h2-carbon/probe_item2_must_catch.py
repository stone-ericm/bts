"""Item 2 adversarial probe: rebuild K1..K10 as amended through real loaders, fetch, CLI, pick loaders, and delivery refusal."""
from __future__ import annotations

import datetime as dt
import json
import socket
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
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
    from bts import cli, contest_fetch, contest_state, orchestrator, picks, scheduler
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

    with pinned_offline(now), tempfile.TemporaryDirectory(prefix="h2c-item2-") as tmp:
        root = Path(tmp)

        # --- K1, K2, K3, K9: Dated observations through real loaders (live + historical), precedence, switch 1..5, saver ---
        d_picks = root / "k1_k3" / "picks"
        d_picks.mkdir(parents=True)
        _Clock.at = now
        r = runner.invoke(cli.cli, [
            "set-contest-streak", "--picks-dir", str(d_picks),
            "--date", "2027-07-15", "--streak", "400", "--best-streak", "400", "--saver", "used",
        ])
        fail_if(r.exit_code != 0, f"K3 400/400 CLI failed: {r.output}")
        man_path = d_picks / "account_state" / "contest_streak.manual.json"
        good_dated = json.loads(man_path.read_text())

        # Precedence: put an invalid 401/401 automatic file and corrupt saver_state behind valid dated 400/400
        auto_path = d_picks / "account_state" / "contest_streak.json"
        auto_path.write_text(json.dumps({"active_streak": 401, "best_streak": 401, "source_date": "2027-07-14"}))
        (d_picks / "account_state" / "saver_state.json").write_text("{corrupt")
        live_st = contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
        fail_if(
            (live_st.streak, live_st.best_streak, live_st.best_status, live_st.observation_kind, live_st.saver_available)
            != (400, 400, "trusted", "dated", False),
            f"K3/K9 dated 400/400 precedence failed: {live_st}",
        )
        fail_if(health_cs.check(d_picks, expected=True, now=now) != [], "K9 live health on valid dated 400/400 alerted")
        fail_if(
            health_cs.check(d_picks, expected=True, now=REAL_DATETIME(2027, 7, 16, 0, 5, tzinfo=ET), historical_day=day) != [],
            "K9 historical health on valid dated 400/400 alerted",
        )

        # Now test K1, K2, K3 refusals through load_decision_streak_state (req=False & True) AND health_cs.check (live & historical)
        bad_dated_cases = [
            (5, "active_streak", -1, "must be an integer from 0 to 400"),
            (5, "active_streak", True, "must be an integer from 0 to 400"),
            (5, "active_streak", False, "must be an integer from 0 to 400"),
            (5, "active_streak", "12", "must be an integer from 0 to 400"),
            (5, "active_streak", 12.0, "must be an integer from 0 to 400"),
            (5, "active_streak", None, "must be an integer from 0 to 400"),
            (5, "active_streak", "__DEL__", "must be an integer from 0 to 400"),
            (5, "active_streak", [12], "must be an integer from 0 to 400"),
            (5, "active_streak", {"v": 12}, "must be an integer from 0 to 400"),
            (5, "active_streak", 401, "must be an integer from 0 to 400"),
            (6, "best_streak", 399, "must be an integer from active_streak to 400"),
            (6, "best_streak", -1, "must be an integer from active_streak to 400"),
            (6, "best_streak", True, "must be an integer from active_streak to 400"),
            (6, "best_streak", "400", "must be an integer from active_streak to 400"),
            (6, "best_streak", 400.0, "must be an integer from active_streak to 400"),
            (6, "best_streak", None, "must be an integer from active_streak to 400"),
            (6, "best_streak", "__DEL__", "must be an integer from active_streak to 400"),
            (6, "best_streak", [400], "must be an integer from active_streak to 400"),
            (6, "best_streak", 401, "must be an integer from active_streak to 400"),
        ]
        for chk_no, field, val, msg_tail in bad_dated_cases:
            body = dict(good_dated)
            if field == "best_streak" and val == 401:
                body["active_streak"] = 57
            if val == "__DEL__":
                del body[field]
            else:
                body[field] = val
            man_path.write_text(json.dumps(body))
            for req in (False, True):
                try:
                    contest_state.load_decision_streak_state(d_picks, require_contest_state=req, now=now)
                    problems.append(f"K1-K3 failed to refuse {field}={val!r} req={req}")
                except contest_state.ContestStateError as exc:
                    fail_if(f"check {chk_no}, {field}: {msg_tail}" not in str(exc), f"K1-K3 wrong msg: {exc}")
            live_alerts = health_cs.check(d_picks, expected=True, now=now)
            fail_if(
                len(live_alerts) != 1 or live_alerts[0].level != "CRITICAL" or f"check {chk_no}, {field}: {msg_tail}" not in live_alerts[0].message,
                f"K1-K3 live health mismatch on {field}={val!r}: {live_alerts}",
            )
            hist_alerts = health_cs.check(d_picks, expected=True, now=REAL_DATETIME(2027, 7, 16, 0, 5, tzinfo=ET), historical_day=day)
            expected_hist = f"The dated manual contest streak file failed validation for {field}."
            fail_if(
                len(hist_alerts) != 1 or expected_hist not in hist_alerts[0].message or "`bts " in hist_alerts[0].message,
                f"K1-K3 historical health mismatch on {field}={val!r}: {hist_alerts}",
            )

        # K9 switch refusal reasons 1..5 (live & historical) + saver rules at streak 58
        man_path.unlink()
        auto_path.unlink()
        (d_picks / "account_state" / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "used"}))
        sw_path = d_picks / "account_state" / "season_contract.json"

        # Reason 3: switch absent
        try:
            contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
            problems.append("K9 switch reason 3 absent not refused")
        except contest_state.ContestStateError as exc:
            fail_if("season 2027 switch is absent" not in str(exc), f"K9 reason 3 wrong text: {exc}")

        # Reason 1: corrupt switch
        sw_path.write_text("{bad")
        try:
            contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
            problems.append("K9 switch reason 1 corrupt not refused")
        except contest_state.ContestStateError as exc:
            fail_if("corrupt season switch, check 1" not in str(exc), f"K9 reason 1 wrong text: {exc}")

        # Reason 4: switch automatic, no automatic observation
        sw_path.unlink()
        runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(d_picks)])
        sw_snap = json.loads(sw_path.read_text())
        try:
            contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
            problems.append("K9 switch reason 4 missing auto not refused")
        except contest_state.ContestStateError as exc:
            fail_if("no automatic observation for this activation" not in str(exc), f"K9 reason 4 wrong text: {exc}")

        # Reason 5: automatic observation at 58/58 with wrong activation_id / wrong season / wrong schema
        for override, snippet in (
            ({"schema_version": "bts_contest_streak_auto_v1"}, "schema_version is 'bts_contest_streak_auto_v1'"),
            ({"season": 2026}, "automatic observation season is 2026, not 2027"),
            ({"activation_id": "f" * 32}, "not the switch's"),
        ):
            payload = {
                "schema_version": "bts_contest_streak_auto_v2",
                "active_streak": 58, "best_streak": 58,
                "source_date": "2027-07-14", "recorded_at": "2027-07-15T14:00:00Z",
                "source": "mlb_bts_profile", "season": 2027,
                "activation_id": sw_snap["activation_id"],
                **override,
            }
            auto_path.write_text(json.dumps(payload))
            try:
                contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
                problems.append(f"K9 switch reason 5 {override} not refused")
            except contest_state.ContestStateError as exc:
                fail_if(snippet not in str(exc), f"K9 reason 5 wrong text: {exc}")

        # Reason 2: unexpired legacy manual override at 58/58 wins in load_contest_streak_state over corrupt 401 auto,
        # but is refused for live decision!
        auto_path.write_text(json.dumps({"active_streak": 401, "best_streak": 401}))
        man_path.write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 58, "best_streak": 58,
            "source_date": "2027-07-15", "override_expires_at": "2027-07-16T00:00:00Z",
        }))
        opt_st = contest_state.load_decision_streak_state(d_picks, require_contest_state=False, now=now)
        fail_if((opt_st.streak, opt_st.best_streak, opt_st.best_status) != (58, 58, "trusted"), f"K9 legacy opt_st wrong: {opt_st}")
        try:
            contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
            problems.append("K9 switch reason 2 legacy manual not refused on live path")
        except contest_state.ContestStateError as exc:
            fail_if("selected legacy manual override is no longer accepted" not in str(exc), f"K9 reason 2 wrong text: {exc}")
        man_path.unlink()

        # Saver rules at streak 58: active -> True, not_earned -> False, used -> False
        for sav, expected_avail in (("active", True), ("not_earned", False), ("used", False)):
            runner.invoke(cli.cli, [
                "set-contest-streak", "--picks-dir", str(d_picks),
                "--date", "2027-07-15", "--streak", "58", "--best-streak", "58", "--saver", sav,
            ])
            st58 = contest_state.load_decision_streak_state(d_picks, require_contest_state=True, now=now)
            fail_if(st58.saver_available is not expected_avail, f"K9 saver {sav} at 58 gave {st58.saver_available}")
        man_path.unlink()
        notes.append("K1-K3, K9 verified (live + historical, precedence, switch 1..5, saver).")

        # --- K4: Automatic file reads (live + historical, require_contest_state False + True) ---
        k4_picks = root / "k4" / "picks"
        k4_picks.mkdir(parents=True)
        runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(k4_picks)])
        k4_sw = json.loads((k4_picks / "account_state" / "season_contract.json").read_text())
        (k4_picks / "account_state" / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "used"}))
        k4_auto = k4_picks / "account_state" / "contest_streak.json"
        base_auto = {
            "schema_version": "bts_contest_streak_auto_v2",
            "active_streak": 400, "best_streak": 400,
            "source_date": "2027-07-14", "recorded_at": "2027-07-15T14:00:00Z",
            "source": "mlb_bts_profile", "season": 2027,
            "activation_id": k4_sw["activation_id"],
        }
        k4_auto.write_text(json.dumps(base_auto))
        ok400 = contest_state.load_decision_streak_state(k4_picks, require_contest_state=True, now=now)
        fail_if((ok400.streak, ok400.best_streak, ok400.best_status) != (400, 400, "trusted"), f"K4 400/400 failed: {ok400}")

        # Absent/null/non-int best_streak read as None (missing) when active=400
        for non_int_best in (None, True, False, "400", 400.0, [400], {"b": 400}, "__DEL__"):
            body = dict(base_auto)
            if non_int_best == "__DEL__":
                del body["best_streak"]
            else:
                body["best_streak"] = non_int_best
            k4_auto.write_text(json.dumps(body))
            st_m = contest_state.load_decision_streak_state(k4_picks, require_contest_state=True, now=now)
            fail_if((st_m.streak, st_m.best_streak, st_m.best_status) != (400, None, "missing"), f"K4 non-int best {non_int_best!r} wrong: {st_m}")

        # Refused active / best combinations across live (req=False, True) and historical health
        k4_bad = [
            ({"active_streak": -1, "best_streak": 10}, "invalid streak"),
            ({"active_streak": True, "best_streak": 10}, "invalid streak"),
            ({"active_streak": "10", "best_streak": 10}, "invalid streak"),
            ({"active_streak": 10.0, "best_streak": 10}, "invalid streak"),
            ({"active_streak": None, "best_streak": 10}, "invalid streak"),
            ({"active_streak": "__DEL__", "best_streak": 10}, "invalid streak"),
            ({"active_streak": 401, "best_streak": None}, "invalid streak or best streak"),
            ({"active_streak": 401, "best_streak": "__DEL__"}, "invalid streak or best streak"),
            ({"active_streak": 401, "best_streak": 401}, "invalid streak or best streak"),
            ({"active_streak": 400, "best_streak": 399}, "invalid streak or best streak"),
            ({"active_streak": 0, "best_streak": -1}, "invalid streak or best streak"),
            ({"active_streak": 58, "best_streak": 401}, "invalid streak or best streak"),
        ]
        for upd, hist_snip in k4_bad:
            body = dict(base_auto)
            for fk, fv in upd.items():
                if fv == "__DEL__":
                    body.pop(fk, None)
                else:
                    body[fk] = fv
            k4_auto.write_text(json.dumps(body))
            for req in (False, True):
                try:
                    contest_state.load_decision_streak_state(k4_picks, require_contest_state=req, now=now)
                    problems.append(f"K4 failed to refuse {upd} req={req}")
                except contest_state.ContestStateError:
                    pass
            ha = health_cs.check(k4_picks, expected=True, now=REAL_DATETIME(2027, 7, 16, 0, 5, tzinfo=ET), historical_day=day)
            fail_if(len(ha) != 1 or hist_snip not in ha[0].message, f"K4 hist health mismatch {upd}: {ha}")
        notes.append("K4 verified (live + historical, non-int best as missing, bounds).")

        # --- K5 & K10: Fetch (dry-run & real; with & without DM recipient; inside & outside cooldown; 57->58->401) ---
        k10_picks = root / "k10" / "picks"
        k10_picks.mkdir(parents=True)
        frame = walk._predictions(day, 0.96).head(2)
        games = list(frame["game_pk"])
        statuses = {int(g): {"abstract": "P", "detailed": "Scheduled"} for g in games}
        cfg = walk._config(k10_picks)
        cfg["orchestrator"]["heartbeat_path"] = str(root / "k10" / ".heartbeat")
        cfg["bluesky"] = {"dm_recipient": "synthetic"}

        # Seed accepted 57/57 at 09:00 ET (3 hours ago)
        _Clock.at = REAL_DATETIME(2027, 7, 15, 9, 0, tzinfo=ET)
        assert walk.install_state(k10_picks, day, walk.State("automatic", 57, 57, "not_earned", 0.96)) == []
        _Clock.at = now
        obs_file = k10_picks / "account_state" / "contest_streak.json"
        led_file = k10_picks / "account_state" / "contest_ledger.jsonl"
        sav_file = k10_picks / "account_state" / "saver_state.json"
        st_file = k10_picks.parent / "health_state" / "contest_streak_fetch_status.json"
        dms_sent: list[tuple[str, str]] = []

        def invoke_fetch(a, b, *, recipient: str | None = "synthetic", dry_run: bool = False):
            session = SimpleNamespace(user_id=walk.UID, username=walk.ACCOUNT, xsid="walk_xsid")
            prof = {"activeStreak": a, "seasonBestStreak": b, "predictions": [{"roundId": 1, "result": "hit"}]}
            with ExitStack() as st:
                st.enter_context(patch.object(auth, "load_session_cookies", lambda: {}))
                st.enter_context(patch.object(auth, "extract_uid", lambda c: walk.UID))
                st.enter_context(patch.object(auth, "fetch_login_session", lambda **kw: session))
                st.enter_context(patch.object(contest_fetch, "fetch_profile", lambda *args: prof))
                st.enter_context(patch.object(cli, "_fetch_rounds", lambda: {1: day - dt.timedelta(days=1)}))
                st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (dms_sent.append((r, m)) or "dm-ok")))
                argv = ["fetch-contest-streak", "--picks-dir", str(k10_picks), "--expected-username", walk.ACCOUNT]
                if recipient:
                    argv.extend(["--dm-recipient", recipient])
                if dry_run:
                    argv.append("--dry-run")
                res = runner.invoke(cli.cli, argv)
                return res.exit_code, res.output

        # Dry-run at 58/58 does not write
        before_obs = obs_file.read_bytes()
        before_led = led_file.read_bytes()
        c_dry, out_dry = invoke_fetch(58, 58, dry_run=True)
        fail_if(c_dry != 0 or "[dry-run]" not in out_dry or obs_file.read_bytes() != before_obs or led_file.read_bytes() != before_led,
                f"K5/K10 dry-run 58/58 modified state: code={c_dry}")

        # Real fetch at 58/58 replaces 57/57, appends ledger, auto-earns saver (not_earned -> active since best 58 >= 10)
        c_ok, _ = invoke_fetch(58, 58, dry_run=False)
        fail_if(c_ok != 0 or dms_sent != [], f"K10 58/58 fetch failed or paged: code={c_ok} dms={dms_sent}")
        fail_if(len(led_file.read_text().splitlines()) != 2, "K10 ledger did not gain 1 row")
        fail_if(json.loads(sav_file.read_text())["state"] != "active", "K10 58/58 did not auto-earn saver")
        st_58 = contest_state.load_decision_streak_state(k10_picks, require_contest_state=True, now=now)
        fail_if((st_58.streak, st_58.best_streak, st_58.best_status) != (58, 58, "trusted"), f"K10 loader after 58/58: {st_58}")

        # Selection at 58 uses longest_streak
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            sel58 = scheduler.run_single_check(str(day), games, set(), cfg, 0.01)["selection"]
        fail_if(sel58.streak != 58 or sel58.decision.objective != "longest_streak", f"K10 selection wrong: {sel58}")

        snap_obs = obs_file.read_bytes()
        snap_led = led_file.read_bytes()
        snap_sav = sav_file.read_bytes()

        # K5/K10 failure WITHOUT recipient: exits 2, writes status, does NOT consume cooldown!
        for dry_flag in (True, False):
            c_no_rec, out_no_rec = invoke_fetch(401, 401, recipient=None, dry_run=dry_flag)
            fail_if(c_no_rec != 2 or "seasonBestStreak must be <= 400" not in out_no_rec, f"K10 401 no-rec dry={dry_flag}: {c_no_rec}")
            fail_if(dms_sent != [], f"K10 sent DM without recipient: {dms_sent}")
            fail_if(obs_file.read_bytes() != snap_obs or led_file.read_bytes() != snap_led or sav_file.read_bytes() != snap_sav,
                    "K10 401 overwrote obs/ledger/saver")
            st_doc = json.loads(st_file.read_text())
            fail_if(st_doc.get("last_alert_at") is not None or st_doc.get("last_alert_at_by_category") != {},
                    f"K10 no-recipient consumed cooldown: {st_doc}")

        # Failure WITH recipient outside cooldown: sends 1 DM and records stamp
        c_rec1, _ = invoke_fetch(57, 401, recipient="synthetic", dry_run=False)
        fail_if(c_rec1 != 2 or len(dms_sent) != 1 or "seasonBestStreak must be <= 400" not in dms_sent[0][1],
                f"K10 401 with recipient outside cooldown failed: {dms_sent}")

        # Inside cooldown (+5h59m): does NOT re-send DM
        dms_sent.clear()
        _Clock.at = now + dt.timedelta(hours=5, minutes=59)
        c_in_cd, _ = invoke_fetch(401, 400, recipient="synthetic", dry_run=False)
        fail_if(c_in_cd != 2 or dms_sent != [], f"K5/K10 inside cooldown sent DM: {dms_sent}")

        # Outside cooldown (+6h01m): DOES send DM again; and check stale reading age (>14h vs <=14h)
        _Clock.at = now + dt.timedelta(hours=6, minutes=1)
        c_out_cd, _ = invoke_fetch(401, 401, recipient="synthetic", dry_run=True)
        fail_if(c_out_cd != 2 or len(dms_sent) != 1, f"K5/K10 outside cooldown dry-run did not send DM: {dms_sent}")

        # Stale reading check tied to accepted 58/58 recorded at 12:00 ET:
        # At +14h00m: still fresh (stale=False); at +14h01m: stale=True, reason="old"
        st_at_14h = contest_state.load_decision_streak_state(k10_picks, require_contest_state=True, now=now + dt.timedelta(hours=14))
        ar_14h = contest_state.automatic_reading_record(st_at_14h, now + dt.timedelta(hours=14))
        st_past_14h = contest_state.load_decision_streak_state(k10_picks, require_contest_state=True, now=now + dt.timedelta(hours=14, minutes=1))
        ar_past_14h = contest_state.automatic_reading_record(st_past_14h, now + dt.timedelta(hours=14, minutes=1))
        fail_if(ar_14h["stale"] is not False or ar_past_14h["stale"] is not True or ar_past_14h["reason"] != "old",
                f"K10 stale boundary wrong: 14h={ar_14h}, 14h01m={ar_past_14h}")
        _Clock.at = now
        notes.append("K5 & K10 verified (dry-run, real, recipient on/off, cooldown in/out, 14h stale boundary).")

        # --- K6: CLI rejection before writes (including preexisting files untouched) ---
        k6_picks = root / "k6" / "picks"
        k6_Empty = root / "k6_empty"
        k6_Empty.mkdir()
        for args in (
            ["--date", "2027-07-15", "--streak", "12", "--best-streak", "11", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "401", "--best-streak", "400", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "-1", "--best-streak", "12", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "57", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "401", "--best-streak", "401", "--saver", "used"],
            ["--date", "2027-07-14", "--streak", "58", "--best-streak", "58", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "58", "--saver", "used", "--source", "x"],
            ["--date", "2027-07-15", "--streak", "58", "--saver", "used"],
            ["--date", "2027-07-15", "--streak", "58", "--best-streak", "58"],
            ["--date", "2027-07-15", "--streak", "bad", "--best-streak", "58", "--saver", "used"],
        ):
            r6 = runner.invoke(cli.cli, ["set-contest-streak", "--picks-dir", str(k6_Empty), *args])
            fail_if(r6.exit_code != 2 or list(k6_Empty.iterdir()) != [], f"K6 failed on {args}: code={r6.exit_code}")
        notes.append("K6 verified.")

        # --- K7: All 4 pick loaders + saved_pick_refusal ---
        k7_picks = root / "k7" / "picks"
        k7_picks.mkdir(parents=True)
        runner.invoke(cli.cli, [
            "set-contest-streak", "--picks-dir", str(k7_picks),
            "--date", "2027-07-15", "--streak", "400", "--best-streak", "400", "--saver", "used",
        ])
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            scheduler.run_single_check(str(day), games, set(), walk._config(k7_picks), 0.01)

        pick_file = k7_picks / f"{day}.json"
        shadow_file = k7_picks / f"{day}.shadow.json"
        good_pick_json = json.loads(pick_file.read_text())
        good_chosen = dict(good_pick_json["chosen_on"])
        live_400 = contest_state.load_decision_streak_state(k7_picks, require_contest_state=True, now=now)

        # Valid 400/400 dated and 400 automatic across all 4 loaders
        shadow_file.write_text(json.dumps(good_pick_json))
        for loader_name, loaded in (
            ("parse_daily_pick", picks.parse_daily_pick(pick_file.read_text())),
            ("load_pick", picks.load_pick(str(day), k7_picks)),
            ("load_pick_bytes", picks.load_pick_bytes(str(day), k7_picks)[0]),
            ("load_shadow_pick", picks.load_shadow_pick(str(day), k7_picks)),
        ):
            fail_if(loaded is None or loaded.chosen_on != good_chosen, f"K7 {loader_name} dropped valid 400 chosen_on")
            fail_if(picks.saved_pick_refusal(loaded, live_400) is not None, f"K7 {loader_name} refused matching 400 pick")

        # Invalid chosen_on mutations must load as None across ALL 4 loaders and refuse delivery with "no valid record"
        invalid_chosen_variants = [
            dict(good_chosen, streak=401, dated=dict(good_chosen["dated"], active_streak=401, best_streak=401)),
            dict(good_chosen, dated=dict(good_chosen["dated"], best_streak=401)),
            dict(good_chosen, dated=dict(good_chosen["dated"], best_streak=399)),
            dict(good_chosen, dated=dict(good_chosen["dated"], active_streak=399)),
            dict(good_chosen, dated=dict(good_chosen["dated"], effective_date="2027-7-15")),
            dict(good_chosen, saver_available=True),  # contradicts used saver at 400
            dict(good_chosen, required=1),
            dict(good_chosen, extra_key=1),
            {
                "schema": "bts_pick_chosen_on_v1", "required": True, "kind": "automatic",
                "streak": 401, "saver_available": False, "dated": None,
            },
            {
                "schema": "bts_pick_chosen_on_v1", "required": True, "kind": "automatic",
                "streak": 400, "saver_available": False, "dated": good_chosen["dated"],
            },
        ]
        for idx_v, bad_c in enumerate(invalid_chosen_variants):
            mut_json = json.dumps(dict(good_pick_json, chosen_on=bad_c))
            pick_file.write_text(mut_json)
            shadow_file.write_text(mut_json)
            for loader_name, loaded in (
                ("parse_daily_pick", picks.parse_daily_pick(mut_json)),
                ("load_pick", picks.load_pick(str(day), k7_picks)),
                ("load_pick_bytes", picks.load_pick_bytes(str(day), k7_picks)[0]),
                ("load_shadow_pick", picks.load_shadow_pick(str(day), k7_picks)),
            ):
                fail_if(loaded.chosen_on is not None, f"K7 variant {idx_v} accepted in {loader_name}: {bad_c}")
                refusal = picks.saved_pick_refusal(loaded, live_400)
                fail_if(refusal is None or "there is no valid record of what it was chosen on" not in refusal,
                        f"K7 variant {idx_v} wrong refusal: {refusal}")
        notes.append("K7 verified across all 4 pick loaders and saved_pick_refusal.")

        # --- K8: classify_best_streak via legacy manual >400 and tail policy non-authorization ---
        from bts import strategy
        mdp = strategy._load_mdp()
        # Late season Sept 25 (2 days left), streak=10: max reachable = 14.
        # If best=401 were trusted, effective_best=57 -> tail stops (skip)!
        # Because best=401 is untrusted, _normalize_best degrades to None -> effective_best=10 -> tail PLAYS!
        man_401 = k4_picks / "account_state" / "contest_streak.manual.json"
        man_401.write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 10, "best_streak": 401,
            "source_date": "2027-09-25", "override_expires_at": "2027-09-26T12:00:00Z",
        }))
        sept25 = REAL_DATETIME(2027, 9, 25, 12, 0, tzinfo=ET)
        st_untrusted = contest_state.load_decision_streak_state(k4_picks, require_contest_state=False, now=sept25)
        fail_if((st_untrusted.best_streak, st_untrusted.best_status) != (401, "untrusted"),
                f"K8 legacy manual 401 was not untrusted: {st_untrusted}")
        ctx_401 = strategy.DecisionContext(0.85, 0.84, True, "2027-09-25", True, mdp, st_untrusted.best_streak, st_untrusted.best_status)
        dec_401 = strategy.resolve_policy_decision(ctx_401, 10, False)
        ctx_400_trusted = strategy.DecisionContext(0.85, 0.84, True, "2027-09-25", True, mdp, 400, "trusted")
        dec_400 = strategy.resolve_policy_decision(ctx_400_trusted, 10, False)
        fail_if(dec_401.action == "skip", f"K8 untrusted 401 authorized tail stop! {dec_401}")
        fail_if(dec_400.action != "skip", f"K8 trusted 400 did not stop in tail! {dec_400}")
        notes.append("K8 verified (401 untrusted never authorizes tail stop; 400 trusted does).")

    out_lines = [*notes, f"problems={len(problems)}"]
    for p in problems[:50]:
        out_lines.append(f"PROBLEM: {p}")
    text = "\n".join(out_lines) + "\n"
    (OUT_DIR / "item2_must_catch.log").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Items 4 & 5 adversarial probe: exhaustive policy check with best in 58..400, plus typo / given-up cases."""
from __future__ import annotations

import datetime as dt
import json
import socket
import sys
import tempfile
from contextlib import ExitStack, contextmanager
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
    at: REAL_DATETIME = REAL_DATETIME(2027, 9, 26, 12, 0, tzinfo=ET)

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
    import bts.web

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
    from bts import cli, contest_fetch, contest_state, daily_decision, orchestrator, picks, saver_state, scheduler, strategy, web
    from bts.health import contest_state as health_cs
    from bts.leaderboard import auth
    from scripts.audit.season_walk import walk

    problems: list[str] = []
    notes: list[str] = []

    def fail_if(cond: bool, msg: str):
        if cond:
            problems.append(msg)

    # === ITEM 4: Policy over streaks 0..56 (and 57..400), days 0..185, savers, bin edges ===
    mdp = strategy._load_mdp()
    assert mdp is not None and mdp.get("policy_table") is not None and mdp.get("tail") is not None
    edges = [float(x) for x in mdp["boundaries"]]
    probs = sorted({
        0.0, 0.50, 0.72, 0.799999, 0.80, 0.96, 1.0,
        *[e - 1e-6 for e in edges],
        *edges,
        *[e + 1e-6 for e in edges],
    })
    finale_end = dt.date(2027, 9, 27)  # SEASON_END_DATES[2027]

    # Check every streak 0..56, raw days 0..185 (covers 0, 1..28 tail, 29..180 table, 181..185 clamped),
    # saver False/True, all probs, allow_double=True, has_diff_game=True:
    # Compare trusted best=57 vs trusted best in (58, 100, 400)!
    comparisons = 0
    guard_hits = 0
    tail_stops = 0
    for raw_d in range(0, 186):
        day_str = (finale_end - dt.timedelta(days=raw_d)).isoformat()
        for s in range(0, 57):
            for sav in (False, True):
                for p in probs:
                    ctx57 = strategy.DecisionContext(p, 0.85, True, day_str, True, mdp, 57, "trusted")
                    ref = strategy.resolve_policy_decision(ctx57, s, sav)
                    if ref.guard is not None:
                        guard_hits += 1
                    if ref.objective == "emax_season_best" and ref.action == "skip":
                        tail_stops += 1
                    for high_best in (58, 100, 400):
                        ctx_hi = strategy.DecisionContext(p, 0.85, True, day_str, True, mdp, high_best, "trusted")
                        got = strategy.resolve_policy_decision(ctx_hi, s, sav)
                        comparisons += 1
                        if (
                            got.action != ref.action
                            or got.policy_action != ref.policy_action
                            or got.objective != ref.objective
                            or got.source != ref.source
                            or got.effective_best != ref.effective_best
                            or got.reason != ref.reason
                            or got.pick_bar != ref.pick_bar
                            or got.guard != ref.guard
                            or got.best_status != "trusted"
                            or got.best_supplied != high_best
                        ):
                            problems.append(
                                f"Item 4 diff at s={s} d={raw_d} sav={sav} p={p} best={high_best}: ref={ref} got={got}"
                            )
                            break
                    if problems:
                        break
            if problems:
                break
        if problems:
            break

    # Also check streaks 57..400 (sampled: 57, 58, 75, 100, 399, 400) with best in (s, 100, 400),
    # raw_d in (0, 1, 2, 3, 4, 30, 180, 200), both savers, trust in ("trusted", "untrusted", None):
    high_checks = 0
    for raw_d in (0, 1, 2, 3, 4, 30, 180, 200):
        day_str = (finale_end - dt.timedelta(days=raw_d)).isoformat()
        for s in (57, 58, 75, 100, 399, 400):
            for b, b_st in ((s, "trusted"), (400, "trusted"), (401, "untrusted"), (None, None)):
                for sav in (False, True):
                    for p in (0.01, 0.72, 0.96):
                        ctx = strategy.DecisionContext(p, 0.85, True, day_str, True, mdp, b, b_st)
                        dec = strategy.resolve_policy_decision(ctx, s, sav)
                        high_checks += 1
                        if raw_d == 0:
                            want_act, want_obj = "skip", "reach57"
                        elif raw_d <= 3:
                            want_act, want_obj = "double", "longest_streak"
                        else:
                            want_act, want_obj = "single", "longest_streak"
                        fail_if(
                            (dec.action, dec.objective) != (want_act, want_obj),
                            f"Item 4 high streak s={s} d={raw_d} b={b}/{b_st}: got {dec}",
                        )

    # Also check forced fallback (mdp=None / broken artifacts) in tail with best 58/100/400 vs 57
    for raw_d in range(1, 29):
        day_str = (finale_end - dt.timedelta(days=raw_d)).isoformat()
        for s in range(0, 57):
            if s + 2 * raw_d >= 57:
                continue
            r57 = strategy.resolve_policy_decision(
                strategy.DecisionContext(0.85, 0.80, True, day_str, True, None, 57, "trusted"), s, False
            )
            r400 = strategy.resolve_policy_decision(
                strategy.DecisionContext(0.85, 0.80, True, day_str, True, None, 400, "trusted"), s, False
            )
            fail_if(
                (r57.action, r57.effective_best, r57.reason) != (r400.action, r400.effective_best, r400.reason),
                f"Item 4 forced tail mismatch s={s} d={raw_d}: {r57} vs {r400}",
            )

    notes.append(
        f"Item 4 verified: {comparisons} below-57 comparisons (guard_hits={guard_hits}, tail_stops={tail_stops}) "
        f"and {high_checks} >=57 checks; zero action differences vs best capped at 57."
    )

    # === ITEM 5: What is given up (contract typo cases + 2 of our own) ===
    runner = CliRunner()
    day_finale = dt.date(2027, 9, 26)   # 1 day left
    day_sep25 = dt.date(2027, 9, 25)    # 2 days left
    day_jul15 = dt.date(2027, 7, 15)    # 74 days left

    with pinned_offline(REAL_DATETIME(2027, 9, 26, 12, 0, tzinfo=ET)), tempfile.TemporaryDirectory(prefix="h2c-item5-") as tmp:
        root = Path(tmp)

        # Case 5.1: "--streak 75 --best-streak 57" is STILL caught by best >= active;
        # "--streak 75 --best-streak 75" (meant as 57/57) is accepted:
        p1 = root / "c51" / "picks"
        p1.mkdir(parents=True)
        r_caught = runner.invoke(cli.cli, [
            "set-contest-streak", "--picks-dir", str(p1),
            "--date", "2027-09-26", "--streak", "75", "--best-streak", "57", "--saver", "used",
        ])
        fail_if(r_caught.exit_code != 2 or "at least the active streak" not in r_caught.output,
                f"Case 5.1 75/57 not rejected: {r_caught.output}")

        assert walk.install_state(p1, day_finale, walk.State("dated", 75, 75, "used", 0.96)) == []
        frame = walk._predictions(day_finale, 0.96).head(2)
        games = [int(g) for g in frame["game_pk"]]
        statuses = {g: {"abstract": "P", "detailed": "Scheduled"} for g in games}
        cfg1 = walk._config(p1)
        cfg1["orchestrator"]["heartbeat_path"] = str(root / "c51" / ".hb")
        cfg1["bluesky"] = {"dm_recipient": "synthetic"}
        dms = []
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
            st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games]))
            st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (dms.append(m) or "dm-1")))
            sel_75 = scheduler.run_single_check("2027-09-26", games, set(), cfg1, 0.01)["selection"]
            saved_75 = picks.load_pick("2027-09-26", p1)
            st_sched = scheduler.SchedulerState("2027-09-26", None, [], [], [], False, None, None, None)
            scheduler._deliver_and_lock_pick(saved_75, cfg1, p1, st_sched, "2027-09-26", "lineup", selection=sel_75)
        dec_75 = daily_decision.load_decision("2027-09-26", p1)
        fail_if(dec_75["action"] != "double" or dec_75["streak"] != 75 or "Streak: 75" not in dms[0],
                f"Case 5.1 75/75 unexpected: dec={dec_75} dm={dms}")
        fail_if(health_cs.check(p1, expected=True, now=_Clock.at) != [], "Case 5.1 alerted on 75/75")

        # Also: "--streak 75 --best-streak 75" meant as 50/57 on weak July 15 (p=0.72) turns a reach-57 skip into H's single!
        _Clock.at = REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)
        r_50 = walk.walk_one(day_jul15, walk.State("dated", 50, 57, "used", 0.72))
        r_75_weak = walk.walk_one(day_jul15, walk.State("dated", 75, 75, "used", 0.72))
        fail_if(r_50["action"] != "skip" or r_75_weak["action"] != "single",
                f"Case 5.1b weak-slate 50->75 typo: 50={r_50['action']} 75={r_75_weak['action']}")

        # Case 5.2: Wrong best of 157:
        # (a) 57/157 at 1 day left plays double (H ignores best);
        # (b) 12/157 meant as 12/15 on Sept 25 (2 days left, max reachable 16 > 15!):
        #     12/15 PLAYS in tail, whereas 12/157 SKIPS with effective_best=57, best_streak=157 in record, and "season-best 57" in skip DM!
        r_57_157 = walk.walk_one(day_finale, walk.State("dated", 57, 157, "used", 0.96))
        r_12_15 = walk.walk_one(day_sep25, walk.State("dated", 12, 15, "used", 0.96))
        r_12_157 = walk.walk_one(day_sep25, walk.State("dated", 12, 157, "used", 0.96))
        fail_if(r_57_157["action"] != "double", f"Case 5.2a 57/157 wrong: {r_57_157}")
        fail_if(r_12_15["action"] == "skip" or r_12_157["action"] != "skip",
                f"Case 5.2b 12/15 vs 12/157 wrong: {r_12_15['action']} vs {r_12_157['action']}")
        fail_if("season-best 57 can't be beaten" not in (r_12_157["skip_dm"] or ""),
                f"Case 5.2b skip_dm wrong: {r_12_157['skip_dm']}")

        # Verify finalized end-of-day skip record for 12/157 stores best_streak=157, effective_best=57
        _Clock.at = REAL_DATETIME(2027, 9, 25, 12, 0, tzinfo=ET)
        p2 = root / "c52" / "picks"
        p2.mkdir(parents=True)
        assert walk.install_state(p2, day_sep25, walk.State("dated", 12, 157, "used", 0.96)) == []
        frame25 = walk._predictions(day_sep25, 0.96).head(2)
        g25 = [int(g) for g in frame25["game_pk"]]
        st25 = {g: {"abstract": "P", "detailed": "Scheduled"} for g in g25}
        cfg2 = walk._config(p2)
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            res25 = scheduler.run_single_check("2027-09-25", g25, set(), cfg2, 0.01)
        s_state25 = scheduler.SchedulerState("2027-09-25", None, [], [], [], False, None, None, None)
        sel25 = res25["selection"]
        s_state25.final_skip_candidate = {
            "primary": sel25.primary_candidate, "double": sel25.double_candidate,
            "streak": sel25.streak, "saver_available": sel25.saver_available,
            "state_source": sel25.state_source, "state_status": sel25.state_status,
            "allow_double": sel25.allow_double, "contest_source_date": sel25.contest_source_date,
            **scheduler._selection_decision_meta(sel25),
        }
        scheduler._write_endofday_skip(p2, "2027-09-25", s_state25)
        skip_rec = daily_decision.load_decision("2027-09-25", p2)
        fail_if(
            (skip_rec["action"], skip_rec["objective"], skip_rec["best_streak"], skip_rec["effective_best"])
            != ("skip", "emax_season_best", 157, 57),
            f"Case 5.2b decision.json wrong: {skip_rec}",
        )

        # Case 5.3: Automatic wrong reading 5/157 when saver is not_earned -> auto-earns saver to active;
        # later corrected fetch 5/5 leaves saver_state active (and dashboard warning clears once best=5<=15)!
        p3 = root / "c53" / "picks"
        p3.mkdir(parents=True)
        assert walk.install_state(p3, day_sep25, walk.State("automatic", 5, 5, "not_earned", 0.96)) == []
        fail_if(saver_state.load_saver_state(p3, season=2027).state != "not_earned", "saver setup wrong")
        with walk._fetch_stubs(5, 157, day_sep25 - dt.timedelta(days=1)):
            c_glitch, _ = walk._invoke("fetch-contest-streak", "--picks-dir", p3, "--expected-username", walk.ACCOUNT)
        fail_if(c_glitch != 0 or saver_state.load_saver_state(p3, season=2027).state != "active",
                "Case 5.3 5/157 did not auto-earn saver")
        dash_during = web.saver_dashboard_context(p3, now=_Clock.at)
        fail_if(not dash_during.warning, "Case 5.3 dashboard did not warn while best=157 and saver=active")
        # Now MLB corrects reading back to 5/5:
        with walk._fetch_stubs(5, 5, day_sep25 - dt.timedelta(days=1)):
            c_fix, _ = walk._invoke("fetch-contest-streak", "--picks-dir", p3, "--expected-username", walk.ACCOUNT)
        fail_if(c_fix != 0 or saver_state.load_saver_state(p3, season=2027).state != "active",
                "Case 5.3 corrected 5/5 unexpectedly reverted saver")
        dash_after = web.saver_dashboard_context(p3, now=_Clock.at)
        fail_if(dash_after.warning, "Case 5.3 dashboard still warned at best=5")

        # Case 5.4 (Our own case 1): R33 automatic reuse across 75/75 -> 0/75 reset when fallback cascade is empty:
        # Cached double chosen at 75/75 is still delivered at 0/75 with DM "Streak: 0", even though fresh 0/75 skips!
        p4 = root / "c54" / "picks"
        p4.mkdir(parents=True)
        assert walk.install_state(p4, day_sep25, walk.State("automatic", 75, 75, "used", 0.96)) == []
        cfg4 = walk._config(p4)
        cfg4["bluesky"] = {"dm_recipient": "synthetic"}
        dms4 = []
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            scheduler.run_single_check("2027-09-25", g25, set(), cfg4, 0.01)
        saved_auto75 = picks.load_pick("2027-09-25", p4)
        # Fetch updates automatic reading to 0/75 (miss settled)
        with walk._fetch_stubs(0, 75, day_sep25):
            assert walk._invoke("fetch-contest-streak", "--picks-dir", p4, "--expected-username", walk.ACCOUNT)[0] == 0
        # Fresh policy at 0/75 skips, but empty fallback refresh reuses cached double!
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25.iloc[:0], "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
            st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in g25]))
            st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (dms4.append(m) or "dm-r33")))
            ref4 = scheduler._refresh_pick_at_fallback_decision(cfg4, "2027-09-25", saved_auto75, 0.01)
            s4 = scheduler.SchedulerState("2027-09-25", None, [], [], [], False, None, None, None)
            ok4 = scheduler._deliver_and_lock_pick(ref4.daily, cfg4, p4, s4, "2027-09-25", "final fallback", selection=ref4.selection)
        rec4 = daily_decision.load_decision("2027-09-25", p4)
        fail_if(not ok4 or "Streak: 0" not in dms4[0] or rec4["action"] != "double" or rec4["streak"] is not None or rec4["best_streak"] != 75,
                f"Case 5.4 R33 reuse unexpected: ok={ok4} dms={dms4} rec={rec4}")

        # Case 5.5 (Our own case 2): Dated best-only typo correction 58/157 -> 58/58 after selection,
        # followed by empty final fallback refresh -> saved_pick_refusal blocks delivery (no pick that day)!
        p5 = root / "c55" / "picks"
        p5.mkdir(parents=True)
        assert walk.install_state(p5, day_sep25, walk.State("dated", 58, 157, "used", 0.96)) == []
        cfg5 = walk._config(p5)
        cfg5["bluesky"] = {"dm_recipient": "synthetic"}
        dms5 = []
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            scheduler.run_single_check("2027-09-25", g25, set(), cfg5, 0.01)
        saved_dated58 = picks.load_pick("2027-09-25", p5)
        # Operator corrects best_streak typo 157 -> 58
        assert runner.invoke(cli.cli, [
            "set-contest-streak", "--picks-dir", str(p5),
            "--date", "2027-09-25", "--streak", "58", "--best-streak", "58", "--saver", "used",
        ]).exit_code == 0
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25.iloc[:0], "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in g25]))
            st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (dms5.append(m) or "dm-ref")))
            ref5 = scheduler._refresh_pick_at_fallback_decision(cfg5, "2027-09-25", saved_dated58, 0.01)
            s5 = scheduler.SchedulerState("2027-09-25", None, [], [], [], False, None, None, None)
            ok5 = scheduler._deliver_and_lock_pick(ref5.daily, cfg5, p5, s5, "2027-09-25", "final fallback", selection=ref5.selection)
        fail_if(ok5 is not False or daily_decision.load_decision("2027-09-25", p5) is not None,
                f"Case 5.5 dated best-only correction unexpectedly delivered: ok={ok5}")
        fail_if(len(dms5) != 1 or "best streak 157" not in dms5[0] or "best streak 58" not in dms5[0],
                f"Case 5.5 refusal alert text wrong: {dms5}")

    notes.append("Item 5 verified: 75/57 caught, 75/75 & 57/157 & 12/157 & auto 5/157 saver earn & R33 75->0 reuse & dated 58/157->58/58 correction match contract.")

    out_lines = [*notes, f"problems={len(problems)}"]
    for p in problems[:50]:
        out_lines.append(f"PROBLEM: {p}")
    text = "\n".join(out_lines) + "\n"
    (OUT_DIR / "item4_5_policy_and_typos.log").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

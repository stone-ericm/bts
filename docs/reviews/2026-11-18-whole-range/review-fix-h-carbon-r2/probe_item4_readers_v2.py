"""Adversarial probe for Item 4 (v2, isolated parent dirs per case): every reader of objective on longest_streak picks and records (both single 57->58 and double 57->59)."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, time, timedelta
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bts import cli, contest_fetch, daily_decision as dd, entry_receipt, orchestrator, picks, scheduler, skip_policy_shadow, strategy, web
from bts.health import mdp_policy_alignment, post_failure
from bts.simulate import mdp
from scripts.audit import boundary_shadow_census as census
from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent


def main() -> None:
    tail_sha = strategy._load_mdp()["tail"].sha256

    with tempfile.TemporaryDirectory(dir=HERE, prefix="readers2-") as tmp_str:
        root = Path(tmp_str)

        for day_str, want_action, next_streak in (("2027-07-15", "single", 58), ("2027-09-26", "double", 59)):
            day = date.fromisoformat(day_str)
            case_root = root / f"case_{day_str}"
            pdir = case_root / "picks"
            pdir.mkdir(parents=True)
            frame = walk._predictions(day, 0.84).head(2)
            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
            cfg = walk._config(pdir)
            cfg["orchestrator"]["heartbeat_path"] = str(pdir / ".heartbeat")
            cfg["bluesky"] = {"dm_recipient": "synthetic"}

            with (
                walk.offline(datetime.combine(day, time(12), walk.ET)),
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")),
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
                patch.object(scheduler, "count_new_confirmations", lambda *a: 6),
                patch.object(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(True, None, True)),
                patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
                patch.object(scheduler, "_void_committed_games", lambda *a: {}),
                patch("bts.dm.send_dm", return_value="dm-ok"),
            ):
                assert walk.install_state(pdir, day, walk.State("dated", 57, 57, "used", 0.84)) == []
                res = scheduler.run_single_check(day_str, list(statuses), set(), cfg, 0.01)
                saved = picks.load_pick(day_str, pdir)
                meta_live = scheduler._selection_decision_meta(res["selection"], saved)
                meta_saved = scheduler._selection_decision_meta(None, saved)
                assert meta_live["objective"] == meta_saved["objective"] == "longest_streak"
                assert meta_live["tail_policy_sha256"] == meta_saved["tail_policy_sha256"] == tail_sha

                align_pre = mdp_policy_alignment.compute_metrics(pdir, mdp.DEFAULT_POLICY_PATH, today=day)
                assert align_pre.primary.n == 1

                st = scheduler.SchedulerState(day_str, None, [], [], [], False, None, None, None)
                assert scheduler._deliver_and_lock_pick(saved, cfg, pdir, st, day_str, "lineup", selection=res["selection"])
                rec = dd.load_decision(day_str, pdir)
                assert rec["objective"] == "longest_streak" and rec["action"] == want_action
                assert dd.decision_objective(rec) == "longest_streak"
                assert dd.is_reach57_mdp_skip(rec) is False
                assert dd.is_scoreable_commit(day_str, pdir, saved) is True

                align_post = mdp_policy_alignment.compute_metrics(pdir, mdp.DEFAULT_POLICY_PATH, today=day)
                assert align_post.primary.n == 0 and align_post.double_down.n == 0

                assert skip_policy_shadow.record_skip_from_decision(day_str, pdir) is None
                assert skip_policy_shadow.record_pending_skips(pdir, now=datetime(day.year, day.month, day.day, 23, tzinfo=walk.ET)) == []

                ledger = case_root / "ledger.jsonl"
                ledger.touch()
                creport = census.run_census(
                    picks_dir=pdir, ledger_path=ledger, policy_path=mdp.DEFAULT_POLICY_PATH,
                    saver_active=False, start=day_str, end=day_str, as_of=day_str,
                    output=case_root / "census.json",
                )
                assert creport["boundaries"]["primary"]["n"] == 0
                assert creport["rows"][0]["state_source"] == "excluded_tail_objective"
                assert creport["rows"][0]["diffs"] == {}

                snap = entry_receipt.units_snapshot_dir(pdir)
                snap.mkdir(parents=True, exist_ok=True)
                units = [{"id": 170, "feedId": saved.pick.game_pk, "roundId": 17}]
                pending = [{"roundId": 17, "unitId": 170, "playerId": 800, "number": 1, "result": None}]
                bts_map = {800: saved.pick.batter_id}
                if saved.double_down:
                    units.append({"id": 171, "feedId": saved.double_down.game_pk, "roundId": 17})
                    pending.append({"roundId": 17, "unitId": 171, "playerId": 801, "number": 2, "result": None})
                    bts_map[801] = saved.double_down.batter_id
                (snap / f"{day_str}T120000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": units}).encode()))
                walk._set_clock(datetime.combine(day, time(18, 35), walk.ET))
                with (
                    walk._fetch_stubs(57, 57, day - timedelta(days=1)),
                    patch.object(contest_fetch, "fetch_pending_predictions", return_value=pending),
                    patch.object(cli, "_fetch_rounds", return_value={17: day}),
                    patch.object(cli, "_fetch_bts_to_mlb", return_value=bts_map),
                ):
                    code, out = walk._invoke("check-pick-entered", "--picks-dir", pdir, "--expected-username", walk.ACCOUNT)
                assert code == 0 and "match" in out, out
                rcpt, = entry_receipt.discover(pdir, day_str)
                assert rcpt["selection"]["commit"]["decision_sha256"] == sha256(dd.decision_path(day_str, pdir).read_bytes()).hexdigest()
                assert rcpt["qualification"]["all_confirmed"] is True

                cfg_toml = case_root / "cfg.toml"
                cfg_toml.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
                with patch.object(scheduler, "fetch_schedule", return_value=[{}, {}]):
                    code, out = walk._invoke("day-outcome", "--date", day_str, "--config", cfg_toml,
                                             "--picks-dir", pdir, "--expected-username", walk.ACCOUNT)
                assert code == 0, out
                outcome = json.loads(out)
                assert outcome["state"] == "entered" and outcome["pick"] == saved.pick.batter_name

                with (
                    patch.object(web, "PICKS_DIR", pdir),
                    patch.object(web, "HEALTH_STATE_DIR", case_root / "hs"),
                    patch.object(web, "fetch_bluesky_posts", return_value=[]),
                    patch.object(web, "_build_live_game_data", return_value=([], None)),
                ):
                    assert web._decision_action(day_str) == want_action
                    html = web.render_page()
                    assert saved.pick.batter_name in html
                    if saved.double_down:
                        assert saved.double_down.batter_name in html
                assert post_failure.check(pdir, today=day, now=datetime.combine(day, time(22, 30), walk.ET)) == []

                picks.save_streak(57, pdir, saver_available=False, season=2027)
                walk._set_clock(datetime.combine(day + timedelta(days=1), time(12), walk.ET))
                with patch.object(picks, "check_hit", return_value="hit") as hit_mock:
                    code1, out1 = walk._invoke("check-results", "--date", day_str, "--picks-dir", pdir)
                    assert code1 == 0 and f"Streak: {next_streak}" in out1, out1
                    assert hit_mock.call_count == (2 if want_action == "double" else 1)
                    code2, out2 = walk._invoke("check-results", "--date", day_str, "--picks-dir", pdir)
                    assert code2 == 0 and "Already resolved" in out2
                    assert hit_mock.call_count == (2 if want_action == "double" else 1)
                graded = picks.load_pick(day_str, pdir)
                assert graded.result == "hit" and picks.load_streak(pdir, season=2027) == next_streak
                assert graded.chosen_on == saved.chosen_on and graded.policy_decision == saved.policy_decision

        sdir = root / "synth_skip" / "picks"
        sdir.mkdir(parents=True)
        srec = dd.write_decision(
            "2027-07-20", sdir, action="skip", source="mdp", objective="longest_streak",
            primary={"batter_id": 100, "batter_name": "Batter 0", "team": "T0", "game_pk": 70000, "p_game_hit": 0.84},
            delivery_status="not_applicable", scoreable=False,
        )
        assert srec is not None and not dd.is_reach57_mdp_skip(srec)
        assert skip_policy_shadow.record_skip_from_decision("2027-07-20", sdir) is None
        assert not list(sdir.glob("*.policy_shadow.json"))

    print("OK: All Item 4 readers verified on both single (57->58) and double (57->59) longest_streak records")


if __name__ == "__main__":
    main()

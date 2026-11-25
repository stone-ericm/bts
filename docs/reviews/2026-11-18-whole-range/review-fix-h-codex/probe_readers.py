"""Read an actual longest-streak commit through the consumer entry points."""
from datetime import date, timedelta
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import cli, contest_fetch, entry_receipt, orchestrator, picks, scheduler, skip_policy_shadow, web
from bts import daily_decision as dd
from bts.health import mdp_policy_alignment, post_failure
from bts.simulate import mdp
from probe_delivery import checked_record, clock, deliver, world
from scripts.audit import boundary_shadow_census as census
from scripts.audit.season_walk import walk


def entry_and_outcome(root, day, saved):
    text = str(day)
    snap = entry_receipt.units_snapshot_dir(root)
    snap.mkdir(parents=True)
    (snap / "2027-07-16T000000Z.json.gz").write_bytes(gzip.compress(json.dumps({
        "units": [{"id": 170, "feedId": saved.pick.game_pk, "roundId": 17}]
    }).encode()))
    pending = [{"roundId": 17, "unitId": 170, "playerId": 800, "number": 1, "result": None}]
    walk._set_clock(clock(day, 18, 35))
    with (walk._fetch_stubs(57, 57, day - timedelta(days=1)),
          patch.object(contest_fetch, "fetch_pending_predictions", return_value=pending),
          patch.object(cli, "_fetch_rounds", return_value={17: day}),
          patch.object(cli, "_fetch_bts_to_mlb", return_value={800: saved.pick.batter_id})):
        code, output = walk._invoke("check-pick-entered", "--picks-dir", root,
                                    "--expected-username", walk.ACCOUNT)
    assert code == 0 and "pick entered (match)" in output, output
    receipt, = entry_receipt.discover(root, text)
    dpath = dd.decision_path(text, root)
    assert receipt["selection"]["commit"]["decision_sha256"] == sha256(dpath.read_bytes()).hexdigest()
    assert receipt["selection"]["commit"]["scoreable"] and receipt["qualification"]["all_confirmed"]
    assert [(s["role"], s["batter_id"], s["game_pk"]) for s in receipt["selection"]["slots"]] == [
        ("pick", saved.pick.batter_id, saved.pick.game_pk)]
    config = root.parent / "intent.toml"
    config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    with patch.object(scheduler, "fetch_schedule", return_value=[{},
                                                                 {}]):
        code, output = walk._invoke("day-outcome", "--date", text, "--config", config,
                                    "--picks-dir", root, "--expected-username", walk.ACCOUNT)
    assert code == 0, output
    outcome = json.loads(output)
    assert outcome["state"] == "entered" and outcome["pick"] == saved.pick.batter_name, outcome
    print("check-pick-entered confirmed exact pick and decision bytes; day-outcome", outcome)


def reach_readers(root, day, rec):
    text = str(day)
    assert dd.decision_objective(rec) == "longest_streak"
    assert not dd.is_reach57_mdp_skip(rec)
    assert skip_policy_shadow.record_skip_from_decision(text, root) is None
    alignment = mdp_policy_alignment.compute_metrics(root, mdp.DEFAULT_POLICY_PATH, today=day)
    assert alignment.primary.n == 0
    ledger = root.parent / "synthetic-observations.jsonl"
    ledger.touch()
    census_report = census.run_census(picks_dir=root, ledger_path=ledger, policy_path=mdp.DEFAULT_POLICY_PATH,
                                     saver_active=False, start=text, end=text, as_of=text,
                                     output=root.parent / "census.json")
    row, = census_report["rows"]
    assert row["state_source"] == "excluded_tail_objective" and row["diffs"] == {}
    assert census_report["boundaries"]["primary"]["n"] == 0
    forced_skip = dd.write_decision(text, root, objective=rec["objective"], action="skip", source="mdp",
                                    primary=rec["primary"], delivery_status="not_applicable", scoreable=False)
    assert forced_skip and not dd.is_reach57_mdp_skip(forced_skip)
    assert skip_policy_shadow.record_skip_from_decision(text, root) is None
    assert not list(root.glob("*.policy_shadow.json"))
    print("alignment sample 0; census excluded with empty diffs; no skip-shadow record for synthetic MDP skip")


def main():
    day = date(2027, 7, 15)
    text = str(day)
    with TemporaryDirectory() as temp:
        root = Path(temp) / "picks"
        with world(root, day, "dated") as (cfg, frame, statuses):
            check = scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
            saved = picks.load_pick(text, root)
            assert saved.policy_decision["objective"] == "longest_streak" and not saved.double_down
            with patch("bts.dm.send_dm", return_value="reader-captured"):
                assert deliver(saved, cfg, root, check["selection"])
            rec = checked_record(root, text, saved, "delivered")
            assert rec["double_down"]["batter_id"] == 101   # A4 records the runner-up on a fresh single
            assert dd.is_scoreable_commit(text, root, saved)
            entry_and_outcome(root, day, saved)
            with (patch.object(web, "PICKS_DIR", root),
                  patch.object(web, "HEALTH_STATE_DIR", root.parent / "health_state"),
                  patch.object(web, "fetch_bluesky_posts", return_value=[]),
                  patch.object(web, "_build_live_game_data", return_value=([], None))):
                assert web._decision_action(text) == "single"
                page = web.render_page()
                assert saved.pick.batter_name in page and '<span class="result-skip"' not in page
            print("dashboard single rendered with a batter and no finalized-skip row")
            assert post_failure.check(root, today=day, now=clock(day, 22, 30)) == []
            picks.save_streak(57, root, saver_available=False, season=2027)
            walk._set_clock(clock(day + timedelta(days=1)))
            with patch.object(picks, "check_hit", return_value="hit") as score:
                code, output = walk._invoke("check-results", "--date", text, "--picks-dir", root)
                assert code == 0 and "Streak: 58" in output, output
                score.assert_called_once()
                assert score.call_args.args == (saved.pick.game_pk, saved.pick.batter_id)
                code, output = walk._invoke("check-results", "--date", text, "--picks-dir", root)
                assert code == 0 and "Already resolved" in output
                score.assert_called_once()
            graded = picks.load_pick(text, root)
            assert graded.result == "hit" and graded.slot_results == {"primary": "hit"}
            assert picks.load_streak(root, season=2027) == 58 and graded.chosen_on == saved.chosen_on
            print("check-results graded one slot, advanced the model 57 to 58 once, and preserved choice metadata")
            reach_readers(root, day, rec)


if __name__ == "__main__":
    main()

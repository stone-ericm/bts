"""Check the single-pick score and analytics after a real delivery."""
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import picks, scheduler
from probe_delivery import checked_record, clock, deliver, world
from probe_readers import reach_readers
from scripts.audit.season_walk import walk


def main():
    day = date(2027, 7, 15)
    text = str(day)
    with TemporaryDirectory() as temp:
        root = Path(temp) / "picks"
        with world(root, day, "dated") as (cfg, frame, statuses):
            check = scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
            saved = picks.load_pick(text, root)
            assert saved.policy_decision["objective"] == "longest_streak" and saved.double_down is None
            with patch("bts.dm.send_dm", return_value="reader-captured"):
                assert deliver(saved, cfg, root, check["selection"])
            rec = checked_record(root, text, saved, "delivered")
            walk._set_clock(clock(day + timedelta(days=1)))
            picks.save_streak(57, root, saver_available=False, season=2027)
            with patch.object(picks, "check_hit", return_value="hit") as score:
                code, output = walk._invoke("check-results", "--date", text, "--picks-dir", root)
                assert code == 0 and "Streak: 58" in output, output
                score.assert_called_once()
                assert score.call_args.args == (saved.pick.game_pk, saved.pick.batter_id)
                code, output = walk._invoke("check-results", "--date", text, "--picks-dir", root)
                assert code == 0 and "Already resolved" in output
                score.assert_called_once()
            graded = picks.load_pick(text, root)
            assert graded.result == "hit" and graded.slot_results == {"pick": "hit"}
            assert picks.load_streak(root, season=2027) == 58 and graded.chosen_on == saved.chosen_on
            assert graded.policy_decision == saved.policy_decision
            print("check-results scored exactly one pick, advanced 57 to 58 once and kept metadata")
            reach_readers(root, day, rec)


if __name__ == "__main__":
    main()

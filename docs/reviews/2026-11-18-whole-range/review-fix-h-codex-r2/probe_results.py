from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import picks, web
from bts.health import post_failure
from probe_lifecycle import check_record, clock, deliver, load_pick_after_check, slate
from scripts.audit.season_walk import walk


def check(root, day):
    with slate(root, day) as (cfg, _, statuses):
        saved, selection = load_pick_after_check(root, day, cfg, statuses)
        with patch('bts.dm.send_dm', return_value='results-capture'):
            assert deliver(saved, cfg, root, selection)
        check_record(root, saved, 'delivered')
        with (patch.object(web, 'PICKS_DIR', root),
              patch.object(web, 'HEALTH_STATE_DIR', root.parent / 'health'),
              patch.object(web, 'fetch_bluesky_posts', return_value=[]),
              patch.object(web, '_build_live_game_data', return_value=([], None))):
            assert web._decision_action(str(day)) == 'single'
            page = web.render_page()
            assert saved.pick.batter_name in page and '<span class="result-skip"' not in page
        assert post_failure.check(root, today=day, now=clock(day, 22, 30)) == []
        picks.save_streak(57, root, saver_available=False, season=day.year)
        walk._set_clock(clock(day + timedelta(days=1)))
        with patch.object(picks, 'check_hit', return_value='hit') as grade:
            for expected in ('Streak: 58', 'Already resolved'):
                code, output = walk._invoke('check-results', '--date', str(day), '--picks-dir', root)
                assert code == 0 and expected in output, output
            grade.assert_called_once_with(saved.pick.game_pk, saved.pick.batter_id,
                                          batter_name=saved.pick.batter_name, date=str(day),
                                          team=saved.pick.team, return_status=True)
        graded = picks.load_pick(str(day), root)
        assert (graded.result, graded.slot_results) == ('hit', {'pick': 'hit'})
        assert picks.load_streak(root, season=day.year) == 58 and graded.chosen_on == saved.chosen_on
        assert graded.policy_decision == saved.policy_decision
        print('dashboard single, post-failure cleared; results graded one slot and moved 57 to 58 exactly once')


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        check(Path(tmp) / 'picks', date(2027, 7, 15))

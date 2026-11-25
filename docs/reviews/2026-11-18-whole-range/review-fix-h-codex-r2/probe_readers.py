from datetime import date, timedelta
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import cli, contest_fetch, daily_decision as dd, entry_receipt, picks, scheduler, skip_policy_shadow, web
from bts.health import mdp_policy_alignment, post_failure
from bts.simulate import mdp
from scripts.audit import boundary_shadow_census as census
from scripts.audit.season_walk import walk
from probe_lifecycle import check_record, clock, deliver, load_pick_after_check, slate


def exclusions(root, day, record):
    assert dd.decision_objective(record) == 'longest_streak' and not dd.is_reach57_mdp_skip(record)
    assert mdp_policy_alignment.compute_metrics(root, mdp.DEFAULT_POLICY_PATH, today=day).primary.n == 0
    ledger = root.parent / 'observations.jsonl'
    ledger.touch()
    report = census.run_census(picks_dir=root, ledger_path=ledger, policy_path=mdp.DEFAULT_POLICY_PATH,
                               saver_active=False, start=str(day), end=str(day), as_of=str(day),
                               output=root.parent / 'census.json')
    row, = report['rows']
    assert row['state_source'] == 'excluded_tail_objective' and row['diffs'] == {}
    assert report['boundaries']['primary']['n'] == 0
    skip_root = root.parent / 'synthetic-skip'
    skip = dd.write_decision(str(day), skip_root, objective='longest_streak', action='skip', source='mdp',
                            delivery_status='not_applicable', scoreable=False, primary=record['primary'])
    assert dd.decision_objective(skip) == 'longest_streak' and not dd.is_reach57_mdp_skip(skip)
    assert skip_policy_shadow.record_skip_from_decision(str(day), skip_root) is None
    assert not list(skip_root.glob('*.policy_shadow.json'))
    print('objective accepted, no reach57 skip or shadow; zero alignment and primary census, excluded row')


def entry_and_day_outcome(root, day, saved):
    snapshots = entry_receipt.units_snapshot_dir(root)
    snapshots.mkdir(parents=True)
    (snapshots / '2027-07-16T000000Z.json.gz').write_bytes(gzip.compress(json.dumps({
        'units': [{'id': 925, 'feedId': saved.pick.game_pk, 'roundId': 37}],
    }).encode()))
    pending = [{'roundId': 37, 'unitId': 925, 'playerId': 913, 'number': 1, 'result': None}]
    walk._set_clock(clock(day, 18, 37))
    with (walk._fetch_stubs(57, 57, day - timedelta(days=1)),
          patch.object(contest_fetch, 'fetch_pending_predictions', return_value=pending),
          patch.object(cli, '_fetch_rounds', return_value={37: day}),
          patch.object(cli, '_fetch_bts_to_mlb', return_value={913: saved.pick.batter_id})):
        code, msg = walk._invoke('check-pick-entered', '--picks-dir', root, '--expected-username', walk.ACCOUNT)
    assert code == 0 and 'pick entered (match)' in msg, msg
    receipt, = entry_receipt.discover(root, str(day))
    commit = receipt['selection']['commit']
    slot, = receipt['selection']['slots']
    assert commit['decision_sha256'] == sha256(dd.decision_path(str(day), root).read_bytes()).hexdigest()
    assert commit['scoreable'] and receipt['qualification']['all_confirmed']
    assert (slot['role'], slot['batter_id'], slot['game_pk']) == ('pick', saved.pick.batter_id, saved.pick.game_pk)
    config = root.parent / 'outcome.toml'
    config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    with patch.object(scheduler, 'fetch_schedule', return_value=[{}, {}]):
        code, output = walk._invoke('day-outcome', '--date', str(day), '--config', config,
                                    '--picks-dir', root, '--expected-username', walk.ACCOUNT)
    assert code == 0, output
    outcome = json.loads(output)
    assert (outcome['state'], outcome['pick']) == ('entered', saved.pick.batter_name)
    print('entry verified pick/game and sealed exact decision bytes; day-outcome entered', outcome['pick'])


def display_and_grade(root, day, saved):
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
        grade.assert_called_once_with(saved.pick.game_pk, saved.pick.batter_id)
    graded = picks.load_pick(str(day), root)
    assert (graded.result, graded.slot_results) == ('hit', {'pick': 'hit'})
    assert picks.load_streak(root, season=day.year) == 58 and graded.chosen_on == saved.chosen_on
    assert graded.policy_decision == saved.policy_decision
    print('dashboard single; post-failure cleared; results graded one pick and moved 57 to 58 once')


if __name__ == '__main__':
    day = date(2027, 7, 15)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp) / 'picks'
        with slate(folder, day) as (cfg, _, statuses):
            saved, selection = load_pick_after_check(folder, day, cfg, statuses)
            assert saved.double_down is None
            with patch('bts.dm.send_dm', return_value='review-reader-capture'):
                assert deliver(saved, cfg, folder, selection)
            record = check_record(folder, saved, 'delivered')
            assert record['double_down']['batter_id'] == 101  # The accepted record-writer slot limit.
            assert dd.is_scoreable_commit(str(day), folder, saved)
            exclusions(folder, day, record)
            entry_and_day_outcome(folder, day, saved)
            display_and_grade(folder, day, saved)

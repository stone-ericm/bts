import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bts import cli, contest_state as cs, orchestrator, picks, season_contract as sc, strategy
from bts.health import contest_state as health
from bts.leaderboard import scraper
from probe_live import at, cfg_for, fetch, model_and_switch, state
from probe_validators import call, invoke
from scripts.audit.season_walk import walk as w


DAY = date(2027, 7, 15)
NOW = at(DAY)


def authority(directory):
    model_and_switch(directory, DAY, 12)
    acct = directory / 'account_state'
    switch_path = acct / 'season_contract.json'
    auto_path = acct / 'contest_streak.json'
    manual_path = acct / 'contest_streak.manual.json'
    orig_switch, orig_auto = [json.loads(p.read_text()) for p in (switch_path, auto_path)]
    for kind, switch, auto in [('1', {'status': 'bad'}, {}), ('2', {}, {}),
                               ('3 absent', None, {}), ('3 revoked', {'status': 'revoked', 'revoked_at': NOW.isoformat()}, {}),
                               ('3 other season', {'season': 2026}, {}), ('4', {}, None),
                               ('5 schema', {}, {'schema_version': 'old'}), ('5 season', {}, {'season': 2026}),
                               ('5 ID', {}, {'activation_id': 'b' * 32})]:
        manual_path.unlink(missing_ok=True)
        if switch is None:
            switch_path.unlink(missing_ok=True)
        else:
            switch_path.write_text(json.dumps(orig_switch | switch))
        if auto is None:
            auto_path.unlink(missing_ok=True)
        else:
            auto_path.write_text(json.dumps(orig_auto | auto))
        if kind == '2':
            print('legacy CLI wrote', invoke(['set-contest-streak', '--picks-dir', directory,
                                               '--streak', 401, '--best-streak', 401]))
            legacy = cs.load_decision_streak_state(directory, now=NOW)
            print('legacy optional', legacy.streak, legacy.best_streak, legacy.best_status)
        print('K9 switch', kind, call(lambda: state(directory)))
        print('K9 historical', kind, [(a.level, a.message.split(' This check')[0]) for a in
                                     health.check(directory, expected=True, now=NOW, historical_day=DAY)])
    switch_path.write_text(json.dumps(orig_switch))
    auto_path.write_text(json.dumps(orig_auto))


def select_directory(directory):
    frame = w._predictions(DAY, .96)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
    with patch.object(orchestrator, 'run_cascade', return_value=(frame, 'synthetic')), \
            patch.object(picks, 'get_game_statuses_detailed', return_value=statuses):
        _, selection, _ = orchestrator.run_and_pick(cfg_for(directory), str(DAY))
    decision = selection.decision
    return decision.objective, decision.action, decision.streak, decision.best_supplied, decision.best_status


def other_inputs(directory):
    print('M3 automatic CLI', fetch(directory, DAY, 12, 60, recipient=False), 'selected', select_directory(directory))
    for active, best, saver in [(60, 75, 'used'), (400, 400, 'used'), (58, 58, 'not_earned'), (58, 58, 'active')]:
        result = invoke(['set-contest-streak', '--picks-dir', directory, '--date', DAY,
                         '--streak', active, '--best-streak', best, '--saver', saver])
        flag = state(directory)
        print('M3 dated CLI', active, best, saver, result, 'selected', select_directory(directory),
              'saver', flag.saver_available, flag.saver_flag_state, 'chosen', flag.observation_kind)
    # Parser reads through the valid dated cover even if automatic JSON is broken.
    path = directory / 'account_state/contest_streak.json'
    path.write_text('{')
    print('current dated bypasses broken automatic', state(directory).observation_kind)
    w._set_clock(at(DAY + timedelta(days=1)))
    print('expired dated no longer bypasses broken automatic', call(lambda: state(directory)))
    stats = scraper.parse_user_profile_response(
        {'success': {'seasonBestStreak': 157, 'activeStreak': 100, 'predictions': []}},
        NOW, 4242, SimpleNamespace(rounds={}), username='synthetic')[1]
    print('leaderboard profile', stats.active_streak, stats.best_streak)


if __name__ == '__main__':
    print(cs.__file__)
    with TemporaryDirectory() as tmp, w.offline(NOW):
        directory = Path(tmp) / 'picks'
        authority(directory)
        other_inputs(directory)

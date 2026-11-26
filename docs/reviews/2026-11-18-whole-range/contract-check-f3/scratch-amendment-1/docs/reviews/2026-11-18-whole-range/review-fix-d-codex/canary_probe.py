from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import contest_state, orchestrator, scheduler
from bts.cli import cli
from bts.picks import chosen_on_record
from tests.test_contest_dated_cs4a import observation, write
from tests.test_late_delivery_guard import _daily, _state
from owner_probe import TARGET, configure, guard


HERE = Path(__file__).resolve().parent
ET = ZoneInfo('America/New_York')
runner = CliRunner()
cli_module = import_module('bts.cli')


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mode, p.stat().st_mtime_ns,
            p.read_bytes() if p.is_file() else None) for p in sorted(root.rglob('*'))}


def blocked(*args, **kwargs):
    raise AssertionError('external I/O or scheduler start in canary')


def run(root, name, now, *, mode='dm', intent='enter', scheduler_extra='', suffix='',
        dated=True, rec_day=None, structure='normal', valid=True, reads=1, automatic=None):
    place = root / name
    picks = place / 'picks'
    picks.mkdir(parents=True)
    (picks / 'existing.bin').write_bytes(b'preserve\0fixture')
    (place / '.heartbeat').write_bytes(b'fixture')
    if dated:
        day = rec_day or now.date().isoformat()
        write(picks, observation(day, recorded_at=day + 'T12:00:00-05:00'))
    if automatic is not None:
        active = 'a' * 32
        write(picks, {'schema_version': 'bts_season_contract_v1', 'season': 2027,
                      'status': 'automatic', 'activation_id': active,
                      'switched_at': '2027-07-15T16:00:00Z', 'revoked_at': None, 'note': None},
              'season_contract.json')
        write(picks, {'schema_version': 'bts_contest_streak_auto_v2', 'active_streak': 7,
                      'best_streak': 9, 'season': 2027, 'activation_id': automatic * 32,
                      'source': 'mlb_bts_profile', 'source_date': '2027-07-14'}, 'contest_streak.json')
    cfg = place / 'config.toml'
    orch = f'[orchestrator]\npicks_dir={json.dumps(str(picks))}\n' if structure == 'normal' else ''
    cfg.write_text(orch + f'[scheduler]\nentry_intent={json.dumps(intent)}\n'
                   f'pick_delivery={json.dumps(mode)}\ncontest_state_required=true\n'
                   f'{scheduler_extra}\n[health_checks]\ncontest_state_expected=true\n{suffix}')
    before = snapshot(place)
    with patch.object(cli_module, '_now_et', return_value=now) as clock, \
         patch.object(contest_state, 'load_decision_streak_state',
                      wraps=contest_state.load_decision_streak_state) as reader, \
         patch.object(scheduler, 'run_day', side_effect=blocked), \
         patch('importlib.import_module', wraps=import_module) as imports:
        result = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
    assert clock.call_count == 1 and reader.call_count == reads
    if reads:
        assert reader.call_args.args == (picks,)
        assert reader.call_args.kwargs == {'require_contest_state': True, 'now': now}
    assert (result.exit_code == 0) == valid, (name, result.output, result.exception)
    assert before == snapshot(place), name
    print(name, result.exit_code, result.output.strip(), 'imports', [c.args[0] for c in imports.call_args_list], flush=True)
    return cfg


def deliver_without_recipient(cfg, now):
    picks = cfg.parent / 'picks'
    config = orchestrator.load_config(cfg)
    daily = _daily()
    daily.date = now.date().isoformat()
    daily.pick.game_time = daily.date + 'T23:00:00Z'
    confirmed = _state()
    confirmed.date = daily.date
    original = contest_state.load_decision_streak_state
    decision = original(picks, require_contest_state=True, now=now)
    daily.chosen_on = chosen_on_record(decision, required=True)
    before = snapshot(cfg.parent)
    with patch.object(contest_state, 'load_decision_streak_state',
                      side_effect=lambda p, *, require_contest_state: original(p, require_contest_state=require_contest_state, now=now)), \
         patch.object(scheduler, '_void_committed_games', return_value={}), \
         patch.object(scheduler, '_now_et', return_value=now), patch('bts.dm.send_dm', side_effect=blocked):
        assert scheduler._deliver_and_lock_pick(daily, config, picks, confirmed, daily.date, 'fixture') is False
    assert snapshot(cfg.parent) == before and not confirmed.pick_locked
    print('accepted DM config with valid chosen-on, open cutoff and no recipient: real delivery returned false', flush=True)


def relative_picks(root, cfg, now):
    other = root / 'loaded-unit-workdir'
    (other / 'picks').mkdir(parents=True)
    good = cfg.parent
    cfg.write_text(cfg.read_text().replace(json.dumps(str(good / 'picks')), '"picks"'))
    saved_cwd = Path.cwd()
    try:
        with patch.object(cli_module, '_now_et', return_value=now):
            os.chdir(good)
            result1 = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
            os.chdir(other)
            result2 = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
    finally:
        os.chdir(saved_cwd)
    assert (result1.exit_code, result2.exit_code) == (0, 1), (result1.output, result2.output)
    print('same live config with relative picks path', result1.output.strip(), '/', result2.output.strip(), flush=True)


def across_release(root, cfg, now):
    case = root / 'midnight-owner'
    case.mkdir()
    tool = guard(case)
    tool.ROOT.mkdir(mode=0o755)
    for name in ('writers.lock', 'transaction.lock', 'state.lock'):
        (tool.ROOT / name).touch(mode=0o644)
    (Path(tool.PROJECT) / 'data').mkdir(parents=True)
    configure(tool, case)
    original_candidate = tool.canary
    original_activation = tool.activate_scheduler
    after_midnight = datetime.fromisoformat('2027-07-16T00:00:01').replace(tzinfo=ET)
    sampled = []
    clock = [now]

    def candidate():
        original_candidate()
        result = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
        assert result.exit_code == 0, result.output
        sampled.append(clock[0].isoformat())

    def activation(uid, authorized_at):
        original_activation(uid, authorized_at)
        clock[0] = after_midnight

    tool.canary = candidate
    tool.activate_scheduler = activation
    with patch.object(cli_module, '_now_et', side_effect=lambda: clock[0]), tool.directory() as fd:
        assert tool.owned_deploy(fd, '1815674400-midnight', TARGET, '') == 0
        assert tool.read_hold(fd) is None
        assert len(sampled) == 2 and clock == [after_midnight]
        after = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
        assert after.exit_code == 1 and 'contest state refused' in after.output
    print('midnight model: owner committed after two prior-date real canaries;', after.output.strip(), flush=True)


def main():
    print('Python', os.sys.version.split()[0], 'bts', import_module('bts').__file__)
    with tempfile.TemporaryDirectory(dir=HERE) as temp, patch.object(socket.socket, 'connect', blocked):
        root = Path(temp)
        noon = datetime.fromisoformat('2027-07-15T18:00:00').replace(tzinfo=ET)
        for mode in ('dm', 'public'):
            run(root, mode + '-good', noon, mode=mode)
            run(root, mode + '-missing-state', noon, mode=mode, dated=False, valid=False)
            run(root, mode + '-automatic', noon, mode=mode, dated=False, automatic='a')
            run(root, mode + '-automatic-wrong-generation', noon, mode=mode, dated=False, automatic='b', valid=False)
            run(root, mode + '-wrong-intent', noon, mode=mode, intent='research', valid=False, reads=0)
            run(root, mode + '-malformed-orchestrator', noon, mode=mode, structure='missing', valid=False, reads=0)
        run(root, 'research', noon, mode='private', intent='research', dated=False, reads=0)
        run(root, 'private-enter', noon, mode='private', valid=False, reads=0)
        run(root, 'private-invalid-toml', noon, mode='private', intent='research', suffix='[malformed\n', valid=False, reads=0)
        run(root, 'private-invalid-required-value', noon, mode='private', intent='research',
            scheduler_extra='private_mode="false"', dated=False, reads=0)
        for name, before, after in (
            ('overnight', '2027-07-15T23:59:59', '2027-07-16T00:00:01'),
            ('year-end', '2027-12-31T23:59:59', '2028-01-01T00:00:01'),
        ):
            first = datetime.fromisoformat(before).replace(tzinfo=ET)
            last = datetime.fromisoformat(after).replace(tzinfo=ET)
            boundary_cfg = run(root, name + '-before', first)
            run(root, name + '-after', last, rec_day=first.date().isoformat(), valid=False)
            if name == 'overnight':
                across_release(root, boundary_cfg, first)
        for mode, intent in (('dm', 'enter'), ('public', 'enter'), ('private', 'research')):
            cfg = run(root, mode + '-invalid-startup-budget', noon, mode=mode, intent=intent,
                      scheduler_extra='cascade_budget_min="forty"', reads=int(intent == 'enter'))
            before = snapshot(cfg.parent)
            actual = runner.invoke(cli, ['schedule', '--config', str(cfg), '--date', '2027-07-15'])
            assert actual.exit_code == 1 and isinstance(actual.exception, ValueError)
            assert snapshot(cfg.parent) == before
            print('production CLI', mode, repr(actual.exception), flush=True)

        cfg = run(root, 'dm-missing-tiers', noon)
        before = snapshot(cfg.parent)
        try:
            orchestrator.run_and_pick(orchestrator.load_config(cfg), '2027-07-15')
        except KeyError as exc:
            assert exc.args == ('tiers',)
            print('live pick path with accepted config', repr(exc), flush=True)
        else:
            raise AssertionError('expected missing tier key')
        assert snapshot(cfg.parent) == before
        deliver_without_recipient(cfg, noon)
        relative_picks(root, cfg, noon)


if __name__ == '__main__':
    main()

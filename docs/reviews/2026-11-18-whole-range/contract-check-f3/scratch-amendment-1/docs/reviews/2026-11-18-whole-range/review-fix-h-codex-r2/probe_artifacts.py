from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bts import orchestrator, strategy
from bts.simulate import mdp, tail_policy as tp
from probe_lifecycle import check_record, deliver, load_pick_after_check, slate


DAY = date(2027, 9, 26)


@contextmanager
def files(base, tail):
    with (patch.object(mdp, 'DEFAULT_POLICY_PATH', base),
          patch.object(tp, 'DEFAULT_TAIL_POLICY_PATH', tail),
          patch.object(strategy, '_mdp_cache', None)):
        yield


def paired(folder, payload, tail):
    base_path, tail_path = folder / 'base.npz', folder / 'tail.npz'
    np.savez_compressed(base_path, **payload)
    base_sha = sha256(base_path.read_bytes()).hexdigest()
    tp.save_tail_policy(replace(tail, base_policy_sha256=base_sha), tail_path)
    validated = tp.load_tail_policy(tail_path, expected_base_sha=base_sha)
    assert validated.base_policy_sha256 == base_sha
    return base_path, tail_path, validated.sha256


def choose(policy):
    ctx = strategy.DecisionContext(.84, .83, True, str(DAY), True, policy, 57, 'trusted')
    return strategy.resolve_policy_decision(ctx, 57, False)


def assert_forced(decision):
    assert (decision.action, decision.policy_action, decision.objective, decision.source) == (
        'single', 'single', 'longest_streak', 'mdp')
    assert decision.days_effective == 1 and decision.tail_sha256 is None
    assert decision.degraded_reason.startswith('longest-streak rule unavailable (')
    assert decision.degraded_reason.endswith('); forced single')


def send_pair(root, label, expected, expected_sha):
    folder = root / f'delivery-{label}'
    with slate(folder, DAY) as (cfg, _, statuses):
        _, direct, _ = orchestrator.run_and_pick(cfg, str(DAY))
        saved, selection = load_pick_after_check(folder, DAY, cfg, statuses)
        assert saved.policy_decision == direct.pick_result.daily.policy_decision
        assert direct.action == selection.action == expected
        with patch('bts.dm.send_dm', return_value='captured-pair') as transport:
            assert deliver(saved, cfg, folder, selection)
        transport.assert_called_once()
        message = transport.call_args.args[1]
        assert 'Batter 0' in message and ('Batter 1' in message) == (expected == 'double')
        record = check_record(folder, saved, 'delivered')
        assert record['tail_policy_sha256'] == expected_sha
        assert (record['degraded_reason'] is None) == (expected_sha is not None)
        print('captured', label, record['action'], 'tail SHA', record['tail_policy_sha256'],
              'degraded', record['degraded_reason'])


def check_paired(root, fixture, tail):
    table = fixture['policy_table']
    short = table[:, :2].copy()
    one_bad_cell = table.copy().astype(np.uint64)
    one_bad_cell[-1, 0, 1, -1] = np.iinfo(np.uint64).max
    negative_cell = table.copy().astype(np.int32)
    negative_cell[-1, -1, 0, -1] = -1
    cases = {
        'scalar': {'policy_table': np.array(0, dtype=np.int8)},
        'rank1': {'policy_table': table[0, 0, 0]},
        'rank3': {'policy_table': table[0]},
        'rank5': {'policy_table': table[..., None]},
        'empty-streak': {'policy_table': table[:0]},
        'empty-days': {'policy_table': table[:, :0]},
        'empty-saver': {'policy_table': table[:, :, :0]},
        'empty-bin': {'policy_table': table[..., :0]},
        'saver-size3': {'policy_table': table[:, :, [0, 1, 1], :]},
        'streak-rows56': {'policy_table': table[:56]},
        'days-off-by-one': {'policy_table': table[:, :-1]},
        'bins-off-by-one': {'policy_table': table[..., :-1]},
        'float16': {'policy_table': table.astype(np.float16)},
        'complex64': {'policy_table': table.astype(np.complex64)},
        'bool-table': {'policy_table': table.astype(np.bool_)},
        'structured-table': {'policy_table': table.astype([('action', 'i1')])},
        'uint-overflow-action': {'policy_table': one_bad_cell},
        'negative-action': {'policy_table': negative_cell},
        'text-action': {'policy_table': table.astype('U1')},
        'horizon0': {'season_length': np.array(0)},
        'negative-horizon': {'season_length': np.array(-1)},
        'huge-horizon': {'season_length': np.array(2**64 - 1, dtype=np.uint64)},
        'text-horizon-invalid': {'season_length': np.array('not a number')},
        'vector-horizon': {'season_length': np.array([180, 180])},
        'scalar-boundaries': {'boundaries': np.array(.5)},
        'object-boundaries': {'boundaries': np.array([None] * 4)},
        'healthy-short': {'policy_table': short, 'season_length': np.array(1)},
        'usable-bool-horizon-on-disk': {'policy_table': short, 'season_length': np.array(True)},
        'usable-text-horizon-on-disk': {'policy_table': short, 'season_length': np.array('1')},
        'unicode-boundaries': {'boundaries': np.array(['low', '.70', '.80', '.90'])},
        'nested-boundaries': {'boundaries': fixture['boundaries'][:, None]},
        'complex-boundaries': {'boundaries': fixture['boundaries'].astype(np.complex128)},
    }
    wrong = []
    for label, changes in cases.items():
        folder = root / label
        folder.mkdir()
        base, tail_path, tail_sha = paired(folder, fixture | changes, tail)
        with files(base, tail_path):
            policy = strategy._load_mdp()
            decision = choose(policy)
            if label in {'unicode-boundaries', 'nested-boundaries', 'complex-boundaries'}:
                assert policy['base_sha256'] == policy['tail'].base_policy_sha256
                try:
                    mdp.lookup_action_guarded(policy['policy_table'], policy['boundaries'],
                                              56, 1, False, .84, policy['season_length'])
                except TypeError as exc:
                    print('failed reach-57 lookup', label, exc)
                else:
                    raise AssertionError('malformed boundary contents unexpectedly comparable')
                if decision.action == 'double':
                    assert decision.tail_sha256 == tail_sha and decision.degraded_reason is None
                    wrong.append(label)
                else:
                    assert_forced(decision)
                if label == 'unicode-boundaries':
                    send_pair(root, label, decision.action, decision.tail_sha256)
            elif label.startswith(('healthy-', 'usable-')):
                assert decision.action == 'double' and decision.tail_sha256 == tail_sha
                for boundaries in (tuple(policy['boundaries']), np.array(policy['boundaries'])):
                    assert_forced(choose(policy | {'boundaries': boundaries}))
                for horizon in (True, np.int64(1), '1', 1.0):
                    assert_forced(choose(policy | {'season_length': horizon}))
                readonly = policy['policy_table'].copy()
                readonly.flags.writeable = False
                mmap_path = folder / 'table.npy'
                np.save(mmap_path, readonly)
                mmap = np.load(mmap_path, mmap_mode='r')
                for array in (readonly, mmap):
                    assert choose(policy | {'policy_table': array}).tail_sha256 == tail_sha
                assert_forced(choose(policy | {'base_sha256': None}))
                if label == 'healthy-short':
                    send_pair(root, label, 'double', tail_sha)
            else:
                assert_forced(decision)
                if label == 'scalar':
                    send_pair(root, label, 'single', None)
            print('paired', label, decision.action, decision.degraded_reason)
    return wrong


def check_file_failures(root, fixture, tail):
    hits, both = [.72, .74], [.51, .55]
    two = replace(tail, boundaries=[.82], bin_p_hit=hits, bin_p_both=both, bin_freq=[.5, .5],
                  policy_table=tp.solve_emax_season_best([.5, .5], hits, both).policy,
                  manifest=tp.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2]))
    modes = ('base-missing', 'base-corrupt', 'base-directory', 'base-truncated', 'tail-missing',
             'tail-corrupt', 'tail-truncated', 'tail-directory', 'tail-mismatch', 'tail-two-bin',
             'tail-manifest', 'tail-invalid-rates', 'neither', 'two-byte-oversize')
    for label in modes:
        folder = root / label
        folder.mkdir()
        base, tail_path, _ = paired(folder, fixture, tail)
        target = base if label.startswith('base-') else tail_path
        if label.endswith('-missing'):
            target.unlink()
        elif label.endswith('-corrupt'):
            target.write_bytes(b'broken artifact')
        elif label.endswith('-directory'):
            target.unlink()
            target.mkdir()
        elif label.endswith('-truncated'):
            target.write_bytes(target.read_bytes()[:61])
        elif label == 'neither':
            base.unlink()
            tail_path.unlink()
        elif label == 'tail-mismatch':
            tp.save_tail_policy(replace(tail, base_policy_sha256='a' * 64), tail_path)
        elif label == 'tail-two-bin':
            tp.save_tail_policy(replace(two, base_policy_sha256=sha256(base.read_bytes()).hexdigest()), tail_path)
        elif label in {'tail-manifest', 'tail-invalid-rates'}:
            bad_tail = (replace(tail, manifest=tail.manifest | {'bin_counts': [None]})
                        if label == 'tail-manifest' else replace(tail, bin_p_both=[2.]))
            tp.save_tail_policy(replace(bad_tail, base_policy_sha256=sha256(base.read_bytes()).hexdigest()),
                                tail_path, validate=False)
        assert label != 'two-byte-oversize' or base.stat().st_size > 2
        limit = 2 if label == 'two-byte-oversize' else tp.MAX_ARTIFACT_BYTES
        with files(base, tail_path), patch.object(tp, 'MAX_ARTIFACT_BYTES', limit):
            with slate(folder / 'picks', DAY) as (cfg, _, _):
                _, selection, _ = orchestrator.run_and_pick(cfg, str(DAY))
                assert_forced(selection.decision)
                assert strategy.effective_pick_bar(57, str(DAY), False) == 0.
                print('on disk', label, selection.decision.degraded_reason)


def check_errors(root, shipped):
    for name in ('_load_mdp', '_validate_longest_base', 'longest_streak_action'):
        for error in (ValueError('probe failure'), KeyboardInterrupt(), SystemExit(14)):
            with slate(root / 'errors', DAY) as (cfg, _, _), patch.object(strategy, name, side_effect=error) as call:
                assert strategy.effective_pick_bar(57, str(DAY), False, shipped) == 0.
                call.assert_not_called()
                if isinstance(error, Exception):
                    _, selection, _ = orchestrator.run_and_pick(cfg, str(DAY))
                    assert_forced(selection.decision)
                    assert 'probe failure' in selection.decision.degraded_reason
                else:
                    try:
                        orchestrator.run_and_pick(cfg, str(DAY))
                    except type(error) as seen:
                        assert seen is error
                    else:
                        raise AssertionError(f'{name} swallowed {type(error).__name__}')
                call.assert_called_once()
    print('ordinary errors degrade and both interrupts propagate at loader, validator and rule')


if __name__ == '__main__':
    with np.load(mdp.DEFAULT_POLICY_PATH) as data:
        fixture = dict(data)
    artifact = strategy._load_mdp()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        wrong = check_paired(root, fixture, artifact['tail'])
        check_file_failures(root, fixture, artifact['tail'])
        check_errors(root, artifact)
    assert not wrong, f'malformed paired bases wrongly selected double: {wrong}'

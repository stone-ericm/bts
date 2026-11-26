from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import logging
import os
import shutil
import signal
import sys

import numpy as np


check = Path(__file__).resolve().parent
mode, family = sys.argv[1:]
scratch = check / 'scratch'
os.chdir(scratch)
sys.path.insert(0, str((check / 'base' if mode == 'base' else scratch) / 'src'))
import bts
from bts import strategy
from bts.health import tail_policy as health
from bts.simulate import mdp, tail_policy as tail


print('bts', bts.__file__, flush=True)
logging.disable(logging.CRITICAL)
loaded = strategy._load_mdp()
shipped_base, shipped_tail = Path(mdp.DEFAULT_POLICY_PATH), Path(tail.DEFAULT_TAIL_POLICY_PATH)
shipped = loaded['tail']


def context(p, day, artifact, best=57, status='trusted', allow=True, partner=True, second=.80):
    return strategy.DecisionContext(p, second if partner else None, partner, day, allow, artifact, best, status)


def run(streak, p, day, artifact, **kwargs):
    try:
        result = strategy.resolve_policy_decision(context(p, day, artifact, **kwargs), streak, False)
        return (result.action, result.policy_action, result.objective, result.days_effective,
                result.source, bool(result.tail_sha256), result.degraded_reason, result.guard)
    except Exception as exc:
        return (type(exc).__name__, str(exc))


if family == 'grid':
    totals = Counter()
    for raw, s, saver, p, allow, partner in product(range(1, 272), (57, 58, 60, 100), (False, True),
                                                    (.01, .50, .80, .99), (False, True), (False, True)):
        day = (date(2028, 9, 28) - timedelta(days=raw)).isoformat()
        ctx = context(p, day, loaded, allow=allow, partner=partner)
        result = strategy.resolve_policy_decision(ctx, s, saver)
        rule = 'double' if raw <= 3 else 'single'
        expected = 'double' if rule == 'double' and allow and partner else 'single'
        if mode == 'base':
            assert (result.action, result.policy_action, result.objective) == ('skip', 'skip', 'reach57')
        else:
            assert (result.action, result.policy_action, result.objective) == (expected, rule, 'longest_streak')
            assert (result.source, result.days_effective, result.tail_sha256, result.guard) == (
                'mdp', min(raw, 180), shipped.sha256, None)
        totals[result.action, result.policy_action, result.objective] += 1
    print('H1–H3', sum(totals.values()), totals)
    print('H6 shipped/no base', strategy.effective_pick_bar(57, '2027-07-15', False),
          strategy.effective_pick_bar(57, '2027-07-15', False, mdp=None))
    with patch.object(strategy, '_load_mdp', side_effect=TypeError('synthetic loader exception')):
        for s in (56, 57):
            try:
                print('H6 raising loader', s, strategy.effective_pick_bar(s, '2027-07-15', False))
            except Exception as exc:
                print('H6 raising loader', s, type(exc).__name__)
    for day in ('2027-09-26', '2027-09-27', '2027-12-31', '2028-01-01'):
        print('date', day, strategy._days_remaining(day), run(57, .99, day, loaded), run(57, .99, day, None))
    sys.exit(0)


class ReadTimeout(Exception):
    pass


def read_limit(signum, frame):
    raise ReadTimeout('FIFO read timed out')


with TemporaryDirectory() as folder:
    temp = Path(folder)
    for case in ('tail-missing', 'tail-corrupt', 'tail-directory', 'tail-mismatch', 'tail-two-bin',
                 'base-missing', 'base-corrupt', 'base-negative', 'base-negative-paired', 'all-missing',
                 'tail-malformed-manifest', 'base-fifo', 'tail-fifo'):
        loc = temp / case
        loc.mkdir()
        base_path, tail_path = loc / 'base.npz', loc / 'tail.npz'
        shutil.copyfile(shipped_base, base_path)
        shutil.copyfile(shipped_tail, tail_path)
        target = base_path if case.startswith('base') else tail_path
        if case == 'all-missing':
            base_path.unlink()
            tail_path.unlink()
        elif case.endswith('missing'):
            target.unlink()
        elif case.endswith('corrupt'):
            target.write_bytes(target.read_bytes()[:70])
        elif case.endswith('directory') or case.endswith('fifo'):
            target.unlink()
            if case.endswith('directory'):
                target.mkdir()
            else:
                os.mkfifo(target)
        elif case.endswith('mismatch'):
            base_path.write_bytes(base_path.read_bytes() + b'harmless trailer')
        elif 'negative' in case:
            with np.load(base_path) as f:
                values = dict(f)
            np.savez_compressed(base_path, **(values | {'season_length': np.array(-1)}))
            if case.endswith('paired'):
                tail.save_tail_policy(replace(shipped, base_policy_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest()), tail_path)
        elif case.endswith('two-bin'):
            freq, ph, pb = [.5, .5], [.75, .75], [.5, .5]
            policy = tail.solve_emax_season_best(freq, ph, pb).policy
            tail.save_tail_policy(replace(shipped, policy_table=policy, boundaries=[.8], bin_freq=freq,
                                         bin_p_hit=ph, bin_p_both=pb, manifest=tail.tail_manifest(
                                             n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2])), tail_path)
        elif case.endswith('manifest'):
            with np.load(tail_path) as f:
                values = dict(f)
            manifest = json.loads(str(values['manifest'])) | {'bin_counts': [None]}
            np.savez_compressed(tail_path, **(values | {'manifest': np.array(json.dumps(manifest))}))
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', base_path), patch.object(tail, 'DEFAULT_TAIL_POLICY_PATH', tail_path), \
                patch.object(strategy, '_mdp_cache', None):
            reads = {'selection prep': strategy._load_mdp}
            if case.endswith(('fifo', 'manifest')):
                reads['health'] = lambda: health.check(base_path=base_path, tail_path=tail_path, today=date(2027, 9, 26))
            for name, read in reads.items():
                previous = signal.signal(signal.SIGALRM, read_limit)
                signal.setitimer(signal.ITIMER_REAL, .20 if case.endswith('fifo') else 2)
                try:
                    artifact = read()
                except Exception as exc:
                    print(case, name, type(exc).__name__, str(exc))
                else:
                    if name == 'selection prep':
                        print(case, 'at 57/date 1 p=.01/.99', *[run(57, p, '2027-09-26', artifact) for p in (.01, .99)])
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)

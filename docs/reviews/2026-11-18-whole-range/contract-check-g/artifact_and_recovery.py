from pathlib import Path
import os
import signal
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import strategy
from bts.health import tail_policy as tail_health
from bts.simulate import mdp, tail_policy
from scripts.audit.season_walk import walk

from records_and_interrupts import DAY, NOW, Interrupt, delivery, readers


def artifacts(root):
    base = root / 'inadmissible.npz'
    tail = Path('data/models/mdp_tail_policy.npz').resolve()
    base.write_bytes(b'not an npz')
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', base), patch.object(strategy, '_mdp_cache', None):
        artifact = strategy._load_mdp()
        assert artifact['tail'] is None and 'unverifiable' in artifact['tail_error']
        for day, want in (('2027-09-17', ('emax_season_best', 'single')),
                          ('2027-09-20', ('emax_season_best', 'skip')),
                          (DAY, ('reach57', 'single'))):
            context = strategy.DecisionContext(0.96, None, False, day, True, artifact, 18, 'trusted')
            decision = strategy.resolve_policy_decision(context, 0, False)
            assert (decision.objective, decision.action) == want
    directory = root / 'directory'
    directory.mkdir()
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', directory), patch.object(strategy, '_mdp_cache', None):
        assert 'IsADirectoryError' in strategy._load_mdp()['base_error']

    fifo = root / 'fifo'
    os.mkfifo(fifo)

    def read_deadline(*_):
        raise Interrupt('FIFO reader deadline')

    def expect_block(label, read):
        signal.setitimer(signal.ITIMER_REAL, 0.15)
        try:
            read()
        except Interrupt:
            print('waits at unwritten FIFO:', label)
        else:
            raise AssertionError(f'{label} returned instead of blocking')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    original = signal.signal(signal.SIGALRM, read_deadline)
    try:
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', fifo), patch.object(strategy, '_mdp_cache', None):
            expect_block('strategy base loader', strategy._load_mdp)
        expect_block('tail loader', lambda: tail_policy.load_tail_policy(fifo))
        expect_block('tail health base hash', lambda: tail_health.check(base_path=fifo, tail_path=tail))
    finally:
        signal.signal(signal.SIGALRM, original)


if __name__ == '__main__':
    with TemporaryDirectory(dir='.codex-review/wr-check-g') as name:
        folder = Path(name)
        with walk.offline(NOW):
            readers(folder / 'readers')
            (folder / 'artifacts').mkdir()
            artifacts(folder / 'artifacts')
            delivery(folder / 'delivery')

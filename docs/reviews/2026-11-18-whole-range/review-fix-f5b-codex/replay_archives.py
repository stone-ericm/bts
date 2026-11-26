from importlib import import_module
from pathlib import Path
import os
import runpy
import sys
from tempfile import TemporaryDirectory

from review_support import ARCHIVE, CONTRACT, HERE, LEGACY, ROOT, ROUND2, World, assert_source, load_line, installed


def activate(scratch):
    import alert_probe
    import support
    import marker_health
    import fallback_recovery
    import state_sequences
    import morning_fifo

    def source(version):
        assert_source((scratch / 'base' if version == 'base' else ROOT) / 'src')

    for mod in (alert_probe, support, marker_health, fallback_recovery, state_sequences, morning_fifo):
        mod.ROOT = ROOT
        mod.load_line = load_line
        mod.source = source

    sys.path.insert(0, str(CONTRACT))
    contract = import_module('replay_support')
    contract.HERE = HERE
    contract.ROOT = ROOT
    contract.ARCHIVE = ARCHIVE
    contract.SCRATCH = scratch
    contract.World = World
    contract.source = source
    contract.load_line = load_line
    contract.installed = lambda root, version, intent='enter', original='': installed(
        root, (scratch / 'base' if version == 'base' else ROOT) / 'scripts/cron-setup-hetzner.sh', intent, original)
    contract.install_archive_adapters()


def archived_main(path, args):
    sys.argv = [str(path), *args]
    runpy.run_path(str(path), run_name='__main__')


def every_case(script, path):
    targets = {
        'alert_probe': (('closing_fixture', ()), ('shared_window', ('bind-wrong',)),
                        ('shared_window', ('remove-entry',)), ('shared_window', ('late-commit',)),
                        ('revised_window', (False,)), ('revised_window', (True,)),
                        ('corrected_pitch', ()), ('previous_date_health', ())),
        'state_replay': tuple((n, ()) for n in ('manual_carries_tiers', 'duplicate_clear',
                         'slot_gained_then_lost', 'moving_double', 'early_after_original')),
        'fallback_recovery': (('stale_copy', ()), ('decode_failure', ())),
        'contract_rows': (('revised', ()), ('revised', (True,)), ('revised', (False, True)),
                          ('revised', (False, False, True)), ('clear', ()), ('year_boundary', ())),
        'watchdog_sequences': (('schedule', ()), ('separate', ())),
    }
    module = import_module('state_sequences' if script == 'state_replay' else script)
    with TemporaryDirectory() as dirname:
        for index, (name, args) in enumerate(targets[script]):
            try:
                print('case', name, args, flush=True)
                getattr(module, name)(Path(dirname) / str(index), 'f5', *args)
            except Exception as exc:
                print('original assertion/setup result:', type(exc).__name__, str(exc), flush=True)


if __name__ == '__main__':
    script, folder, variant, scratch_path = sys.argv[1:]
    scratch = Path(scratch_path)
    mode = 'base' if variant == 'base' else 'f5'
    assert_source((scratch / 'base' if mode == 'base' else ROOT) / 'src')
    activate(scratch)
    path = {'r1': LEGACY, 'r2': ROUND2, 'contract': CONTRACT}[folder] / (script + '.py')
    if variant == 'cases':
        every_case(script, path)
    elif script == 'morning_fifo':
        with TemporaryDirectory() as dirname:
            root = Path(dirname)
            world = World(root)
            world.commit_at('10:05', at='09:40')
            assert world.cron('09:45')['receipt']['early_cron_run'] is True
            pick = world.picks / f'{world.day}.json'
            pick.unlink()
            os.mkfifo(pick)
            print('actual morning:', world.morning(), flush=True)
            archived_main(path, [dirname, 'f5'])
    else:
        args = {'mutation_witness': ['candidate'], 'round_one': ['replay'], 'operator_text': [],
                'pitch_correction': [], 'lock_boundary': [], 'mutations': ['ledger']}.get(script, [mode])
        archived_main(path, args)

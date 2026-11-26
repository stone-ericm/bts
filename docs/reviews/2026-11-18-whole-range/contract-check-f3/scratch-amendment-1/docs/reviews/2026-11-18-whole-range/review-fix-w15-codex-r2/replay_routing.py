import configparser
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from review_evidence import Clock, build, seed
from review_features import CACHE, HEAD, pa


REPO = HEAD.parent
SOURCE = 'probable_pitcher_cache'
STATUS = Path('data/health_state') / SOURCE
PICKS = Path('data/picks')
DAY = date(2026, 12, 31)
SCRIPT = str(Path(__file__).resolve())


class Wall(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2027, 1, 1, 5, 1, tzinfo=timezone.utc).astimezone(tz)


def run_scheduler(offday=False):
    from bts import scheduler
    from bts.health import alert, runner

    sent = []
    safe_run = runner._safe_run
    game = dict(gamePk=100, gameDate=f'{DAY}T18:10:00Z', doubleHeader='N', gameNumber=1,
                status={'abstractGameCode': 'P', 'detailedState': 'Scheduled'},
                teams={'away': {'team': {'name': 'A'}}, 'home': {'team': {'name': 'B'}}})

    def transport(recipient, message):
        assert recipient == 'offline'
        sent.append(message)
        return 'synthetic'

    with (patch.object(runner, '_safe_run', lambda source, thunk: safe_run(source, thunk) if source == SOURCE else []),
          patch.object(runner.realized_calibration, '_current_deploy_iso', return_value=None),
          patch.object(runner, 'read_systemd_nrestarts', return_value=None),
          patch.object(runner, 'datetime', Wall), patch.object(alert, 'datetime', Wall),
          patch.object(alert, 'send_dm', transport),
          patch.object(scheduler, 'fetch_schedule', side_effect=[[] if offday else [game], []]),
          patch.object(scheduler, '_now_et', return_value=Wall.now(scheduler.ET)),
          patch.object(scheduler, 'run_single_check', return_value={'skipped': True, 'new_lineups': 0,
                                                                   'pick_result': None, 'should_post': False}),
          patch.object(scheduler, 'notify_ready'), patch.object(scheduler, 'notify_watchdog'),
          patch.object(scheduler, '_idle_until_next_wakeup')):
        scheduler.run_day(str(DAY), {'orchestrator': {'picks_dir': str(PICKS)},
                                     'scheduler': {'pick_delivery': 'private'}, 'tiers': [],
                                     'bluesky': {'dm_recipient': 'offline'}})
    if offday:
        assert not sent
        print('no-games scheduler branch ended with no EOD report')
    else:
        assert len(sent) == 1 and 'first game IDs' not in sent[0]
        print('end-of-day scheduler reached real health and captured one summary')


def run_manual_health():
    from bts.health import alert, runner

    safe_run = runner._safe_run
    with (patch.object(runner, '_safe_run', lambda source, thunk: safe_run(source, thunk) if source == SOURCE else []),
          patch.object(runner.realized_calibration, '_current_deploy_iso', return_value=None),
          patch.object(runner, 'datetime', Wall), patch.object(alert, 'datetime', Wall),
          patch.object(alert, 'send_dm', return_value='offline')):
        _, = runner.run_all_checks(PICKS, CACHE.parent, 'offline', today=DAY)
    print('direct health under a cron-style redirect reached its real logging path')


def run_preview():
    import pandas as pd
    from bts.cli import preview
    from bts.features import compute, park_drag, probable_pitcher_cache
    from bts.model import predict

    compute._probable_pitcher_cache_path = CACHE
    probable_pitcher_cache.datetime = Clock
    raw_pa = pd.DataFrame([pa(100, pd.Timestamp('2025-04-01'), 3, 0)])

    def fixture_pipeline(*args, **kwargs):
        # Exercise the real producer at the boundary reached by preview's pipeline.
        compute.compute_all_features(raw_pa)
        return pd.DataFrame()

    CACHE.parent.mkdir(parents=True)
    CACHE.write_bytes(b'{"100":null}')
    with patch.object(predict, 'run_pipeline', fixture_pipeline), patch.object(park_drag, 'get_table', return_value=None):
        preview.callback(str(DAY + timedelta(days=1)), 'data/processed', str(PICKS), str(CACHE.parent))
    assert len(list(STATUS.glob('*.json'))) == 1


def child(mode):
    from bts.features import probable_pitcher_cache

    assert logging.getLogger().handlers == []
    probable_pitcher_cache.datetime = Clock
    Clock.at = datetime(2027, 1, 1, 4, 59, tzinfo=timezone.utc)
    if mode == 'preview':
        run_preview()
    else:
        seed(b'{"100":null}', feeds=False)
        build()
        (STATUS / f'{DAY}.bad.json').write_bytes(b'{')
        if mode in ('scheduler', 'offday'):
            run_scheduler(offday=mode == 'offday')
        else:
            run_manual_health()


def main():
    unit_path = REPO / 'scripts/systemd/bts-scheduler.service'
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.read(unit_path)
    unit = config['Service']
    assert 'uv run bts schedule' in unit['ExecStart']
    print('scheduler unit command:', unit['ExecStart'])
    print('scheduler unit stream directives:', unit.get('StandardOutput'), unit.get('StandardError'))
    cron = (REPO / 'scripts/cron-setup-hetzner.sh').read_text()
    preview_line = next(line for line in cron.splitlines() if line.startswith('0 3 '))
    assert 'run bts preview' in preview_line and '>> $LOG_DIR/cron.log 2>&1' in preview_line
    print('installed cron preview route:', preview_line)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent / 'offline'), str(HEAD)]),
               PYTHONDONTWRITEBYTECODE='1', TZ='America/New_York')
    details = 'had 1 invalid entry (first game IDs 100)'
    bad = 'Probable-pitcher cache status unreadable at'
    with TemporaryDirectory(dir=HERE, prefix='routing-') as name:
        work = Path(name)
        for mode in ('scheduler', 'offday', 'manual-health', 'preview'):
            folder = work / mode
            folder.mkdir()
            child_env = dict(env, MPLCONFIGDIR=str(folder / 'mpl'), TMPDIR=str(folder))
            command = [sys.executable, SCRIPT, mode]
            if mode in ('scheduler', 'offday'):
                result = subprocess.run(command, cwd=folder, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, timeout=12)
                output = result.stderr
                if mode == 'offday':
                    assert 'No games scheduled' in output and details not in output and bad not in output, output
                else:
                    assert details in output and bad in output and details not in result.stdout, (result.stdout, output)
                print(f'real scheduler {mode} stderr (journal-bound under stock systemd defaults):\n' + output)
            else:
                cron_capture = folder / 'cron.log'
                with cron_capture.open('ab') as stream:
                    result = subprocess.run(command, cwd=folder, env=child_env, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=12)
                output = cron_capture.read_text()
                if mode == 'preview':
                    assert 'WARNING: probable-pitcher cache' in output and details not in output and bad not in output
                else:
                    assert details in output and bad in output
                print(mode, 'combined cron.log stream:\n' + output)
            assert result.returncode == 0, (mode, output)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()

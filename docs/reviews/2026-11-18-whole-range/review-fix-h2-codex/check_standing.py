import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe_standing as standing
from run_standing import bounded


fixture = standing.fixture


def capture_all_dms(stack, *args, **kwargs):
    config, games, frame, dm = fixture(stack, *args, **kwargs)
    stack.enter_context(patch('bts.health.alert.send_dm', dm))
    return config, games, frame, dm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    print(standing.cs.__file__)
    with patch.object(standing, 'describe', bounded), patch.object(standing, 'fixture', capture_all_dms):
        if args.rollback:
            standing.rollback(args.rollback)
        else:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                standing.interruption(root / 'interrupt')
                standing.malformed(root / 'malformed')
                standing.rollover(root / 'year')
                standing.delivery(root / 'delivery')

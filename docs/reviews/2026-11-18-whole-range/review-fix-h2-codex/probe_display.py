from pathlib import Path
from tempfile import TemporaryDirectory

from bts import picks
from probe_day import dashboard
from support import DAY, at, install, walk


if __name__ == '__main__':
    for mode in ('dated', 'automatic'):
        for active in (58, 100, 400):
            with TemporaryDirectory() as tmp, walk.offline(at()):
                directory = Path(tmp) / 'picks'
                install(directory, mode, active, active)
                picks.save_streak(6, directory, saver_available=True, season=DAY.year)
                dashboard(directory, active)
                print(mode, 'dashboard contest streak', active, 'with separate replay streak 6')

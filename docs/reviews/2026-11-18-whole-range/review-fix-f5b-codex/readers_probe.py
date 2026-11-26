import csv
from datetime import date, timedelta
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory

from bts.data.backup import BACKUP_SETS
from review_support import ROOT, World, assert_source


assert_source(ROOT / 'src')
with TemporaryDirectory() as dirname:
    root = Path(dirname)
    instance = World(root / 'data')
    instance.commit_at('06:10', at='04:55')
    attempt = instance.cron('05:00', dm='failed')
    assert attempt['receipt']['early_cron_run'] and instance.early.is_file()
    covered = [parent for parent in BACKUP_SETS['ops'].paths if instance.early.relative_to(root).is_relative_to(parent)]
    assert covered == ['data/health_state']
    print('backup:', instance.early.relative_to(root), 'included by', covered[0], flush=True)
    subprocess.run(['bash', '-n', str(ROOT / 'scripts/cron-setup-hetzner.sh')], timeout=3, check=True)
    data = root / 'short.csv'
    with data.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['date', 'slot', 'p', 'outcome'])
        first = date(2027, 3, 24)
        writer.writerows((first + timedelta(days=n), 1, 0.72, 'miss' if n % 3 == 0 else 'hit') for n in range(21))
    tool = ROOT / 'scripts/audit/pvr_threshold_bootstrap.py'
    subprocess.run([sys.executable, str(tool), str(data)], timeout=15, check=True)

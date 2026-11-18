import os
from pathlib import Path
import runpy
from time import perf_counter


PROBE = Path(__file__).resolve().parent
REPO = PROBE.parent.parent
DATA = Path('/Users/eric/projects/bts/data')
LOOKUP = Path('/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json')
ROOT = PROBE / 'audit-root'

(ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
(ROOT / 'data/models').mkdir(parents=True, exist_ok=True)
for year in (2023, 2024, 2025):
    name = f'pa_{year}.parquet'
    target = ROOT / 'data/processed' / name
    if not target.is_symlink():
        target.symlink_to(DATA / 'processed' / name)
cache = ROOT / 'data/models/probable_pitcher_lookup.json'
if not cache.is_symlink():
    cache.symlink_to(LOOKUP)

os.chdir(ROOT)
start = perf_counter()
runpy.run_path(str(REPO / 'scripts/leakage_audit.py'), run_name='__main__')
print(f'audit including Audit 4 elapsed seconds: {perf_counter() - start:.3f}')

from pathlib import Path
import runpy
import sys


check = Path(__file__).resolve().parent
sys.path.insert(0, str(check / 'scratch'))
from scripts.audit.season_ledger.ids import Parsed

Parsed.observations = property(lambda result: result.rows)
runpy.run_path(str(check / 'decision_probe.py'), run_name='__main__')

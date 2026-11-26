"""Run one group of the brief's exact search matches."""
from pathlib import Path
import subprocess
import sys

import bts
import pytest

PATTERN = (r'check_pick_entered|check-pick-entered|_check_pick_entered|pick_entry_status|entry_receipt|'
           r'EntryReceipt|qualify\(|load_units|target_rows|pick_entry_check|_selection')


def groups():
    matched = subprocess.run(['rg', '-l', PATTERN, 'tests', 'scripts'], capture_output=True,
                             text=True, check=True, timeout=5).stdout.splitlines()
    python_tests = sorted(p for p in matched if p.startswith('tests/') and p.endswith('.py'))
    f1 = [p for p in python_tests if p.startswith('tests/test_wr_f1_')]
    audit = ['tests/test_audit_driver.py']
    other = [p for p in python_tests if p not in f1 + audit]
    return {'f1': f1, 'audit': audit, **{str(i // 8 + 1): other[i:i + 8] for i in range(0, len(other), 8)}}


if __name__ == '__main__':
    print(f'bts.__file__ = {bts.__file__}', flush=True)
    selected = groups()
    if sys.argv[1] == 'list':
        for group, files in selected.items():
            print(group, len(files), *files, sep='\n')
        print('total files', sum(map(len, selected.values())))
    else:
        files = selected[sys.argv[1]]
        assert all(Path(file).is_file() for file in files)
        print(*files, sep='\n', flush=True)
        raise SystemExit(pytest.main(['-p', 'no:cacheprovider', '-q', '--tb=short', *files, *sys.argv[2:]]))

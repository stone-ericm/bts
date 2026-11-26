from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from probe_mutations import (MUTATIONS, REVIEW, SOURCE, boundary_checks, test_changes,
                             verify_fix)


def verify_logs(names, suite):
    for name in names:
        evidence = (REVIEW / f'mutation-{name}-{suite}.log').read_text()
        assert (' failed' in evidence or ' passed' in evidence) and 'ERROR collecting' not in evidence, name


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        # macOS resolves /var to /private/var, including in an imported module path.
        src = Path(tmp).resolve() / 'src'
        shutil.copytree(SOURCE, src, ignore=shutil.ignore_patterns('__pycache__'))
        match sys.argv[1]:
            case 'boundary':
                boundary_checks(src)
            case 'fix':
                verify_fix(src)
            case 'group1':
                names = list(MUTATIONS)[:11]
                test_changes(src, names, False)
                verify_logs(names, 'h')
            case 'group2':
                names = list(MUTATIONS)[11:]
                test_changes(src, names, False)
                verify_logs(names, 'h')
            case 'wider':
                names = sys.argv[2:]
                test_changes(src, names, True)
                verify_logs(names, 'all')

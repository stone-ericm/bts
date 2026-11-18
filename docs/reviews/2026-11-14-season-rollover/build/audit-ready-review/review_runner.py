"""Run each probe twice in independent Python processes and compare the captured output."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
BASE = REVIEW / 'base'
PYTHON = ROOT / '.venv/bin/python'


def archive(revision, target, *paths):
    payload = subprocess.check_output(['git', 'archive', revision, *paths], cwd=ROOT)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as source:
        source.extractall(target, filter='data')


def environment(source):
    return os.environ | {
        'PYTHONPATH': os.pathsep.join(map(str, (REVIEW, source / 'src', source, ROOT))),
        'PYTHONDONTWRITEBYTECODE': '1',
        'TZ': 'America/New_York',
    }


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def run(source, *args, cwd=ROOT):
    return subprocess.run([str(PYTHON), *map(str, args)], cwd=cwd, env=environment(source)).returncode


def revision_probe(probe, compare):
    import sitecustomize

    if len(sys.argv) > 1:
        label = sys.argv[1]
        source = BASE if label == 'base' else ROOT
        import bts.cli

        assert Path(bts.cli.__file__).resolve().is_relative_to(source / 'src')
        sitecustomize.verify()
        directory = REVIEW / Path(sys.argv[0]).stem / label
        directory.mkdir(parents=True, exist_ok=True)
        probe(directory, source)
        return 0
    destination = REVIEW / Path(sys.argv[0]).stem
    results = [run(source, Path(sys.argv[0]).resolve(), label)
               for label, source in [('base', BASE), ('head', ROOT)]]
    if any(results):
        return 1
    compare(destination)
    return 0


if __name__ == '__main__':
    archive('be217307', BASE, 'src', 'scripts')

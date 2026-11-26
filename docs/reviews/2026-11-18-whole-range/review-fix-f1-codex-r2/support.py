"""File-backed review fixture with the repository's captured transport leaves."""
from contextlib import contextmanager
from itertools import count
import json
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory

import bts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVED_REVIEW = REPO / 'docs/reviews/2026-11-18-whole-range/review-fix-f1-codex'
fixture = runpy.run_path(str(REPO / 'tests/test_wr_f1_wrong_game.py'))
World, DAY, PITCH, row = (fixture[name] for name in ('World', 'DAY', 'PITCH', 'row'))
RIGHT = {'id': 77, 'roundId': 1, 'feedId': 101}
WRONG = {**RIGHT, 'feedId': 102}
RIGHT_DD = {'id': 78, 'roundId': 1, 'feedId': 201}
WRONG_DD = {**RIGHT_DD, 'feedId': 202}
MATCH, DOUBLE = [row()], [row(), row(78, 2, 2)]
ALL_CLEAR = f'✅ BTS pick entry confirmed for {DAY} (match).'


def print_source():
    print(f'bts.__file__ = {bts.__file__}', flush=True)


class Arena:
    def __init__(self, tag):
        self.tmp = TemporaryDirectory(prefix=f'{tag}-', dir=HERE)
        self.ids = count()

    def world(self, day=DAY):
        return World(Path(self.tmp.name) / str(next(self.ids)), day)

    def clear(self, world):
        import shutil
        shutil.rmtree(world.root)


@contextmanager
def world(tag, day=DAY):
    with TemporaryDirectory(prefix=f'{tag}-', dir=HERE) as root:
        yield World(Path(root), day)


def read_marker_bytes(w):
    if not w.marker.is_file():
        return None
    raw = w.marker.read_bytes()
    return re.sub(rb'("(?:receipt|checked_at)"\s*:\s*)"[^"]*"', rb'\1"<normalized>"', raw).decode(errors='backslashreplace')


def normalized(value, root):
    if isinstance(value, dict):
        return {key: normalized(item, root) for key, item in value.items()
                if key not in {'receipt', 'attempt_id', 'confirmed_by', 'checked_at', 'started_at',
                               'published_at', 'response_completed_at', 'failed_at'}}
    if isinstance(value, list):
        return [normalized(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), '<fixture>')
    return value


def dump(label, value):
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def current_receipt(w, result):
    marker = result['marker'] or {}
    return next((item for item in result['receipts'] if item['attempt_id'] == marker.get('receipt')),
                result['receipts'][-1] if result['receipts'] else {})

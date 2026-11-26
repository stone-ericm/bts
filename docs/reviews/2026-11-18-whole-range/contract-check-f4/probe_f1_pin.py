"""Verify the F1 comparison hash with just the new null stdout key removed."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

SCRATCH = Path(__file__).resolve().parents[1] / 'scratch-f4'
sys.path[:0] = [str(SCRATCH / 'src'), str(SCRATCH)]

from tests.test_wr_f1_wrong_game import BASE_COMPARISON, World, test_unchanged_comparison_grid

original_morning = World.morning


def old_stdout_projection(world):
    result = original_morning(world)
    assert result['outcome'].pop('streak_problem') is None
    return result


with patch.object(World, 'morning', old_stdout_projection):
    for case in BASE_COMPARISON:
        with TemporaryDirectory() as tmp:
            test_unchanged_comparison_grid(World(Path(tmp)), case)
        print(case, 'original digest matches')

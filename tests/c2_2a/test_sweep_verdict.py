"""The sweep's witness rule (`tests/c2_2a/sweep.py`, `witness_only_loses`): a faulted run's witness may only lose
information against the unfaulted run's. A digest of a sibling part (`samples_sha256` of `samples`, `map_sha256` of
`map`) cannot keep the unfaulted value when that part lost a field, so it is accepted only as exactly the canonical
digest of the faulted run's own published part, which must itself only lose."""
import pytest

from tests.c2_2a.sweep import witness_only_loses
from tests.c2_2a.test_golden import _canon

SAMPLES = [{"file": "2026-06-01.json", "file_sha256": "a" * 64, "p": 0.77, "y": 1},
           {"file": "2026-06-02.json", "file_sha256": "b" * 64, "p": 0.71, "y": 0}]


def _cal(samples):
    return {"calibration": {"samples": samples, "samples_sha256": None if samples is None else _canon(samples)}}


def test_an_unchanged_witness_passes():
    assert witness_only_loses(_cal(SAMPLES), _cal(SAMPLES)) == []


def test_the_digest_of_a_part_that_only_lost_a_field_is_accepted():
    lossy = [dict(SAMPLES[0], file_sha256=None), SAMPLES[1]]
    assert witness_only_loses(_cal(SAMPLES), _cal(lossy)) == []


def test_a_digest_that_is_not_the_published_parts_is_refused():
    lossy = [dict(SAMPLES[0], file_sha256=None), SAMPLES[1]]
    faulted = _cal(lossy)
    faulted["calibration"]["samples_sha256"] = "c" * 64
    assert witness_only_loses(_cal(SAMPLES), faulted)


@pytest.mark.parametrize("changed", [{"p": 0.99}, {"file_sha256": "d" * 64}, {"batter_id": 7}])
def test_a_part_that_says_more_or_otherwise_is_refused_with_its_digest(changed):
    faulted = _cal([dict(SAMPLES[0], **changed), SAMPLES[1]])
    assert witness_only_loses(_cal(SAMPLES), faulted)


def test_a_digest_without_its_part_is_refused():
    faulted = _cal(SAMPLES)
    faulted["calibration"]["samples"] = None
    faulted["calibration"]["samples_sha256"] = "e" * 64
    assert witness_only_loses(_cal(SAMPLES), faulted)

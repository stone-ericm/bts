"""Replay each of the original stopped-branch corpus histories on each archived source tree."""

from datetime import datetime
import importlib.util
import sys
from unittest.mock import patch

from probe_common import ET, REVIEW, examine, run_against_archives, validate


def load_fixture(name):
    spec = importlib.util.spec_from_file_location(f"tests.{name}", REVIEW / "c3" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


def probes(scratch, revision):
    reconcile = load_fixture("test_wr_c3_reconcile")
    with patch.dict(sys.modules, {"tests.test_wr_c3_reconcile": reconcile}):
        corpus = load_fixture("test_wr_c3_corpus")
    results = {}
    for case in corpus.cases():
        for index in range(len(corpus.ORDINARY)):
            name = f"{case.name}-{index}"
            root = scratch / name / "picks"
            corpus.seed_streak(root, season=2026)
            corpus.build_history(root, case, index)
            results[name] = examine(root, clock=lambda: datetime(2026, 8, 20, 2, tzinfo=ET))
    assert len(results) == 630
    return results


if __name__ == "__main__":
    run_against_archives(__file__, probes, validate)

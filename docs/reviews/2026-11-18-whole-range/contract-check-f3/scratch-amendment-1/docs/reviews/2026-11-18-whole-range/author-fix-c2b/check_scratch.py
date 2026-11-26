"""Verify an isolated receipt variant with the named regression assertion."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
from io import StringIO
import signal
import sys

from check_offline import run
from tests import test_wr_c2_receipt_identity as identity
from bts.health import alert


def carbon_base():
    source = Path(__file__).parent / "scratch/base/carbon_original.py"
    spec = importlib.util.spec_from_file_location("carbon_original", source)
    carbon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(carbon)
    counts = []
    for route in ("direct", "classification"):
        for seed in carbon.test_foreign_or_noncanonical_receipt_never_suppresses_c2.pytestmark[0].args[1]:
            counts.append(run(f"{route} {seed}", lambda world, mp, _: (
                carbon.test_foreign_or_noncanonical_receipt_never_suppresses_c2(world, mp, route, seed)
            )))
        stderr = StringIO()
        def double(world, mp, _):
            with patch("sys.stderr", stderr):
                carbon.test_double_down_unconfirmed_locks_double_without_naming_either_batter_and_handles_null_bluesky(
                    world, mp, SimpleNamespace(readouterr=lambda: SimpleNamespace(err=stderr.getvalue())), route,
                )
        counts.append(run(f"{route} double", double))
    counts.append(run("swapped inputs", lambda world, _, root: (
        carbon.test_opt_in_multi_source_receipt_rejects_unsorted_sources_and_dedups_swapped_inputs(world, root)
    )))
    counts.append(run("watchdog", lambda world, mp, _: (
        carbon.test_recommendation_send_returning_none_matches_ledgered_watchdog_divergence(world, mp)
    )))
    assert len(counts) == 18 and all(counts)
    print("18 GREEN")


def main(variant):
    assert Path(alert.__file__).is_relative_to(Path(__file__).resolve().parent / "scratch" / variant)
    if variant == "base":
        carbon_base()
        return
    case = {
        "drop-keys": "extra-error",
        "allow-superset": "extra-unknown",
        "allow-duplicates": "duplicate-status",
        "drop-hash-check": "foreign-body",
        "hash-constant": "exact-body-hash",
        "hash-alert-message": "exact-body-hash",
        "compare-day-only": "foreign-body",
    }[variant]
    for route in ("direct", "classification"):
        def operation(world, monkeypatch, directory):
            encounter = identity.encounter.__wrapped__(world, monkeypatch, route)
            if case == "exact-body-hash":
                identity.test_receipt_hash_matches_exact_transport_body(encounter)
            elif case == "foreign-body":
                identity.test_foreign_dispatcher_receipts_do_not_suppress_c2(encounter, case, directory)
            else:
                document = next(item.values[0] for item in identity.extra_cases() if item.id == case)
                identity.test_rejects_invalid_receipt(encounter, document)
        assert not run(f"{route} {case}", operation)
    if variant == "allow-duplicates":
        with TemporaryDirectory() as root:
            try:
                identity.test_reader_rejects_duplicate_members_inside_nested_objects(Path(root))
            except AssertionError:
                print("RED reader-rejects-nested-duplicates")
            else:
                raise AssertionError("Nested duplicate mutant survived")


if __name__ == "__main__":
    signal.alarm(50)
    main(sys.argv[1])

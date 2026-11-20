"""Run receipt fixtures directly while the shared pytest wrapper is busy."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import signal
from unittest.mock import patch

from pytest import MonkeyPatch

from tests import test_wr_c2_r2_carbon as carbon
from tests import test_wr_c2_receipt_identity as identity


def run(name, operation):
    with TemporaryDirectory() as root, MonkeyPatch.context() as monkeypatch:
        directory = Path(root)
        world = identity.world.__wrapped__(directory, monkeypatch)
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                operation(world, monkeypatch, directory)
        except AssertionError:
            print(f"RED {name}")
            return False
    print(f"GREEN {name}")
    return True


def main():
    counts = []
    for route in ("direct", "classification"):
        def check(test, arg):
            def operation(world, monkeypatch, directory):
                encounter = identity.encounter.__wrapped__(world, monkeypatch, route)
                if test is identity.test_foreign_dispatcher_receipts_do_not_suppress_c2:
                    test(encounter, arg, directory)
                elif test is identity.test_receipt_hash_matches_exact_transport_body:
                    test(encounter)
                else:
                    test(encounter, arg)
            return operation

        for case in [*identity.extra_cases(), *identity.wrong_field_cases()]:
            counts.append(run(f"{route} {case.id}", check(identity.test_rejects_invalid_receipt, case.values[0])))
        for seed in ("foreign-body", "legacy-cutoff", "legacy-with-id", "other-slate"):
            counts.append(run(f"{route} {seed}", check(identity.test_foreign_dispatcher_receipts_do_not_suppress_c2, seed)))
        for rewrite in ("none", "order", "whitespace"):
            counts.append(run(f"{route} genuine-{rewrite}", check(identity.test_genuine_receipt_deduplicates, rewrite)))
        counts.append(run(f"{route} exact-body-hash", check(identity.test_receipt_hash_matches_exact_transport_body, None)))
    for route in ("direct", "classification"):
        for seed in carbon.test_foreign_or_noncanonical_receipt_never_suppresses_c2.pytestmark[1].args[1]:
            counts.append(run(f"carbon {route} {seed}", lambda world, mp, _: (
                carbon.test_foreign_or_noncanonical_receipt_never_suppresses_c2(world, mp, route, seed)
            )))
    counts.append(run("carbon swapped", lambda world, _, root: (
        carbon.test_opt_in_multi_source_receipt_rejects_unsorted_sources_and_dedups_swapped_inputs(world, root)
    )))
    counts.append(run("carbon watchdog", lambda world, mp, _: (
        carbon.test_recommendation_send_returning_none_matches_ledgered_watchdog_divergence(world, mp)
    )))
    for route in ("direct", "classification"):
        stderr = StringIO()
        class Capture:
            def readouterr(self):
                from types import SimpleNamespace
                return SimpleNamespace(err=stderr.getvalue())
        def double(world, mp, _):
            with patch("sys.stderr", stderr):
                carbon.test_double_down_unconfirmed_locks_double_without_naming_either_batter_and_handles_null_bluesky(
                    world, mp, Capture(), route,
                )
        counts.append(run(f"carbon double {route}", double))
    print(f"{sum(counts)} GREEN, {len(counts) - sum(counts)} RED")


if __name__ == "__main__":
    signal.alarm(50)
    main()

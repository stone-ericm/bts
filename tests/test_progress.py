"""Tests for the in-process progress beacon (H5b truthful heartbeat)."""
import logging

from bts import progress


def setup_function(_fn):
    # beacon is module-global; isolate tests by retiring any active run
    if progress._run_id is not None:
        progress.end_run(progress._run_id)


def test_begin_run_seeds_cascade_starting():
    rid = progress.begin_run("primary")
    snap = progress.snapshot(rid)
    assert snap is not None
    assert snap["stage"] == "cascade_starting"
    assert snap["kind"] == "primary"
    assert snap["stage_age_s"] >= 0
    assert snap["generation"] == 0


def test_mark_is_stage_entry_and_closes_previous():
    rid = progress.begin_run("primary")
    progress.mark("computing_features")
    snap = progress.snapshot(rid)
    assert snap["stage"] == "computing_features"
    assert snap["generation"] == 1
    rows = progress.drain_transitions(rid)
    assert [r["stage"] for r in rows] == ["cascade_starting"]
    assert rows[0]["duration_s"] >= 0
    assert rows[0]["generation"] == 0
    # drained -> gone
    assert progress.drain_transitions(rid) == []


def test_mark_without_active_run_is_noop():
    progress.mark("orphan_stage")  # must not raise
    assert progress.snapshot("anything") is None


def test_snapshot_rejects_foreign_and_retired_run_ids():
    rid = progress.begin_run("primary")
    assert progress.snapshot("not-" + rid) is None
    progress.end_run(rid)
    assert progress.snapshot(rid) is None


def test_begin_run_supersedes_prior_run():
    rid1 = progress.begin_run("primary")
    rid2 = progress.begin_run("shadow")
    assert progress.snapshot(rid1) is None
    assert progress.snapshot(rid2)["kind"] == "shadow"


def test_end_run_returns_final_transitions():
    rid = progress.begin_run("primary")
    progress.mark("selecting_pick")
    final = progress.end_run(rid)
    assert [r["stage"] for r in final] == ["cascade_starting", "selecting_pick"]
    # idempotent / foreign-safe
    assert progress.end_run(rid) == []


def test_drain_with_mismatched_run_id_returns_empty_without_clearing():
    rid = progress.begin_run("primary")
    progress.mark("stage_b")
    assert progress.drain_transitions("wrong") == []
    assert len(progress.drain_transitions(rid)) == 1


def test_history_bound_drops_oldest(caplog):
    rid = progress.begin_run("primary")
    with caplog.at_level(logging.WARNING):
        for i in range(progress.HISTORY_BOUND + 10):
            progress.mark(f"stage_{i}")
    rows = progress.drain_transitions(rid)
    assert len(rows) == progress.HISTORY_BOUND
    # oldest (cascade_starting + stage_0..stage_9) dropped; newest retained
    assert rows[-1]["stage"] == f"stage_{progress.HISTORY_BOUND + 8}"
    assert any("overflow" in r.message for r in caplog.records)

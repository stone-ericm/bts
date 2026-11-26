"""Run the four original cases, five original writer producers, and nine Carbon cases."""

from datetime import datetime
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch

from corpus_probe import load_fixture
from probe_common import REVIEW, RUN, examine, receipt, run_against_archives, validate


def probes(scratch, revision):
    import pytest

    from bts import picks
    from tests import test_wr_c3_r2_entered_pick_blockers as carbon
    from tests import test_wr_c3v_delivery as delivery
    from tests.test_wr_c3v_corpus import dump, load, pick_data, seed_streak, stagger

    reconcile = load_fixture("test_wr_c3_reconcile")
    with patch.dict(sys.modules, {"tests.test_wr_c3_reconcile": reconcile}):
        corpus = load_fixture("test_wr_c3_corpus")
        with patch.dict(sys.modules, {"tests.test_wr_c3_corpus": corpus}):
            first_round = load_fixture("test_wr_c3_delivery")

    old_support = {
        "compare_probes": SimpleNamespace(BASE=None, HEAD=None, archive=None, offline=None, python=None, scratch=None),
        "delivery_probe": delivery,
        "probe_support": SimpleNamespace(dump=dump, load=load, pick_data=pick_data, plant=delivery.plant,
                                         run_cli=None, seed_streak=seed_streak, stagger=stagger),
        "reconcile_probe": SimpleNamespace(menu=None, receipts=None),
    }
    spec = importlib.util.spec_from_file_location("prior_writers", REVIEW / "c3" / "writer_probes.py")
    prior = importlib.util.module_from_spec(spec)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return RUN.astimezone(tz)

    results = {}
    with patch.dict(sys.modules, old_support), patch.object(picks, "datetime", FixedDatetime), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}):
        spec.loader.exec_module(prior)
        for scenario in (
            "lost_ack", "skip_conflict", "private_crash", "manual_run_crash", "lost-ack-eod",
            "private-preview-restart", "private-preview-quarantine", "scheduler-preview-ack",
            "orchestrator-post-interrupted",
        ):
            root = scratch / scenario / "picks"
            for offset in range(9):
                delivery.plant(root, stagger("2027-04-01", offset), "hit", bluesky_posted=True)
            seed_streak(root, streak=9, saver=True)
            if scenario == "manual_run_crash":
                events = first_round.manual_run_commit(root)
            elif scenario in {"lost_ack", "skip_conflict", "private_crash"}:
                events = first_round.scheduler_commit(root, scenario)
            elif scenario == "orchestrator-post-interrupted":
                events = prior.orchestrator_commit(root)
            else:
                if scenario == "lost-ack-eod":
                    delivery.plant(root, delivery.ENTERED)
                elif scenario == "scheduler-preview-ack":
                    prior.preview(root)
                events = prior.scheduler_commit(root, scenario)
            assert len(events) == 1 and picks.load_pick(delivery.ENTERED, root).result is None
            delivery.plant(root, delivery.DAY, bluesky_posted=True)
            with patch.object(picks, "resolve_daily_slot_results", return_value={"pick": "hit"}):
                scored = delivery.check_cli(root, delivery.DAY)
                assert scored.exit_code == 0 and "Streak: 10" in scored.stdout, scored.output
            with patch.object(picks, "resolve_daily_slot_results", return_value={"pick": "miss"}):
                results[scenario] = examine(root, lookback=1)
            assert picks.load_pick(delivery.DAY, root).result == "miss"

        class Compared(BaseException):
            pass

        for index, scenario in enumerate((
            "lost_ack_then_fallback_skip_auto", "lost_ack_then_fallback_skip_dated",
            "manual_run_crash_with_daemon_eod_skip", "live_private_preview_lock_restart",
            "private_research_fresh_pick_restart", "live_private_preview_lock_quarantined",
            "orchestrate_not_live_post_crash", "preview_overwrites_unmarked_entered_pick", "b2",
        ), start=1):
            name = "b2" if scenario == "b2" else f"N{index}"

            def compare_here(root, monkeypatch, base_modules, *, clock, lookback, result_feed):
                with patch.object(picks, "resolve_daily_slot_results", result_feed):
                    results[name] = examine(root, clock, lookback)
                raise Compared

            with pytest.MonkeyPatch.context() as monkeypatch, patch.object(carbon, "compare_reconcile", compare_here):
                try:
                    if scenario == "b2":
                        carbon.test_b2_symlinked_decision_parent_keeps_unmarked_live_pick(scratch / name, monkeypatch, None)
                    else:
                        carbon.test_entered_pick_never_skipped_by_season_replay(scratch / name, monkeypatch, None, scenario)
                except Compared:
                    pass
                else:
                    raise AssertionError(f"producer did not reach reconcile: {name}")
    return results


def check(base, current):
    counts = validate(base, current)
    assert counts == {"unavailable": 18}, counts
    for name, evidence in current.items():
        fold, code, error, before, after = evidence["strict"]
        assert (fold, code, error) == (["return", None], 0, None), name
        assert before == after and before is not None, name
        assert receipt(evidence)["replay"] == {
            "state": "unavailable", "blocked_date": "2027-04-16" if name == "b2" else "2027-04-18",
            "reason": "no final result on the pick",
        }, name
        if name != "b2":
            assert "hit -> miss\nStreak retained: 10\n" in evidence["stdout"], name
    return counts


if __name__ == "__main__":
    run_against_archives(__file__, probes, check)

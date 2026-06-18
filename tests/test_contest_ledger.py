"""Tests for the per-round MLB contest ledger parser (bts.contest_ledger).

Each ledger line is one fetch snapshot {recorded_at, active_streak, predictions:[...]}.
The latest fetch's predictions are the rounds; a round is `stable` only when the same
(result, streak) for that roundId also appeared in the PREVIOUS fetch (two-read
confirmation), because the latest values are provisional and can still change.
"""
import json

from bts.contest_ledger import parse_latest_ledger, infer_saver, LedgerRound


def test_parses_per_round_with_pre_and_post_streak(tmp_path):
    led = tmp_path / "contest_ledger.jsonl"
    # two fetch rows; the LATEST row's predictions are parsed
    led.write_text("\n".join(json.dumps(r) for r in [
        {"recorded_at": "2026-06-16T17:00:00Z", "active_streak": 7, "predictions": []},
        {"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 8, "predictions": [
            {"roundId": 904, "result": "hit", "streak": 6, "streakIncrease": 1,
             "roundPredictions": [{"playerId": 1, "result": "hit"}]},
            {"roundId": 905, "result": "hit", "streak": 7, "streakIncrease": 1,
             "roundPredictions": [{"playerId": 2, "result": "hit"}]},
            {"roundId": 903, "result": "hit", "streak": 5, "streakIncrease": 2,
             "roundPredictions": [{"playerId": 3, "result": "hit"}, {"playerId": 4, "result": "hit"}]},
        ]},
    ]))
    rounds = parse_latest_ledger(led)
    assert [r.round_id for r in rounds] == [903, 904, 905]   # sorted by roundId
    r905 = rounds[-1]
    assert isinstance(r905, LedgerRound)
    assert r905.post_streak == 7 and r905.pre_streak == 6      # pre = prior round's post
    assert rounds[0].is_dd is True and rounds[1].is_dd is False  # 903 had 2 slots


def test_missing_ledger_returns_empty(tmp_path):
    assert parse_latest_ledger(tmp_path / "nope.jsonl") == []


def test_round_is_stable_when_unchanged_across_two_fetches(tmp_path):
    led = tmp_path / "contest_ledger.jsonl"
    round_a = {"roundId": 904, "result": "hit", "streak": 6, "streakIncrease": 1,
               "roundPredictions": [{"playerId": 1, "result": "hit"}]}
    led.write_text("\n".join(json.dumps(r) for r in [
        {"recorded_at": "2026-06-16T17:00:00Z", "active_streak": 6, "predictions": [round_a]},
        {"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 6, "predictions": [round_a]},
    ]))
    rounds = parse_latest_ledger(led)
    assert len(rounds) == 1 and rounds[0].stable is True


def test_round_is_unstable_when_value_changed_or_single_fetch(tmp_path):
    # (a) value changed between the two fetches -> provisional
    led = tmp_path / "contest_ledger.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in [
        {"recorded_at": "2026-06-16T17:00:00Z", "active_streak": 5, "predictions": [
            {"roundId": 904, "result": None, "streak": 5,
             "roundPredictions": [{"playerId": 1, "result": None}]}]},
        {"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 6, "predictions": [
            {"roundId": 904, "result": "hit", "streak": 6,
             "roundPredictions": [{"playerId": 1, "result": "hit"}]}]},
    ]))
    assert parse_latest_ledger(led)[0].stable is False

    # (b) only one fetch -> cannot confirm -> provisional
    led2 = tmp_path / "single.jsonl"
    led2.write_text(json.dumps(
        {"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 6, "predictions": [
            {"roundId": 904, "result": "hit", "streak": 6,
             "roundPredictions": [{"playerId": 1, "result": "hit"}]}]}))
    assert parse_latest_ledger(led2)[0].stable is False


def test_pre_streak_not_fabricated_after_missing_post(tmp_path):
    # a round with a missing post streak must break the chain: the next round's
    # pre_streak is unrecoverable (None), not the stale prior value (adjustment #3).
    led = tmp_path / "contest_ledger.jsonl"
    led.write_text(json.dumps(
        {"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 8, "predictions": [
            {"roundId": 901, "result": "hit", "streak": 5,
             "roundPredictions": [{"playerId": 1, "result": "hit"}]},
            {"roundId": 902, "result": "void", "streak": None,
             "roundPredictions": [{"playerId": 2, "result": "void"}]},
            {"roundId": 903, "result": "hit", "streak": 7,
             "roundPredictions": [{"playerId": 3, "result": "hit"}]},
        ]}))
    by_id = {r.round_id: r for r in parse_latest_ledger(led)}
    assert by_id[902].post_streak is None
    assert by_id[903].pre_streak is None    # NOT 5 -- chain broke at the None-post round


# --- saver inference ---

def _r(rid, result, pre, post, is_dd=False, stable=True):
    return LedgerRound(rid, result, pre, post, None, is_dd, stable)


# best_streak < 10: the account never reached the 10-15 zone this season, so the saver was
# never consumable -> provably available, regardless of ledger contents (best_streak is a
# reliable season-max counter, immune to ledger windowing).

def test_saver_available_when_best_streak_below_zone():
    assert infer_saver([_r(1, "hit", 7, 8), _r(2, "hit", 8, 9)], best_streak=9) == "available"


def test_saver_available_when_best_streak_below_zone_without_any_rounds():
    assert infer_saver([], best_streak=5) == "available"


# best_streak >= 10: the account reached the saver zone, so the saver MAY have been consumed.

def test_saver_consumed_on_stable_miss_at_10_15_that_did_not_reset():
    # not_hit at pre-streak 12, post 12 (didn't reset) -> the mulligan absorbed it
    assert infer_saver([_r(1, "hit", 11, 12), _r(2, "not_hit", 12, 12)], best_streak=12) == "consumed"


def test_saver_unknown_when_reached_zone_but_no_confirmed_consumption():
    # reached 12 but no visible consuming miss -> can't confirm the season saver survived
    # without complete coverage (Phase 2b) -> unknown (conservatively unavailable)
    assert infer_saver([_r(1, "hit", 10, 11), _r(2, "hit", 11, 12)], best_streak=12) == "unknown"


def test_saver_unknown_on_ambiguous_dd_miss_at_10_15():
    # a not_hit at 10-15 that didn't reset BUT is a DD (one slot may have missed) -> ambiguous
    assert infer_saver([_r(1, "hit", 11, 12), _r(2, "not_hit", 12, 12, is_dd=True)], best_streak=12) == "unknown"


def test_saver_unknown_on_unstable_consuming_round():
    # a clear-looking consumption that is still provisional (single-read) -> unknown (adj #2)
    assert infer_saver([_r(1, "hit", 11, 12), _r(2, "not_hit", 12, 12, stable=False)], best_streak=12) == "unknown"


def test_saver_unknown_when_pre_streak_unrecoverable():
    assert infer_saver([_r(1, "not_hit", None, 11)], best_streak=12) == "unknown"


def test_saver_unknown_on_empty_or_no_best_streak():
    # no best_streak signal and no confirmed consumption -> conservatively unavailable (adj #1)
    assert infer_saver([]) == "unknown"
    assert infer_saver([], best_streak=12) == "unknown"


def test_saver_consumed_evidence_overrides_low_best_streak():
    # ledger evidence of a 10-15 consumption wins over best_streak < 10 -- guards an
    # under-reported best_streak (stale counter / wrong manual override) from a false-available
    assert infer_saver([_r(1, "hit", 11, 12), _r(2, "not_hit", 12, 12)], best_streak=9) == "consumed"


def test_saver_not_available_at_best_streak_exactly_10():
    # reaching exactly streak 10 makes the saver consumable, so best_streak == 10 is NOT
    # auto-available (the < 10 boundary is correct)
    assert infer_saver([_r(1, "hit", 9, 10)], best_streak=10) == "unknown"


def test_saver_zone_edges_inclusive_10_to_15():
    # consumption fires at the inclusive pre-streak edges 10 and 15...
    assert infer_saver([_r(1, "not_hit", 10, 10)], best_streak=12) == "consumed"
    assert infer_saver([_r(1, "not_hit", 15, 15)], best_streak=15) == "consumed"
    # ...but a miss at pre-streak 16 is above the zone -> a reset, not a saver consumption
    assert infer_saver([_r(1, "not_hit", 16, 0)], best_streak=16) == "unknown"

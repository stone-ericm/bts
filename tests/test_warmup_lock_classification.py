"""Warmup lock-classification semantics (2026-08-13 incident + Codex round-2).

Warmup (statusCode PW) is the only MLB state that is abstract-Live but
pre-first-pitch. An existing undelivered candidate must stay deliverable through
it — but the carve-out must NOT reopen picks whose durable markers say a
delivery was attempted (crash idempotency), and must NOT widen the fresh-
candidate pool into warmup games (contest entry closes T-5; selection stays
conservative).
"""
from unittest.mock import patch

from bts.picks import (
    DailyPick,
    Pick,
    classify_pick_lock_state,
    get_game_statuses_detailed,
    pick_candidate_status_is_available,
    save_pick,
)

DATE = "2026-08-13"
WARMUP = {"abstract": "L", "detailed": "Warmup"}
PREVIEW = {"abstract": "P", "detailed": "Scheduled"}


def _daily(**kw):
    d = DailyPick(
        date=DATE, run_time="2026-08-13T16:24:00+00:00",
        pick=Pick(batter_name="Arraez", batter_id=650333, team="PHI",
                  lineup_position=4, pitcher_name="Bradley", pitcher_id=671737,
                  p_game_hit=0.767, flags=[], projected_lineup=True,
                  game_pk=823669, game_time="2026-08-13T23:30:00Z"),
        double_down=None, runner_up=None,
    )
    for k, v in kw.items():
        setattr(d, k, v)
    return d


class TestWarmupClassification:
    @patch("bts.picks.get_game_statuses_detailed", return_value={823669: WARMUP})
    def test_undelivered_pick_in_warmup_not_locked(self, _s):
        state = classify_pick_lock_state(_daily(), DATE)
        assert state.locked is False
        assert state.stale is False

    @patch("bts.picks.get_game_statuses_detailed", return_value={823669: WARMUP})
    def test_delivery_attempted_pick_locked_even_in_warmup(self, _s):
        """Crash-gap idempotency: delivery_attempted=True with no confirmed
        delivery means an unconfirmed send. Reclassifying it as refreshable
        would let select_pick overwrite the marker and re-send (Codex r2 #1)."""
        state = classify_pick_lock_state(_daily(delivery_attempted=True), DATE)
        assert state.locked is True
        assert state.reason == "delivery_attempt_unconfirmed"

    @patch("bts.picks.get_game_statuses_detailed", return_value={823669: PREVIEW})
    def test_delivery_attempted_pick_locked_in_preview_too(self, _s):
        """The same erase-the-marker hazard existed in Preview before the
        warmup carve-out; the durable marker must win over any game status."""
        state = classify_pick_lock_state(_daily(delivery_attempted=True), DATE)
        assert state.locked is True
        assert state.reason == "delivery_attempt_unconfirmed"

    def test_statuscode_pw_recognized_without_detailed_text(self):
        """statusCode is the machine signal; detailedState is display text.
        A PW status with unexpected display text must still classify warmup."""
        with patch("bts.picks.get_game_statuses_detailed", return_value={
            823669: {"abstract": "L", "detailed": "", "code": "PW"},
        }):
            state = classify_pick_lock_state(_daily(), DATE)
        assert state.locked is False
        assert state.stale is False


class TestCandidateAvailability:
    def test_warmup_game_not_available_for_fresh_candidates(self):
        """Fresh-selection pool stays conservative: entry closes T-5 before
        scheduled first pitch, so warmup games are not offered to the optimizer
        even though an existing committed candidate remains deliverable."""
        assert pick_candidate_status_is_available(WARMUP) is False

    def test_preview_game_available(self):
        assert pick_candidate_status_is_available(PREVIEW) is True

    def test_started_game_not_available(self):
        assert pick_candidate_status_is_available(
            {"abstract": "L", "detailed": "In Progress"}) is False


class TestDetailedStatusFetchShape:
    def test_fetch_includes_status_code(self, monkeypatch):
        import io
        import bts.picks as picks_mod

        payload = (b'{"dates": [{"games": [{"gamePk": 823669, "status": '
                   b'{"abstractGameCode": "L", "detailedState": "Warmup", '
                   b'"statusCode": "PW"}}]}]}')
        monkeypatch.setattr(picks_mod, "retry_urlopen",
                            lambda *a, **k: io.BytesIO(payload))
        statuses = get_game_statuses_detailed(DATE)
        assert statuses[823669]["code"] == "PW"
        assert statuses[823669]["abstract"] == "L"
        assert statuses[823669]["detailed"] == "Warmup"

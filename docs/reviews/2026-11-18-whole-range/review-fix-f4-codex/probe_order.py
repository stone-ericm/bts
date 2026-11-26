"""Expose an early streak snapshot by changing the file during real day classification."""

import json
from unittest.mock import patch

from bts import day_outcome
from report_harness import _box, _write_decision, report


def test_reads_once_after_skip_review(tmp_path):
    picks, config = _box(tmp_path)
    _write_decision(picks, action="skip", scoreable=False, streak=1, saver_available=True)
    path = picks / "streak.json"
    path.write_bytes(b"{}")
    events = []
    really_review = day_outcome._skip_review
    really_read = day_outcome.read_regular_file_text

    def classify_skip(*args, **kwargs):
        outcome = really_review(*args, **kwargs)
        events.append("classified")
        path.write_bytes(b"{")
        return outcome

    def read_streak(candidate, **kwargs):
        assert candidate == path
        events.append("read")
        return really_read(candidate, **kwargs)

    with patch.object(day_outcome, "_skip_review", classify_skip), \
         patch.object(day_outcome, "read_regular_file_text", read_streak):
        result, chat = report(picks, config, 1)
    assert events == ["classified", "read"], (events, result.stdout, chat.attempted)
    assert result.exit_code == 1 and json.loads(result.stdout)["streak_problem"] == "not valid JSON"
    assert len(chat.accepted) == 1 and "MODEL STREAK FILE NEEDS REVIEW" in chat.accepted[0]

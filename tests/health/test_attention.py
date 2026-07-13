from datetime import date

from bts.health.alert import Alert
from bts.health.attention import apply_warn_attention_policy


def test_repeated_warn_enters_attention_on_second_consecutive_day(tmp_path):
    state_path = tmp_path / "warn_attention.json"
    first = [Alert("WARN", "predicted_vs_realized", "drift +0.08")]
    policy, attention = apply_warn_attention_policy(
        first,
        state_path=state_path,
        today=date(2026, 5, 20),
    )
    assert policy == []
    assert attention == []

    second = [Alert("WARN", "predicted_vs_realized", "drift +0.11")]
    policy, attention = apply_warn_attention_policy(
        second,
        state_path=state_path,
        today=date(2026, 5, 21),
    )
    assert policy == []
    assert len(attention) == 1
    assert attention[0].source == "predicted_vs_realized"
    assert "2nd consecutive day" in attention[0].message


def test_calibration_drift_warn_enters_attention_on_repeat(tmp_path):
    state_path = tmp_path / "warn_attention.json"
    apply_warn_attention_policy(
        [Alert("WARN", "calibration_drift", "top-1 P below floor")],
        state_path=state_path,
        today=date(2026, 5, 20),
    )

    policy, attention = apply_warn_attention_policy(
        [Alert("WARN", "calibration_drift", "top-1 P below floor")],
        state_path=state_path,
        today=date(2026, 5, 21),
    )

    assert policy == []
    assert [a.source for a in attention] == ["calibration_drift"]
    assert "2nd consecutive day" in attention[0].message


def test_projected_lineup_warn_enters_attention_on_repeat(tmp_path):
    state_path = tmp_path / "warn_attention.json"
    apply_warn_attention_policy(
        [Alert("WARN", "projected_lineup", "high projected lineup share")],
        state_path=state_path,
        today=date(2026, 5, 20),
    )

    policy, attention = apply_warn_attention_policy(
        [Alert("WARN", "projected_lineup", "high projected lineup share")],
        state_path=state_path,
        today=date(2026, 5, 21),
    )

    assert policy == []
    assert [a.source for a in attention] == ["projected_lineup"]


def test_leaderboard_freshness_warn_enters_attention_on_repeat(tmp_path):
    state_path = tmp_path / "warn_attention.json"
    apply_warn_attention_policy(
        [Alert("WARN", "leaderboard_freshness", "leaderboard scrape lagging")],
        state_path=state_path,
        today=date(2026, 5, 20),
    )

    policy, attention = apply_warn_attention_policy(
        [Alert("WARN", "leaderboard_freshness", "leaderboard scrape lagging")],
        state_path=state_path,
        today=date(2026, 5, 21),
    )

    assert policy == []
    assert [a.source for a in attention] == ["leaderboard_freshness"]


def test_always_attention_warn_enters_digest_immediately(tmp_path):
    policy, attention = apply_warn_attention_policy(
        [Alert("WARN", "postponed_pick", "status lookup failed")],
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert policy == []
    assert len(attention) == 1
    assert attention[0].source == "postponed_pick"


def test_live_forward_resolution_warn_enters_digest_immediately(tmp_path):
    policy, attention = apply_warn_attention_policy(
        [Alert("WARN", "live_forward_resolution", "canonical resolution stalled")],
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert policy == []
    assert len(attention) == 1
    assert attention[0].source == "live_forward_resolution"


def test_memory_warn_with_oom_evidence_promotes_policy_critical(tmp_path):
    alerts = [
        Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
        Alert("CRITICAL", "analytics_artifacts_missing", "Result=oom-kill"),
    ]
    policy, attention = apply_warn_attention_policy(
        alerts,
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert any(a.source == "memory_oom_correlation" for a in policy)
    assert attention == []


def test_warn_oom_evidence_alone_promotes_policy_critical(tmp_path):
    alerts = [
        Alert("WARN", "analytics_artifacts_missing", "shadow missing Result=oom-kill"),
    ]
    policy, attention = apply_warn_attention_policy(
        alerts,
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert any(a.source == "analytics_job_oom" for a in policy)
    assert len(attention) == 1


def test_oom_evidence_does_not_match_innocent_words(tmp_path):
    alerts = [
        Alert("WARN", "analytics_artifacts_missing", "shadow worker room unavailable"),
    ]
    policy, attention = apply_warn_attention_policy(
        alerts,
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert not any(a.source == "analytics_job_oom" for a in policy)
    assert len(attention) == 1


def test_existing_critical_oom_does_not_duplicate_generic_oom_alert(tmp_path):
    alerts = [
        Alert("CRITICAL", "analytics_artifacts_missing", "capture Result=oom-kill"),
    ]
    policy, attention = apply_warn_attention_policy(
        alerts,
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert not any(a.source == "analytics_job_oom" for a in policy)
    assert attention == []


def test_memory_warn_with_critical_oom_adds_correlation_context(tmp_path):
    alerts = [
        Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
        Alert("CRITICAL", "analytics_artifacts_missing", "capture Result=oom-kill"),
    ]
    policy, attention = apply_warn_attention_policy(
        alerts,
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert any(a.source == "memory_oom_correlation" for a in policy)
    assert attention == []


def test_memory_warn_with_restart_spike_enters_attention(tmp_path):
    alerts = [
        Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
        Alert("CRITICAL", "restart_spike", "NRestarts +7"),
    ]
    policy, attention = apply_warn_attention_policy(
        alerts,
        state_path=tmp_path / "warn_attention.json",
        today=date(2026, 5, 21),
    )
    assert policy == []
    assert len(attention) == 1
    assert attention[0].source == "memory_growth"


def test_new_f1_f3_sources_are_always_attention():
    # Codex review #4: WARNs from pick_entry / scheduler_state_integrity must
    # reach the DM channel, not just the journal — both are rare + actionable.
    from bts.health.attention import ALWAYS_ATTENTION_WARN_SOURCES
    assert "pick_entry" in ALWAYS_ATTENTION_WARN_SOURCES
    assert "scheduler_state_integrity" in ALWAYS_ATTENTION_WARN_SOURCES


def test_with_streak_preserves_incident_key():
    # Round-2 review #4 follow-up: reconstruction must not drop the dedup
    # identity, or distinct incidents sharing a source degrade to
    # source-level dedup downstream.
    from bts.health.attention import _with_streak
    a = Alert("WARN", "s", "m", incident_key="s:x")
    assert _with_streak(a, 2).incident_key == "s:x"


def test_realized_calibration_is_attention_listed():
    # 2026-07-12: the DD-band bucket makes realized_calibration the absolute
    # -level monitor for chronic slot miscalibration; without attention
    # membership its WARNs never reach the DM digest no matter how long
    # they persist.
    from bts.health.attention import REPEATED_ATTENTION_WARN_SOURCES
    assert "realized_calibration" in REPEATED_ATTENTION_WARN_SOURCES


def test_streaks_keyed_by_incident_not_source(tmp_path):
    # r2#3 of the DD-band work: realized_calibration's two buckets are
    # distinct incidents. Bucket A warning yesterday must not make bucket
    # B's first WARN today read as "2nd consecutive day".
    a = Alert("WARN", "realized_calibration", "75-80% over",
              incident_key="realized_calibration:75-80%")
    b = Alert("WARN", "realized_calibration", "70-75% DD over",
              incident_key="realized_calibration:70-75% DD-leg")
    state = tmp_path / "warn_state.json"
    policy_day1, att1 = apply_warn_attention_policy(
        [a], state_path=state, today=date(2026, 7, 16))
    policy_day2, att2 = apply_warn_attention_policy(
        [b], state_path=state, today=date(2026, 7, 17))
    # Day-2 bucket B is streak 1 → below the min streak → NOT in attention.
    assert att2 == []

"""since_deploy_iso pick filter must compare INSTANTS, not ISO strings.

run_time is written in +00:00; since_deploy_iso comes from git `%cI` which uses
the committer's local offset (e.g. -04:00). A lexicographic compare of the two
strings is wrong whenever the offsets differ — it can keep a genuinely
pre-deploy pick or drop a post-deploy one.
"""
from bts.health.realized_calibration import _iso_before


def test_post_deploy_pick_not_treated_pre_deploy_across_offsets():
    # pick 19:00-04:00 == 23:00Z is AFTER deploy 20:00Z — but "19" < "20" lexicographically
    assert _iso_before("2026-04-29T19:00:00-04:00", "2026-04-29T20:00:00+00:00") is False


def test_pre_deploy_pick_correctly_before_across_offsets():
    # pick 18:00Z == 14:00 ET is BEFORE deploy 14:30-04:00 == 18:30Z — but "18" > "14"
    assert _iso_before("2026-04-29T18:00:00+00:00", "2026-04-29T14:30:00-04:00") is True


def test_unparseable_run_time_treated_as_pre_deploy():
    assert _iso_before("", "2026-04-29T00:00:00+00:00") is True
    assert _iso_before("not-a-date", "2026-04-29T00:00:00+00:00") is True


def test_naive_run_time_assumed_utc():
    assert _iso_before("2026-04-29T10:00:00", "2026-04-29T12:00:00+00:00") is True

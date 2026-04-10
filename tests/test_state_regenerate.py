"""Tests for bts state regenerate.

These tests deliberately import the real formatters from bts.posting and feed
their output through the parser. That guarantees the parser tracks any future
change to the post format instead of drifting against fictional fixtures.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bts.posting import format_post, format_skip_post
from bts.state.regenerate import (
    fetch_bluesky_posts,
    parse_pick_from_post,
    ParsedPost,
)


def _make_post(text: str, uri: str, created_at: str, is_reply: bool = False):
    """Build a fake atproto post response."""
    post = MagicMock()
    post.post.uri = uri
    post.post.record.text = text
    post.post.record.created_at = created_at
    if is_reply:
        post.post.record.reply = MagicMock()
    else:
        post.post.record.reply = None
    return post


def test_fetch_bluesky_posts_returns_posts_in_order():
    fake_posts = [
        _make_post("pick 3", "at://post3", "2026-04-03T12:00:00Z"),
        _make_post("pick 2", "at://post2", "2026-04-02T12:00:00Z"),
        _make_post("pick 1", "at://post1", "2026-04-01T12:00:00Z"),
    ]
    fake_response = MagicMock()
    fake_response.feed = fake_posts
    fake_response.cursor = None

    with patch("bts.state.regenerate._bluesky_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.get_author_feed.return_value = fake_response
        mock_client_factory.return_value = mock_client

        posts = fetch_bluesky_posts(handle="test.bsky.social", from_date="2026-04-01")

    # Should be sorted chronologically
    assert len(posts) == 3
    assert posts[0].uri == "at://post1"
    assert posts[-1].uri == "at://post3"


def test_parse_pick_post_extracts_single_pick():
    # Use the real formatter so the parser is exercised against production text.
    text = format_post("Nico Hoerner", "CHC", "Test Pitcher", 0.783, streak=2)
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.batter_name == "Nico Hoerner"
    assert parsed.team == "CHC"
    assert parsed.is_skip is False
    assert parsed.is_double_down is False
    assert parsed.double_down_batter is None
    assert parsed.streak_at_time == 2


def test_parse_pick_post_extracts_double_down():
    text = format_post(
        "Jose Altuve", "HOU", "Pitcher A", 0.82, streak=5,
        double="Kyle Tucker", double_p_game=0.80,
        double_team="HOU", double_pitcher="Pitcher B",
    )
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.batter_name == "Jose Altuve"
    assert parsed.team == "HOU"
    assert parsed.is_double_down is True
    assert parsed.double_down_batter == "Kyle Tucker"
    assert parsed.double_down_team == "HOU"
    assert parsed.streak_at_time == 5


def test_parse_double_down_without_second_team():
    # The formatter omits the team parens on the second line if double_team is None.
    text = format_post(
        "Jose Altuve", "HOU", "Pitcher A", 0.82, streak=5,
        double="Kyle Tucker", double_p_game=0.80,
    )
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.is_double_down is True
    assert parsed.batter_name == "Jose Altuve"
    assert parsed.double_down_batter == "Kyle Tucker"
    assert parsed.double_down_team is None


def test_parse_skip_post():
    text = format_skip_post("Top Batter", "NYY", 0.765, streak=3)
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.is_skip is True
    assert parsed.batter_name == "Top Batter"
    assert parsed.team == "NYY"
    assert parsed.streak_at_time == 3


def test_parse_unrecognized_post_returns_none():
    text = "random promotional content, not a pick"
    parsed = parse_pick_from_post(text)
    assert parsed is None


from bts.state.regenerate import (
    parse_result_from_reply,
    reconstruct_pick_timeline,
    Timeline,
    HistoricalPickRecord,
)


def test_parse_result_reply_hit():
    from bts.posting import format_result_reply
    text = format_result_reply("hit", 3)
    parsed = parse_result_from_reply(text)
    assert parsed.is_result is True
    assert parsed.result == "hit"
    assert parsed.streak_after == 3


def test_parse_result_reply_miss():
    from bts.posting import format_result_reply
    text = format_result_reply("miss", 0)
    parsed = parse_result_from_reply(text)
    assert parsed.is_result is True
    assert parsed.result == "miss"
    assert parsed.streak_after == 0


def test_parse_result_reply_not_a_result():
    parsed = parse_result_from_reply("Random reply text")
    assert parsed.is_result is False


def test_reconstruct_timeline_alternates_picks_and_results():
    posts = [
        ParsedPost(uri="at://p1", created_at="2026-04-01T12:00:00Z",
                   text="pick A", is_reply=False,
                   batter_name="A", team="NYY"),
        ParsedPost(uri="at://r1", created_at="2026-04-01T23:00:00Z",
                   text="hit reply", is_reply=True,
                   is_result=True, result="hit", streak_after=1),
        ParsedPost(uri="at://p2", created_at="2026-04-02T12:00:00Z",
                   text="pick B", is_reply=False,
                   batter_name="B", team="BOS"),
        ParsedPost(uri="at://r2", created_at="2026-04-02T23:00:00Z",
                   text="miss reply", is_reply=True,
                   is_result=True, result="miss", streak_after=0),
    ]

    timeline = reconstruct_pick_timeline(posts)
    assert len(timeline.pick_records) == 2
    assert timeline.pick_records[0].date == "2026-04-01"
    assert timeline.pick_records[0].batter_name == "A"
    assert timeline.pick_records[0].result == "hit"
    assert timeline.pick_records[0].bluesky_uri == "at://p1"
    assert timeline.final_streak == 0
    assert timeline.pick_records[1].result == "miss"


def test_reconstruct_timeline_handles_unresolved_last_day():
    posts = [
        ParsedPost(uri="at://p1", created_at="2026-04-01T12:00:00Z",
                   text="pick A", is_reply=False,
                   batter_name="A", team="NYY"),
        ParsedPost(uri="at://r1", created_at="2026-04-01T23:00:00Z",
                   text="hit", is_reply=True,
                   is_result=True, result="hit", streak_after=1),
        ParsedPost(uri="at://p2", created_at="2026-04-02T12:00:00Z",
                   text="pick B", is_reply=False,
                   batter_name="B", team="BOS"),
        # No reply for p2 — still in progress or regeneration runs mid-day
    ]
    timeline = reconstruct_pick_timeline(posts)
    assert len(timeline.pick_records) == 2
    assert timeline.pick_records[1].result is None
    assert timeline.final_streak == 1  # Last known resolved streak


def test_saver_consumed_on_first_miss_at_streak_10():
    """Saver fires on first miss at streak 10-15 per MDP rules."""
    posts = [
        # Build up to streak 10
        *[
            ParsedPost(
                uri=f"at://p{i}", created_at=f"2026-04-{i:02d}T12:00:00Z",
                text=f"pick {i}", is_reply=False,
                batter_name=f"B{i}", team="NYY",
            )
            for i in range(1, 11)
        ],
        *[
            ParsedPost(
                uri=f"at://r{i}", created_at=f"2026-04-{i:02d}T23:00:00Z",
                text=f"hit {i}", is_reply=True,
                is_result=True, result="hit", streak_after=i,
            )
            for i in range(1, 11)
        ],
    ]
    # Sort by timestamp
    posts.sort(key=lambda p: p.created_at)

    # Day 11 = miss at streak 10 — saver should fire, streak stays at 10
    posts += [
        ParsedPost(uri="at://p11", created_at="2026-04-11T12:00:00Z",
                   text="pick 11", is_reply=False,
                   batter_name="B11", team="NYY"),
        ParsedPost(uri="at://r11", created_at="2026-04-11T23:00:00Z",
                   text="MISS — Streak: 10 (saver used)", is_reply=True,
                   is_result=True, result="miss", streak_after=10),
    ]

    timeline = reconstruct_pick_timeline(posts)
    # Saver consumed — no longer available
    assert timeline.saver_available_at_end is False
    assert timeline.final_streak == 10


def test_regenerate_composes_snapshot_and_bluesky(tmp_path: Path):
    """Regeneration should produce state files using snapshot + Bluesky data."""
    # Initial snapshot file
    snapshot = {
        "version": 1,
        "exported_at": "2026-04-01T00:00:00Z",
        "cutoff_date": "2026-03-31",
        "streak_at_cutoff": 5,
        "saver_available": True,
        "historical_picks": [
            {
                "date": "2026-03-31",
                "pick": {
                    "batter_name": "Aaron Judge",
                    "batter_id": 100, "team": "NYY",
                    "pitcher_name": "X", "pitcher_id": 200,
                    "game_pk": 111, "game_time": "2026-03-31T19:05:00-04:00",
                    "p_game_hit": 0.85,
                },
                "double_down": None,
                "result": "hit",
                "bluesky_posted": True,
                "bluesky_uri": "at://old/post",
            },
        ],
    }
    snapshot_path = tmp_path / "initial-state.json"
    snapshot_path.write_text(json.dumps(snapshot))

    # Bluesky timeline from 2026-04-01 onward
    timeline = Timeline(
        pick_records=[
            HistoricalPickRecord(
                date="2026-04-01",
                batter_name="Nico Hoerner",
                team="CHC",
                is_double_down=False,
                double_down_batter=None,
                double_down_team=None,
                bluesky_uri="at://new/post",
                result="hit",
                streak_after=6,
            ),
        ],
        final_streak=6,
        saver_available_at_end=True,
    )

    from bts.state.regenerate import compose_state_from_snapshot_and_timeline

    out_dir = tmp_path / "regenerated"
    compose_state_from_snapshot_and_timeline(
        snapshot_path=snapshot_path,
        timeline=timeline,
        out_picks_dir=out_dir,
    )

    # Pre-cutoff pick file
    old_pick = json.loads((out_dir / "2026-03-31.json").read_text())
    assert old_pick["pick"]["batter_name"] == "Aaron Judge"
    assert old_pick["result"] == "hit"
    assert old_pick["bluesky_uri"] == "at://old/post"

    # Post-cutoff pick file
    new_pick = json.loads((out_dir / "2026-04-01.json").read_text())
    assert new_pick["pick"]["batter_name"] == "Nico Hoerner"
    assert new_pick["result"] == "hit"
    assert new_pick["bluesky_uri"] == "at://new/post"

    # Streak file
    streak = json.loads((out_dir / "streak.json").read_text())
    assert streak["streak"] == 6
    assert streak["saver_available"] is True


def test_regenerated_pick_files_are_loadable_by_load_pick(tmp_path: Path):
    """Integration test: regenerated files must work with load_pick().

    Both snapshot-originated (pre-cutoff) and Bluesky-originated (post-cutoff)
    pick files must be parseable into a valid DailyPick with Pick sub-objects.
    """
    from bts.picks import load_pick
    from bts.state.regenerate import compose_state_from_snapshot_and_timeline

    snapshot = {
        "version": 1,
        "cutoff_date": "2026-04-01",
        "streak_at_cutoff": 1,
        "saver_available": True,
        "historical_picks": [{
            "date": "2026-04-01",
            "pick": {
                "batter_name": "Snapshot Batter",
                "batter_id": 100,
                "team": "NYY",
                "lineup_position": 3,
                "pitcher_name": "Pitcher",
                "pitcher_id": 200,
                "p_game_hit": 0.85,
                "flags": [],
                "projected_lineup": False,
                "game_pk": 999,
                "game_time": "2026-04-01T19:05:00-04:00",
                "pitcher_team": "BOS",
            },
            "double_down": None,
            "result": "hit",
            "bluesky_posted": True,
            "bluesky_uri": "at://test/snapshot-post",
        }],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))

    # Post-cutoff: Bluesky-originated pick (minimal info)
    timeline = Timeline(
        pick_records=[
            HistoricalPickRecord(
                date="2026-04-02",
                batter_name="Bluesky Batter",
                team="CHC",
                is_double_down=False,
                double_down_batter=None,
                double_down_team=None,
                bluesky_uri="at://test/bluesky-post",
                result="hit",
                streak_after=2,
            ),
        ],
        final_streak=2,
        saver_available_at_end=True,
    )

    out_dir = tmp_path / "picks"
    compose_state_from_snapshot_and_timeline(
        snapshot_path=snapshot_path,
        timeline=timeline,
        out_picks_dir=out_dir,
    )

    # Snapshot-originated pick: load_pick must not crash
    snap_pick = load_pick("2026-04-01", out_dir)
    assert snap_pick is not None
    assert snap_pick.pick.batter_name == "Snapshot Batter"
    assert snap_pick.pick.lineup_position == 3
    assert snap_pick.pick.pitcher_team == "BOS"

    # Bluesky-originated pick: load_pick must not crash
    bs_pick = load_pick("2026-04-02", out_dir)
    assert bs_pick is not None
    assert bs_pick.pick.batter_name == "Bluesky Batter"
    assert bs_pick.pick.team == "CHC"
    assert bs_pick.pick.lineup_position == 0  # backfilled default
    assert bs_pick.pick.flags == []
    assert bs_pick.result == "hit"


def test_regenerated_snapshot_without_new_fields_still_loadable(tmp_path: Path):
    """Older snapshots missing lineup_position/flags/pitcher_team are backfilled."""
    from bts.picks import load_pick
    from bts.state.regenerate import compose_state_from_snapshot_and_timeline

    # Snapshot with a pick missing the newer fields
    snapshot = {
        "version": 1,
        "cutoff_date": "2026-04-01",
        "streak_at_cutoff": 1,
        "saver_available": True,
        "historical_picks": [{
            "date": "2026-04-01",
            "pick": {
                "batter_name": "Legacy Batter",
                "batter_id": 100,
                "team": "NYY",
                "pitcher_name": "Pitcher",
                "pitcher_id": 200,
                "game_pk": 999,
                "game_time": "2026-04-01T19:05:00-04:00",
                "p_game_hit": 0.85,
                # No lineup_position, flags, projected_lineup, pitcher_team
            },
            "double_down": None,
            "result": "hit",
            "bluesky_posted": True,
            "bluesky_uri": "at://test/old-post",
        }],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))

    out_dir = tmp_path / "picks"
    compose_state_from_snapshot_and_timeline(
        snapshot_path=snapshot_path,
        timeline=Timeline(),
        out_picks_dir=out_dir,
    )

    pick = load_pick("2026-04-01", out_dir)
    assert pick is not None
    assert pick.pick.batter_name == "Legacy Batter"
    assert pick.pick.lineup_position == 0
    assert pick.pick.flags == []
    assert pick.pick.pitcher_team is None

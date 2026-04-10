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

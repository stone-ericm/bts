"""Tests for bts state regenerate."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

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
    text = "Today's BTS pick: Nico Hoerner (CHC) vs RHP Test Pitcher — 78.3% 🎯\n\nStreak: 2"
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.batter_name == "Nico Hoerner"
    assert parsed.team == "CHC"
    assert parsed.is_double_down is False
    assert parsed.double_down_batter is None


def test_parse_pick_post_extracts_double_down():
    text = (
        "Today's BTS pick: Jose Altuve (HOU) vs RHP Pitcher A — 82.0% 🎯\n"
        "Double down: Kyle Tucker (HOU) vs Pitcher B — 80.0%\n\n"
        "Streak: 5"
    )
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.batter_name == "Jose Altuve"
    assert parsed.is_double_down is True
    assert parsed.double_down_batter == "Kyle Tucker"


def test_parse_skip_post():
    text = "Today's BTS pick: SKIP — top prob 76.5%, below 80% threshold. Streak holds at 3."
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.is_skip is True


def test_parse_unrecognized_post_returns_none():
    text = "random promotional content, not a pick"
    parsed = parse_pick_from_post(text)
    assert parsed is None

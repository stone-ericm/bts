"""Pydantic schemas for the BTS leaderboard watcher data model.

Mirrors the parquet column schemas. Validation happens on parsing —
bad rows are rejected before they reach storage. round_id was added
in Phase 1 since the BTS API returns roundId per pick (not date) and
we map roundId -> date via the static rounds.json file.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


TabName = Literal["active_streak", "all_season", "all_time", "yesterday"]
HomeAway = Literal["home", "away"]


class LeaderboardRow(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    tab: TabName
    rank: int = Field(gt=0)
    username: str = Field(min_length=1)
    streak: int | None = Field(default=None, ge=0)
    hits_today: int | None = Field(default=None, ge=0)


class PickRow(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    round_id: int = Field(ge=0)
    pick_date: date
    pick_number: int = Field(ge=1, le=2)  # 1=primary, 2=double-down
    unit_id: int = Field(ge=0)
    bts_player_id: int = Field(ge=0)
    result: str  # 'hit' | 'not_hit' (string for forward compat with new statuses)
    at_bats: int = Field(ge=0)
    hits: int = Field(ge=0)
    streak_after: int = Field(ge=0)

    # Resolved at scrape time when static lookups (players.json/units.json/squads.json)
    # are available. May be None if scraper couldn't resolve.
    batter_id: int | None = None        # MLB person_id (= players.json[playerId].feedId)
    batter_name: str | None = None
    batter_team: str | None = None       # squad abbreviation, e.g. "NYM"
    opponent_team: str | None = None
    home_or_away: HomeAway | None = None


class SeasonStats(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    username: str = Field(min_length=1)
    best_streak: int = Field(ge=0)
    active_streak: int = Field(ge=0)
    pick_accuracy_pct: float = Field(ge=0.0, le=100.0)

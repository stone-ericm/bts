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
    batter_name: str = Field(min_length=1)
    batter_team: str = Field(min_length=1)
    opponent_team: str = Field(min_length=1)
    home_or_away: HomeAway
    at_bats: int = Field(ge=0)
    hits: int = Field(ge=0)
    streak_after: int = Field(ge=0)
    batter_id: int | None = None


class SeasonStats(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    username: str = Field(min_length=1)
    best_streak: int = Field(ge=0)
    active_streak: int = Field(ge=0)
    pick_accuracy_pct: float = Field(ge=0.0, le=100.0)

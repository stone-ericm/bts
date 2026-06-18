"""Parse the per-round MLB ledger (contest_ledger.jsonl) into a per-round series.

Each ledger line is one fetch snapshot: {recorded_at, active_streak, ..., predictions:
[...]}. A prediction row carries the POST-round `streak`; pre_round_streak is the prior
settled round's post_streak. Finality and scoring-correction state are NOT in the rows,
so callers must treat the latest fetch's values as provisional: a round is `stable` only
when the same (result, streak) for that roundId also appeared in the PREVIOUS fetch (a
two-read confirmation). Saver inference (see infer_saver) acts only on stable rounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LedgerRound:
    round_id: int
    result: str | None          # hit / not_hit / void / None
    pre_streak: int | None      # prior round's post_streak (None for the earliest row / after a gap)
    post_streak: int | None
    streak_increase: int | None
    is_dd: bool                 # two entered slots
    stable: bool                # same (result, streak) seen in the previous fetch too


def parse_latest_ledger(ledger_path: Path) -> list[LedgerRound]:
    """Parse the most recent fetch's predictions into rounds sorted by roundId.

    `stable` is True when the round's (result, streak) is unchanged from the previous
    fetch snapshot; a single-fetch ledger or a newly-appeared/changed round is provisional
    (stable=False). A round whose post streak is missing breaks the pre_streak chain, so a
    later round does not inherit a stale (and now unrecoverable) pre-streak.
    """
    if not ledger_path.exists():
        return []
    lines = [ln for ln in ledger_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    try:
        latest = json.loads(lines[-1])
    except json.JSONDecodeError:
        return []

    # Previous fetch (if any) -> {roundId: (result, streak)} for the stability check.
    prev_by_id: dict[int, tuple] = {}
    if len(lines) >= 2:
        try:
            previous = json.loads(lines[-2])
        except json.JSONDecodeError:
            previous = None
        if previous is not None:
            for p in previous.get("predictions", []):
                rid = p.get("roundId")
                if rid is not None:
                    prev_by_id[int(rid)] = (p.get("result"), p.get("streak"))

    preds = sorted(
        (p for p in latest.get("predictions", []) if p.get("roundId") is not None),
        key=lambda p: int(p["roundId"]),
    )
    rounds: list[LedgerRound] = []
    prev_post: int | None = None
    for p in preds:
        rid = int(p["roundId"])
        post = p.get("streak")
        result = p.get("result")
        rounds.append(LedgerRound(
            round_id=rid,
            result=result,
            pre_streak=prev_post,
            post_streak=post if isinstance(post, int) else None,
            streak_increase=p.get("streakIncrease"),
            is_dd=len(p.get("roundPredictions", [])) >= 2,
            stable=prev_by_id.get(rid) == (result, post),
        ))
        # Carry post only when known; a missing post breaks the chain (the next round's
        # pre_streak is then correctly unrecoverable rather than a stale prior value).
        prev_post = post if isinstance(post, int) else None
    return rounds

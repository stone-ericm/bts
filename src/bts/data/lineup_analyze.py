"""Analyze lineup posting time distributions from collected JSONL logs.

Reads files written by `bts data collect-lineup-times` and computes
statistics on how many minutes before first pitch each lineup was
confirmed. Used to inform scheduler timing configuration.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Distribution:
    """Percentile summary of minutes-before-first-pitch."""
    n: int
    p10: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    mean: Optional[float]
    minutes: list[int]


def compute_minutes_before_first_pitch(
    lineup_time_utc: str,
    game_time_et: str,
) -> int:
    """Compute minutes between a lineup confirmation and first pitch.

    Positive = confirmed before first pitch (normal).
    Negative = confirmed after (anomaly, should be rare).
    """
    lineup_dt = datetime.fromisoformat(lineup_time_utc)
    game_dt = datetime.fromisoformat(game_time_et)
    delta_sec = (game_dt - lineup_dt).total_seconds()
    return round(delta_sec / 60)


def compute_distribution(samples: Iterable[int]) -> Distribution:
    """Compute percentile distribution from a collection of minute values."""
    data = sorted(samples)
    n = len(data)
    if n == 0:
        return Distribution(
            n=0, p10=None, p25=None, p50=None, p75=None,
            p90=None, p95=None, p99=None, mean=None, minutes=[],
        )

    def percentile(p: float) -> float:
        # Linear interpolation between closest ranks
        k = (n - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(data[f])
        return data[f] + (data[c] - data[f]) * (k - f)

    return Distribution(
        n=n,
        p10=percentile(10),
        p25=percentile(25),
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
        mean=sum(data) / n,
        minutes=data,
    )


def load_samples_from_jsonl(
    in_dir: Path,
    from_date: str,
    to_date: str,
) -> list[int]:
    """Load all lineup-time samples from JSONL files in a date range.

    Reads both first_away_confirmed_utc and first_home_confirmed_utc per
    game as separate samples. Skips nulls. Returns a list of
    minutes-before-first-pitch integers ready to feed into compute_distribution.
    """
    samples: list[int] = []
    for jsonl in sorted(in_dir.glob("*.jsonl")):
        date_str = jsonl.stem
        if date_str < from_date or date_str > to_date:
            continue
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            game_time_et = row["game_time_et"]
            for field in ("first_away_confirmed_utc", "first_home_confirmed_utc"):
                confirmed = row.get(field)
                if confirmed is None:
                    continue
                samples.append(
                    compute_minutes_before_first_pitch(confirmed, game_time_et)
                )
    return samples

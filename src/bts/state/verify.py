"""State drift detection: compare live state to what regeneration produces.

Runs periodically (weekly cron on the Fly machine) to catch bit-rot in
the regeneration logic before it matters in a real recovery event. Also
catches silent corruption in the live state files.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


@dataclass
class DriftReport:
    """Result of comparing two state directories."""
    issues: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.issues


def diff_pick_files(live_dir: Path, regenerated_dir: Path) -> DriftReport:
    """Compare live state to regenerated state. Returns a DriftReport.

    Compares pick files (by date), result field, batter_name, and streak.
    Does not compare fields that cannot be recovered from Bluesky alone
    (batter_id, p_game_hit, etc.) because those are expected to differ.
    """
    report = DriftReport()

    # Collect pick dates from both
    live_picks = {p.stem: p for p in live_dir.glob("*.json")
                  if DATE_PATTERN.match(p.name)}
    regen_picks = {p.stem: p for p in regenerated_dir.glob("*.json")
                   if DATE_PATTERN.match(p.name)}

    # Dates in live but missing in regen
    for date in sorted(set(live_picks) - set(regen_picks)):
        report.issues.append(f"pick {date} exists in live but not in regenerated")
    for date in sorted(set(regen_picks) - set(live_picks)):
        report.issues.append(f"pick {date} exists in regenerated but not in live")

    # Common dates — compare recoverable fields
    for date in sorted(set(live_picks) & set(regen_picks)):
        live = json.loads(live_picks[date].read_text())
        regen = json.loads(regen_picks[date].read_text())
        if live.get("result") != regen.get("result"):
            report.issues.append(
                f"pick {date} result mismatch: "
                f"live={live.get('result')}, regen={regen.get('result')}"
            )
        if live.get("pick", {}).get("batter_name") != regen.get("pick", {}).get("batter_name"):
            report.issues.append(
                f"pick {date} batter name mismatch: "
                f"live={live.get('pick', {}).get('batter_name')}, "
                f"regen={regen.get('pick', {}).get('batter_name')}"
            )

    # Streak
    try:
        live_streak = json.loads((live_dir / "streak.json").read_text())
        regen_streak = json.loads((regenerated_dir / "streak.json").read_text())
        if live_streak.get("current") != regen_streak.get("current"):
            report.issues.append(
                f"streak mismatch: live={live_streak.get('current')}, "
                f"regen={regen_streak.get('current')}"
            )
    except FileNotFoundError as e:
        report.issues.append(f"streak file missing: {e}")

    return report

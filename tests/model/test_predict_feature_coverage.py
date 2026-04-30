"""Test the FEATURE_COL → predict() contract by source-code inspection.

Catches the bpm-class bug shipped 2026-04-29 → fixed 2026-04-30: a feature
was added to FEATURE_COLS without corresponding row[col] population in
predict(), causing every prediction to crash with KeyError.

We use a "registry of known populators" — every FEATURE_COL must have an
entry. When you add a new feature, this test fails until you also document
where predict() populates it. That review step is the forcing function that
prevents another silent inference break.
"""
import pandas as pd
import pytest
import re
from pathlib import Path

from bts.features.compute import FEATURE_COLS


def _synthetic_pa_frame() -> pd.DataFrame:
    """Build a tiny PA frame with enough structure for compute_all_features + lookups."""
    rows = []
    base = pd.Timestamp("2025-04-01")
    # 50 batter-pitcher matchups across 30 days
    for d in range(30):
        date = base + pd.Timedelta(days=d)
        for bid in range(100, 110):
            for pid in range(200, 205):
                rows.append({
                    "batter_id": bid,
                    "pitcher_id": pid,
                    "date": date,
                    "season": 2025,
                    "is_hit": (bid + pid + d) % 4 == 0,
                    "game_pk": 8000 + d * 10 + (pid % 5),
                    "is_home": (bid % 2 == 0),
                    "venue_id": 1 + (pid % 3),
                    "pitch_hand": "R" if pid % 2 == 0 else "L",
                    "pitcher_throws": "R" if pid % 2 == 0 else "L",
                    "weather_temp": 70.0,
                    "weather_wind_speed": 8.0,
                    "weather_wind_dir": "out to cf",
                    "roof_type": "open",
                    "atm_pressure": 1013.0,
                    "humidity": 50.0,
                    "is_out": (bid + pid + d) % 4 != 0,
                    "career_pas": 500,
                    "is_swing": True,
                    "is_swinging_strike": (bid + d) % 5 == 0,
                    "balls": 1,
                    "strikes": 2,
                    "is_ground_ball": (bid + d) % 3 == 0,
                    "pitch_type": "FF",
                    "hp_umpire_id": 50 + (d % 3),
                    "opp_pitching_team_id": (pid % 4) + 10,
                    "event_type": "single" if (bid + pid + d) % 4 == 0 else "strikeout",
                })
    return pd.DataFrame(rows)


class TestFeatureCoverage:
    def test_every_feature_col_has_registered_populator(self):
        """Every FEATURE_COL must have an entry in the populator registry below.

        This is the forcing function: when you add a feature to FEATURE_COLS,
        this test fails until you update the registry AND verify predict()
        actually populates row[col] for that feature. The registry is a
        **source-code review prompt**, not a runtime check — the value of
        each entry is a brief description of where predict() picks it up.
        """
        # Where predict() (in src/bts/model/predict.py) gets each row[col].
        # Update when adding features. Each value is the row[col] = ... source.
        populator_registry = {
            "batter_hr_7g": 'lookups["batter"][col].get(bid)',
            "batter_hr_30g": 'lookups["batter"][col].get(bid)',
            "batter_hr_60g": 'lookups["batter"][col].get(bid)',
            "batter_hr_120g": 'lookups["batter"][col].get(bid)',
            "batter_whiff_60g": 'lookups["batter"][col].get(bid)',
            "batter_count_tendency_30g": 'lookups["batter"][col].get(bid)',
            "batter_gb_hit_rate": 'lookups["batter"][col].get(bid)',
            "platoon_hr": 'lookups["platoon"].get((bid, slot["pitcher_hand"]))',
            "pitcher_hr_30g": 'lookups["pitcher_hr"].get(slot["pitcher_id"])  # debut→league avg',
            "pitcher_entropy_30g": 'lookups["pitcher_ent"].get(slot["pitcher_id"])',
            "weather_temp": 'slot["weather_temp"]',
            "park_factor": 'lookups["park"].get(slot["venue_id"])',
            "pitcher_catcher_framing": 'lookups.get("pitcher_framing", {}).get(slot["pitcher_id"])',
            "opp_bullpen_hr_30g": 'lookups.get("bullpen", {}).get(slot.get("opp_team_id"))',
            "days_rest": "computed inline from lookups['last_date'] in predict()",
            # Added 2026-04-29 (commit 7afee63), inference path wired 2026-04-30 (commit ee4190f).
            # The gap between those commits caused 1 day of broken production predictions.
            "batter_pitcher_shrunk_hr": 'lookups.get("batter_pitcher_hr", {}).get((bid, slot["pitcher_id"]), 0.2195)',
        }

        unregistered = [c for c in FEATURE_COLS if c not in populator_registry]
        assert not unregistered, (
            f"FEATURE_COLS contains {unregistered} with no entry in this test's "
            f"populator_registry. Adding a feature to FEATURE_COLS without also "
            f"populating row[col] in predict() crashes every inference (see the "
            f"bpm bug 2026-04-29). Update the registry here AND verify "
            f"src/bts/model/predict.py:predict() actually populates each row dict."
        )

        # Also verify the reverse: nothing stale in the registry that was removed
        # from FEATURE_COLS. Catches dangling entries.
        stale = [c for c in populator_registry if c not in FEATURE_COLS]
        assert not stale, (
            f"populator_registry contains {stale} no longer in FEATURE_COLS. "
            f"Remove these entries; they're documenting code that's no longer needed."
        )

    def test_predict_source_actually_populates_each_feature_col(self):
        """Strong test: scan predict.py and confirm each FEATURE_COL is referenced as row[col]."""
        predict_src = Path("/Users/stone/projects/bts/src/bts/model/predict.py").read_text()
        # Locate the predict() function body
        pred_start = predict_src.find("def predict(")
        assert pred_start > 0, "couldn't find predict() definition"
        # End of predict() is approximated by next top-level def or end of file
        pred_end = predict_src.find("\ndef ", pred_start + 1)
        if pred_end == -1:
            pred_end = len(predict_src)
        predict_body = predict_src[pred_start:pred_end]

        missing = []
        for col in FEATURE_COLS:
            # Either row[col] = ... OR row["col_name"] = ... — both work
            if f'row["{col}"]' not in predict_body and f"row['{col}']" not in predict_body:
                # Some features are in iterated lists (e.g., batter_cols loop).
                # Check if col appears in the predict body at all.
                if col not in predict_body:
                    missing.append(col)
        assert not missing, (
            f"FEATURE_COLS {missing} not referenced in predict() body. "
            f"This is the bpm-bug check: every FEATURE_COL must appear by name "
            f"in predict() so the row dict has a key for it before the model "
            f"does its lookup."
        )

    def test_feature_cols_count_matches_architecture_doc(self):
        """ARCHITECTURE.md must claim the right feature count (it drifted before)."""
        from pathlib import Path
        arch = Path("/Users/stone/projects/bts/ARCHITECTURE.md").read_text()
        # The header line "## Features (N, provably leak-free)" must match.
        import re
        m = re.search(r"## Features \((\d+), provably leak-free\)", arch)
        assert m, "ARCHITECTURE.md doesn't have the 'Features (N, ...)' header"
        claimed = int(m.group(1))
        assert claimed == len(FEATURE_COLS), (
            f"ARCHITECTURE.md claims {claimed} features but FEATURE_COLS has "
            f"{len(FEATURE_COLS)}. Update the doc when adding/removing features."
        )

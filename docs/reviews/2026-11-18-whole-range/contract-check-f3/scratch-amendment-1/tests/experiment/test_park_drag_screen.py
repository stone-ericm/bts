"""Registry tests for the park_drag 2026 screen."""
import numpy as np
import pandas as pd
import pytest

from bts.experiment import park_drag_screen as pds


def _pa(n_dates=8, venues=(1, 2, 3)):
    rows = []
    for d in range(n_dates):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=d)
        for v in venues:
            for b in range(3):  # 3 batters per venue-date (game-mates)
                rows.append({
                    "date": date, "season": 2026, "venue_id": v,
                    "game_pk": 10000 + d * 10 + v, "batter_id": 100 + b,
                    "lineup_position": b + 1,
                    "is_hit": int((d + v + b) % 3 == 0),
                    "park_drag_delta": -0.01 * v + 0.001 * d,
                    "park_drag_delta_expanding": -0.008 * v,
                    "rolling_outcome_pf": 0.30 + 0.01 * v,
                })
    df = pd.DataFrame(rows)
    df.loc[df.date < "2026-05-03", "park_drag_delta"] = np.nan  # early NaN
    return df


class TestArms:
    def test_baseline_flags_only(self):
        frame, cols = pds.build_arm_frame("baseline", _pa())
        assert cols == ["has_park_drag_delta", "has_park_drag_delta_expanding",
                        "has_rolling_outcome_pf"]

    def test_pd_anchored_adds_column(self):
        _, cols = pds.build_arm_frame("pd_anchored", _pa())
        assert "park_drag_delta" in cols and "has_park_drag_delta" in cols

    def test_mask_only_destroys_values_keeps_pattern(self):
        pa = _pa()
        frame, cols = pds.build_arm_frame("ctl_mask_only", pa)
        m = frame["mask_park_drag_delta"]
        assert set(m.dropna().unique()) == {1.0}
        assert (m.isna() == pa["park_drag_delta"].isna()).all()

    def test_permuted_is_venue_block(self):
        pa = _pa()
        frame, _ = pds.build_arm_frame("ctl_permuted", pa)
        col = "perm_park_drag_delta"
        # game-mates share one value
        assert (frame.groupby(["date", "venue_id"])[col].nunique(dropna=False) <= 1).all()
        # per-date multiset of venue values preserved
        for d, day in frame.groupby("date"):
            orig = sorted(pa[pa.date == d].groupby("venue_id")["park_drag_delta"]
                          .first().fillna(-999).tolist())
            perm = sorted(day.groupby("venue_id")[col].first().fillna(-999).tolist())
            assert orig == pytest.approx(perm)
        # and the mapping actually changed somewhere (seeded)
        changed = (frame[col].fillna(-999) != pa["park_drag_delta"].fillna(-999)).any()
        assert changed

    def test_sentinels_require_driver_columns(self):
        for arm in ("ctl_sentinel_gross", "ctl_sentinel_soft", "ctl_sentinel_leaky"):
            with pytest.raises(ValueError):
                pds.build_arm_frame(arm, _pa())

    def test_unknown_arm_raises(self):
        with pytest.raises(KeyError):
            pds.build_arm_frame("nope", _pa())


class TestRollingOutcomePF:
    def test_strictly_prior(self):
        rows = []
        for i in range(7):
            for b in range(4):
                rows.append({"venue_id": 1, "season": 2026,
                             "date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i),
                             "is_hit": 1 if (i + b) % 2 == 0 else 0})
        pa = pd.DataFrame(rows)
        pf = pds.rolling_outcome_pf(pa)
        pf = pf.sort_values("date").reset_index(drop=True)
        # dates 0-4: fewer than 5 prior venue-dates -> NaN
        assert pf.loc[:4, "rolling_outcome_pf"].isna().all()
        # date 5 = mean of the first 5 venue-date hit rates (all 0.5 here)
        assert pf.loc[5, "rolling_outcome_pf"] == pytest.approx(0.5)

    def test_value_excludes_same_date(self):
        rows = []
        for i in range(6):
            hit = 1 if i < 5 else 0  # date 6 is an 0-fer
            for b in range(4):
                rows.append({"venue_id": 1, "season": 2026,
                             "date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i),
                             "is_hit": hit})
        pa = pd.DataFrame(rows)
        pf = pds.rolling_outcome_pf(pa).sort_values("date").reset_index(drop=True)
        # last date's value reflects the five 1.0 days, NOT its own 0.0
        assert pf.iloc[-1]["rolling_outcome_pf"] == pytest.approx(1.0)

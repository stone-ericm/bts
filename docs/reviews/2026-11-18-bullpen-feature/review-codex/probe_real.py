"""Run production computations for historical identity, reference, and serving."""

import gc
import pickle
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from bts.features import compute

from probe_common import (
    DATA, EXACT, FEATURE, HERE, TEAM, assert_aligned, assert_reference,
    attach_probe_ids, compare_versions, load_base, lookup_frozen, run, season, stage_base,
)


def coverage(frame, lookup):
    pids = frame["game_pk"]
    away = frame["is_home"]
    print(
        f"lookup games={pids[pids.isin(lookup)].nunique()}/{pids.nunique()}",
        *(f"{name}={pids.map({k: v.get('away' + suffix) for k, v in lookup.items()}).where(away, pids.map({k: v.get('home' + suffix) for k, v in lookup.items()})).notna().sum()}"
          for name, suffix in (("team", "_tid"), ("probable", ""))),
        flush=True,
    )


def compare(label, frame, base, lookup):
    print(label, flush=True)
    coverage(frame, lookup)
    old = run(base, frame, lookup)
    new = run(compute, frame, lookup)
    assert_aligned(old, frame)
    assert_aligned(new, frame)
    compare_versions(old, new)
    return old, new


def metric(label, left, right):
    finite = left.notna() & right.notna()
    delta = (left[finite] - right[finite]).abs()
    print(label, "both finite", finite.sum(), "unequal", delta.ne(0).sum(),
          "left only", (left.notna() & right.isna()).sum(),
          "right only", (left.isna() & right.notna()).sum(),
          "both missing", (left.isna() & right.isna()).sum(),
          "mean abs", delta.mean(), "max abs", delta.max(), flush=True)


def simulate_mornings(old, new):
    sort_cols = [TEAM, "date"]
    for frame in (old, new):
        grouped = frame.groupby(sort_cols)[FEATURE]
        assert grouped.nunique(dropna=False).le(1).all()
    by_date = old[sort_cols + [FEATURE]].rename(columns={FEATURE: "old_training"})
    by_date["new_training"] = new[FEATURE]
    by_date = by_date.drop_duplicates(sort_cols).sort_values(sort_cols).reset_index(drop=True)
    for arm in ("old", "new"):
        by_date[f"{arm}_lookup"] = by_date.groupby(TEAM)[f"{arm}_training"].transform(
            lambda series: series.ffill().shift(1)
        )
    affected = by_date["old_training"].isna() & by_date["new_training"].notna()
    following = by_date.loc[affected, sort_cols].rename(columns={"date": "filled_date"})
    next_rows = by_date.groupby(TEAM)["date"].shift(-1)
    following["date"] = next_rows[affected]
    following = following.merge(by_date, on=sort_cols, how="left", validate="1:1")
    assert following["date"].notna().all()
    for left, right in (
        ("old_training", "old_lookup"), ("new_training", "old_lookup"),
        ("new_training", "new_lookup"), ("old_lookup", "new_lookup"),
    ):
        metric(f"{left} against {right}", by_date[left], by_date[right])
    metric("next games", following["old_lookup"], following["new_lookup"])
    np.testing.assert_array_equal(following["new_lookup"], following["new_training"])
    following.to_csv(HERE / "following-games.csv", index=False)
    by_date.to_csv(HERE / "morning-parity.csv", index=False)
    return by_date


def compare_lookups(old, new, mornings):
    from bts.model.predict import _build_feature_lookups

    rows = []
    team_ids = sorted(new[TEAM].dropna().unique())
    for cutoff in ("2025-03-19", "2025-04-14", "2025-07-25", "2025-09-21", "2025-12-31"):
        cutoff = pd.Timestamp(cutoff)
        old_out = _build_feature_lookups(old.loc[old["date"] < cutoff])
        new_out = _build_feature_lookups(new.loc[new["date"] < cutoff])
        old_bp = old_out.pop("bullpen")
        new_bp = new_out.pop("bullpen")
        assert pickle.dumps(old_out, protocol=5) == pickle.dumps(new_out, protocol=5)
        print(cutoff.date(), "other lookup byte equality", sorted(old_out), flush=True)
        for team in team_ids:
            targets = mornings.loc[mornings[TEAM].eq(team) & mornings["date"].eq(cutoff)]
            if not targets.empty:
                for label, actual in (("old_lookup", old_bp), ("new_lookup", new_bp)):
                    np.testing.assert_equal(targets[label].iat[0], actual.get(team, np.nan))
            rows.append((cutoff, team, old_bp.get(team), new_bp.get(team)))
    result = pd.DataFrame(rows, columns=["strict_cutoff", TEAM, "old_lookup", "new_lookup"])
    result.to_csv(HERE / "actual-lookups.csv", index=False)
    for cutoff, rows in result.groupby("strict_cutoff"):
        metric(str(cutoff.date()), rows["old_lookup"], rows["new_lookup"])


def large(arm):
    # Read only years explicitly in scope and take peak RSS in a fresh process.
    year_frames = [season(year) for year in range(2017, 2026)]
    frame = pd.concat(year_frames, ignore_index=True)
    del year_frames
    gc.collect()
    lookup = lookup_frozen()
    module = load_base() if arm == "base" else compute
    start = time.perf_counter()
    result = run(module, frame, lookup)
    seconds = time.perf_counter() - start
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    assert len(result) == len(frame)
    print(f"large {arm}: rows={len(result)} seconds={seconds:.3f} peak_mib={peak_mib:.1f} "
          f"missing={result[FEATURE].isna().sum()}", flush=True)


def main():
    stage_base()
    print("pandas", pd.__version__, "data", DATA, flush=True)
    base = load_base()
    lookup = lookup_frozen()
    year24 = season(2024)
    year25 = season(2025)
    compare("2024", year24, base, lookup)
    old, new = compare("2025", year25, base, lookup)
    print("2025 reference max error", assert_reference(new, year25, lookup), flush=True)
    mornings = simulate_mornings(old, new)
    compare_lookups(old, new, mornings)
    del old, new
    compare("2024-2025", pd.concat([year24, year25], ignore_index=True), base, lookup)
    del year24
    frame = attach_probe_ids(year25)
    compare("2025 shuffled", frame.sample(frac=1, random_state=215), base, lookup)
    compare("2025 non-default duplicate labels",
            frame.set_axis(np.arange(len(frame)) // 2 + 401), base, lookup)
    compare("2025 duplicated rows",
            pd.concat([frame, frame.iloc[::337]], ignore_index=True), base, lookup)
    strings = frame.assign(date=pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"))
    old_s, new_s = compare("2025 date strings", strings, base, lookup)
    old_t, new_t = compare("2025 timestamps", frame.assign(date=pd.to_datetime(frame["date"])),
                           base, lookup)
    pd.testing.assert_frame_equal(old_s, old_t, **EXACT)
    pd.testing.assert_frame_equal(new_s, new_t, **EXACT)
    del old_s, new_s, old_t, new_t, frame, strings, year25
    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        large(sys.argv[1])
    else:
        main()
        for version in ("base", "branch"):
            subprocess.run([sys.executable, __file__, version], check=True)

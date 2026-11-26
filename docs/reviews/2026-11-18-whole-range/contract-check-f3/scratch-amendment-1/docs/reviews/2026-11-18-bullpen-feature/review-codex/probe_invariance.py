"""Perturb complete real 2025 dates through the production feature entry point."""

import numpy as np
import pandas as pd
from bts.features import compute

from probe_common import EXACT, FEATURE, TEAM, attach_probe_ids, classify, lookup_frozen, run, season


def target_values(before, after, day):
    before = before.loc[before["date"].eq(day)].set_index("review_row")
    after = after.loc[after["date"].eq(day)].set_index("review_row")
    shared = before.index.intersection(after.index)
    pd.testing.assert_series_equal(before.loc[shared, FEATURE], after.loc[shared, FEATURE],
                                   check_exact=True)
    expected = before.drop_duplicates(TEAM).set_index(TEAM)[FEATURE]
    np.testing.assert_array_equal(after[FEATURE], after[TEAM].map(expected))
    return len(before), len(after), before[FEATURE].notna().sum()


def same_day(frame, baseline, lookup, label, day):
    prefix = frame.loc[frame["date"].le(day)]
    today = prefix["date"].eq(day)
    games = prefix.loc[today, "game_pk"].unique()
    explicit = lookup.copy()
    for game in games:
        game_info = explicit[game].copy()
        for side in ("home", "away"):
            if pd.isna(game_info.get(side)):
                game_info[side] = 30_001_000
        explicit[game] = game_info
    starter = prefix.copy()
    starter.loc[today, "pitcher_id"] = [
        classify(row, explicit)[1] for row in starter.loc[today].itertuples()
    ]
    starter.loc[today, "is_hit"] = 1 - starter.loc[today, "is_hit"]
    starter_result = run(compute, starter, explicit)
    print(label, day.date(), "all starter, all hits flipped",
          target_values(baseline, starter_result, day), flush=True)

    relief = prefix.copy()
    relief.loc[today, "pitcher_id"] = 30_002_000
    relief.loc[today, "is_hit"] = 1 - relief.loc[today, "is_hit"]
    second_game_key = int(frame["game_pk"].max() + 1)
    today_rows = relief.loc[today]
    second_game = today_rows.loc[today_rows["game_pk"].eq(games[0])].copy()
    second_game["game_pk"] = second_game_key
    second_game["review_row"] = np.arange(len(second_game)) + len(frame)
    explicit[second_game_key] = explicit[games[0]].copy()
    relief = pd.concat([relief.drop(index=today_rows.iloc[::7].index), second_game], ignore_index=True)
    relief_result = run(compute, relief, explicit)
    print(label, day.date(), "all relief, hits flipped, removed PAs, additional game",
          target_values(baseline, relief_result, day), flush=True)


def future(frame, baseline, lookup, label, day):
    after = frame["date"].gt(day)
    cutoff = baseline["date"].le(day)
    if not after.any():
        appended = frame.loc[frame["date"].eq(day)].copy()
        appended["date"] += pd.Timedelta(days=1)
        appended["review_row"] += len(frame)
        frame = pd.concat([frame, appended], ignore_index=True)
        after = frame["date"].gt(day)
        label += " (synthetic later date)"
    mutated = frame.copy()
    mutated.loc[after, "is_hit"] = 1 - mutated.loc[after, "is_hit"]
    mutated.loc[after, "pitcher_id"] = 30_003_000
    mutated.loc[after, "batter_id"] = 30_004_000
    mutated.loc[after, "launch_angle"] = 25.0
    mutated.loc[after, "launch_speed"] = 95.0
    mutated = mutated.drop(index=mutated.loc[after].iloc[::7].index)
    result = run(compute, mutated, lookup)
    pd.testing.assert_frame_equal(baseline.loc[cutoff], result.loc[result["date"].le(day)], **EXACT)
    print(label, day.date(), "future: all columns equal on", cutoff.sum(), "PAs", flush=True)


def main():
    frame = attach_probe_ids(season(2025).reset_index(drop=True))
    frame["date"] = pd.to_datetime(frame["date"])
    lookup = lookup_frozen()
    baseline = run(compute, frame, lookup)
    target = pd.Timestamp("2025-07-23")
    gap = frame.loc[frame["date"].eq(target)]
    team_side = [row for row in gap.itertuples() if classify(row, lookup)[0] == 113]
    assert team_side and all(pd.isna(probable) or row.pitcher_id == probable
                             for row in team_side for _, probable in [classify(row, lookup)])
    teams = baseline[[TEAM, "date", "game_pk"]].drop_duplicates()
    max_games = teams.groupby("date").apply(
        lambda day: day.groupby(TEAM)["game_pk"].nunique().max()
    )
    middle = max_games.loc["2025-05-01":"2025-05-31"]
    for label, day in (
        ("gap (team 113)", target),
        ("ordinary", middle[middle.eq(1)].index[0]),
        ("doubleheader", middle[middle.gt(1)].index[0]),
        ("first", frame["date"].min()),
        ("last", frame["date"].max()),
    ):
        same_day(frame, baseline, lookup, label, day)
        future(frame, baseline, lookup, label, day)


if __name__ == "__main__":
    main()

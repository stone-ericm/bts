# C1 4b: calendar and profile-coverage check (outcome-free, before X-31)

**Date:** 2026-10-04. **Status:** a finding from build task T1, made before any outcome-bearing read. It changes nothing in the frozen registration; the decision below is Eric's.

## What was checked (dates and game ids only; no hit outcomes read)
- **The 2021–2025 MLB regular-season schedules:** fetched from the public statsapi, one request per season, .
  - A date counts as a game day if it has at least one game that was not postponed or cancelled.
  - The sha256 of each response is listed below.
- **The Sun Oct  4 15:48:50 EDT 2026 column of all 120 estimated-PA profiles** (, 24 seeds × 5 seasons).
- **The  column of .**

## Findings

| Season | Game days (schedule) | Profile dates (identical across all 24 seeds) | Game days with no profile rows |
|---|---|---|---|
| 2021 | 182 (4/01–10/03) | 182 | 0 |
| 2022 | 179 (4/07–10/05) | 179 | 0 |
| 2023 | 183 (3/30–10/02) | 182 (3/30–10/01) | **1: 2023-10-02** |
| 2024 | 185 (3/20–9/30, including the Seoul Series on 3/20–3/21) | 185 | 0 |
| 2025 | 184 (3/18–9/28, including the Tokyo Series on 3/18–3/19) | 184 | 0 |

- **No profile date falls on a non-game day.**
- **The one gap:** 2023-10-02 had a single game (gamePk 716404, status "Completed Early"). It is **absent from ** (0 rows), so the gap is upstream of the profiles. It cannot be filled without re-acquiring that game's feed.

## Contest calendar evidence
- **Historical BTS rules (2021–2025) could not be retrieved:** the archive was unreachable from this session.
- **The only verified wording is 2026's:** the entry period ends "upon the conclusion of the final game of the 2026 MLB regular season". Under that wording, 2023-10-02 (that season's final regular-season game) would be a contest date.

The build therefore uses the MLB regular-season game days as each season's opportunity calendar, a convention consistent with the 2026 rule. Opening is the first game day (including international openers); the final date is the last game day; the exclusive end is the day after the final date.

## Consequence under the frozen registration
§7 says: "Unknown playable-date coverage makes acceptance inconclusive; a conditional replay may be shown descriptively with those dates explicitly treated as no play, retaining calendar time."

So, as frozen, the 4b screen's disposition will be **inconclusive** because of this one final-day date. The full trade tables are still produced and shown to Eric descriptively.

**Owner decision needed before any outcome read:**
- **(a)** Keep the frozen rule: inconclusive by construction, with descriptive tables.
- **(b)** Record a pre-outcome deviation: treat 2023-10-02 as an evidenced data gap (no play, calendar time retained) for the acceptance disposition. Disclose it in every table.

Nothing outcome-bearing has been read. The choice does not depend on any result.

## Schedule response hashes
```
52dfa75e1e3543a6f22c8e8726b783ecc96e8706610f7079a396a00f4fcdf5f5  sched_2021.json
de0090ef4e4372cc940410744d788a28d7fc5fbf13fef864014ad30206a03a36  sched_2022.json
83ad5f099c8b7e58ad5ada0051b6525601f36f1e1f9aeea0a5e7a102e6e314ce  sched_2023.json
2aa5b058acb36c49bb3dca9ac6e38df37597825c1e9fe6e436fce710e742580b  sched_2024.json
9f1cdf422e3458570a2de1785291813425b44f15d6855d9bf6fbd040f7a4f972  sched_2025.json
```

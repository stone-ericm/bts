# C1 rank 3 build plan: the 2021–2025 plate-appearance count table and starter workload history (historical half)

**Implements:** the FROZEN registration `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md`, §§1–4 and the freeze manifest (X-E1).
- **Scope:** only the historical count build. The 2027 shadow archive (§5) is production code under its own plan: reviewed until SIGN, with D7.
- **Rule:** every task is test-first, on synthetic feeds.
- **Exposure:** X-34 and the outcome-free freeze manifest are published before the build reads any feed or PA parquet for fitting. Before X-34, only code, the registration and outcome-free schema facts are used.

**Code:** `scripts/audit/c1_r3/` (beside the acquirer). **Tests:** `tests/scripts/c1_r3/`.

**Facts the build rests on (from code; no data read):**
- **Re-acquired feeds:** `data/raw_c1/<season>/<gamePk>.json.gz` (MLB v1.1 live feed). There are 12,148 games, each bound to a `stored` receipt (`--verify`, 10/05). The five 10/04 schedules carry no receipts and are pinned only by sha256 (cycle index, rank-3 row).
- **The production parser:** `src/bts/data/build.py:parse_game_feed`.
  - It dates PAs by `gameData.datetime.officialDate`.
  - It flags the resumed portion with `resumeDateTime`.
  - Its `lineup_position` is `battingOrder // 100`, which **mixes starters and substitutes**. That is why the registration requires feed metadata.
- **Starting identity in the feed:**
  - **Starting batter:** a player whose `boxscore.teams[side].players[*].battingOrder` is an exact multiple of 100 (`"100"`…`"900"`) started in slot `battingOrder / 100`. `X01`+ codes are substitutes.
  - **Starting pitcher:** `boxscore.teams[side].pitchers[0]`, cross-checked against the first pitcher faced by the opposing lineup in `allPlays`.

## Tasks
| # | Module | What it does | Key tests (synthetic, red first) |
|---|---|---|---|
| T1 | `count_meta.py` | Per feed: the game identity, home/away, the certified starting nine per side (slot ← `X00`), the starting pitcher per side, and the chronological PA list (`allPlays` in `atBatIndex` order, PA-ending events as `PA_ENDING_EVENTS`, resumed-portion flag as in `parse_game_feed`) | starters vs substitutes; a missing or duplicate slot; a starting pitcher mismatch between the boxscore and the plays; a suspended/resumed game; a non-monotonic `atBatIndex` |
| T2 | `count_verify.py` | Quarantine rules (§4): feed identity vs file and receipt; each side's 9 unique slots; starters appear as batters or have an evidenced no-PA; substitutions coherent; chronology; PA-event completeness against the production PA parquet's rows for that game (scoring definition). Every quarantined game is counted with its reason. The build **stops if quarantined > 1%** of eligible games | each quarantine reason; the 1% stop; the count is reported, never silently dropped |
| T3 | `count_table.py` | The count table: slot × home/away → distribution of min(N, 8) given N ≥ 1, where N = the certified starter's scoring-definition PAs, resumed portion excluded; add-one smoothing; the historical overflow (N > 8) count reported, not dropped | hand-computed tables; overflow in category 8; N = 0 excluded from the conditional; smoothing |
| T4 | `count_bf.py` | Per certified starting pitcher, per start: the complete PA-event count while pitching (legitimate resumed PA **retained**), keyed by official date with source provenance; the fixed 2021–2025 league median per-start BF | the resumed-PA retention contrast with T3; per-start attribution after a relief change; the median |
| T5 | `count_build.py` | Orchestration: the X-34 gate (as 4b's admission: `admission.json`, reviewed commit, X-34 row in its own commit); the outcome-free freeze manifest (feed and receipt digests, PA parquet digests, code commit, `uv.lock`); one pass over the feeds; T2's stop; outputs written to `data/hetzner_results/c1/r3/count_build/<sha>-<run>/` (restic-backed): the table, the BF history, the manifest and the quarantine log, each with sha256 | the gates refuse; the manifest is written before any read; a quarantine over 1% stops; synthetic end to end |
| T6 | review and run | Codex code review (≤2 rounds), then X-34, then the box run through the launcher (`c1-r3-build`, declared **6 CPU-hours**), then a results note | — |

**Order:** T1 → T2 → T3 → T4 → T5 → T6.

**Out of scope:** the 2027 shadow archive and its run/selection rules (§5); evaluation (§6, after 2027 date 90); any new acquisition (the feeds' schedules stay unreceipted, and a resume would refuse them).

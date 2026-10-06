# Rank 3 result: the 2021–2025 historical count build (C2 item (b), step 1)

**What this is:** the results note for rank 3's historical count build, registration `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` §4 (FROZEN). The build produces the frozen inputs of the 2027 forecast: a plate-appearance count table and each starter's batters-faced history.
- **This note does not compare forecasts.** The registered comparison is prospective: 2027 forecasts are scored after 2027 date 90 (§6). No 2026 data was read.
- **Under the registrations' cross-design rule,** none of these outputs is supplied to 4a or 4b (correction in `docs/sota_audit/2026-10-06-c2-cycle-index.md`).

**Status:** the build ran once and exited cleanly. **Independent acceptance is pending** (one Codex acceptance, at most 2 rounds, per the C2 proposal §4). Nothing here approves any production use; that needs §7's acceptance chain and Eric's D7.

## The run
| Item | Value |
|---|---|
| Code | `2dab83afbc4b11247c408346408324298b64b772` (the reviewed closure `bbafbef`, plus `admission.json` only) |
| Review | C2 code review r2, plain **SIGN.** (`docs/audit/2026-10-06-c2-r3-build-codex-r2.md`, sha256 `518c2c51…`) |
| Exposure row | X-34, published in `2107d32`; never edited |
| Unit | `c1-c2-r3-build-20261006T191700Z-a31bb602`, through the C1 launcher from `~/projects/bts-c1` at `2dab83a` |
| Claimed run | `2dab83a-20261006T191701Z`; `CLAIM.json` sha256 `3b4139c1…` |
| Time | 2026-10-06 19:17:00 → 19:19:12 UTC |
| Compute | 130.0 CPU-seconds (guard receipt: `exit`, rc 0, no leftover processes); C2 ledger total 0.036 CPU-hours of 15 |
| Output root | `data/hetzner_results/c1/r3/count_build/2dab83a-20261006T191701Z/` on the box (restic-backed) |

## Inputs (pinned before the run)
- **Feeds:** 12,148 receipt-bound 2021–2025 regular-season game feeds under `data/raw_c1/`. The inventory sha256 is `87ce6d8b…`.
  - These are the C1 re-acquisition's legacy per-game receipts. That evidence's limits are in the C1 cycle index, rank-3 row.
  - The 10/04 schedules carry no receipts and were not used.
- **PA parquets:** five files, each pinned by sha256 in `scripts/audit/c1_r3/admission.json`. X-34 binds their digest, `b024936f…`.

| File | sha256 | Bytes |
|---|---|---|
| `pa_2021.parquet` | `8f570f0e…a16b3f` | 25,626,512 |
| `pa_2022.parquet` | `37fae457…87e03bf` | 26,580,530 |
| `pa_2023.parquet` | `d4acf380…433ed9f4` | 27,041,597 |
| `pa_2024.parquet` | `f27a7def…a91029fa` | 26,723,195 |
| `pa_2025.parquet` | `fa69bb09…dd244d44e` | 26,838,461 |

## Census
- **Eligible:** 12,018 games; 130 feeds are ineligible under the build's eligibility rules.
- **Certified:** 11,949. **Quarantined:** 69 (rate 0.574%), under the registered 1% stop, so the build proceeded (`census.json`, sha256 `bcf4905b…`).
- **Quarantine reasons, counted by script from `census.json`** (games per reason; 3 games carry more than one reason):
  - **Chronology, 37 games:** a play's `startTime` goes backwards. The rule refuses any non-monotonic play time.
  - **Starting pitcher's official batters faced ≠ the completed turns he faced in the feed, 21 games:** home 12, away 9.
  - **The listed starting pitcher was not the first to face the opposing side, 9 games:** away 7, home 2.
  - **One game each:**
    - a play with `isComplete` false;
    - a play result `game_advisory` (not a supported completed-turn or open-turn result);
    - the last play not complete;
    - a lineup person appearing for both sides.
- Every quarantined game is counted with its reasons, and none was retained. The registration requires this whatever the rate.

## Outputs
| File | sha256 | What it holds |
|---|---|---|
| `count_table.json` | `15167e3b…9d9ee14` | slot × home/away → the distribution of min(N, 8) given N ≥ 1, with add-one smoothing over categories 1–8 (cap 8, conditioning N ≥ 1, as registered) |
| `bf_starts.json` | `ebf720d7…a4e33ce7` | 23,898 certified starts' batters-faced counts (legitimate resumed PA retained), with source provenance; league median BF per start = 23.0 |
| `census.json` | `bcf4905b…35efa715` | eligibility, certification and quarantine reasons |
| `provenance.json` | `28356ead…bf9929df4` | per-game source binding |
| `manifest.json` | `698c3368…` | the pre-read manifest: code, admission, accepted review, closure tree ids, registration hash, inventory and pins |
| `results.json` | `ffa04f89…` | the run's summary, binding the four outputs above by sha256 |

**Count-table support** (descriptive; computed from `count_table.json`):

| Cell | n (N ≥ 1) | zero-PA excluded | N > 8 overflow | mean min(N, 8) |
|---|---|---|---|---|
| 1 away | 11949 | 0 | 0 | 4.575 |
| 1 home | 11947 | 2 | 0 | 4.395 |
| 2 away | 11948 | 1 | 0 | 4.472 |
| 2 home | 11946 | 3 | 0 | 4.289 |
| 3 away | 11948 | 1 | 0 | 4.360 |
| 3 home | 11948 | 1 | 0 | 4.190 |
| 4 away | 11947 | 2 | 0 | 4.268 |
| 4 home | 11946 | 3 | 0 | 4.097 |
| 5 away | 11945 | 4 | 0 | 4.140 |
| 5 home | 11938 | 11 | 0 | 3.977 |
| 6 away | 11944 | 5 | 0 | 4.022 |
| 6 home | 11941 | 8 | 0 | 3.851 |
| 7 away | 11944 | 5 | 0 | 3.876 |
| 7 home | 11940 | 9 | 0 | 3.714 |
| 8 away | 11941 | 8 | 0 | 3.711 |
| 8 home | 11941 | 8 | 0 | 3.543 |
| 9 away | 11896 | 53 | 0 | 3.416 |
| 9 home | 11843 | 106 | 0 | 3.239 |

**Plausibility, not a test:** the means fall down the order, and home trails away. That fits the home team often not batting in the 9th, which also gives the larger zero-PA count at home slot 9. The slot-1 mean (4.58 away, 4.40 home) is close to production's flat `est_pas` of 4.5 for slot 1.

## What follows
1. **Independent acceptance** (Codex, at most 2 rounds) of these outputs against the registration's §4 definitions and the run's own binding.
2. **The 2027 shadow archive** (§5) is C2 item (b) step 2: production code, reviewed until SIGN, then D7, and deployed before 2027 date 1. It consumes `count_table.json` and the BF history by hash.
3. **The evaluation** runs after 2027 date 90 (§6, 0.5 CPU-hours).

## Limits (from the registration)
- The table is fitted on 2021–2025 and may not transfer to 2027 rules.
- The starter split is a deterministic batting-order cutoff, not a fitted hazard.
- The probability of not starting is not modelled.
- The 2021–2025 feed evidence has per-game receipts only (no response-level byte retention), and the schedules are unreceipted.

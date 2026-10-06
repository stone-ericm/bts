## Verdict

**ACCEPT WITH CORRECTIONS**

Accept the supplied historical artifacts as rank 3's frozen input, subject to the exact results-note edits below. No artifact repair, refit or rerun is required by this review. This verdict approves no production use, no 2027 run and no deployment; registration §7's acceptance chain and Eric's D7 remain required.

Protocol disclosure: my first tool call read the complete prompt and then searched `/Users/eric/.codex/memories/MEMORY.md` before I had inspected the prompt's isolation instruction. I disclosed that error in commentary before continuing. The search returned earlier BTS review entries, including rank-3 and rank-4b topics. I made no further outside-checkout source search, read no referenced memory summaries or shared corpus, and used no memory facts as evidence for this verdict. This round cannot be described as strictly blind.

## Findings

### 1. Binding passes on the supplied bytes and local Git history

HEAD is the requested detached `428b13a492221e323ab46b0e53dd690b8c4ae0ec`; the tracked tree was clean at entry and after the checks. The executable closure is unchanged between the run commit `2dab83afbc4b11247c408346408324298b64b772` and this HEAD. Between reviewed `bbafbef11beb530b22e8ba8232eb874dbd4b1510` and the run commit, the only closure change is `scripts/audit/c1_r3/admission.json`.

All four `results.json.outputs` digests match the supplied bytes. Independently computed digests for the complete supplied file set are:

| File | SHA-256 |
|---|---|
| `CLAIM.json` | `3b4139c184f0efbfeb8d17b88e4500be5e6c053f49a07018c4142691e37be842` |
| `manifest.json` | `698c3368bfd5927319f9dcab368d9e935d779d32e3a2c05a0839961c126af71f` |
| `inventory.json` | `87ce6d8bf1babc0e9a8e0ba2a1715bae23188ea0d5bee266b32791a9af9a1b6c` |
| `census.json` | `bcf4905b4ac333c31d72e85c963945e1e7441901be7db5a60532e05d35efa715` |
| `count_table.json` | `15167e3bff47a9c050ad565367b4ddc8b0b5b0edfa7b78aba60ff919b9d9ee14` |
| `bf_starts.json` | `ebf720d7b7f67f4a5e03dbde3375ae37c68caf15b9ac6e95ec97a104e4a33ce7` |
| `provenance.json` | `28356eadf40fdd379d037055d42139f8ccb3f12a1bfebcecea313ebdf9929df4` |
| `results.json` | `ffa04f89addf50e4fbe02e2b9697634f71b6cfd71fee96bb7a8135f807f20937` |

The manifest contains the fields emitted by `count_build.py:390–398` and binds:

- The admission record exactly, including its byte digest `82ad180ad46dc9eaed377b86efa02f1de118bbbaf2816de555e79f73c6eab38b`.
- The accepted report at the exposure commit: digest `518c2c51dbd4658f6fc9d19771a3398cd0c299681159b6944f75222d145d3ea9`, plain SIGN and the exact reviewed-commit subject. The current report has the same digest.
- All eight closure Git object IDs at the run commit, including the dependency-file IDs; I compared object metadata without opening configuration contents.
- Registration SHA-256 `e682e1cba7b125787fb9a68b455a5b60c5e54e293c4d88fcc0b9e0faefadbc2a`, and the event-definition hash independently reconstructed from the source's literal `PA_ENDING_EVENTS`.
- The supplied inventory's byte hash and 12,148-entry count; exactly five expected parquet pins; and the independently recomputed pins digest `b024936f4c4661dd343b931a3545bd863a6de6a7415c25bdaa23b46d026c1ef3`.

X-34 first appears at `2107d32ab8b32ebcb25ba4b1b90ba02f652e3db9`, is absent from that commit's parent and is unchanged now. Its complete structured description cell agrees with the admission and manifest on scope, report, report hash, reviewed commit and input-pins digest. Reviewed → exposure → run → current HEAD ancestry passes. The current read-only admission gate also passes.

Claim, manifest and results agree on the full run code. The claim names `2dab83a-20261006T191701Z`, has a positive integer PID and records `2026-10-06T19:17:02.012905+00:00`, consistent with the run name and reported interval.

These are byte/identity checks, not an independent execution witness. `results.json` hashes the four derived outputs; it does not hash `manifest.json` or `CLAIM.json`. The manifest separately hashes the inventory. Supplied files cannot prove claim-before-read timing, a unique box run, launcher execution, exit status or backup completion. The code's publication order and synthetic tests support the declared procedure; operational receipts were not supplied. Correction C5 makes that distinction explicit.

### 2. Census, inventory and provenance reconcile

The 12,148 unique, canonically ordered inventory games are distributed as 2,429 / 2,430 / 2,430 / 2,429 / 2,430 over 2021–2025. Every inventory entry has a canonical season/game path, matching feed URL, valid source digests and a nonempty attempt ID. All supplied availability flags are true, and every timezone-aware retrieval time precedes the claim.

`provenance.json` has exactly 12,018 eligible game identities. Each joins to its inventory entry on decoded hash, retrieval time, attempt ID and URL. Its false certification flags identify exactly the 69 quarantined games; its true flags identify 11,949 certified games. Inventory minus provenance contains exactly 130 games, matching the ineligible-feed count. Thus `12,148 = 12,018 + 130` and `12,018 = 11,949 + 69`.

The stop rule uses eligible games, not all feeds: `69 / 12,018 = 0.005741387918122816`, or 0.5741387918%. `stop=false` is correct under the registered strict `> 1%` rule (`count_verify.py:156–166`). Exactly 1% is permitted by that rule; the passing synthetic boundary test checks this explicitly.

All **73 reason occurrences** are counted in the census's prefix-based `reasons` dictionary and reproduced in `results.json`. The note's semantic counts also reconcile:

- Chronology: 37 games, **39** occurrences.
- Official starter BF versus completed turns: 21 games, home 12 / away 9.
- Starter not first opposing pitcher: 9 games, away 7 / home 2.
- Incomplete interior play, unsupported `game_advisory` and incomplete last play: each occurs in the same game, 661565.
- A lineup person on both sides: one game.

The three games with multiple reason strings are 633545 (2), 662155 (2) and 661565 (3). Nothing was lost by the note's games-per-reason presentation. The census's prefix summary is coarse, but the complete strings remain available.

Every quarantined identity is absent from BF history, verified directly. The count table has no game identities, so exclusion from its individual contributors cannot be independently proved from that aggregate. Its support totals are consistent with exclusion, and the inspected executing path appends count rows and BF starts only under `if not reasons` (`count_build.py:451–460`). That is source evidence, not a raw-data recount.

### 3. Count-table arithmetic passes; the note must identify its empirical means

There are exactly 18 cells, one for each slot 1–9 × away/home. Each has exactly categories 1–8, nonnegative integer counts, and `n_starts=11,949`. In every cell, category counts plus zero-PA exclusions equal the certified-game count.

For every cell I recomputed `n = sum(counts)` and every probability as `(counts[k] + 1) / (n + 8)`. All **144** probabilities agree exactly with the supplied values, and every distribution normalizes within floating-point tolerance. Cap 8, conditioning `N >= 1` and add-one smoothing match both manifest and registration. The code places all `N >= 8` in category 8 while separately counting `N > 8` (`count_table.py:25–36`); its passing synthetic overflow test includes N=9 and N=11.

Across the cells there are 215,082 starting-batter opportunities, 230 zero-PA exclusions and 214,852 positive-count observations. The supplied overflow counters are all zero. Three observations are in category 8; with the declared overflow counters these are N=8, not N>8. The overflow counters themselves cannot be independently remeasured without starting-batter/game rows or raw feeds.

Every support-table number in the note agrees with the output counts. Every displayed mean equals the **unsmoothed empirical conditional mean**, rounded to three decimals. It is not the mean of the fitted `p` distribution. For example, slot 9 home is 3.23887528498 empirically (displayed 3.239), versus 3.23972660535 under smoothed `p` (3.240). Correction C3 clarifies the heading without changing the numbers.

### 4. BF history supports the lag predicate, with two consumer requirements

The history has exactly 23,898 rows: one away starter and one home starter for each certified game, covering 844 pitcher IDs. No game/side or pitcher/official-date start identity is duplicated. Both starts in each game share the official date and have different pitcher IDs. Dates are valid ISO dates within the inventory's season. Every source hash, retrieval timestamp and attempt ID agrees with the game's certified provenance and inventory.

BF is a positive integer in [1,37]. Every `bf_resumed` is an integer between zero and BF; **all supplied values are zero**. The registered reader includes resumed PA in BF, unlike the contest count target; that behavior passes synthetic resumed-game fixtures. A positive resumed contribution in this historical artifact was not demonstrated.

The independently recomputed median of all 23,898 registered BF values is **23.0**. Do not substitute `box_batters_faced` for `bf`: 23,433 starts have equal values, 450 have official BF greater by one, and 15 by two. Registered BF sums to 521,863 versus official BF 522,343. This is consistent with the distinct definitions in the reviewed code: BF counts `meta.pas`, whose event set excludes `intent_walk` and `batter_interference`; official completeness counts completed turns including both (`count_bf.py:15–21`, `count_meta.py:62–65,275–281`, `count_verify.py:111–127`). Raw events are unavailable, so I cannot independently attribute each difference to its event. Correction C2 prevents the note from implying ordinary boxscore BF.

The fields support §2's 2027 selection: match pitcher identity, require `official_date < forecast_date` and actual retrieval availability before the forecast, then choose the latest five available official dates. Certified-source retrieval times range from 2026-10-04 23:16:11.128685 UTC to 2026-10-05 08:29:14.969719 UTC. They support historical-source availability before 2027; they do **not** support simulated availability before the original 2021–2025 games. An independent selection over these fields returned no sources for an as-of-2025 timestamp.

The file is ordered by season/game ID and side, **not chronologically**: 668 pitchers have an official-date reversal in their emitted rows. A consumer must filter and sort; taking the last five rows would violate the registration. No consumer lag implementation is being accepted here. Its strict earlier-date, actual-availability, fewer-than-five and median-fallback fixtures remain part of §5's production review.

### 5. Results note: no forecast or production overclaim; specific wording corrections required

All supplied hash abbreviations, five parquet byte counts, census figures, quarantine summaries, start count, median and 18 support-table rows agree with the supplied files. The production slot-1 reference of 4.5 agrees with `src/bts/model/predict.py:730–731`. The displayed 4.58 / 4.40 rounded empirical means are accurate. The declared evaluation budget of 0.5 CPU-hours and prospective date-90 evaluation match the registration; these are declarations, not measured forecast results.

The note expressly withholds forecast conclusions and production approval and respects the cross-design rule. Three distinctions need clearer language: no 2026 fitting versus 2026 provenance reads (C1), registered PA-event BF versus official BF (C2), and empirical means versus smoothed means (C3). The claim that omitted home ninth innings "also gives" the larger slot-9 zero-PA count is not established by the outputs (C4). Fewer innings can explain fewer opportunities, but the artifacts do not identify the causes of zero PA. Launcher, timing, CPU, process-exit, ledger, single-run and backup assertions are author-reported operational facts, not independently verified supplied-output measurements (C5). The reported 130 CPU-seconds would equal 0.036111 CPU-hours, consistent with 0.036 after rounding; I did not verify the premise.

### 6. Test result: one admission-state-dependent failure, not a build-artifact blocker

The exact permitted command completed with **122 passed, 1 failed in 7.69s**. The failing `test_admission_is_required` (`test_count_build_e2e.py:139–142`) sets only a synthetic DATA root and expects admission refusal. It relies on the repository admission being closed. This checkout's populated admission passes; the synthetic world supplies only 2023, so the test instead reaches the premanifest stat for absent `pa_2021.parquet` and raises `FileNotFoundError`.

A separate scratch control reran that exact test with only `load_admission` replaced in memory by a closed admission record: **1 passed in 0.07s**. I did not edit the test or claim the original suite was green. This distinguishes a stale fixture assumption from admission bypass or an output arithmetic failure. The earlier code review's 184-test suite was not reproduced; the requested subdirectory contains 123 tests.

## Corrections

Apply these edits only to `docs/sota_audit/2026-10-06-result-c2-r3-count-build.md`. The replacements below are verbatim; no output or registration changes are required.

**C1 — fitting years versus provenance years.** Replace:

> No 2026 data was read.

with:

> The admitted fitting sources are 2021–2025 only; 2026 acquisition receipts and build records provide provenance, not 2026 count or outcome fitting.

**C2 — BF definition.** Replace the complete BF output-table row with:

```markdown
| `bf_starts.json` | `ebf720d7…a4e33ce7` | 23,898 certified starts' registered PA-event BF counts (`PA_ENDING_EVENTS`, excluding `intent_walk` and `batter_interference`; resumed portions included by definition), with source provenance; league median BF per start = 23.0 |
```

Immediately after the output table and before `**Count-table support**`, insert:

```markdown
**BF definition:** registered `bf` differs from official `box_batters_faced` in 465 starts: 450 by one and 15 by two, with official BF higher. The artifact retains both fields. The 2027 consumer must use registered `bf`, filter by strictly earlier official date and actual source availability, and sort by official date before selecting the latest five starts; file order is by game ID.
```

**C3 — identify the mean.** Replace the support-table header row with:

```markdown
| Cell | n (N ≥ 1) | zero-PA excluded | N > 8 overflow | empirical mean min(N, 8) given N ≥ 1 (before smoothing) |
```

**C4 — remove the unsupported zero-PA explanation.** Replace the complete plausibility paragraph with:

```markdown
**Plausibility, not a test:** the empirical conditional means fall down the order, and home trails away. Omitting the home ninth inning is one plausible contributor to fewer home opportunities, but these aggregate outputs do not identify the causes of either the mean differences or the larger zero-PA count at home slot 9. The slot-1 empirical mean (4.58 away, 4.40 home) is close to production's flat `est_pas` of 4.5 for slot 1; that comparison establishes no forecast improvement.
```

**C5 — distinguish reported operations from supplied evidence.** Replace:

> **Status:** the build ran once and exited cleanly.

with:

> **Status:** the author reports that the build ran once and exited cleanly; the supplied files contain one claim and completed outputs.

Immediately after `## The run` and before its table, insert:

```markdown
**Evidence boundary:** code, claim identity, input pins, inventory and output hashes can be checked from the supplied files and local Git history. Launcher execution, start/end times, CPU usage, clean process exit, the ledger total, single-run history and backup status are the author's operational account; their guard, ledger, run-inventory and backup receipts were not supplied for this acceptance.
```

## What was checked

**Independently recomputed:** strict JSON parsing for duplicate keys and nonfinite constants; all eight supplied file hashes; all four result-bound digests; inventory hash/count; admission, report, registration and event-set hashes; every closure object ID and closure delta; commit ancestry and X-34's first-publication/full-cell/unchanged-row checks; all inventory/provenance/BF joins and identities; census conservation, exact quarantine flags, rate, strict stop decision and all 73 reason occurrences; 18 cell conservation equations and all 144 smoothed probabilities; support, zero and overflow totals and bounds; every note support-table mean; BF identity coverage, ranges, official-BF differences, emitted ordering and median; and all five note parquet hash abbreviations/byte counts. The independent output checker did not call the production count-table or BF aggregation helpers. It used only supplied JSON, permitted source and local Git objects.

**Source/synthetic checks:** the raw-input pin-before-parse path, quarantine-before-aggregation branch, scoring/resumed definitions and BF event set were inspected at the unchanged run code. The permitted rank-3 suite and the separately labelled closed-admission control ran with no real data or provider calls. The first, exact permitted pytest invocation used pytest's default temporary test directory; the later control used `/private/tmp/c2-r3acc-a1-tLXQQk/`. Task-generated temporary files were cleaned up. No tracked file was edited.

**Taken from supplied evidence, not independently regenerated:** input-source digests and parquet metadata; the inventory's receipt-derived retrieval times and source claims; certification judgments; historical starting identities, chronological play/substitution/completeness judgments; per-batter N, resumed classification, zero-PA and overflow membership; and per-start BF values. Without raw feeds, original receipts and PA parquets I cannot hash the original sources, reconstruct the eligible set from parquet rows, re-certify games, rebuild the category counts or workload rows, identify individual contributors to the aggregate table, or independently verify each official-date/availability witness against original source bytes. The result-file hashes prove equality to this supplied bundle, not equality to uninspected box originals.

**Not checked or authorized:** box/network/SSH/gh access, real `data/` contents, credentials, backup restoration, guard/ledger receipts, process lifecycle, production deployment or a 2027 forecast/evaluation. The supplied manifest's environment versions describe the reported run and were not measured on the box. The 2027 consumer and forecast comparison still need their registered reviews and owner gates.

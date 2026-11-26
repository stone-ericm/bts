## Round

design r1 — `docs/superpowers/specs/2026-09-28-season-ledger-design.md` — HEAD `5e44bc1` (verified).

Reviewed the spec against W1.1 and source/test code. No real data, snapshots, network, or outcome artifacts were accessed; no tests were run. The inventory counts and coverage statements in the spec are inputs to this review, not independently verified facts. Examples below are synthetic failure cases.

## Findings

1. **BLOCKER — §§3, 5–6, 8: cache chronology does not establish the BTS cutoff outcome.**

   The proposed evidence bundle freezes bytes now; it does not establish when those bytes were acquired or whether they were subsequently replaced. `src/bts/data/pull.py:40–46` discovers only games already marked Final, and `download_game_feed` (`:53–79`) skips existing files and writes response bytes without a capture receipt. A scheduled 03:00 pull therefore does not prove that every file was acquired at 03:00, was final then, or still contains its original bytes. Even a proven final 03:00 observation can precede a valid 06:00 correction. It is evidence of an earlier result, not necessarily the 08:00 result.

   The fallback compounds this: §5 writes a post-cutoff observation into `bts_result`, although §10 says early caches cannot provide that result. A source flag does not prevent downstream users from counting it as a BTS outcome. Likewise, `bts_result != mlb_final_result` proves disagreement, not that a change occurred after the cutoff.

   **Change:** define authority and temporal confidence separately. Preserve observed feed grades with capture-time evidence, game status, and unknown chronology where necessary; reserve authoritative BTS outcomes for qualifying contest evidence or an explicitly justified cutoff reconstruction. Keep post-cutoff-only observations out of the authoritative outcome. Rename the generic flag to `outcome_disagreement`; require additional temporal evidence for `rescored_after_cutoff`. Fixtures must include an overwritten cache, late-finishing game, unknown capture time, and a 03:00-to-06:00 correction, not just files hand-labelled pre/post-cutoff.

2. **BLOCKER — §§4–5, 8: the named grader does not implement the stated scoring rule.**

   `picks.grade_pick_in_feed` (`src/bts/picks.py:1003–1016`) delegates normal games to `_boxscore_hit` (`:970–989`), which tests only hits and defaults missing batting hits to zero. It returns a miss for a rostered nonparticipant or a batter whose only appearances were walks; neither establishes an at-bat or sacrifice fly. It also does not check game status. The suspension helper (`:931–967`) treats any pre-resumption `PA_ENDING_EVENTS` event as enough for a miss, including walks, hit-by-pitches, and sacrifice bunts. Missing play times can instead become a false void. Reusing this function as specified cannot produce §5's pass/no-hit semantics.

   **Change:** specify an audit grader with explicit hit / qualifying no-hit / pass / pending / unknown rules, required fields, status checks, and strict game/player identity. Missing statistics must remain unknown rather than default to zero. Establish equivalent exclusion of resumed PA to `filter_out_resumed_portion`; that exclusion alone does not establish the AB/SF rule. Exercise raw synthetic feeds through the real parser and grader: DNP, BB/HBP-only, sacrifice fly versus bunt, live/postponed games, suspended BB-only, resumed-only hit, and missing resumption/play timing. Do not silently change production grading as part of this audit build.

3. **BLOCKER — §§3, 5–6, 8: contest authority lacks a lossless parsing and state contract.**

   The existing parsers are unsuitable adapters without explicit replacement. `contest_ledger.parse_latest_ledger` retains only round-level fields and the last two snapshots; its module documentation explicitly says finality is unavailable. `leaderboard.scraper` (`src/bts/leaderboard/scraper.py:305–348`) retains slots but coerces null streak/hits/atBats to zero and drops predictions with unmapped rounds. A fixture that supplies normalized per-slot grades and saver state directly would bypass these losses.

   The design also leaves the relationship between round `void`, slot `void`, slot hit/no-hit, partial-DD validity, and saver protection undefined. A round label must not automatically become both legs' grade. `contest_ledger.likely_save` explicitly warns that saver availability cannot soundly be inferred from its windowed history. `saver_transitions` records attempted transitions, including rejected ones (`src/bts/saver_state.py:77–101`); an operator's write time is not necessarily the consumption time. Latest settled contest state also is not the state available when a historical decision was made.

   **Change:** parse raw fetch records and nested slots losslessly, preserving absent/null/zero, round and slot labels, `streakIncrease`, observation time, and source-record identity. Define deterministic selection among conflicting snapshots, coverage/finality status, and separate decision-as-of state from later observed state. Give round/slot/saver cases an explicit truth table; preserve unknown saver availability and pre-streak when continuity is unproved. Replay only successful state writes, with their actual evidential meaning. Add raw fixtures for null statistics, a round disappearing between fetches, missing intermediate state, saver-held miss, partial-void DD, and rejected saver transitions.

4. **BLOCKER — §§3, 5–6, 8: batter-only matching cannot establish entry for a particular game, and local rows cannot cover contest-only entries.**

   W1.1 requires round/unit, game, batter, slot, and revision identity. The proposed keys omit unit/game/batter, and §6's batter match cannot distinguish the same batter in two games on one date or a replacement selection. A contest grade can consequently attach to the wrong local game. The raw slot has `unitId`, but §3 freezes only rounds and players lookups. Current static units are not a promised historical crosswalk (`src/bts/leaderboard/static_capture.py:12` describes their current/upcoming scope).

   Moreover, “entered” requires a graded slot: a pending entered slot becomes indistinguishable from absent evidence. Starting from local picks also loses a manually entered or otherwise contest-only slot. Absence from the latest fetch is not proof of non-entry.

   **Change:** retain round/unit/player identity and establish a documented unit-to-game mapping where evidence permits. Unresolved or ambiguous matches must not receive another game's grade. Build from the union of source observations, retain contest-only/unmapped slots, and expose join cardinality and unmatched records. Use entry status such as confirmed / absent-with-coverage / unknown independently of grading status. A doubleheader fixture must reach this join with raw IDs and incomplete lookups; supplying `game_pk` directly in normalized contest rows would conceal the defect.

5. **BLOCKER — §§1–3, 5, 8: overwritable files and timestamps cannot recover “every decision.”**

   “One row per date × stream × slot” conflicts with revision retention. `save_pick` overwrites the daily pick; `write_decision` overwrites the daily decision and stamps a fresh `finalized_at` (`src/bts/daily_decision.py:99–103`). A later crash-guard/classification write can describe the same selection, while an earlier preview may no longer exist. Its timestamp is not a stable selection identifier. A synthetic fixture containing two complete revisions would not demonstrate that the listed real source types preserve both.

   The input list also omits W1.1's late/refused-delivery archives. `_archive_and_remove_pick` (`src/bts/scheduler.py:2125–2139`) writes the full candidate and removes the top-level pick; deferred and refused candidates can survive only there. `append_lineup_evolution` (`src/bts/picks.py:370–401`) provides additional, thinner observations, not a complete decision history.

   **Change:** explicitly inventory supported archives and distinguish source observation IDs, selection identity, later status updates, and the selected canonical revision. Retain recoverable revisions with links to their evidence; mark unrecoverable history rather than reconstructing it from the surviving commit. Define whether the output contains all revisions or a final view over a persisted observation table. Test the actual overwrite/archive shapes, repeated finalization of one selection, and a preview for which only partial evidence survives.

6. **BLOCKER — §§5, 8: unrestricted legacy day fallback fabricates DD leg outcomes.**

   A DD day-level miss does not say which leg missed; a hit on a partial-void day does not prove both legs hit. Copying `result` into both missing slot results makes invented observations look measured. W1.1 allows justified legacy **single** fallback. The established measurement helper already enforces this: `scripts/audit/build_slot_dataset.py:34–45` uses day result only for a single primary and leaves legacy DD unresolved.

   **Change:** retain `local_day_result` separately. Permit per-slot fallback only for a proven single-pick observation under a documented recipe. Leave missing DD leg grades unknown unless independent slot evidence resolves them. Test day miss with one hit leg, partial void, and a legacy DD with no slot evidence; the latter must not emit two resolved local grades merely because `legacy_day_fallback` is flagged.

7. **SHOULD — §§5–6, 8: field definitions need version and stream rules before implementation.**

   `objective=absent` conflates legacy reach57 with invalid v3 metadata: `daily_decision.decision_objective` (`:40–50`) deliberately returns reach57 for v1/v2 and unknown for invalid v3. `action_source` also mixes source with fallback/objective: `PolicyDecision` records mdp/heuristic separately from objective and degradation. Shadow v1/v2 has no explicit schema column here; production's daily decision cannot supply the counterfactual action or recipe for every stream. Calendar absence cannot establish an explicit skip, and a declined skip candidate is not a production selection.

   W1.1 also requires delivery/entry/outcome timestamps and game eligibility; §5 supplies booleans and game start but not those fields. `finalized_at`, recovered lock time, provider confirmation, observed contest entry, and actual entry time have different meanings. `pick_was_delivered` (`src/bts/picks.py:321–329`) requires a DM identifier as well as the sent flag. Finally, §6 permits journal-only fallback while §10 prohibits journals as the only field source.

   **Change:** publish a small source/version/stream mapping table, including unknown/not-applicable states, shadow version, decision candidates versus executed selections, and explicit skip versus unobserved calendar day. Separate event timestamps from observation timestamps and leave unobserved events null. Define delivery evidence and eligibility at the relevant selection/attempt time; resolve the journal contradiction. Test invalid v3 objective, legacy v1, a crash lock without confirmation, and a missing day with no skip evidence.

8. **BLOCKER — §§1, 7–8: the reconciliation can pass while excluding the records that explain the discrepancy.**

   “For every ... slot in the pick files” is an insufficient universe when the disputed tally may include archives, other streams, or duplicate revisions. Only the primary scorecard rule is described; the DD membership/grading rule is not. Current frozen results can also differ from the results that the historical recipe saw. Giving every surviving row an exclusion reason and listing a residual can satisfy the present acceptance text without identifying any historical membership. Also, 191 versus 156 does not by itself prove which files or iterations the earlier count used.

   **Change:** freeze each recoverable recipe's file selection, date window, as-of source version, slot inclusion rule, and grading rule. Account for the union of candidate source records, including duplicates, archives, and contest-only observations. Require counts to partition into emitted, explicitly excluded, and quarantined/unresolved records, with anti-join and cardinality checks. Distinguish exact historical membership, partial reconstruction, and unrecoverable membership; do not label guessed exclusions as explanations. Keep original-recipe outcomes separate from corrected canonical outcomes. Synthetic golden membership sets should fail when one record is omitted, duplicated, or matched to two games. Preserve the completed P-01 read recorded in the exposure register; this build must not silently replace its frozen recipe.

9. **BLOCKER — §§2–3, 5, 8: D8 can admit captures that its existing contract rejects.**

   A dated research directory and a top-ranked row are insufficient. `research_sidecar_state` (`scripts/live_forward_capture_once.py:925–997`) validates contract hashes, candidate identity, exporter commit binding, start/completion ordering, and a matching `research_capture.accepted.json` publication before first pitch. W1.1's parent plan explicitly requires that two-phase acceptance. The proposed parser/fixtures never require it. The exporter contains both production and candidate ranks (`src/bts/experiment/artifacts.py:451–504`), so “research-only top-1” also needs an explicit variant selection.

   **Change:** require the same read-only acceptance checks before admitting a research observation; retain research-only/ineligible-for-official-read labels and reconcile the provisional skip trigger with final day classification. Select the intended variant/rank and join outcomes by batter and game. Add missing acceptance marker, mismatched hash/head, late publication, wrong candidate, and later production-pick fixtures. Alternatively defer this stream to a second phase; do not weaken its existing acceptance contract to fit the generic parser.

10. **BLOCKER — §§3, 5, 8: recipe and slate references can be assigned to the wrong prediction.**

    A checkout reflog is not evidence of which code a running process or cached prediction used. The proposed epoch tuple omits W1.1's feature schema, blend/calibration configuration, PA aggregation, training/retrain rule, and policy boundaries. `FEATURE_ENV_DEFAULTS` (`src/bts/picks.py:25–42`) covers five environment settings and explicitly excludes non-environment model composition. A code/env hash alone cannot certify the larger recipe.

    A same-date slate is similarly insufficient. `save_slate` is last-write-wins (`src/bts/slate.py:10–12, 61–75`) and the orchestrator writes it before selection (`src/bts/orchestrator.py:276–280`). A subsequent failed attempt can overwrite the slate after an earlier usable prediction was created. Its hash proves which file was read, not that it was the selectable slate for the committed decision.

    **Change:** separate prediction provenance from delivery/deployment context. Use only supported historical recipe components, with per-component unknowns and explicit evidence for epoch boundaries. Preserve missing dimensions rather than asserting a complete epoch from the reduced tuple. A slate needs a defensible binding to the selection; otherwise label it an unbound same-day observation, unsuitable for decision-time slate analysis. Fixtures should cover a cached prediction across a deploy, missing calibration configuration, and a later slate overwrite after an unsuccessful selection.

11. **SHOULD — §§3–4, 9–10: separate evidence acquisition from deterministic compilation.**

    Fetching today's MLB record is acceptable for an explicitly dated comparison, but “once at build time” mixes acquisition and reconstruction. Partial failures/restarts can silently mix acquisition times. Hashing only the two new input classes also leaves the build's complete dependency set and the persistence of normalized source observations unspecified. Fixed sorting alone does not guarantee byte-identical Parquet across serialization versions or absolute evidence-root paths.

    **Change:** make acquisition produce a sealed manifest with the complete input inventory, lookup/config/source hashes, request identity, fetch times, and explicit missing/failed responses. Compile without network access from that manifest, with output fields referencing their winning source observations. Bind a builder version and serialization environment; define stable types, nulls, timestamps, ordering, and logical relative paths. Test rebuilds from relocated fixture roots and reordered discovery, plus rejection of changed/missing declared inputs. A later acquisition is a new evidence version, not a silent update to the old ledger.

12. **SHOULD — §§1–2, 9: restore a bounded first pass.**

    The design makes production reconciliation, three counterfactual streams, current MLB acquisition, and a recipe registry one deliverable. W1.1 explicitly permits a minimum pass identifying DD legs and their inclusion status. This broader scope adds acceptance and provenance problems before the core reconciliation is trustworthy.

    **Change:** first deliver production/source observations, game-specific contest matching, justified per-slot local results, conservative outcome authority, and auditable historical membership. Report unresolved evidence honestly. Add shadow versions, policy shadow, validated D8, current-record comparisons, and richer epoch/slate attribution in separately accepted increments. Keep one shared schema where useful; do not require speculative reconstruction to complete the initial ledger. The already completed formal DD read remains intact.

## Verdict

BLOCK — resolve outcome authority, grading, identity/history, and reconciliation acceptance before implementing the canonical ledger.

## Round

design r2 — spec v2, `docs/superpowers/specs/2026-09-28-season-ledger-design.md`; HEAD `58b90a7` + working tree (verified).

Reviewed source, test code, the revised spec, and the prior review. No real data or snapshots were accessed. No network requests or tests were run. Inventory statements remain unverified inputs to this design review. Counterexamples below are synthetic.

## Findings

**Disposition of every r1 finding**

| r1 | Disposition in v2 |
|---|---|
| 1 — feed authority/chronology | Authority issue resolved: feeds no longer populate authoritative BTS outcomes. The chronology label still overstates its evidence; see finding 7. |
| 2 — reuse of incompatible grader | Resolved by a separate audit grader. Its incomplete-input and suspension contract needs the clarification in finding 8. |
| 3 — contest parsing/state | Partly resolved: raw slots, null preservation, and separate decision-time state are specified. Coverage, snapshot qualification, and state continuity remain open; finding 3. |
| 4 — game identity/entry | Partly resolved: unit IDs and contest-only observations are retained. The schedule fallback can assign a wrong game; findings 1–3. |
| 5 — revisions/archives | Partly resolved: archives and observation revisions are included. Canonical skip rules and completeness claims still fail; findings 1 and 6. |
| 6 — legacy DD fallback | The direct DD fabrication is resolved. The proof of a single-pick day remains dependent on incomplete history; finding 6. |
| 7 — versions/fields | Stream-specific issues are safely deferred and the journal rule is clarified. Public-delivery evidence, unknown action source, and required fields need correction; finding 9. |
| 8 — reconciliation | Still open. Count matching has replaced evidence of recipe identity, and the source accounting universe is incomplete; finding 4. |
| 9 — D8 admission | Safely deferred, with its existing acceptance contract retained. No Phase 1 dependency. |
| 10 — epochs/slate binding | Safely deferred; raw provenance can be retained without assigning epochs or binding slates. No Phase 1 dependency. |
| 11 — acquisition/compiler split | Resolved in architecture. Clarify declared missing inputs and source occurrences in the implementation plan; findings 4 and 7. |
| 12 — bounded first pass | Resolved in scope. Production-only Phase 1 is buildable after the corrections below; the deferred streams need not return. |

1. **BLOCKER — §§4–5, 9–10: define a canonical day/selection rule that cannot promote a skip candidate or provisional intent.**

   A skip decision contains a `primary` candidate: `_write_endofday_skip` writes that declined candidate while setting `action="skip"`, `scoreable=False` (`src/bts/scheduler.py:759–785`). Thus “the selection named by the surviving decision file” needs an explicit action-qualified exception. Otherwise a literal implementation can emit both the declined primary and the `slot=none` skip row as canonical production decisions.

   Scheduler state is more problematic: it has `final_skip_candidate`, not a separate final-skip receipt. That field is persisted during intraday checks (`:2677–2699`). Finalization checks both `committed_pick_written` and the surviving scoreable decision before recording a skip (`:769–773`). A state containing a skip candidate plus a committed pick must not yield a skip; an abandoned midday state containing only a skip candidate proves intent, not end-of-day finalization. A date with an archive or unfinished attempt but no surviving final decision is also neither “no evidence at all” nor a proven skip.

   The canonical key needs one more distinction: a production primary A and an unmatched contest entry B can coexist on the same date. B has no known local primary/DD role, so it cannot be forced into the same `(date, slot)` key or assigned a role from contest `number`.

   **Smallest change:** specify a day-status/row-kind table. A validated explicit skip produces one day row; its candidates remain observations. A scheduler skip candidate alone produces `observed_unfinalized`/unknown final action. A pick-file-only candidate remains explicitly unconfirmed as a commit unless supported. Give contest-only rows their own stable identity and unknown local slot. On decision/pick conflicts, preserve the preferred source view but mark finalization unresolved; never attach B's delivery, probability, or result to A. Add fixtures for each case, including a decision-only committed selection and conflicting surviving files.

2. **BLOCKER — §§3, 5, 9: a current player-team lookup plus one scheduled game does not prove a historical game.**

   Consider a player traded from team A to team B in July. The 9/27 player lookup associates him with B. In April both teams played one game. For an April contest slot with no unit mapping, the proposed fallback selects B's April game and can attach the slot's official grade to that wrong game. A schedule has team/game identity, not a historical batter-team relationship. The existing static parser derives team through the player's `squadId` (`src/bts/leaderboard/scraper.py:318–345`); it does not make that relationship historical. Contest-only slots have no local pick to supply a contemporaneous team either.

   Current schedule responses also need explicit treatment of postponed/rescheduled games and resumed games; “exactly one game” must not mean only one remaining Final game after filtering away a second scheduled unit. Conflicting unit captures must not silently collapse to the last dictionary value.

   **Smallest change:** remove the schedule fallback from authoritative matching in Phase 1. Retain unmapped unit IDs and an ambiguous match. If the fallback is retained, require independently evidenced team membership for that date, an exhaustive game candidate set including postponed/resumed cases, a unique mapping, and an explicit inferred-match status. A current team assignment is insufficient. Test a traded player and conflicting unit mappings, not only the easy doubleheader ambiguity.

3. **BLOCKER — §§5–7, 9: round presence is not a coverage certificate, and “latest” is not a qualification rule.**

   `derive_source_date` is the maximum date among settled predictions, not a declaration that every round or slot through that date is present (`src/bts/contest_fetch.py:79–97`). The profile is described as settled-only, while pending entries use a separate endpoint (`:41–52`). The ledger writer persists raw predictions but no completeness certificate (`src/bts/cli.py:1966–1975`); `validate_fetch` checks streak counters, not nested slot completeness. `contest_ledger.py` explicitly warns that finality is unavailable.

   Counterexample: record R1 has round X with settled A; R2 has X with A and newly available B. Applying “a ledger record ... shows the round without this selection” to R1 can label B absent despite R2 proving entry. An unresolved player/unit in a present round likewise cannot establish absence of a particular local selection. Conversely, a later malformed or truncated round can replace an earlier valid slot set under the unconditional latest-containing rule. Retaining the earlier value in a revision table does not repair the selected outcome or absence claim.

   “Consecutive ... continuous” also needs a definition: adjacency among observed rounds does not prove that intervening account state is known, and an unbroken observed window does not establish saver availability at its start.

   **Smallest change:** define one deterministic snapshot qualification/selection policy shared by entry, grades, and state: structural validity, timestamps/order and tie handling, slot identity, and treatment of regressions/dropped slots. Preserve a last-observed official grade with its observation time and qualification status; do not call a pre-cutoff observation settled merely because it is the latest retained one. For Phase 1, leave nonmatches unknown unless complete round-slot coverage is independently established; unknown identities prohibit negative entry claims. Define the predecessor/initial-state evidence needed for streak/saver inference, or leave those derived fields unknown. Add partial-round growth, mapped/unmapped mixed slots, a later malformed round, and an older positive versus newer omission fixture.

4. **BLOCKER — §§1, 8–9: matching 191/157 cannot identify a historical recipe, and accounting omits decision-only facts.**

   Two candidate rules can both emit 191 primaries and 157 DD legs while selecting different dates or revisions. Pre-declaring the rules prevents one kind of fitting but cannot identify which rule was used historically. Even a unique match within that list is not evidence that the original rule belongs to the list. Calling the result `reconstructed (rule X)` without retaining unknown historical membership overstates what the test establishes. Also, 191 versus 156 does not prove the earlier tally counted other file types: the earlier file population may differ or the counting procedure may duplicate records.

   The proposed union includes O3 **skip** decisions but omits O3 pick decisions lacking an O1 file, and O4-only facts that §5 permits to generate rows. A canonical decision-only selection can therefore escape the “every production source record” accounting checks. Source observations and unique selections also have different multiplicities; repeated contest snapshots must not become extra historical picks.

   **Smallest change:** report candidate rules as `matches published aggregate totals; historical membership unverified`, retain all matching alternatives, and keep per-record historical membership partial/unknown until independent recipe evidence exists. Separate observed historical result from the result of rerunning a hypothesis on final files. The cited C-03 change on 8/26 predates 9/11, so it is not an example of a post-scorecard change; use the actual evidence interval, not the game's date.

   Define an exhaustive source-occurrence accounting table, including non-skip decisions and scheduler-only observations, separately from deduplicated selections and recipe membership. Give occurrences path + record/slot locator identity as well as content hashes, so byte-identical files at different paths cannot disappear from accounting. Fixtures must include two different rules with identical totals, a decision-only pick, and duplicate source bytes at distinct paths.

5. **BLOCKER — §§6–7, 9: the disagreement flags compare different vocabularies and different units.**

   Local results are preserved as written: `DailyPick` uses `hit/miss/void` (`src/bts/picks.py:223–224`), whereas the proposed contest and feed columns use `no_hit`. A local `miss` and contest `not_hit → no_hit` agree semantically, but literal comparison flags a C-03-style disagreement. `feed_grade=pass` versus contest slot `void` needs an explicit semantic mapping before comparison. “Any two non-null outcome columns” also includes `local_day_result`: a DD day miss with primary hit is an ordinary aggregate/leg relationship, not conflicting evidence about the primary.

   **Smallest change:** preserve raw labels and add a closed normalization/comparison table. Compare only known, semantically comparable **slot** outcomes for the same selection/game; exclude day aggregates, unknowns, and unresolved status labels. Define slot `void` explicitly—the round's `void` meaning is irrelevant to this mapping. Add fixtures for local miss/contest not_hit agreement, pass/void comparability, and primary hit plus DD miss/day miss. This is needed even if feed grading is deferred.

6. **SHOULD — §§2–5, 6, 9: `history_incomplete` cannot default to “complete,” and the single-day fallback needs a bounded proof.**

   V2 flags incompleteness only when evidence shows a lost revision. Overwrite-on-write means that absence of such evidence is not proof that no revision existed. In particular, “no double-down in any revision of that date” can only establish no DD in the retained revisions. It cannot prove the whole day was single when the history is unknown. Conversely, an unrelated discarded DD preview should not automatically invalidate a day result demonstrably bound to a later single selection.

   The spec also still omits the thin append-only lineup-evolution source written by `save_pick` (`src/bts/picks.py:300–315, 370–401`). That source can contain a surviving earlier batter/game/probability observation even when the pick file was overwritten; it cannot recover all commit metadata.

   **Change:** use history status complete / known incomplete / unknown, with the completeness criterion stated. Either include retained evolution observations or explicitly narrow Phase 1's history claim and disclose their exclusion. Bind any legacy-single derivation to the particular single-pick observation and its day result, with a derivation source; do not populate the raw `local_slot_result` column with a derived value. Where that binding is uncertain, keep the derived slot unknown. Test missing history and a discarded DD preview followed by an independently evidenced single.

7. **SHOULD — §§3, 6, 9: `captured_final_before_cutoff` still claims more than mtime plus a log proves.**

   A successful pull log and an old file mtime do not bind the current bytes to that pull. Replacing/restoring bytes with preserved timestamps passes both predicates. `download_game_feed` writes bytes without a contemporaneous hash receipt (`src/bts/data/pull.py:53–79`). The bundle hash protects the later frozen copy, not historical capture. Separating feed observations from BTS authority fixes the major r1 problem, but the chronology label remains too strong.

   **Change:** distinguish proven capture identity from `mtime/log suggest pre-cutoff`, and use unknown when the bytes cannot be bound. Preserve the actual evidence fields rather than deriving a certified capture claim. Apply the same evidential rule to `captured_after_cutoff`; a copied file's mtime is not automatically a fetch time. Include preserved-mtime replacement in the overwrite fixture.

   Also clarify that an explicit manifest `missing` entry is valid evidence of absence. Compilation should refuse a file declared present that is absent/hash-mismatched, not refuse every bundle containing one of the expected-but-missing inputs that acquisition is required to record.

8. **SHOULD — §§6, 9: finish the audit grader's incomplete-input rules before coding fixtures.**

   The new AB/SF rule fixes the old grader's central defect. The remaining edge is deriving zero from absent or partial plays: a Final suspended game's boxscore can contain the batter while `allPlays` is missing or empty. Summing the available pre-resumption events yields zero and can falsely return pass. “Missing play timing” does not cover missing plays or a missing resumption boundary. The existing build filter excludes unplaceable PA (`src/bts/data/build.py:158–196`); it does not prove that the retained PA list is complete. Its behavior is not a completeness certificate for the audit grader.

   **Change:** require feed game identity to match the requested game, define validated input completeness and the suspension boundary, and return unknown when the evidence cannot establish a zero-AB/SF outcome. Give an explicit event-to-hit/AB/SF table, including sacrifice double plays; distinguish unsupported/malformed status from a known pending game. State the raw expectations for a rostered DNP with an empty batting object: absent stats remain unknown, not zero. Add missing/empty/truncated plays, absent boundary, unknown event type, and wrong embedded game ID fixtures. No production-grader change is needed.

9. **SHOULD — §§4, 10: restore the public-delivery branch and publish the remaining Phase 1 field contract.**

   The mapping lists sent+DM-ID, `delivered_at`, or a delivered decision, but omits the legacy public `bluesky_posted` branch that `pick_was_delivered` actually implements (`src/bts/picks.py:321–329`). A public pick before `delivered_at` and decision files existed can therefore lose its only supported delivery signal. The action-source enum also still omits `unknown`, which commit/classification writers explicitly persist (`src/bts/scheduler.py:683, 706, 753`); `forced` belongs in a separate derivation/fallback interpretation rather than replacing the raw source.

   **Change:** specify the historical public-delivery predicate and preserve conflicts with private/lock evidence; test a public-only legacy record. Retain raw and normalized source with an unknown value. Before the implementation plan, list the canonical output columns and nullability, including attempt versus confirmation, game eligibility, and entry/outcome observation timestamps required by W1.1. A contest round date is not an entry timestamp. These basic fields cannot be deferred with the recipe/slate work.

**Phase 1 and the smallest signable revision:** the production-only scope is now appropriate. Resolve findings 1–5 with explicit row, qualification, and comparison rules. The shortest route is to omit the schedule fallback, keep unsupported negative-entry and finalization claims unknown, and label aggregate-matching recipes as hypotheses. Fold findings 6–9 into the field/fixture contract without expanding scope. If further reduction is useful, defer cached-feed grading and chronology too; the production observations, local slot results, contest evidence, and honest reconciliation can stand on their own. Preserve the completed P-01 read. No new historical outcomes or production changes are needed to repair this design.

## Verdict

BLOCK — Phase 1 is bounded, but canonicalization, contest matching/qualification, outcome comparison, and reconciliation claims need the explicit corrections above before planning implementation.

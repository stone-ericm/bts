## Round

design r3 — spec v3, `docs/superpowers/specs/2026-09-28-season-ledger-design.md`; HEAD `77e4e8b` (verified).

Code/docs/test-source review only. No real data or snapshots accessed; no network requests or tests run. Inventory claims were not independently verified. This review reports only remaining contracts that need correction before implementation planning.

## Findings

**Disposition of r2 BLOCKERs**

| r2 | v3 disposition |
|---|---|
| #1 — canonical rows and skip intent | Skip candidates, scheduler-only intent, conflicting selections, and contest-only identity are addressed. The no-decision date restriction and commit inference still need finding 1. |
| #2 — historical game matching | Pick-time team addresses the trade counterexample; inferred matches are explicitly labelled and contest-only inference is excluded. Conflicting mappings must not fall through to inference: finding 3. |
| #3 — qualification/coverage | No negative entry claims, qualified observations, first/last seen, and `dropped_later` address the main blocker. Keeping a dropped slot as a dated last-positive observation is acceptable under the explicit absence of a finality claim. Saver inference still needs finding 4. |
| #4 — accounting/recipe reconstruction | Resolved at design level: occurrences, selections, and membership are separate; equal totals remain hypotheses and alternatives survive. |
| #5 — result vocabulary/comparison | Resolved at design level: raw labels survive, normalization is explicit, and only comparable slot outcomes are compared. HOLD must remain an outcome classification; it cannot establish game eligibility (finding 4). |

1. **BLOCKER — §§5, 9, 11: the row table still omits later missing decisions and treats account entry as evidence of a system commit.**

   The surviving-pick/no-decision rule is restricted to “before 6/23.” Decision persistence is best-effort on later dates too: `write_decision` returns `None` on failure (`src/bts/daily_decision.py:61–105`). `_write_commit_decision` sets `committed_pick_written` only after a successful write (`src/bts/scheduler.py:683–697`). A September pick can therefore have a successful delivery and a surviving pick file but no decision file. The table has no selection-row rule for it; routing it to “attempts without a surviving final record” loses the canonical selection despite its delivery evidence.

   Separately, a contest match proves account entry, not that the system committed that prediction. For example, an undelivered preview A survives and the owner independently enters A. Its `entry_status` is confirmed; the proposed `commit_status=entered` must not turn that preview into a committed system decision. Private commits also require a distinct status: the scheduler explicitly commits `private_locked` without delivery (`src/bts/scheduler.py:1013–1026`).

   **Smallest change:** apply the no-decision/surviving-pick rule on every date. Define commit status independently of entry: evidenced system commit, unconfirmed, or conflicted, with the source/basis retained. A contest match or generic `pick_locked` flag alone does not prove a system commit. Explicit private commits remain committed without claiming delivery or entry. Add one later-date failed-decision-write fixture and one entered-but-undelivered preview fixture. No extra source or production change is needed.

2. **BLOCKER — §5 history criteria: present and consistent files cannot certify complete history.**

   Concrete producer-supported counterexample: saving selection A succeeds, but its lineup-evolution append fails; saving selection B later succeeds, including the append, followed by B's decision/state writes. All four sources named in §5 are present and consistent, and evolution contains no other selection. The proposed predicate returns `complete`, although A was lost. `save_pick` expressly swallows audit-append failures after overwriting the pick (`src/bts/picks.py:300–317`). Consistency among surviving files cannot detect this case.

   **Smallest change:** do not infer `complete` from source presence/consistency. Phase 1 should emit `unknown` unless independent evidence establishes a lossless history; emit `known_incomplete` when a missing earlier record is evidenced. It is acceptable for no Phase 1 row to qualify as complete. Include the failed-append-then-overwrite sequence in the synthetic fixture contract. This needs no new capture machinery.

3. **BLOCKER — §§6, 11: conflicting unit evidence can still transfer a grade through the schedule fallback.**

   The rule uses unit evidence when there is exactly one `feedId`, **else** tries inference. Suppose unit U has conflicting captures mapping to G1 and G2; the surviving local selection is G1 and its team's acquired schedule lists only G1 on that date. The fallback produces `match=inferred` and transfers U's grade to G1 despite positive contradictory unit evidence. Merely retaining the conflicting captures does not make that transfer sound. The “conflicting unit captures” fixture currently has no required disposition.

   **Smallest change:** distinguish missing from conflicting/invalid mapping evidence. Only an absent mapping may proceed to the documented inference rule. Multiple distinct game mappings, incompatible round identity, or other contradictory unit evidence must produce `ambiguous` and transfer no grade. Give the conflicting-unit fixture an explicit expected result. The existing pick-time-team inference can otherwise remain a labelled inference; it must not be presented as direct unit evidence.

4. **BLOCKER — §§6–7, 9: two derived state claims still exceed the observations.**

   **Saver availability:** two rounds with consistent `streakIncrease` do not establish the availability at the beginning of the observed chain. A round taking the streak from 12 to 13 looks identical whether the saver is still available or was consumed before the retained window. Both rounds can be present and internally consistent in either history. `contest_ledger.likely_save` warns that a consumption can disappear from the window without a chain gap (`src/bts/contest_ledger.py:84–91`); `maybe_auto_earn_saver` likewise refuses to initialize active from an unknown state merely because the best streak is at least 10 (`src/bts/saver_state.py:169–175`). The parenthetical qualification in §6 is insufficient for saver availability.

   **Game eligibility:** §7 deliberately merges Pass and voided-slot outcomes into HOLD, but §9 turns a “voided” slot into `ineligible_evidenced`. An eligible player can produce a Pass; a game can also become postponed after a valid selection. The final slot label alone does not establish that the game or selection was ineligible when chosen or submitted. Normalized outcome agreement should not silently become an eligibility exclusion.

   **Smallest change:** require an evidenced initial saver state plus complete relevant transitions before deriving availability; otherwise emit unknown. Preserve directly reported round/streak facts separately. Define the time and meaning of `game_eligibility`, and remove bare slot `void`/HOLD as proof of ineligibility. Keep an evidenced postponement or refusal with its reason/time; unsupported selection-time eligibility stays unknown. Add same-visible-rounds/different-saver-history and valid-selection-then-Pass fixtures.

**Smallest signable revision:** make the four changes above, retaining unknowns instead of adding reconstruction work. Phase 1 is otherwise bounded enough for an implementation plan. Feed grading/chronology, research streams, epochs, and slate binding can remain deferred. The three reconciliation tables and the slot comparison contract do not need redesign. As a wording correction alongside these edits, §3 must also stop saying acquisition needs no network while it requires schedule fetches; the compiler is the offline step.

## Verdict

BLOCK — correct the four bounded row, history, matching, and state contracts above; no expansion of Phase 1 is required.

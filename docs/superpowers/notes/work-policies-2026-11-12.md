# BTS work policies, adopted from the EoR project (2026-11-12)

**Owner words (verbatim, 2026-11-12):** "Update all our work policies to match those developed in the eor project". Given in answer to the roadmap's question about review weight (`docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`, decision 2).

**This note governs how BTS work is done from 2026-11-12.** Where it conflicts with an older working rule (review until SIGN, the October manager rules, per-launch custody steps), this note wins. `CLAUDE.md` § Work policies states the rules in force; this note holds the source, the mapping and what is superseded.

**Sources in the EoR repo** (`~/projects/eve_of_revolution`): `CLAUDE.md` §§ 4, 5, 8, 9; `docs/superpowers/notes/review-effort-2026-10-21.md`; `docs/superpowers/notes/overhead-cut-2026-10-31.md`.

## Amendment, 2026-11-14: reviews by another agent are more frequent

**Owner words (verbatim, 2026-11-14):** "Whatever we decide on Gemini, let's make sure these types of reviews are more frequent going forward". Said after he asked whether adversarial reviews by another agent were being done, and learned that two days of pick-path changes had been verified only by their author until a Codex review was started that night. That review blocked twice, on defects the author's own tests and the gate run had not shown.

**What changes** (this replaces rule 5's single whole-range review, and narrows "no review" in rule 1):
- **Tier A:** a fresh adversarial Codex review of each change set the day it is written, before it is pushed.
- **Tiers B and C:** everything that changed goes into a daily adversarial Codex batch review, one round, besides the pre-result review of a result-deciding tool.
- **Before the pre-season deploy:** still one whole-range review, by a reviewer that has seen none of the earlier rounds.
- **Every review is adversarial:** assume the author is wrong; concrete counterexamples run against the real code; false greens asked for; the reviewer's probes become regression tests.
- **Reviewer settings:** Codex on the best model at the highest reasoning effort (`model_reasoning_effort=xhigh`), memories off. The first three review rounds under this note (the framing aggregate, pick path rounds 1 and 2) ran at the default `high`, which was a lapse against the standing rule in memory `feedback-codex-max-model-reasoning`.
- **A second model family** joins tier A only after passing a known-answer test against Codex's findings.

**Measured basis (2026-11-13/14, one project, small sample):** three Codex rounds on work the author had tested, mutation-checked and gated: 3 BLOCKs, 6 blocking findings, all demonstrated by running the code, all in interactions between changes; none found by the author's tests. **Cost if wrong:** more Codex quota and about ten minutes per round; a day's push waits for its review.

**The frequency and tiering above are the lead's reading of his words, not his wording.**

## Why it transfers

EoR measured that its real defects were found by cheap end-to-end runs (boots, save readings) and one review of runtime logic, and that most review rejections were about the verification apparatus itself (9 of 11). BTS's October cycle has the same shape, though it was not measured the same way: the watchdog, 4a, rank 3, 4b and the 2026 framing test each stopped on review findings about evidence controls, launch accounting or test tooling, while the maths and the deployed behaviour were mostly unchallenged. That reading is my interpretation of the cycle index and the review archives, not a count.

## The mapping

| EoR | BTS |
|---|---|
| The boot: one cheap hands-off run that decides a commit | **The gate run:** the focused tests for the change, then the fast suite (`-m "not slow"`, the four heavy directories ignored). For anything in the daily pick path, also a fixture-day dry run (`bts run --date <pinned> --dry-run`) once roadmap thread A has built the fixture. |
| The save reading: look at the real outcome | **The real-artifact check:** run the changed code against real stored data (`python -c`, a real `decision.json`, a real slate) and read the result. On the box, the failure-direction check: induce the failure, see the alarm, restore. |
| Tier A, runtime behaviour a boot cannot see | **Tier A, pick-path behaviour:** which pick is chosen, delivery, entry, streak and saver state, scheduler timing, the tail and base policies. |
| Tier B, content data | **Tier B, captures and instrumentation:** receipt logs, archives, slate fields, capture writers and readers. |
| Tier C, acceptance-deciding tools | **Tier C, tools that decide acceptance:** study runners and scorers, the compute launcher, admission and aggregate scripts. |
| Tier D, tests, docs, records | **Tier D:** tests, docs, register rows, plans, memos. |
| The driver | The session leading BTS work. |
| Push to pi5 inside the landing tool | Push to `origin main` after a green gate run. |
| Publication and spending | Deploys to the box, anything on the real contest account, activation, spending. |

## Rules in force

1. **Review by tier.**
   - **A:** RED-first focused tests, the gate run, a real-artifact check, and ONE fresh Codex review of behaviour, at most two rounds.
   - **B:** tests, the gate run and a real-artifact check. No review, unless the run shows something the author cannot explain. The limits of each capture are written down beside it.
   - **C:** RED to GREEN on excerpts of real artifacts, and the author runs the tool end to end on the real retained input before handing it over. No review, except the one tool that decides whether a commit lands or a study result is accepted. Mutant ledgers and fail-closed matrices are kept here only.
   - **D:** no review.
   - A mixed change takes its highest tier.
2. **Reviewer prompts name the tier** and carry this sentence: "Review behaviour and the change's stated contract. A finding about test-code or evidence-tooling quality is a NOTE, not a BLOCK, unless it makes a false claim about behaviour that no gate run or real-artifact check covers." A NOTE never counts toward a stop.
3. **Three BLOCKs in one session stop that session, never the work.** Record the obligations, write a kickoff brief, start a fresh session, and change something material: a narrower contract, less apparatus, a different approach or a fresh reviewer.
4. **Commit on green.** A green gate run is the commit test. Push to `origin main` follows. No per-launch custody steps: no sealed-outcome announcements, tuple checks, marker hashes, heads-up messages or relay ledgers.
5. **Draft sprint for preseason work.** Remaining 2027 preparation is written as best informed guesses without design-review rounds. Each change still passes its gate run. One fresh whole-range review covers everything in tier A before the deploy that precedes activation.
6. **Standing design authority.** The lead session decides design questions and reports them to Eric once a day when work is running; he can overrule afterwards. Rulings are ledgered with their cost if wrong.
7. **Owner-only actions** (never under design authority): spending, including restoring the box; any deploy to the box; anything on the real contest account (cookie capture, entry, the field-capture footprint); delivery mode and activation; reading 2027 outcomes for a study with no exposure-register row; cutting scope; publication.
8. **A stall never idles the wall hours.** Make a safe, reversible provisional ruling and record it, or switch to other work. Owner-only actions still wait.
9. **Agents.** Lanes, reviewers and helpers are Codex sessions with memories off and Remote Control off; Claude for a second-model opinion or what Codex cannot do. Check `ai-usage` before staffing and spend each ration close to its reset. Long runs go to a remote machine, not the Mac.
10. **Daily multi-model workflow review** whenever BTS runs agent work across more than one day: a facts-only evidence brief, a fresh Codex reviewer and a Claude subagent on another model asked separately for the largest workflow flaws and false greens, a synthesis, and a delivery estimate with the critical path.
11. **Owner decisions** are written in plain language first (situation, options, evidence, cost if wrong), identifiers last.

## Kept unchanged

- Every rule under `CLAUDE.md` § Safety Rules: leakage guards, suspended-game scoring, and the real-account request hygiene with its 403/429 kill switch.
- **The exposure register and pre-registration.** A 2027 outcome is read for a study only after its row is published. This is Eric's D2 ruling about evidence, not a work policy, and it is what makes 2027 data usable as a test.
- Test first, verify in the failure direction, pinned `today=` in date-relative tests.
- The frozen 2026 artifacts: the season ledger recipe, the incident register tooling at `f453283`, the frozen wrap memos. They are records, not working code.
- Deploy mechanics: push to `deploy`, land inside a scheduler sleep window, canary and auto-rollback.

## Superseded

- "Reviewed until SIGN" for production code, and the deploy-gating review as an open-ended loop. Replaced by rule 1 tier A and the whole-range review in rule 5.
- The October manager rules: no push to main without a full suite at that exact commit, suite serialization across worktrees, per-launch heads-up and post-launch tuple messages, sealed outcomes with an unsealing announcement, completion markers and brief hashes for reviewer handovers, BLOCK and process-slip counters per lead.
- Design review before build as a standing requirement (memory `feedback_codex_design_review_before_build`). Under the draft sprint, design questions are decided by the lead and reviewed once with the code.
- C1's two-round cap on registrations stays; a registration without a SIGN after two rounds is frozen with its limits stated, not deferred.

## Where BTS differs from EoR, and the cost if wrong

EoR's ruling rested on "nothing is published, so fixes stay cheap". That holds for BTS until activation and stops holding afterwards: in season a pick-path bug can cost the streak, and a capture bug loses data that cannot be collected again.

- **Before activation:** the draft sprint applies in full. Cost if wrong: a defect surfaces in the rehearsal or the whole-range review instead of earlier, and is fixed then.
- **From activation to the end of the contest:** tier A changes are hotfixes from a clean branch at the deployed commit, each with its ONE review before deploy, and each deploy is Eric's. Captures in tier B get a failure-direction check on the box in their first week. This is my adaptation, not EoR's wording; it is the smallest change that keeps the real account protected.

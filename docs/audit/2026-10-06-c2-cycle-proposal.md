# C2 cycle proposal: scope, order, caps and stop rules for 2027 Cycle 2 (for Eric's approval)

**Status:** **DRAFT 2026-10-06**, sent to Eric through the herdr manager (`projects-30`). Nothing in it is built before his recorded approval. Like the C1 proposal, it is not Codex-reviewed: each item's designs and code get their own reviews. Written by the BTS lead session (bts-lead2) from the kickoff brief `docs/ops/2026-10-06-bts-lead-kickoff.md` §3.

## What is already decided
- **Eric's order (10/06):** (a) the rank-2 watchdog; (b) 4a together with rank 3; (c) 4b; (d) P3 / W-state stays deferred. One at a time.
- **Rulings that carry over** (register §C):
  - D1, Option 2: a 4b policy needs Eric's trade table under D7.
  - D2: one named change at a time, with independent acceptance.
  - D3 RESERVE: no 2026 outcome reads.
  - D6: no activation decided.
  - D7: every production change and every research result goes to Eric; a review SIGN approves neither.
- **C1's rounds are spent.** C1 deferred every candidate after its last round. C2 does not resume those rounds; it starts each item from the fixes its last review required.

## The proposal, in plain words
- **One new cycle, C2.** No new money. A small box budget: **15 CPU-hours** of research jobs, of which the plan uses about 11.
- **One item at a time, in Eric's order.** Each item restarts from the exact list of fixes its last review asked for. Nothing is redesigned.
- **Reviews are capped.** Each review unit (a design, or a piece of research or production code) gets at most 2 Codex rounds. Without a SIGN after 2, the item stops and comes back to Eric with options. A third round needs a recorded ruling, as in C1.
- **Each item has a finish-by date.** An item not done by its date stops and comes back to Eric, so no item can use up the time the later ones need before the season.
- **The hard deadline is the 2027 season start.**
  - MLB's 2027 season opens on **Wednesday 2027-03-24** (traditional Opening Day 3/25), the earliest ever.
  - Everything that must run from the first game is deployed by **2027-03-10**.
  - The players' labor contract expires 2026-12-01 and a lockout is expected. If the season moves, these dates move with it (two weeks before contest date 1). They never move earlier.
- **Nothing ships or changes picks without Eric's D7.** Research results go to him separately. A 4b policy affects picks only after he has seen and approved its trade table.

**What Eric gets if it all works:**
- **(a)** A signed watchdog. The plan is to deploy it in private research mode, so it runs for the whole off-season before any activation. That deploy is its own D7 request.
- **(b)** Rank 3's 2021–25 count table built and accepted. The pick-path capture both studies need is deployed before the first game: rank 3's pre-lock forecast archive and 4a's serving record. 4a's study code is signed, so its fit (after 2027 date 40) and its test (after date 90) can run.
- **(c)** 4b's trade table (the chance of 57 given up, against the expected longest streak gained), for his decision. If he approves a policy, wiring it into production is a further production change under D7.

**What it costs if an item fails:**
- A stopped item is deferred, not forced. 2027 then runs without it, exactly as production runs today: the reach-57 base policy plus the tail policy, no watchdog and no new capture.
- **The one loss that can't be recovered later:** 4a and rank 3 need records taken before each lock. If the capture is not deployed by the first game, the 2027 data they need is never recorded.

## 1. Name
**C2** ("2027 Cycle 2").
- **Cycle index:** `docs/sota_audit/<date>-c2-cycle-index.md`.
- **New plans and specs:** `docs/superpowers/{plans,specs}/<date>-c2-<item>-*.md`. **Reviews:** `docs/audit/<date>-c2-<item>-codex-r<N>.md`.
- **Box units:** `c1-c2-<item>-<step>` (§3 says why the `c1-` prefix stays).

## 2. Money cap: $0 of new spend
The same as C1: the existing box and the existing Claude and Codex subscriptions only. Anything that would cost money stops the cycle and goes to Eric.

## 3. Compute cap: 15 CPU-hours of box research jobs, through the C1 launcher, unchanged
**Recommendation: no new ledger root.**
- **Why:** the C1 launcher, guard and ledger were signed on 10/06 (infrastructure review r4 SIGN on `3d6ae3f`). They have one canonical root by design (`data/hetzner_results/c1`); infrastructure review r1 B3 removed the alternate-root option. The 50/100 CPU-hour gates are hard-coded (`ledger.py`). A C2 root or a C2 cap in code would be an infrastructure change, and so would need a new infrastructure review.
- **How C2 is still counted:** C2 jobs launch with `--name c2-<item>-<step>`, so their units are `c1-c2-…` and their ledger rows can be summed on their own.
- **The shared ledger:** it held **0.45 CPU-hours** (16 rows) when read on 2026-10-06. The launcher's 50-hour checkpoint and 100-hour stop stay as they are, for the combined C1 + C2 total.
- **How the 15-hour cap is enforced:**
  - the guard enforces each job's declared budget (it kills the job's processes at that budget);
  - before each launch, the lead checks that C2's used hours plus the declared budget stay within 15.

**Planned jobs (declared CPU-hours):**

| Job | When | Declared |
|---|---|---|
| (a) W-restore rehearsal | before the watchdog's deploy review | 0.1 |
| (b) rank-3 count build | after X-34 | 6 |
| (c) 4b run (3 wall-hours) | after X-31 | 4 |
| (b) 4a fit and 4a evaluation | after 2027 dates 40 and 90 | 0.25 + 0.25 |
| (b) rank-3 evaluation | after 2027 date 90 | 0.5 |
| **Total** | | **11.1** |

- **Stop and report to the manager before launching** any job not on this list, and any rerun. The launcher already pauses every job after an overrun, a guard error, a timeout or an out-of-memory kill, and blocks a same-name relaunch after an ordinary failure.
- **Before the first C2 job:** the box worktree `~/projects/bts-c1` is at `1fa872a`, which is older than the signed launcher `3d6ae3f`. It is re-pinned to each job's reviewed commit, with `uv sync --extra model --frozen`. The production tree `~/projects/bts` is never used for research jobs.
- **Per job:** one research job at a time on the box, with the launcher's limits (nice 10, `MemoryMax=12G`, `OOMScoreAdjust=1000`). During the 2027 season, jobs run only inside the scheduler's sleep window.
- **The Mac:** test suites are not counted, and no multi-hour run goes on the Mac.

## 4. Review cap (attention)
- **Every review unit gets at most 2 Codex rounds.**
  - **Without a SIGN after 2:** the item stops, and the lead reports to the manager with options (a third round, a later cycle, or a narrower scope).
  - **A third round** needs a recorded ruling by Eric, or by the manager under the 10/03 delegation. Its provenance is written exactly as given.
- **Production code is still reviewed until SIGN before it can ship.** The cap means an unsigned piece does not ship; it never means a piece ships after fewer rounds.
- **Before each deploy:**
  - a whole-range deploy-gating review by a fresh Codex session, with Part 1 blind;
  - the affected W1.5 certificates re-run with the frozen tooling (`f453283`), as on 10/06;
  - then D7.
- **Each research result** gets one independent Codex acceptance memo, at most 2 rounds.
- **Before each review:** a pinned mutation-test ledger, run as the C1 lessons describe (`python -B`, a fresh `PYTHONPYCACHEPREFIX`, restore by hash).
- **A disclosed limit:** C1's reviewers searched the memory registry three times despite the ban. Any recurrence is recorded, and that review is labelled not strictly blind.

## 5. Order, scope, budgets and finish-by dates
**What counts as done for the order:** an item is done when its last code review signs and its D7 request has gone to Eric. Its deploy and commissioning wait on Eric, so they may overlap with the next item's work.

**Where code lives:**
- **Production code:** on a branch cut from a fresh `origin/main` until its D7. The next `main:deploy` ships everything, so production code merges only with its approval (or the reviewed SHA is deployed directly).
- **Research code** (`scripts/audit/`): may be on main. The admission gate keeps it closed until it is admitted.

### (a) The rank-2 watchdog: every unit signed by **2026-11-30**
**Starts from:**
- W0 at `c429c41`, parked off main by `f882411`, plus W0 review r3's required changes R3-1 to R3-4;
- the W1 evidence map and spec draft (both unreviewed);
- the FROZEN registration and the build plan.

The producers C1/C2, P4, P1 and P2 are now deployed (`f882411`). So W2 and W4 can be tested against receipts the real producers emit, which is C1's producer-to-reader lesson.

**Four review units.** Each gets a spec round where no reviewed spec exists, then code rounds; each kind is capped at 2:

| Unit | Pieces | Notes |
|---|---|---|
| 1. W0 repair | R3-1 a stable checker identity; R3-2 delivery order per target; R3-3 reverse completeness; R3-4 the gate's treatment of child processes | **Hardest first: R3-4.** It decides how the confinement gate handles child processes (W-mode's `crontab -l` is one). Its approach goes into a short design note, reviewed with the W0 code |
| 2. The 5-minute job | W1 (deliver, postponed, decision, mode), W5 (restart), W7 (self) | W1's spec draft gets its spec round first |
| 3. The receipt readers | W2 (entry, from P1 receipts), W4 (grade, from the contest ledger and P2 receipts) | fixtures produced by the deployed producers' code |
| 4. The rest | W6 (restore, one 0.1 CPU-hour rehearsal), W8 (capture-stop), W9 (the coverage matrix over all 119 incident records) | W-state is recorded as deferred in W9. **W8 depends on §7 question 2** |

**Then:**
- the deploy-gating review and the certificate re-runs;
- D7;
- install through the intent-aware cron, with `entry_intent = "research"`;
- commissioning: Eric's dedicated Healthchecks check, a missed-completion rehearsal, and the producer receipts verified on the box.

Activation stays a separate A8 approval. The D7 request will state the recurring costs: the weekly read-only R2 restore must fit the existing free allowance, or it stops at $0.

**Kill condition (registration):** the watchdog adds state races, or it cannot tell private picks from contest entries.

**P3 / W-state stays deferred.** It reopens only if W9's coverage matrix shows a defect class that only W-state can catch, and then only by a recorded scope decision.

### (b) Rank 3 with 4a: all code signed and the rank-3 build accepted by **2027-01-31**; capture deployed by **2027-03-10**
Four steps, in this order.

**1. The rank-3 historical build (research).**
- **The fix:** R4-1 only (receipt witnesses validated before joins, across every completion outcome), meeting review r4's three required changes.
- **The review:** at most 2 rounds. The admission gate needs a plain SIGN.
- **Then:** the hash-only input capture, X-34, the build (6 CPU-hours, through the launcher), a results note and an independent acceptance.
- **Why it goes first (reasoning, not yet tested):**
  - 4a's R1 fix reuses rank 3's team plate-appearance accounting. The build runs that accounting over 12,148 real feeds and publishes a quarantine census by reason.
  - That census tests C1's hardest open assumption on real feed structure: that complete-support accounting does not make most games unknown. It does so under X-34, before 4a builds on it.
  - X-34 should state that the census by reason is shared with 4a, and that the count table is not.

**2. The pick-path capture (production).** It holds three pieces:
- rank 3's pre-lock run archive (registration §5);
- 4a's serving record (S1–S3, resumed from `e66b440`);
- R10's explicit `projected=false` for a posted lineup.

**Recommendation (reasoning only): one design and one production review sequence.**
- **Why:** all three write at the same point of the serving path (`orchestrator.predict_local` / `predict.py`). One sequence means the pick path is reviewed, re-certified and deployed once. The design review may still split them.
- **The gate:** the registration's shadow on/off byte-equality fixtures (pick, slate, decision and delivery traces, including slow and failing capture).
- **The fixtures** are built by running the real producer, not by hand. This is C1's `projected` lesson.

**3. The 4a study code (research).**
- **The fixes:** R1 (reusing rank 3's accounting), R2, R3, R4, R10's `parsed` counter, and the `~/projects/bts-c1` launch path.
- **The review:** at most 2 rounds.

**4. Before the first 2027 capture:**
- checklist A3 (the 2027 rules and calendar) feeds 4a's calendar and serving-contract freezes;
- X-32 is published before the first 2027 read.

### (c) 4b: trade table to Eric by **2027-02-15**; any chosen policy deployed by **2027-03-10**
**Starts from:** the code at `3648120` / `ad29869`, and review r3's required changes 1–8:
- **B1–B3:** adopt the shared admission gate `scripts/audit/c1/admission.py`, as rank 3 did.
- **B4/B5:** fixed by the signed C1 infrastructure. The review is shown this, not asked to assume it.
- **B6:** `--verify` in the shared rank-3 acquirer. It is fixed only **after** rank 3's build has run, so rank 3's reviewed code is not disturbed before its run.
- **B7:** the parity call runs only the three registered Δ = 0 anchors.
- **F4/F6:** output and inventory validation.
- **Documentation:** as r3 required.

**The review:** a fresh session on the whole range, at most 2 rounds; the admission gate needs a plain SIGN.

**Then:** X-31; the run (4 CPU-hours, 3 wall-hours); the results memo; an independent acceptance; and the trade table to Eric under D7.

**If Eric picks a policy:** its artifact and routing are a production change, reviewed until SIGN and then D7. They must pass the A2 tail/base pairing check. If (c) runs late, 2027 starts on today's policies; that is the safe fallback.

### (d) P3 / W-state: stays deferred
It reopens only under the condition in (a).

## 6. Stop rules
**Cycle-level stops.** Any one of these pauses all of C2, and resuming needs Eric:
1. **Money:** any cost above $0.
2. **Compute:** C2 reaching 15 CPU-hours, or the launcher's own 50/100 gates.
3. **Production safety:** a research job that delays or disrupts a scheduler run, the dashboard, the nightly backup or a pick is killed at once and not restarted without Eric.
4. **MLB blocking:** any 403 or 429 stops at once, with no rerun without Eric.
5. **Scope:** no item joins mid-cycle. Reopening P3, a rank-1 blend, or ranks 5–7 each needs a recorded scope decision.
6. **Calendar:** C2 ends on the last day of the 2027 contest (recorded under A3). An item without a disposition then is recorded inconclusive and does not ship.

**Item-level stops:** the review cap (§4) and the finish-by date (§5). Either one stops the item and brings it to Eric with options; the next item then starts.

**Per-item kill conditions** (from the FROZEN registrations; unchanged):

| Item | Stops if | Done when |
|---|---|---|
| (a) watchdog | it adds state races, or cannot tell private picks from contest entries | failure/recovery fixtures red then green, every unit SIGN, deploy-gating SIGN, D7; operational acceptance after commissioning |
| (b) rank 3 | it needs actual-exposure inputs; the hit target changes inconsistently; or nothing beyond the June PA-tilt null | the count table is built and accepted; pre-lock forecasts archived from the first 2027 game; scored on later untouched dates and independently accepted |
| (b) 4a | held-out proper scores don't improve by the registered threshold | fit and test windows scored and independently accepted; a separate boundary check before it may change play |
| (c) 4b | registration §6: a negative disposition (the mean longest-streak contrast ≤ 0 at Δ = 0, or a breached reach-20 or individual-season regression guardrail); incomplete coverage or a failed validation is inconclusive, never positive | Eric has the complete tables; a positive screen also needs independent acceptance, and a concrete policy needs his D7 approval after the trade table |

**Data rules:**
- X-34, X-32 and X-31 are each published before their outcome-bearing read.
- No 2026 outcome is read (D3).

## 7. What this asks of Eric
1. **Approve C2 as written:** its scope, order, the 15 CPU-hour cap on the shared C1 ledger, the review cap, the finish-by dates and the 2027-03-10 deploy deadline. It includes two recommendations he may overrule:
   - reuse the C1 launcher and ledger rather than give C2 its own root;
   - one pick-path capture design for rank 3 and 4a.
2. **A gap found in planning: the rank-1 capture prerequisites.**
   - **The facts:** Eric approved them as part of C1 (D4: "only the rank-1 prerequisites alongside" ranks 2, 4a and 3). Their design is FROZEN (`docs/sota_audit/2026-10-04-prereg-c1-mlb-capture.md`). They were never built and never deferred; no register row disposes of them.
   - **Part A** gives the existing MLB static capture a receipt per fetch and a permanent stop on 403/429. It is production code, and it must run from the first 2027 capture or that evidence is lost.
   - **Part B** is a one-off, outcome-free 0.5 CPU-hour diagnostic, which needs X-33.
   - **The dependency:** W8 (capture-stop) watches the stop marker that Part A writes, and the watchdog's gate cannot pass on a receipt no producer emits.
   - **The options:**
     - **(i) Recommended:** build Part A inside (a)'s unit 4, beside W8, and run Part B after (c) (+0.5 CPU-hours, total 11.6).
     - **(ii)** Add both as a new item (e) after 4b; W8 then waits for it, and so does the watchdog's deploy.
     - **(iii)** Leave both out: W8 is recorded as deferred in W9, and 2027 has no receipted MLB-forecast record.
   - **The cost if wrong:**
     - (iii) loses a season of rank-1 evidence for good;
     - (i) adds one small production change to the watchdog's last unit.

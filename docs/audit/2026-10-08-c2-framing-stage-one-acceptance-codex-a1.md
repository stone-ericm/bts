## Verdict
**ACCEPT WITH CORRECTIONS.**

Accepted-commit: 2e077a36a22dfe2b692c05957d89103abdd3b0de

The supplied artifacts establish **inconclusive for A and B**, conditional on their retained labels. Accept the stage-one result and the note with the exact corrections below: correct the diagnostic count and B sd/interpretation, make the secondary rule explicit, restore input-availability limits, and qualify unsupported operational/budget claims. Producer-platform execution and original-input label correctness were not independently verified. This review approves no production change, stage two, or further run.

Blindness disclosure: before receiving this task, the conversation automatically supplied a memory summary naming the earlier C2 framing r9 runner-containment BLOCK (`prompt-r9`, `confcutdir`, allowed imports, local module shadowing, symlink, HookCaller, R9-1), with the guidance that collection must be confined, imports bound to reviewed physical bytes, and linked targets covered. It also supplied general BTS review guidance about provenance and the distinction between local evidence and runtime admission. No measured framing-screen outcome was in that summary. I did not open any memory file, session record, or previous review before writing Part 1. The authorized pre-registration itself describes earlier reviews; I read that description, not those reports. No commit subject was displayed. Before Part 1 was written, the archived r10 report was only hashed as expressly allowed, without decoding or interpreting its bytes. Apart from the task checkout, reads were limited to the Python/uv runtime and installed packages, and task-owned scratch; there was no outside project or remote read.

## Part 1 (blind)

This section was written to disk before opening the results note, its evidence directory, index row (e) or ledger, or any prior framing review. The independent reconstruction and all nine full rescoring checks below were completed first. The unmodified aggregate invokes an admission gate that reads the archived r10 review internally, so its invocation is deliberately deferred until this section is on disk. That subsequent procedural measurement will be recorded separately without rewriting these blind findings.

Binding passed. HEAD was exactly the accepted commit and `git status --porcelain=v1 --untracked-files=no` was empty before analysis. The supplied hash list contains exactly 45 unique paths and exactly matches the inventory under the copies' root: three run directories with 15 files apiece. Every copied file's SHA-256 matches its listed hash. This establishes copy-to-list integrity; the author's statement that this is the production box's list is not independently verified.

The only seeds are 2273360, 260991262, and 1746737973, in registered order. Their run names are `22d31f2-20261008T061514Z`, `11e0cfd-20261008T134306Z`, and `11e0cfd-20261008T163019Z`. In each run, CLAIM names the directory and manifest HEAD, its byte hash matches `claim_sha256`, and manifest/results seed and HEAD agree. Each run has all six units in baseline/A/B then 2024/2025 order, no STOPPED record, and `units.json` exactly equals results' units.

All manifests have exactly the admitted five-field identity. The reviewed commit is `a3f5e3e89d8b2615e50d8c4aed5c08cbfe02a56f`, exposure commit `105561842e2bdf2580d3ecc867e1ee5e57df04f2`, archived review hash `ba182a95171c19664540271734dbe6c79ddfe5322d0c9a404879a655de130b75`, and admission-record hash `588472f1e8ce73cd9e761ee89ee70877d4d8a4e28325cbeaa6f89c35df43eb7c`. All ten declared input pins exactly match admission.json; their independently hashed canonical digest is `0be317b4da8ec27f6e77b25e4fc0d562cb44cd7f312087ccf9e0e4382665a6bb`. X-35 binds that digest, reviewed commit, review path and review hash, equals its exposure-commit version, and is absent from that commit's parent. These checks bind declarations; I did not read or rehash the original inputs.

The reviewed commit precedes the exposure commit, which precedes each recorded run HEAD. Subject-free git ancestry and closure diffs verified this. For each run HEAD and current HEAD, the executable closure differs from the reviewed commit only at admission.json; the pre-registration is unchanged. Seed 1's full HEAD is `22d31f23a0c2683a7dfe530ea3380861c090667f`; seeds 2–3 use `11e0cfd73acb01be77826dd7d05a587a8e636852`. The latter commit's direct-parent diff contains only the exposure register. The entire interval from seed 1 to seeds 2–3 also contains a cycle-index documentation change, with no executable change. The release row parses exactly as release after seed-1 run `22d31f2-20261008T061514Z`, 30 CPU-hours per seed, with Eric as its source's first token. There is exactly that one seed-1 directory in the copies.

Manifests agree on estimated_pa, retraining every 7 days, test seasons 2024/2025, feature settings 20/7, deterministic and force_row_wise LightGBM, and scoring at 10,000 trials/180 days. Their environment declares the matching seed, deterministic=1 and America/New_York. All self-checks report identical=true on 1,516,736 rows, with null max_abs_diff. These are consistent retained records, not independent re-execution of training or the real-input self-check.

Using my own pandas calculations, I selected rank=1 in each retained season profile, counted its hits and days, and took mean(actual_hit). All 18 season files have 10 rows per day, ranks 1..10 without duplicate date/rank pairs, complete scored columns, dates/seasons matching the unit, binary hits, and finite numeric probabilities in [0,1]. Days match between variants within each seed. Each 2024 file has 1,850 rows/185 days; each 2025 file has 1,840 rows/184 days. The P@1 values exactly equal results.json, scorecard P@1, and scorecard season precision at rank 1.

| Seed | Baseline 2024 hits/185 (P@1 %) | A 2024 hits (P@1 %) | B 2024 hits (P@1 %) | Baseline 2025 hits/184 (P@1 %) | A 2025 hits (P@1 %) | B 2025 hits (P@1 %) |
|---|---:|---:|---:|---:|---:|---:|
| 2273360 | 146 (78.918919) | 146 (78.918919) | 143 (77.297297) | 124 (67.391304) | 129 (70.108696) | 132 (71.739130) |
| 260991262 | 138 (74.594595) | 144 (77.837838) | 142 (76.756757) | 123 (66.847826) | 137 (74.456522) | 127 (69.021739) |
| 1746737973 | 144 (77.837838) | 142 (76.756757) | 141 (76.216216) | 125 (67.934783) | 132 (71.739130) | 133 (72.282609) |

I subtracted each seed's own baseline and independently evaluated the rule: strict improvement in both seasons, or both deltas >= -0.003 with mean_max_streak delta >= 0 and exact P(57) delta > 0. I also evaluated the symmetric |delta| <= 0.003 fallback. Both readings give identical pass/fail results. Every retained diff exactly equals a diff reconstructed from the stored scorecards. I additionally checked each numeric baseline/variant/delta triple directly against its source scorecard and subtraction; the stored summary, secondary values, boolean and reason match the retained diff and the code's rule.

| Variant | Seed | 2024 delta (pp) | 2025 delta (pp) | d (pp) | Pass | Deciding reason |
|---|---:|---:|---:|---:|---|---|
| A | 2273360 | 0.000000 | +2.717391 | +1.358696 | No | 2024 is tied; exact P(57) delta is exactly zero, so the code's neutral fallback fails. |
| A | 260991262 | +3.243243 | +7.608696 | +5.425969 | Yes | Strict improvement in both seasons. |
| A | 1746737973 | -1.081081 | +3.804348 | +1.361633 | No | 2024 drops beyond 0.3pp; both secondary deltas are negative. |
| B | 2273360 | -1.621622 | +4.347826 | +1.363102 | No | 2024 drops beyond 0.3pp; exact P(57) delta is zero. |
| B | 260991262 | +2.162162 | +2.173913 | +2.168038 | Yes | Strict improvement in both seasons. |
| B | 1746737973 | -1.621622 | +4.347826 | +1.363102 | No | 2024 drops beyond 0.3pp; both secondary deltas are negative. |

The following uses sample standard deviation (n-1), with each seed's d the equally weighted mean of its two season deltas. These independent quantities also agree with screen.disposition.

| Variant | Mean 2024 delta (pp) | Mean 2025 delta (pp) | m (pp) | sd(d) (pp) | t | Passes | Disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| A | +0.7207207207 | +4.7101449275 | +2.7154328241 | 2.3473940335 | 2.0036123245 | 1/3 | inconclusive |
| B | -0.3603603604 | +3.6231884058 | +1.6314140227 | 0.4647296526 | 6.0802919708 | 1/3 | inconclusive |

A satisfies the positive season means, practical m and t gates but fails the two-seed pass gate. B fails both the positive 2024 mean gate and the two-seed pass gate. Neither is negative because m > 0. B's high t measures small variation in the two-season averages, not consistency of positive improvement in each season: seeds 1 and 3 have identical B hit-count changes (-3 in 2024, +8 in 2025), hence identical d. These are three algorithm seeds on two consumed seasons, not three independent season samples or a significance test.

I recomputed all nine complete scorecards with the registered settings, from the copied profiles only, and recursively compared every field except timestamp. On this arm64 Mac the only unequal fields are p_57_exact and, in seven cards, p_57_mdp. There are 16 unequal scalar fields in total. All P@1 values, streak metrics, precision, miss analysis, calibration and proper scoring values match exactly. The differences are consistent with floating-point numerical variation; this check does not isolate architecture from other runtime numerical differences.

| Seed | Variant | Absolute p_57_exact difference | Absolute p_57_mdp difference |
|---|---|---:|---:|
| 2273360 | baseline | 1.6155871339e-27 | 4.0657581468e-20 |
| 2273360 | A | 1.6155871339e-27 | 1.3552527156e-20 |
| 2273360 | B | 1.6155871339e-27 | 1.6940658945e-21 |
| 260991262 | baseline | 4.9630836753e-24 | 6.7762635780e-21 |
| 260991262 | A | 1.0339757657e-25 | 2.7105054312e-20 |
| 260991262 | B | 1.0339757657e-25 | 1.3552527156e-20 |
| 1746737973 | baseline | 3.3087224502e-24 | 0 |
| 1746737973 | A | 1.0339757657e-25 | 0 |
| 1746737973 | B | 5.1698788285e-26 | 2.0328790734e-20 |

For seed 1 baseline, stored/local exact P(57) is 7.742570144719881e-12 / 7.74257014471988e-12; stored/local MDP P(57) is 7.042195951832376e-05 / 7.04219595183238e-05. The strict JSON equality validator will refuse this card. This is a verified portability limitation, not evidence of corrupted profiles. No tolerance or scorecard substitution was used to obtain a passing validation. Reconstructing diffs from the locally rescored cards leaves all six rule verdicts unchanged: exact P(57) deltas for seed 1 remain zero, and seeds 2–3 remain negative. P@1, d, m, sd, t and pass counts are unchanged, so neither disposition changes.

The supplied records cannot independently establish exact rescoring equality on the producing x86_64 host. They are consistent with such equality, but neither a copied hash list nor local numerical proximity proves the producing platform's execution or admission. The author-reported box aggregate, unit rc=0, guard CPU totals and shared ledger remain operational attestations pending comparison with the supplied Part 2 evidence; there is no box verification here.

For subsequent reconciliation, retained process CPU totals are 65,611.290507, 48,006.548197 and 47,565.328916 seconds, totaling 44.7731021167 CPU-hours. Unit sums are 65,518.949761, 47,914.381352 and 47,473.596069 seconds. The gaps from process totals include feature CPU (86.010138, 86.003547 and 85.408672 seconds) and other process overhead; unit sums must not be substituted for process totals. First baseline-2024 units are below the 7.5-hour stop. All units declare zero label changes and zero void rows. Manifests record resumed_portion_rows={} (the recording code returns this if the source column is absent as well as if there are no flagged rows, so it does not prove no resumed PA).

No artifact-binding defect was found. The substantive stage-one result is independently reconstructible and both variants are inconclusive. Formal aggregate acceptance on this Mac is limited by its unchanged strict numerical equality requirement.

Post-write procedural measurement: after the initial report was on disk (SHA-256 `3202d8a4c0dff83280c4197340ba7d185aec9b72f2b7c007e38f5aff39beea3a`), I invoked the unmodified `screen.aggregate` on the three copied directories with `_test_out_root` set to their copies root. It raised `RunInvalid`: seed 1, variant baseline's scorecard does not match recomputation from retained profiles. A separate unmodified admission_gate call passed at the accepted HEAD and returned exactly the identity above. Both gate calls internally read the archived r10 review after the blind findings had been saved; no prior-review content was displayed. PyArrow emitted sandbox sysctl permission warnings, but profile reads and rescoring completed. No escalation was requested. This confirms the predicted strict-equality refusal; it is not a passing Mac aggregate. The blind findings above have been preserved verbatim.

## Part 2 (against the note)

All six rows of the note's P@1 table (lines 44–49), all hit counts and hit-count deltas, all §5 quantities (55–65), all six per-seed reasons (72–76), and all secondary values (82–84) agree with Part 1 at their displayed precision. The supplied aggregate.box.json also exactly matches the independently reconstructed seed order, HEADs, identity, process CPU total, season means, d values, m, t, pass counts, booleans, reasons and dispositions. Its SHA-256 is `8d02d0bc078bf6f9edfd4e5eae0d9fdb7ac273605b2634edb8796b754e373e2b`, matching the note's prefix. The evidence-directory runs.sha256 is byte-identical to the acceptance copy's list. No numerical outcome correction is needed.

The wording-gap explanation is correct. With the symmetric fallback, seed 1 A fails on the 2025 gain exceeding 0.3pp; with the reviewed one-sided fallback it reaches the secondary test and fails because exact P(57) is unchanged. The other five cases already have the same deciding strict-improvement or 2024-drop outcome under either interpretation. All six pass/fail booleans and both dispositions remain unchanged. One small but consequential notation correction is required at note line 69: the nonnegative test applies to the mean_max_streak **delta**, not its positive absolute level.

The cross-platform diagnosis is substantially right, with a verified counting error. I measured nine unequal exact-P(57) fields and seven unequal MDP-P(57) fields, totaling **16**, not 17 (note line 103). Summing the counts in the author's own rescore_diff.out also gives 16, and its listed stored/local values exactly match my independently calculated differences. The maximum relative difference is 5.8513274134e-16, consistent with the displayed 5.85e-16. Local Python is 3.12.13 on arm64, numpy 2.4.3 and pandas 3.0.1. The box's architecture and Python version remain the author's account. All other fields match exactly. Exact-P(57) is a potential input to the fallback generally; only seed 1 A reaches that deciding check in these six cases, and its delta is identically zero on both sets of recorded numbers. MDP-P(57) enters no rule. The no-decision-change claims are verified against both stored and locally recomputed cards.

The producer-platform exact-equality claim has supporting supplied records, but is not independently established here. aggregate.box.stderr.txt records start 14:26:52 EDT, rc=0, end 14:27:11 and user/sys CPU 18.834/0.210 seconds. The unchanged aggregate code would perform exact reconciliation if run as described, and the recorded output has exactly the independently expected result. Those facts support accepting the reported producing-platform check as an operational attestation. They do not prove execution on that host or that the transferred files were the actual process inputs. The same limit applies to clean producing HEAD, launcher/nice settings, rc=0 guard records, reconciliation status, and lack of production interference. No box access or independently retained guard records were supplied to this review.

The cost account correctly distinguishes process CPU from guard/ledger CPU. I reconciled all 18 unit records with results.json; each unit's CPU-hours rounds to the values in index row (e), including seed 1 A 2024 at 7.0428699111 CPU-hours versus roughly two hours for 2024 peers. Seeds 2–3 span 1.9832870050–2.4160619342 hours per walk-forward, matching 1.98–2.42. Each first baseline walk-forward is below 7.5 hours, and process/author-reported guard totals fit the declared 45/30/30 budgets. Process totals sum to 161,183.167620 seconds, or 44.7731021167 hours, matching aggregate and note line 112. The displayed guard totals sum to 161,213.4 seconds whereas the note says 161,213.5: the 0.1-second gap is within rounding of three one-decimal per-seed values and cannot be resolved without the raw guard totals. Either gives 44.78 hours. I have not treated that rounding gap as an outcome defect.

The reported starting shared ledger 0.4813 plus reported framing guard CPU gives 45.2628 hours rounded. Adding the supplied aggregate stderr's 19.044 user+sys seconds gives 45.26809 hours, agreeing with the note/index's 19.04 seconds, 0.0053-hour addition and 45.2681 effective total. C2 cumulative rows 18.2640, 31.6022, 44.8176 and 44.8229 reconcile with the 0.0361-hour pre-framing C2 job and those additions at their printed precision. The resulting total is below 50, so the stated arithmetic does not require that checkpoint. Whether the ledger is complete, guard CPU covers all process descendants, or an acknowledgement existed is not independently proved by these copies. The earlier rc=127 attempt and the manager's accounting ruling are likewise stated history.

Projecting seven further seeds from the two reported uncontended guard totals gives 92.9376388889 hours; 100 minus the effective shared total leaves 54.7319111111 hours. Thus stage two does not fit, even before any other planned jobs. The note's separate 11.1-hour planned-C2 reservation has no supporting current budget breakdown in the supplied evidence. The index itself says 11.6 originally planned, and its linked proposal is absent from this checkout. I require removing the unsupported 11.1 figure while preserving the already-supported cap conclusion. No stage-two cost projection is a measured seven-seed cost.

The "Things to weigh" section correctly identifies uniformly positive 2025 deltas and mixed 2024 deltas. Its baseline-noise headline is too broad: baseline P@1 spans 74.594595–78.918919% in 2024, but only 66.847826–67.934783% in 2025. Reporting both avoids obscuring the common 2025 gains. B's t explanation needs a substantive correction. The identical integer hit-count changes in seeds 1 and 3 are verified, as are their identical d values. But small sd is precisely consistency of the registered two-season averages; it does not establish the matching counts are a "coincidence," and it does not establish consistency within each season. The exact sample sd is **0.4647296526pp**, rounding to **0.46pp**, not 0.47pp. The replacement explains both the consistent positive seed averages and the negative 2024 deltas, without claiming significance from three seeds.

The listed streak-metric caution is fair. Both variants have decreasing mean_max_streak and exact P(57) on seeds 2 and 3 despite positive two-season average P@1 changes. Mean streak deltas over all three seeds are -1.1759 (A) and -0.7297 (B); mean exact-P(57) deltas are -5.6409494329e-09 and -5.7141623224e-09. Seed 1's streak means increase and its exact P(57) ties, so this is a mixed per-seed pattern and a negative average, not universal worsening. The note's bullets identify seeds 2 and 3 and do not claim every streak-related quantity worsens. These iid/bootstrap strategy summaries are the registered secondary metrics, not verified deployed milestone probabilities or measurements on untouched seasons.

The contention explanation should remain a bounded inference. The unit records establish the elevated cost, but the alleged preview overlap comes from the author's index account, not supplied process logs. I checked the source's model creation and the installed LightGBM documentation/code: omitted n_jobs normally defaults to a physical-core count, rather than adapting to load. That supports the proposed mechanism but does not establish the effective producing-host thread count or unchanged model output under contention. The patch describes the missing effective-thread and uncontended-replay evidence without changing the experiment result.

The procedural §7 order is retained: results files, aggregate, note, independent acceptance pending, then owner delivery. No delivery or further run was performed by this reviewer. For the requested numerical reporting order (per-seed/mean deltas, dispositions, cost), the note currently leads its Result section with the disposition before its P@1 table; the patch moves that sentence after the §5 table. The reading chronology is internally consistent: CLAIM timestamps follow the stated launches, and the scorecard timestamps end at 05:21:45, 11:37:06 and 14:23:16 EDT, before the reported 14:27 opening. The supplied aggregate stderr also has the stated interval. Neither timestamps nor file mtimes prove who first read an outcome, that nobody else read it, or that it was not sent. The reading-record patch explicitly attributes those claims to the lead's account. No contrary read/distribution evidence was found.

One material omission from the note's Limits must be restored before a stage-two decision: pre-registration §8's event-availability restriction. Official-date shift(1) can admit later resumed-game events before they happened, and all variants share that convention. The note also needs to state the limit of the empty resumed-row record. All three manifests contain resumed_portion_rows={}, which screen.resumed_counts returns both when no rows are flagged and when the source column is absent. filter_out_resumed_portion returns an unflagged input unchanged. Zero relabel changes and zero void rows therefore do not independently prove original-portion label correctness. I did not read the pinned inputs and cannot decide which case occurred. This is an unresolved input-evidence limit to disclose, not a verified wrong-label finding or a reason to invent a different disposition.

The note already correctly limits catcher identity to a postgame game-level proxy, requires independently specified pregame catcher identity for a 2027 test, records missing pre-2019 catcher history, consumed seasons, outcome-stratified seed selection and separate unadjusted A/B screens. With the exact edits below, the note accurately reports the supplied stage-one artifacts and the limits of what they establish. No repair to frozen executable code, admission policy, or scoring rules is approved by this review.

## Corrections

Apply the following exact unified diff to the results note. It has been verified with `git apply --check`; no tracked change was made.

```diff
--- a/docs/sota_audit/2026-10-08-result-c2-framing-stage-one.md
+++ b/docs/sota_audit/2026-10-08-result-c2-framing-stage-one.md
@@ -10,7 +10,7 @@
 **Status:** all three registered seeds ran; `aggregate` passed. **Independent acceptance: pending.** Per pre-registration §7, nothing in this note goes to Eric or job-search-52 before the acceptance. Nothing here approves a production change, a deploy or stage two.
 
 ## Reading record
-Who read what, and when (all 2026-10-08, EDT):
+Who read what, and when (all 2026-10-08, EDT), according to the lead's account. The supplied artifacts support the production chronology, but do not independently audit reads or distribution:
 - **Until 14:27:11 EDT, only costs and label counts were read:**
   - the `cpu_s`, `wall_s` and `labels` fields of units.json;
   - the progress lines of the log tails;
@@ -31,11 +31,10 @@
 - **Exit and records.** Every unit exited rc 0, and its guard records `exit` and RECONCILED with no problems. Every walk-forward reports labels changed 0 and void 0.
 - **Code.** `11e0cfd` adds only Eric's release row to the register: it is a metadata-only descendant of `22d31f2`. `validate_run`'s descent-and-closure rule admitted both commits.
 - **Seed 1's slow walk-forward.** Seed 1's A 2024 walk-forward cost 7.04 CPU-h against about 2 for its peers, because it collided with production's 03:00 `bts preview` (C2 index, row (e)).
-  - The collision is observed. That it changed nothing but cost is inferred: LightGBM ran with `deterministic=True` and `force_row_wise=True`, with no thread count among the manifest's recorded parameters, so the thread count does not depend on load. This is not tested.
+  - The collision is reported in the C2 index. That it changed nothing but cost is inferred, not tested: the manifests record `deterministic=True` and `force_row_wise=True`, but omit the effective thread count and provide no uncontended replay of this run.
 - **Output root:** `data/hetzner_results/c2/framing_screen/seed_<seed>/<run>/` on the box (restic-backed archive set).
 
 ## Result
-**Both variants are inconclusive.** Neither is positive, so neither reaches "worth testing on untouched 2027 data".
 
 **P@1 by seed** (top-ranked pick hit rate; 185 test days in 2024, 184 in 2025; hits in brackets):
 
@@ -60,13 +59,15 @@
 | Per-seed rule passed | **1 of 3** (seed 2) | **1 of 3** (seed 2) |
 | **Disposition** | **inconclusive** | **inconclusive** |
 
+**Both variants are inconclusive.** Neither is positive, so neither reaches "worth testing on untouched 2027 data".
+
 **Why:**
 - **A** meets three of the four positive conditions: both season means are above 0, m ≥ +0.3pp, and t ≥ 1.5. It fails the fourth: the per-seed rule holds on only 1 seed, not the 2 required. It is not negative because m > 0.
 - **B** fails two positive conditions: its 2024 mean is below 0, and the per-seed rule holds on only 1 seed. It is not negative because m > 0.
 
 **The per-seed rule** (`bts.experiment.runner.evaluate_pass_fail`) passes a seed when either:
 1. P@1 improves in both seasons; or
-2. the neutral fallback holds: no season's P@1 drops more than 0.3pp (delta ≥ −0.003), `mean_max_streak` ≥ 0, and exact P(57) strictly improves.
+2. the neutral fallback holds: no season's P@1 drops more than 0.3pp (delta ≥ −0.003), the `mean_max_streak` delta is ≥ 0, and exact P(57) strictly improves.
 
 Seed by seed:
 - **Seed 2** passes for both variants on condition 1.
@@ -85,11 +86,11 @@
 
 ## Things to weigh in the result
 - **The seasons disagree.** Every seed improves 2025 under both variants, by +2.2 to +7.6pp. 2024 is mixed: A gives 0.00, +3.24 and −1.08pp; B gives −1.62, +2.16 and −1.62pp.
-- **The seed noise is as large as the deltas.** The baseline's own P@1 spans 74.6–78.9% in 2024 across the three seeds.
-- **B's t of 6.08 comes from a coincidence of whole hit counts, not from consistency.**
+- **Baseline P@1 also varies by seed.** It spans 74.6–78.9% in 2024 and 66.8–67.9% in 2025 across the three seeds. The spread matters particularly to the mixed 2024 result; every seed still improves 2025 under both variants.
+- **B's t of 6.08 reflects a small spread in the two-season seed averages.** Those averages are consistently positive across seeds, while the 2024 deltas are mixed.
   - Seeds 1 and 3 give B the same deltas: −3 of 185 in 2024 and +8 of 184 in 2025.
   - The underlying counts differ: B 143 and 132 hits against a baseline of 146 and 124, versus B 141 and 133 against 144 and 125.
-  - Two of the three seed-level values are therefore identical, so the standard deviation is small (0.47pp) and t is large.
+  - Two of the three seed-level values are identical. Their sample standard deviation is 0.46472965pp (0.46pp rounded), giving t = 6.08. The matching counts are observed; calling them a coincidence would be an inference.
   - With 3 seeds, t ≥ 1.5 is a screening convention, not a significance test (pre-registration §5).
 - **The streak metrics point the other way.**
   - On seeds 2 and 3, exact P(57) falls for both variants.
@@ -100,7 +101,7 @@
 - **On the box, it passes.** `aggregate` ran on the box from `~/projects/bts-c1` at `11e0cfd` (clean) at 14:26:52–14:27:11 EDT, with rc 0. It ran under the passing admission gate and validated all three runs: claim, manifest, pins, identity, code descent and closure, season evidence, and every scorecard, diff and summary recomputed from the retained profiles with exact equality.
   - Its output is `aggregate.box.json` (sha256 `8d02d0bc…`), in `docs/audit/2026-10-08-c2-framing-stage-one-evidence/`.
 - **On the Mac, exact equality fails.** Re-running it there on the hash-matched copies, with the copies as the namespace root, refuses seed 1's baseline scorecard: the stored value does not equal the Mac's recomputation.
-  - **The cause, measured** (`rescore_diff.py`, output `rescore_diff.out`): over the nine scorecards, 17 values differ, and only two fields: `p_57_exact` and `p_57_mdp`. Every difference is in the last digits, with relative size at most 5.85e-16. Every other field is equal, including every P@1, streak metric, calibration and precision value.
+  - **The cause, measured** (`rescore_diff.py`, output `rescore_diff.out`): over the nine scorecards, 16 values differ, and only two fields: `p_57_exact` and `p_57_mdp`. Every difference is in the last digits, with relative size at most 5.85e-16. Every other field is equal, including every P@1, streak metric, calibration and precision value.
   - **The platforms:** the box is x86_64 with Python 3.12.6; the Mac is arm64 with Python 3.12.13. Both have numpy 2.4.3 and pandas 3.0.1.
   - **Mechanism:** platform floating-point differences, inferred and not tested.
 - **No decision changes.**
@@ -121,7 +122,7 @@
 1. **Independent acceptance** of this note and the run artifacts, by a fresh Codex session that has not reviewed this item.
 2. **Then the result goes to Eric through job-search-52** (§7): per-seed and mean P@1 deltas per season for A and B, the dispositions and the measured cost. The fallback is `~/projects/job-search/mets-2026-10-06/catcher-experiment-result.md`, with the herdr manager told.
 3. **Stage two is Eric's call.** Under §5, only an inconclusive variant could justify stage two (to 10 seeds), and both variants are inconclusive. Stage two needs his second go-ahead; nothing here asks for it or approves it.
-   - **The cost arithmetic** (projection from the two uncontended seeds, n = 2): 7 more seeds at about 13.2–13.3 CPU-h each come to about 93 CPU-h. Only about 54.7 remain under the 100 cap, and C2's planned jobs need about 11.1 CPU-h of it.
+   - **The cost arithmetic** (projection from the two uncontended seeds, n = 2): 7 more seeds at about 13.2–13.3 CPU-h each come to about 93 CPU-h. Only about 54.7 remain under the 100 cap, before reserving any CPU for planned C2 jobs.
    - Stage two therefore does not fit under the current cap, and it would also need the 50 acknowledgement. Raising the cap is a separate decision, never made here (§6).
 4. **The Mets application stays held** until Eric has this result.
 
@@ -129,5 +130,7 @@
 - **Seasons:** 2024 and 2025 are consumed seasons. Three seeds measure the algorithm's seed sensitivity, not new season samples.
 - **Catcher identity** is a postgame, game-level proxy: one catcher per side per game, which need not be the starter. A 2027 test needs a pregame catcher identity, specified independently.
 - **History:** there is no catcher history before 2019; the pitcher feature has it.
+- **Event availability:** feature history keeps resumed-portion PAs at the original official game date. Date-level shift(1) can therefore admit events before they happened. Baseline and both variants share this convention; the screen does not establish unconditional pregame availability (pre-registration §8).
+- **Labels:** the run rebuilds labels through `filter_out_resumed_portion` and drops void profile rows. All units report zero label changes and zero void rows, and all manifests have `resumed_portion_rows={}`. That empty record can mean no flagged rows or an absent flag column; the filter returns the input unchanged if the column is absent. These copies do not independently certify original-portion label correctness.
 - **Multiplicity:** A and B are reported separately, with no multiplicity adjustment.
 - **The seeds:** they are positions 0–2 of `canonical-n10.json`, which is neither an outcome-independent random sample nor full-range coverage.
```

## What was run

I ran all commands from the accepted checkout root. Before Part 2, I read the task, pre-registration, admission record, X-35 and release row, screen/shared admission code, scorecard and per-seed-rule code, and the scorer's relevant simulation code. Subject-free git rev-parse/status, merge-base ancestry, rev-list and diff --name-only checks confirmed HEAD, initial tracked cleanliness, code descent and closure. Hash-only access to the archived review bound its identity without opening its text before Part 1. I did not search memory, history or other review reports.

My scratch blind.py verified all 45 hashes, exact inventory and bindings; independently recomputed rank-1 hit counts/means and deltas with pandas; checked retained diff arithmetic, summaries and both per-seed-rule readings; computed d, sample sd, m, t, passes and dispositions; and rescored all nine cards at 10,000 trials/180 days. All assertions passed. The first guarded attempt stopped on the shared helper's ordinary /dev/null opening; allowing that nonpersistent device let the retry complete. The full rescoring completed before any Part 2 read. The initial Part 1 report was written and hashed before opening the note, evidence or index.

After that write, aggregate_check.py invoked unmodified aggregate with the copies root as _test_out_root. It captured the expected seed-1 baseline RunInvalid; a separate admission_gate passed at the accepted HEAD. These calls internally read the archived r10 review only after Part 1 was saved. The helper process exited zero because it deliberately caught the refusal; this is not aggregate success. I read all five evidence-directory files and compared aggregate.box.json's values with the independent calculations (exact agreement), its file hash with the note (agreement), and the two runs.sha256 lists with cmp (identical). I counted the author's diagnostic differences (16), checked numeric runtime versions, unit/cost/ledger arithmetic and artifact chronology, and inspected the installed LightGBM n_jobs documentation/code and original-portion filter fallback. Source lookup probes for simulate/profiles.py, classifier.py, blend_walk_forward.py, model/blend.py, simulate/backtest.py and the linked C2 proposal returned file-not-found; actual load_profiles/model code was located under monte_carlo.py, predict.py and backtest_blend.py. The missing proposal left the 11.1-hour reservation unverified.

Every Python command used `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run --offline python -B`, with TMPDIR and MPLCONFIGDIR routed to task-owned `/private/tmp/c2-framing-a1-review`. No network resolution, box/SSH/gh, credential/configuration read, data/ read, walk-forward, model training, tracked edit, commit, push or escalation occurred. Writes were limited to this report and that scratch directory; no copied run was modified. The report's unified diff was generated against the accepted note and `git apply --check` passed, without applying it. Final verification passed: the five required sections and sole Accepted-commit line are exact; the original blind findings are preserved verbatim; the embedded diff passes git apply --check; all 45 copied-file hashes still match; all three full LightGBM parameter dictionaries agree; HEAD is unchanged and the tracked tree is clean. The task-owned scratch directory was deleted and its absence verified.

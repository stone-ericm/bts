# C2 side item (e): catcher-grouped framing screen, stage two — pre-registration addendum

**Status:** draft for review. It is frozen at the reviewed commit once signed. It amends nothing in the stage-one pre-registration `docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md` (frozen at `a3f5e3e`). It adds what stage two needs: the seeds, the 10-seed rule, budgets, the launch window, and how the two stages' runs are bound.

**Authority:** Eric, 2026-10-08, relayed by job-search-52.
- Register row **C2-framing-stage-two**: "Expand to 10 runs"; the Mets application is no longer held.
- The cap and the per-seed budget: the row **C2-framing-stage-two-cap** (below), still to be ruled.
- The 50 CPU-hour checkpoint: his written acknowledgement, still to be given.
- Register row **C2-framing-side-item**, unchanged: no production change and no deploy; stage two runs to 10 seeds and needs his second go-ahead, which is the row above.
- Pre-registration §5: "Only an inconclusive variant could justify stage two". Stage one gave **both** variants inconclusive (results note `docs/sota_audit/2026-10-08-result-c2-framing-stage-one.md`, accepted in `docs/audit/2026-10-08-c2-framing-stage-one-acceptance-codex-a1.md`).

**Disclosure: written after stage one was opened.**
- The lead opened stage one's outcomes at 14:27 EDT on 2026-10-08, and Eric received them before this addendum was written.
- So that no choice here can depend on what stage one showed, every rule below is the mechanical extension of the stage-one pre-registration:
  - the same run, inputs, labels, settings, scoring and validator;
  - the same quantities and thresholds;
  - the seeds that follow in the same file, in its order;
  - a majority of the seeds, exactly as §5's "at least 2 of 3" is a majority of 3.
- The rule would read the same had either variant come out positive, negative or inconclusive in stage one.
- No analysis is added, removed or reweighted.

**What it can support:** unchanged. At most "worth testing on untouched 2027 data", never "ship it". The ten seeds measure the algorithm's seed sensitivity on two consumed seasons. They are not ten season samples.

## 1. What does not change
- **The run.** `run`'s computation is byte-for-byte the stage-one computation: §4 steps 1–8 of the stage-one pre-registration. The stage-two code changes only the seed admission, the launch wrapper, the aggregate and the disposition's seed count.
- **The inputs.** The same ten pinned inputs (inputs digest `0be317b4…`, exposure row X-35). A stage-two run must carry exactly the pins of the stage-one runs; the validator refuses anything else.
- **The rest.** Labels, feature settings, LightGBM parameters (deterministic), the scoring settings (10,000 trials, 180 days), `validate_run`'s checks, and the first-walk-forward stop (7.5 CPU-hours).
- **The order of work.** One box job at a time, through the C1 launcher. Outcomes are opened only after all seeds have run.

## 2. Seeds and order
- **Seeds 4–10** are positions 3–9 of `data/seed_sets/canonical-n10.json`: 2048, 3629294338, 1277948386, 3219332220, 2207587974, 3170105529, 2675988121.
  - They are the rest of the file stage one took its first three from, in the file's order.
  - The file's limits stand (stage-one pre-registration §4): its positions come from an outcome-ranked, stratified historical baseline distribution. They are not an outcome-independent random sample.
  - **Five of the seven exceed 2³¹−1.** LightGBM 4.6.0 (the locked version) reads such a seed with 32-bit wraparound. Measured on the Mac: 3629294338 trains the same model as −665672958, with no warning, and repeats exactly. The seven seeds stay distinct after wraparound. The April multi-seed screening used this file.
- **One at a time, in that order.** Seed k (k ≥ 4) is admitted only when:
  - each stage-one seed has exactly its accepted run;
  - each earlier stage-two seed has exactly one complete run that validates.
- **Stage one's runs.** They are the three accepted runs: `22d31f2-20261008T061514Z` (seed 2273360), `11e0cfd-20261008T134306Z` (260991262) and `11e0cfd-20261008T163019Z` (1746737973).
  - Their files must equal, byte for byte, the accepted hash list `docs/audit/2026-10-08-c2-framing-stage-one-evidence/runs.sha256` (sha256 `05343931…`).
  - They must validate under stage one's own identity: review r10's report, its reviewed commit `a3f5e3e`, exposure commit `1055618` and admission record sha256 `588472f1…`.
- **The justification is checked, not assumed.** Stage two is admitted only when stage one's dispositions, recomputed from those runs, include an inconclusive variant (§5's condition).

## 3. Budgets, the cap and the launch window
- **Per-seed declared budget:** the number in Eric's row C2-framing-stage-two-cap (proposed 20 CPU-hours).
  - The measured costs it is set against: uncontended seeds 13.22 and 13.34 CPU-hours; seed 1, which collided with production's 03:00 preview, 18.23.
- **The shared cap:** the number in the same row (proposed 165 CPU-hours). The launcher's cap constant must equal it: the stage-two admission refuses a row whose cap differs from the launcher's.
- **The 50 checkpoint:** Eric's written acknowledgement, recorded in the register. The launcher reads only the file `CHECKPOINT_50_ACK.json`, which the lead writes after his row, with his permission.
- **The launch window.** No stage-two seed launches from 00:45 to 03:10 America/New_York.
  - Production's nightly chain trains its blend at about 03:04. A seed overlapping it cost seed 1 about 5 CPU-hours, and slowed production's preview from about 1 minute to about 70.
  - Uncontended seeds measured 1 h 53 min and 1 h 54 min of wall time, so a seed launched before 00:45 is expected to finish before 03:00. The launch wrapper refuses inside the window.
- **Stops:** a stopped or killed seed, or any 403/429, pauses stage two. It is reported to Eric, with no rerun without his decision. A stopped seed is not a complete seed, and stage two is then incomplete.

## 4. Identities
- **Each stage keeps its own admitted identity.** Stage two's code is a new reviewed commit with its own exposure row (X-36) and admission record. Stage one's runs carry stage one's identity.
  - Stage-one runs are validated against stage one's identity (§2).
  - Stage-two runs are validated against the stage-two admission.
  - `validate_run` and `head_admitted` are unchanged. Each run's recorded HEAD must descend from its own stage's exposure commit, with its own stage's reviewed executable closure.
- **Both stages must agree** on the input pins, inputs digest, LightGBM parameters, feature settings, basis, retrain interval and test seasons. The aggregate refuses otherwise.

## 5. Dispositions over the ten seeds
The stage-one quantities and thresholds, over exactly the ten registered seeds:
- For each seed, d = the mean of its 2024 and 2025 P@1 deltas; m = the mean of the ten d; t = m / (sd / √10), with sd the sample standard deviation (the same `_t`).
- **positive:** all of:
  - the mean 2024 delta and the mean 2025 delta are both above 0;
  - m ≥ +0.003;
  - t ≥ 1.5;
  - the per-seed screening rule (`evaluate_pass_fail`, unchanged) holds on a majority of the seeds: **at least 6 of 10**.
- **negative:** m ≤ 0, and the per-seed rule holds on fewer than 6 seeds.
- **inconclusive:** anything else.
- **incomplete:** any input other than exactly the ten registered seeds' runs.

**What these are not:** with ten seeds, t ≥ 1.5 remains a screening convention, not a significance test. A and B are reported separately, with no multiplicity adjustment. No disposition approves a production change.

## 6. Reporting
- **While stage two runs:** a heads-up to job-search-52 and the manager at each launch (time and ledger), and an immediate report of any stop or of a cost well off the projection. Only costs and label counts are read.
- **After seed 10:**
  1. the per-seed `results.json` files, then the ten-seed `aggregate`;
  2. a results note;
  3. an independent acceptance;
  4. then the result to Eric through job-search-52: per-seed and mean P@1 deltas per season for A and B, the dispositions and the measured CPU cost.
- **Disclosure in that note:** stage one's three seeds were opened before stage two ran. The note discloses that, and reports the ten-seed result as the result.

## 7. Limits
The stage-one pre-registration's §8 limits apply unchanged, and so do the two the stage-one acceptance restored:
- event availability (resumed-portion PAs at the original official date);
- the empty resumed-row record, which cannot tell "no flagged rows" from "no flag column".

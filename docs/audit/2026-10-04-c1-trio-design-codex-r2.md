# C1 trio — final design review, round 2 of 2

Reviewed B/C rev 2 as published in `89bb791`. Their current bytes matched that commit during this review: B SHA256 `27055e6304c949aa0257e0cbb195e34a07db1389cd2582dc355afe4f064b9230`; C SHA256 `b7912c7fbe7f74de4acac7baab74a8393d1f6385d37f0db02950744e265c65c5`. Other main/index updates occurred during the review; the B/C documents were unchanged. References are checkout-relative paths and rev-2 line numbers.

The archived round-1 report is byte-identical to the requested scratch report, SHA256 `4e47a380ef8875d0051799734c56405d8b3b4db216a9ab757099d084f09d17f9`. A read-only Python check verified every quoted paragraph of B-E1–B-E5, C-E1–C-E3 and X-E1 in its intended document, allowing the stated bullet/whitespace formatting. No `data/` reads, SSH, network, `gh`, production runs or tracked-file edits. Synthetic selector/state examples below use invented inputs, not measured behavior of unbuilt code. Frozen A was read only for B's imported definitions, not re-reviewed.

## Verdicts

| Design | Final verdict | Required action |
|---|---|---|
| B: PA count | **SIGN WITH EDITS** | Apply B-R2-E1–E3 verbatim, then freeze. The authoritative-metadata precondition remains mandatory; unavailable metadata means the build is deferred, not that first-seen reconstruction becomes admissible. |
| C: rank-1 prerequisites | **SIGN WITH EDITS** | Apply C-R2-E1–E2 verbatim, then freeze. Broader-sheet binding remains **not established** absent the separately reviewed independent contract already required by Part B. |

These are final design dispositions under the two-round rule. No further design round is requested. The residual issues have concrete, limited text fixes; neither requires a new candidate or another scientific search. Signatures authorize neither production implementation acceptance, deployment, acquisition, activation nor a change to picks. B/C production code still goes through its stated review until SIGN and Eric's D7 approval. The relevant index exposure cells correctly distinguish X-34's historical/prospective reads from X-33's outcome-free diagnostic and receipt-only capture.

## Findings (B)

**Round-1 dispositions.**

| R1 finding | Rev-2 disposition |
|---|---|
| B1: unavailable starting/order evidence | Resolved in §4, lines 38–40: authoritative metadata and provenance are required; unresolved games are quarantined; missing metadata defers the build; acquisition remains separately gated. This review does not establish that the required metadata exists. |
| B2: target/formula/input gaps | Resolved in §§2–3, lines 26–35: capped-count definition, conditional historical target, starter/reliever conventions, strictly earlier-date BF, sparse-history fallback and baseline fallback are stated. Line 80 correctly describes the conditional-hazard approximation and limits Jensen. The leftover `shift(1)` wording needs B-R2-E2. |
| B3: post-commit information and invented slates | Actual-commit/cutoff boundaries, one common run, identity/provenance and rank-before-outcome rules are present at lines 49–53. One completion-selection ambiguity remains; close it with B-R2-E1. |
| B4: file-byte-only parity | Resolved at lines 55–57: copied inputs, nonblocking failure/slow-work behavior, fixed-clock decision/transport/state comparisons and a mutation leaving pick/slate bytes unchanged are required. The surviving “negligible” claim at line 88 needs B-R2-E2. These are implementation gates, not evidence that the code already passes them. |
| B5: boundary remapping and June-null overclaim | Resolved at lines 78–83: positive means forecast scores only; the residual-lever condition remains open; no global remapping is presumed; independent acceptance, downstream replay/stress and D7 are separate. |

**B-R2-1 — Medium: selecting a “complete run” can still select an older forecast because the latest candidate computation failed.** Section 5, line 49 lists baseline and candidate p in a complete snapshot; line 51 selects the latest complete archive. Lines 53/55 instead require a missing candidate to use baseline p and forbid selecting an older run. Both interpretations are currently possible.

Synthetic example: r1 and r2 have complete baseline forecasts at 12:00 and 14:00, both before a 15:00 commit and 17:00 cutoff. r1 has candidate output; r2 does not. Requiring candidate completion to make a run complete selects r1. Selecting the baseline run independently selects r2 and applies the registered baseline fallback. Candidate success must not control the selected information set. This ambiguity survived my round-1 replacement; B-R2-E1 closes it without changing the candidate or thresholds. A partial baseline archive also cannot silently justify choosing an earlier run when a later eligible production run is evidenced.

**B-R2-2 — Low: leftover outputs/leakage/compute wording weakens the replacement contract.** Lines 43/45 still say BF is obtained with date-level `shift(1)`, although line 28 explicitly requires strictly earlier official dates. A genuine date-level implementation can comply, so this is not a blocker, but remove the shorthand rather than leave a row-shift interpretation available. Also distinguish the capped count artifact from uncapped BF workload counts, and remove the unmeasured “negligible” statement at line 88. B-R2-E2 makes these existing requirements consistent; it changes no lag, support floor or scientific specification.

**B-R2-3 — Low: the surviving primary description and imported rules need an explicit boundary.** Line 62 calls the population “all served candidates with known outcomes,” while §5 restricts it to the eligible common-run pool. Line 71 imports 4a's rules, whose separate population starts from the last retained daily slate. B must import calendar/completeness/validity/ranking rules, not A's last-write population, 30/60 split or 50-date support floor. The detailed B requirements can already be read correctly, but B-R2-E3 removes the alternative reading and reconciles the earlier “after date 90 is graded” bullet with the fixed 08:00 freeze.

The remaining arms, conditional target, no_pa/unknown exclusion, common baseline-rank-1 guardrail, direct date-vector bootstrap, 75-date floors and disposition table are consistent. The shared-date comparison remains separate from 4a/4b; X-E1 prevents baseline recipe changes from silently turning this into a combined comparison. A positive score result still cannot close the June-null condition or permit production use by itself.

## Findings (C)

**Round-1 dispositions.**

| R1 finding | Rev-2 disposition |
|---|---|
| C1: retained-byte/marker binding | Resolved at lines 30–32: wire/decoded/stored digests, immutable retained paths, verified unchanged objects, time-local marker results and unavailable/error states are explicit. |
| C2: native HTTP errors, serialization and cycle stop | Native urllib classification, one-writer lock, abort/exit behavior, skipped-feed handling and synthetic transport tests are present at lines 33–40. W-capture-stop is also explicitly carried in the current watchdog design. One restart failure path remains; close it with C-R2-E1. Launcher/watchdog enforcement is required future integration, not currently verified runtime behavior. |
| C3: concordance is not all-player identity | Resolved at lines 48–54/65: no Bindable gate, no missing-value inequality, no searched rollover promotion, coverage/staleness disclosure, independent identity/target requirements and conservative inferred-game checks. |
| C4: X-33/new-field exposure | Resolved at lines 54/70 and the index: X-33 precedes the new outcome-free read; receipt-only capture has no outcome exposure. |

**C-R2-1 — Medium: a limit or unresolved attempt can survive without the separate marker, yet the next invocation only explicitly checks that marker.** Line 33 aborts on receipt/stop publication failure and leaves an incomplete attempt pending reconciliation, but its next-invocation rule is only “while stopped.” A durable completion reporting HTTP 429 followed by failure to publish STOP_403_429.json has no marker. Marker-only preflight could then resume traffic without Eric. An intent-only crash has a similar ambiguity: unavailable evidence is not an instruction preventing the next request.

Synthetic state: `stop_marker_exists=false`, persisted receipt `rate_limited/http_status=429`, no recorded reset. Marker-only refusal is false; the persisted limit still requires refusal. C-R2-E1 makes unresolved receipt evidence part of preflight, preserves Eric's control over limits/ambiguous attempts, and adds next-invocation fixtures. This does not require a live rate-limit probe or a new library. The same known-limit predicate must feed the required watchdog/cycle gate so deleting or failing to create the convenience marker cannot silently resume C1.

**C-R2-2 — Low: the summary still promises a binding decision the replacement refuses to infer.** Line 17 says the concordance read “settles whether” the all-player number belongs to a game; line 59 says a later blend inherits “this binding decision.” The detailed Part B instead conservatively reports not established and requires a separately specified/reviewed positive identity contract. Correct the summary/inheritance wording rather than weakening that gate. Also reconcile A4's “only to within one capture interval” with the new per-feed receipt times: cadence limits observations between polls, while qualified receipts timestamp actual retrieval. It is not a guaranteed bound through failed/crashed fetches. C-R2-E2 fixes those statements.

The remaining Part B “What is known” paragraphs and Execution paragraph are consistent with the diagnostic: round-tagged most-selected rows are the comparator, not a provider identity contract; the same single 0.5-CPU-hour diagnostic supports plain/gzipped inputs. The 99%/95% figures cannot certify binding, and no undefined rollover function can pass an identity gate. No blend, outcome-based identity selection or new acquisition is admitted. The stdlib capture can implement the declared receipt/locking/stop contract; code acceptance must verify it before the first admitted capture.

## Verbatim edits

Apply only these limited replacements/additions, then freeze B/C. Do not alter frozen A or the already fixed candidate, split, thresholds, support floors or production approval gates.

### B-R2-E1 — Clarify §5's first paragraph and replace its second paragraph (lines 49/51)

Append to the first paragraph:

> For run selection, completeness means completeness of the baseline snapshot, independent of candidate-computation success. Candidate p may initially be null with an unavailable reason; separately appended candidate completion/failure records retain the same run identity and their actual availability times. Candidate completion is never a condition for choosing a baseline run.

Replace the second paragraph with:

> Primary and ranking comparisons use one common baseline run per date. Identify eligible production runs from the bound run/decision inventory without examining candidate results; select the latest whose forecast inputs were available strictly before both the first terminal production commit, if one occurred, and the common earliest run-known submission cutoff of the registered date's game pool. On an evidenced no-commit/skip date use that common earliest cutoff. The selected run must have a complete baseline archive available before that boundary; otherwise exclude/count the date rather than choose an earlier run. Missing run-inventory coverage, commit-boundary evidence or required eligibility evidence also excludes/counts the date. Use candidate output only if its linked completion was available before that same boundary; otherwise use baseline p as the candidate fallback and count the reason. No candidate failure or late completion selects an older run. Do not stitch different runs per candidate or substitute post-run realized times/starters. Add a fixture where the latest eligible baseline run has missing/late candidate output: that run remains selected with baseline fallback. Add a fixture where its baseline archive is incomplete: the date is unavailable, not replaced by an older run.

### B-R2-E2 — Replace §4's first two output bullets, its leakage bullet, and §8's shadow bullet

> - the count table (slot × home/away → distribution over min(N,8), conditioned on N≥1), with the declared historical overflow count;
> - each certified starter's complete per-start BF workload counts, retaining legitimate resumed PA, plus their source/availability provenance; lagged BF uses only starts available on strictly earlier official dates as specified in §2;
> - **Leakage:** the count table is a fixed historical aggregate applied to 2027 only. BF history is filtered by strict official_date < forecast_date and actual source availability before the forecast, then the latest five available starts are selected under §2's fallback rules. A row shift is not a substitute for that predicate. `scripts/leakage_audit.py` is unaffected because no PA-model feature changes; this is stated, not run. The shadow's same-date-history fixture remains required.
> - **The shadow:** the formula sums at most 8 count terms per candidate per model. Total BF/provenance/archive and prediction overhead is unmeasured at design stage; implementation acceptance must verify §5's production-safety requirements rather than presume negligible overhead.

Keep the existing third output bullet (artifact SHA256 and input-hash manifest).

### B-R2-E3 — Replace §6's Test/Primary bullets and its first sentence after the disposition table

> - **Test:** fixed 2027 contest-calendar dates 1–90; one freeze and analysis at 08:00 ET the following day after date 90, with the stated unknown exclusions/support floors and no extension.
> - **Primary:** C minus B log loss, averaged within each date's known eligible rows from §5's selected common baseline run and then equally across scoreable dates. Candidate baseline-fallback rows remain in that same paired pool. Use the registered direct date-difference bootstrap, seed 20270102 and 95% percentile interval; candidate results never choose the run or support.
> Import only 4a's fixed calendar numbering, settled-source completeness, probability-validity/clipping and rank-before-outcome/tie/no-replacement rules; B's run population, 1–90 test window, seeds and 75-date support floors remain those specified here, not 4a's daily last-write population or fit/test windows.

Keep the remainder of that paragraph, beginning “Test dates are 1–90, with no extension,” and the existing primary thresholds, support floors, guardrail and dispositions.

### C-R2-E1 — Append to A3 and to the Gate paragraph

Append to A3:

> The canonical marker is _receipts/STOP_403_429.json under the capture output root. Under the same writer lock, preflight also validates instrumented intent/completion reconciliation and recorded resets before any network request. A prior unreset 403/429 receipt or stop record refuses with exit 3 even if the marker is absent; rebuild the marker when possible without fetching. A missing completion, malformed/unreadable receipt or stop state, or failed required state publication refuses further requests with a nonzero exit pending recorded reconciliation. Missing/unreadable state is never treated as evidence of no stop. If independent retained evidence cannot rule out a limit for an unresolved attempt, capture resumption requires Eric's fresh recorded decision. Deleting a marker is not itself a reset; the recorded reset identifies the stop/attempts it resolves. Known unresolved limit evidence is also a stop input to the required watchdog and C1 cycle gate, independent of marker presence; capture reset and cycle resumption retain their separate owner decisions.

Append to Gate:

> Add next-invocation tests for a durable 403/429 receipt followed by marker-publication failure or marker removal, an intent-only crash, and malformed/unreadable reconciliation state. In each unresolved case, assert no transport request; after a properly bound recorded reconciliation/reset, assert the permitted behavior with transport patched. Verify that known-limit evidence without the marker still reaches the required watchdog/cycle stop integration. No live probe is authorized.

### C-R2-E2 — Replace the plain-words binding item, A4, the final Out-of-scope sentence, and the coarse-timing limit

> - **A concordance diagnostic.** One outcome-free read of the 2026 captures describes agreement with round-tagged most-selected rows. It does not establish broader-sheet game identity; without independent binding evidence, that prerequisite is reported not established.
> - **A4, cadence:** unchanged, every 30 minutes, now registered. Qualified receipts timestamp actual per-feed retrieval. Between attempts availability is unobserved; failed/crashed attempts can make the gap longer than one interval. Retrieval witnesses do not establish provider generation age or actual use by a pick decision.
> A later rank-1 registration carries this capture contract and the not-established broader-sheet disposition unless it first supplies the independently specified, reviewed, outcome-free identity/target contract required by Part B; this diagnostic selects no blend or binding rule.
> - **Observation gaps:** planned 30-minute cadence; actual successful retrieval times come from qualified receipts, and failures can leave longer gaps.

Keep the other Out-of-scope statements, Part B execution budget and run-start-filename limit unchanged.

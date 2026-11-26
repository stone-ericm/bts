## Recommendation

**Choose option 2 now, with a targeted version of option 3 later.** Stop issuing general in-process absence certificates in Phase 1. Give the affected defence links an explicit `unavailable (absence not certifiable by the Phase 1 recorder)` entry. Preserve their regression tests, inspected semantic mutants and any supported positive decision evidence. Continue toward the register using event/return witnesses whose actual identity, invocation and execution conditions can be established.

I would next amend §9.3 and plan rulings 2/8 to make that boundary explicit, inventory affected links, and make the runner refuse absence requests mechanically. Then review the smaller positive-witness implementation and run the prepared evidence. If an important missing-alert or missing-pick link needs stronger certification, build one contract-specific recorder outside the application process, or a positive witness of the wrong decision, for that link. I would not start by building a general Python attribute-dispatch monitor.

The immediate loss is smaller than the consultation's framing might suggest: **two of the nine prepared specs are absence-kind; four are return-kind and three event-kind.** The two are already labelled `component`. Their deferral does not erase the incidents, their fixes, deployment evidence or their ordinary regression tests. It does withhold a particular causal certificate. The full 62-link worklist has not been fully specified, so this two-of-nine count cannot be extrapolated to the whole register.

Dropping absence is not automatic approval of the remaining recorder. Observer purity, executable closure and callable attribution also matter to positive witnesses. The r6 cached-hook example was an accepted pair over a changing executable closure, and r7's audit callbacks challenge the shared observational contract. Those obligations need resolution, enforceable exclusion or explicit unavailability in the retained modes too. I would not simply remove one certificate branch and declare the existing code ready.

This is an opinion about the next scope, not a new code-review verdict. I read plan rev 7 at `f6d881f`, design §9, the retained r7 report, and the nine prepared specs/worklist at branch pin `b991993`. Main was `0e25aa0bf59de493d2c970053f82d35f042beea2` when consulted; its plan/design are unchanged from f6d881f. I ran no new probes or evidence executions for this consultation. Previous measurements below remain the r7 measurements, not new results.

## Is the loop converging?

**It is converging on individual defects, but not yet on a stable certification boundary.** The old counterexamples generally become refused, their regression controls are sensitive, and substantial useful work survives: closure handling, qualification of reports, deployment bounds, fixture predicates and lifecycle cleanup. The r7 report's 309 passes/22 XFAILs and five new false acceptances coexist; neither result cancels the other.

The recurring mistake is treating a necessary condition as sufficient: a watched dictionary as all attribute resolution; a code object as a function identity; exact namespace types as stable dispatch; Python starts as every opportunity to change dispatch; identical test states/failure locations as identical application behavior. Each patch handles the exposed mechanism, while the claim still covers other mechanisms. Some fixes also enlarge the observer's own surface: the GC scan closed clone attribution but invoked application audit callbacks.

I cannot infer a decreasing escape rate from three rounds. The probes are selected counterexamples, not random samples with a denominator. “Every known probe is refused” is useful regression evidence, not a convergence measurement or proof that the next round will finish.

**Bounded absence is achievable under a closed model or at a trusted receiving boundary.** It is not inherently impossible just because Python is dynamic. A recorder can establish that zero qualifying receipts arrived during a defined, completely observed interval. A sufficiently trusted interpreter-level implementation could also expose more execution semantics than this callback-based recorder.

**I would not promise complete detection for arbitrary helpers/transports sharing this Python runtime and admitting arbitrary native behavior.** The present strategy has no justified way to promote its own missing events into complete absence across all those behaviors. If the program may also tamper with the monitor, its evidence or native memory, the same-process recorder is not an independent trust boundary at all; that stronger attacker is beyond my actual r5–r7 probes. Even without such tampering, those probes already defeat the narrower “ordinary Python and stdlib calls, frozen observer” claim.

So my answer is not a theorem that every possible in-process implementation must fail. It is that complete coverage needs a specified execution model and a credible trust boundary. Replacing Python callbacks with a trusted modified interpreter would be a different, much larger engineering project. Unbounded “no call ever happens later” is also a different claim from absence before a controlled cutoff.

## The adversary model

The model that fits this retrospective register is **cooperative, pinned code with potentially wrong behavior and a potentially wrong test harness**. We need to detect a real regression, an accidental masking mock, a fixture that follows the wrong path, a wrong invocation/selection, and an observer that misses or creates an effect. We do not need to defend this audit against an attacker intentionally corrupting the interpreter or forging its evidence files.

My crafted type swaps, fallback transports and send-on-audit hook are not evidence that BTS actually uses those mechanisms or that its developers are malicious. Several are outside a sensible everyday BTS execution model. I should keep that distinction explicit: the measured escapes refute the stated general coverage guarantee; they do not establish a production incident in the nine actual specs.

However, they were inside the **declared certification promise**: the framework claimed unsupported shapes become unavailable, yet accepted them. And their underlying mechanisms are not all exotic. Helpers rebind functions, tests patch module attributes, mocks have subclasses, imports affect execution closure, and workers can outlive the tested invocation. The actual prepared tests use both `monkeypatch` and `unittest.mock.patch`; for example the I-083 alert test installs a Mock inside the call phase (`tests/test_scheduler_decision_record_integration.py:272`). A blanket “no rebinding” model would exclude an intended fixture.

A narrower model should be stated approximately as follows:

> Certificates concern the named invocation and declared contract in this pinned synthetic execution, under an enumerated set of supported bindings, fixture replacements, dependencies and execution modes. They are not guarantees over arbitrary Python programs. Eligibility covers production, the semantic mutant, frozen tests/helpers/conftest/plugins, startup hooks and native dependencies. An unreviewed closure change, unsupported binding/call implementation, unaccounted worker/task/process, or observer callback with application effects makes the relevant witness unavailable. Deliberate evidence tampering and unreviewed native code are outside the model and outside any claim of certification.

To keep this from becoming a blind spot:

- Define allowed operations and trust assumptions before seeing a desired result. Name the module/function bindings, supported mock implementation and the fixture's authorized replacement points/lifetime.
- Check the **whole executed closure**, including the mutant and test infrastructure. A production-only scan would miss nearly every helper/transport escape I constructed.
- Bind eligibility to hashes and an explicit review record. Changed or unknown code invalidates eligibility; an `I trust this` boolean is not sufficient.
- Use static checks as a conservative screen plus review of a small admitted closure. Banning a few strings is not sound: aliases, reflective access, generated code, imports and native packages can perform the same operations.
- Enforce observable runtime restrictions independently where possible. Unsupported paths must produce a machine refusal, or have an explicitly reviewed precondition; they cannot quietly retain the old unconditional coverage wording.
- Keep counterexamples. Under the new policy they should be refused before certification or explicitly rejected as outside the admitted model. Do not delete them because they were constructed for review. Reassess the model when a real fixture uses an excluded mechanism.

A manually reviewed closed fixture model can support a qualified audit result. It is not the same thing as a static analyzer proving arbitrary Python free of those mechanisms. I would accept that distinction in the specification; I would not let “non-adversarial” silently replace the complete-coverage requirement while leaving its certificate label unchanged.

## Different approaches

Costs here are relative implementation/review scope, not elapsed-time estimates.

| Approach | What it proves | Residual gaps | Rough cost |
|---|---|---|---|
| Continue the general in-process recorder | If made sound for its stated model, entry/branch plus zero qualifying boundary calls in the bounded interval | More dispatch, native, reentrancy, worker and observer-effect cases; completeness still requires a model | **High, open-ended:** more interpreter-specific machinery and continuing independent probes |
| Disable absence certificates | An honest record of where the Phase 1 witness stops; retain independently supported positive witnesses | Two prepared causal links lack the requested negative witness; shared observer defects still need handling | **Low initial scope:** rejection gate, record/memo wording and refusal controls; no new general tracer |
| Static preconditions on the closure | Eligibility for a named restricted model, provided the checks/review are sound for that admitted subset | Production-only or keyword checks miss helpers, aliases and native/startup behavior; a precondition is not itself a boundary observation | **Medium for a deliberately small fixture closure; high for full scheduler closures or a general checker:** closure review, explicit exclusions, invalidation and runtime support |
| Independent sink process/socket collector | Zero qualifying receipts at that sink throughout a closed interval, plus a baseline receipt control | It must mediate the relevant observable channel; internal invocation, enqueue, attempted send and confirmed delivery differ; alternate routes and late work must be handled | **Medium per contract:** collector/protocol, interval/completion controls and revised fixture setup; potentially high for many channels |
| Positive wrong-decision/returned-state witness | The named invocation took a particular wrong decision or returned a contract-violating value after the mutant branch | The wrong decision may be corrected later; a skip/defer branch does not prove no send happened elsewhere | **Low to medium where a real return/state exists:** new contract sheet and linked oracle; more work if it requires new instrumentation or APIs |
| Small contract-specific control-flow proof plus fixture | Under stated inputs and immutable callees, a particular guard suppresses the required next step | Does not cover arbitrary run_day paths, external effects or changing dependencies; assumptions need separate support | **Medium for a small guard; high for the whole scheduler:** inspect every relevant path and keep the proof tied to source/input hashes |

The out-of-process candidate is the strongest alternative when the requirement really is a missing external effect. I would prototype it for one missing-pick or missing-alert contract before generalizing it. The parent owns receipt storage, run identity and interval closure; receipts carry the relevant date/selection/category; the baseline must produce the required receipt. Incomplete execution, an unaccounted producer or an incomplete interval yields unavailable. “No messages happened to be in a file at test return” is insufficient.

A child-written log file is not an independent recorder merely because the parent reads it. A collector-owned socket/process is a better trust boundary. But moving the sink does not automatically move every possible send through it. Keeping the real transport code and replacing its external I/O can preserve more path fidelity than replacing `send_dm` itself; neither establishes real Bluesky delivery. In Phase 1 these would remain local synthetic executions. The existing I-0813-b mock replaces the internal `_alert_missed_pick`, so an internal alert-request sink would prove a request at that component boundary; proving operator delivery needs a different boundary and fixture.

Positive restatement is worthwhile when the contract is genuinely about a decision. The I-0830-b mutant positively returns `FallbackPlan("defer", ...)` from `plan_fallback_action`. A linked witness of that wrong plan, with the enterable candidate and infeasible confirmation window pinned, could certify a planner regression. It would not certify “no pick DM reached the operator.” The existing incident test also checks a persisted fallback action (`tests/test_incident_2026_08_30.py:115`), but changing a snapshot label alone would not establish the causal path or absence of delivery.

Likewise, “the skip path ran” cannot replace an absence claim when another path can send. Avoid replacing a difficult observable contract with an easy internal reason string and then reporting the original contract as verified.

## What the register loses under each option

The following inventory is read directly from `b991993:docs/audit/2026-09-29-incident-register-evidence/current_defence/specs/`. All nine are **prepared inputs**, not accepted defence results; the README and plan Task 6 explicitly say they have not run.

| Prepared spec | Incident/link | Kind | Requested witness |
|---|---|---|---|
| I-0811-a | I-082 link 1 | return | Hard auth error instead of transient handling |
| I-0811-b | I-082 link 2 | event | Cookie-refresh advice on a transient outage |
| I-0813-a | I-083 link 1 | return | Wrong Warmup lock classification |
| **I-0813-b** | **I-083 link 2** | **absence** | Missing E3 alert after an undelivered classification lock |
| I-0830-a | I-084 link 2 | event | Pick send at/after cutoff |
| **I-0830-b** | **I-084 link 1** | **absence** | Missing timely pick after infeasible contender-window deferral |
| I-0830-c | I-084 link 3 | event | Extra sleep on cached feeds |
| I-0830-d | I-084 link 4 | return | Empty late-delivery health-alert result |
| I-0903-a | I-085 link 1 | return | Wrong reach-57 objective in the tail regime |

Under **continued general iteration**, none of these capabilities is intentionally removed, but publication remains dependent on sound tooling and actual accepted/reviewed runs. Fixing the five known r7 issues is not evidence that any of these nine has already passed.

Under **option 2**, I-083 link 2 and I-084 link 1 carry an explicit unavailable current-defence certificate, with the reason and retained regression evidence. The other seven remain eligible for their own checks; they are not automatically accepted. I-083's classification link and I-084's other three prepared links remain separate evidence opportunities. Their missing links must appear in coverage tables rather than being hidden by an incident-level “verified” flag.

For those two links I would retain: fix identity, semantic-mutant patch, inspected trigger/branch, the frozen test's observed failure if a run actually establishes it, restored baseline, certificate level and reviewer notes. Label this **regression evidence without a certified absence witness**. It cannot occupy the accepted-certificate field. Both corresponding branch drafts currently have empty fixture arrays; there is no accepted defence certificate here to revoke.

Under a **restricted model**, qualifying links could eventually have model-qualified component certificates; excluded links carry unavailable and the failed precondition. The cost to the register is limited generality and an explicit reliance on closure review. It would be dishonest to give them the same blanket “complete boundary coverage” description as before.

Under an **external sink**, the two links could gain bounded synthetic receipt-absence evidence, once the sink covers the actual declared channel and the runner/reviewer accept the run. They still retain their component limitations. An alert request receipt is not operator receipt; a DM transport receipt is not durable delivery confirmation unless the chosen boundary actually supplies that fact.

Under a **positive-decision rewrite**, the link carries the narrower wrong-decision certificate, with the original missed-alert/missed-pick effect still unavailable unless separately established. The four return specs already demonstrate why symptom wording and certificate kind differ: I-0830-d is described as a “missing alert,” yet its requested witness is a concrete returned empty list. Disabling absence-kind does not forbid every incident whose narrative contains the word “missing.”

Historical replay stays a separate axis. Its harness-compatibility and observable-contract requirements still apply; an ordinary zero-call assertion does not become a historical causal certificate by being moved out of current defence. Nor does an unavailable present-day witness alter the historical incident's qualified occurrence evidence or imply its fix failed. The full worklist remains **30 fixed Tier-A incidents / 62 fixed links**, with the yet-unprepared absence fraction unknown.

## If Eric still chooses to iterate

I would first fix the target of iteration. **Under an unchanged demand for general completeness against all the probe behaviors, I cannot honestly give a finite round estimate.** Another round can fix the known examples, but I have no evidence that it finishes the semantic surface. I would not promise that instruction monitoring plus an audit-hook census settles it.

The smallest sequence I could see reaching SIGN is:

1. **Agree on a written, bounded model and claims.** Specify the eligible bindings/closures/mocks, controlled interval, completion condition and unsupported modes. Include helpers, mutants and startup/native assumptions. Decide whether the two negative links remain deferred or get a separately mediated boundary. This changes the review target openly.
2. **Implement refusal before expanding observation.** Close the deletion/unresolved-path gap and unverified mock-call admission. Handle census failures without fabricated zero counts. Prove or exclude observer-triggered application callbacks. For transient dispatch, either enforce the reviewed stable-binding precondition for the actual admitted closure or implement genuine operation coverage; sampled types cannot be the evidence for that precondition.
3. **Review a real eligible pilot and its exclusions.** Check an actual prepared fixture's entire closure, observed/off behavior at the relevant observable, and the semantic mutant. All r5–r7 counterexamples must still fail to obtain a certificate under the declared policy, with refusal reasons that explain the boundary. Also use ordinary supported fixture controls to show the gate has not become “reject everything.”
4. **Freeze the model and run the planned evidence.** Verify publication pins and per-link outcomes. Any still-unproved negative link stays unavailable. Results and the register require their separate review; tooling approval is not approval of those results.

**My estimate is two further tooling-review rounds if Eric first accepts that bounded target; that is a guess, not a forecast based on measured convergence.** One focused implementation review might suffice if clean; the second allows for a defect found in that review. The subsequent planned result/publication review is separate. With the present unrestricted target, my estimate is simply unknown.

If “still iterate” means retain every current broad claim while refusing scope limits, the minimum is all known r7 corrections plus an independently justified completeness argument for the recorder's supported semantics, not merely a larger mutant count. I have no evidence supporting a small finite number of rounds for that path. For a retrospective incident register, I would spend the next review on clearly bounded evidence that answers the incident contracts, while publishing the remaining gaps honestly.

DONE

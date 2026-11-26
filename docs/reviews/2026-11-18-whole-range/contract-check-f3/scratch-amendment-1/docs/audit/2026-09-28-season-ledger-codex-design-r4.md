## Round

design r4 — spec v4, `docs/superpowers/specs/2026-09-28-season-ledger-design.md`; HEAD `4220056` (verified). The working-tree spec matches the committed version.

Confirmation limited to the four r3 blockers, the §3 wording correction, and contradictions introduced by those edits. No real data or snapshots accessed; no network requests or tests run.

## Findings

1. **r3 #1 — RESOLVED (§§5, 9, 11).** The surviving-pick rule now applies on every date, including a failed decision write after June. Commit and entry are independent; a contest match or generic lock flag alone cannot establish a system commit. Explicit private commits do not imply delivery or entry. The new fixtures cover the later missing decision, entered-but-undelivered preview, and private commit. Selection-specific binding remains required by “naming the selection” / “for it”; a date-level scheduler flag is not permission to attach another selection's commit evidence.

2. **r3 #2 — RESOLVED (§§5, 11).** Surviving file presence and consistency no longer establish complete history. Phase 1 uses unknown or evidenced incompleteness, and the failed-append/overwrite fixture explicitly expects unknown. This closes the counterexample without adding capture machinery.

3. **r3 #3 — RESOLVED (§§6, 11).** Conflicting game mappings or contradictory round identity require ambiguous matching and prohibit grade transfer. Schedule inference is available only when the unit has no capture at all. The conflicting-unit fixture now has an explicit ambiguous/no-inference/no-grade result.

4. **r3 #4 — RESOLVED (§§6, 9, 11).** Saver availability requires an evidenced initial state and every relevant transition; consistent visible rounds alone are insufficient. Eligibility is tied to lock/submission time and timed evidence. HOLD/Pass is expressly excluded as eligibility evidence. The added fixtures cover indistinguishable saver histories and a Pass or postponement after a valid selection.

5. **§3 wording correction — RESOLVED.** Acquisition explicitly includes schedule fetches; compilation remains offline. The evidence and compilation descriptions agree.

6. **Contradiction check — none requiring another design change.** The revised rules and their required fixtures agree. The earlier bounded scope, observation provenance, hypothesis-labelled reconciliation, and deferred increments remain intact. No further design correction is required before writing the implementation plan.

## Verdict

SIGN — spec v4 at `4220056` is ready for the implementation plan; implementation and data correctness remain to be verified.

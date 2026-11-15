# Ruling R24: the season-boundary design the build follows (lead session, 2026-11-14)

Inputs: `design-carbon.md` (marker `SRD1-f560f9bf`), `design-attack-codex.md` (marker `SRD2-60dfbf62`, verdict BUILD WITH THESE CHANGES), the reviewer's probes in `probes-codex/`. Neither document's claims about the code have been re-run by the lead; the reviewer's six demonstrated counterexamples are in its `results.json`.

## Decision
Build the reviewer's alternative C as the frame, keeping the parts of the designer's proposal that the attack left standing. In one sentence: **during the season boundary the decision runs on an explicit observation for that contest date, and automatic observations are trusted only after someone has looked at what the contest's endpoints actually return and switched the system over.**

Why not the designer's improved option A as written: its `proven_season` passes confidence from one response to a later, different one (a zero response followed by a carried counter; a current prediction row beside carried counters; an old anchor followed by a weaker snapshot). The reviewer ran all three. No rule in code can tell a real 7 from a carried 7 in otherwise identical responses, and nobody can observe the endpoints before March.
Why not alternative B (automatic zero for the first three days): it still refuses in the hard cases, writes a streak the system did not observe, and can zero a real streak if the rounds calendar is ever truncated.

## Rulings on the attack's findings
| # | Finding | Ruling |
|---|---|---|
| F1 | A valid manual observation still fails when `streak.json` is corrupt | Accepted. A valid contest or manual observation never depends on parsing model state; unreadable model state is reported separately. |
| F2 | `proven_season` launders confidence across observations | Accepted. The fetch records what each response contained (an attempt record, with the mapping it used) apart from the last accepted observation. No chaining. An automatic counter is accepted for live use only while a season-scoped marker says the counter contract has been verified by an operator. An absent `predictions` key is a schema failure, not an empty list. |
| F3 | A same-season restore reads as fresh and can resurrect a spent saver | Accepted. A restore needs an explicit acknowledgment; observation age is shown and judged without reference to local pick history; after a restore the saver's continuity is unverified, is treated as spent, and is not promoted until reconciled. |
| F4 | A multi-day zero has no safe renewal | Accepted. A manual observation covers one contest date and carries the actual streak, best and saver state; re-issuing it for the same date is idempotent; nothing reasserts zero silently. Undated automatic state is rejected for live use. Every delivery path enforces the requirement, or activation verifies the configuration. The runbook ships with the guard (the checklist rows to edit are B8 and the activation steps; B5 is the season-end table). |
| F5 | Writing an override does not wake a selection | Accepted. Readiness of the streak state is checked whether or not predictions exist, and recovery includes an explicit command that re-runs selection, with a stated latest start. The dependency on the unmerged extra-attempts work (`sr-lastcheck`) is named, not assumed. |
| F6 | Health and alert claims overstated; `not_earned` at 10 to 15 unwarned | Accepted. Health and selection share one classification; the warning is added; each alert is specified as a log line or a DM through a channel that exists; a new refusal with new instructions is a distinct incident. |

Kept from the designer's proposal: `not_earned` below 10 reaches the policy as in reserve and `uninitialized` as spent; the season tag and reset for `streak.json` (tested on real legacy bytes); a fetch that returns zero rounds fails and leaves the accepted observation alone; the fetch status keeps the last success beside the last error; the saver is never earned from an unaccepted observation; defensive reads of pick and state files.

## What the owner has to accept (reported to Eric; the build of the uncontested parts does not wait on it)
1. For the opening stretch, until the endpoints have been observed and the switch to automatic is made, each day's pick needs the current streak supplied for that date. If nobody supplies it, that day has no pick and an alert. Default until he says otherwise: no pick on unverified state.
2. After a mid-season restore of the box, he confirms whether the saver has been used; until then it counts as spent.
3. Reading the real account's responses across the boundary in March is his to authorise, as now.

## Owner decision (Eric, 2026-11-14 evening)
Asked whether, on a mid-season day when the contest reading is stale, the system should keep picking on the last confirmed streak or skip, with the lead recommending keep picking plus a DM: **"i prefer keep picking"**.

- Scope: an ordinary in-season day after the switch to automatic, when an accepted current-season observation exists but is behind (a failed fetch, a late contest settlement, an ambiguous new response). The decision uses the last accepted value and a pick is made.
- What changes from today's code: the stale case sends a DM (today it is a log-only warning), and the pick's record says it was made on a stale reading.
- Not covered by this decision, and unchanged from the ruling: the opening stretch before the switch (there is no accepted current-season value to fall back on, so a streak is supplied for the date or there is no pick), and the saver after a restore (counted as spent until he confirms). This reading of the scope is the lead's; he was asked about the mid-season case.

## Next
The designer revises its document into the build specification under these rulings (`design-v2.md`); the uncontested change sets (defensive reads with F1, the `streak.json` season tag, the saver mapping) go first, each to a fresh author who runs every test touching the modules it changes, with a same-day adversarial review before push.

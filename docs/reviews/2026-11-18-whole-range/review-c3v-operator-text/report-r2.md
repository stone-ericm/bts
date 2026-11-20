# Verdict

**SIGN WITH EDITS** on the docs-only patch against `bab11792`. Two checklist sentences need tighter scope.

| Finding | Resolved? |
| --- | --- |
| M1 | **Yes.** The warning quotation is exact. Warning and failure both trigger explicit handoff to `bts-lead` or Eric, with the 08:00 ET cutoff and the next affected job covered. CLAUDE points to the same procedure. |
| M2 | **No.** The checklist applies “This run leaves `streak.json` unreplayed” to failures as well as warnings. The required-live and model-dependent distinctions are preserved, and CLAUDE's retained-state sentence is accurate with its B8b cross-reference. |
| M3 | **Yes.** Both versions retain the insufficient-evidence rule and describe the private lock and delivery windows as possibilities. |
| M4 | **Yes.** CLAUDE conditions the summary on a successful model read and the raw receipt fields on publication, allows pre-replay failure, and accurately describes the inherited write rules. |
| M5 | **No.** Verification and recovery are now explicit, but the stand-alone “Move nothing and delete nothing” can still bar a file move during verified recovery. |

## Anything new

- **M2.** The CLI can exit unsuccessfully after replay saves. In the probe a permission change before the summary caused `PermissionError` after the saved count changed from 9 to 1 and the sealed receipt recorded `replay.state=saved`. [Save and receipt order](../../src/bts/cli.py#L2937)
  **Replace “This run leaves …” with** “When this warning appears, the replay leaves `streak.json` unrecalculated and later grading can advance an incorrect model count or saver. On other failures, verify whether replay saved before deciding what needs recovery.”
- **M5.** Initial preservation is safe. Extending it through recovery can leave the round-one A-decision/B-file mismatch in service, where the watchdog confirmed B, `day-outcome` falsely cleared A, and grading scored B. [Round-one evidence](report.md#must-fix)
  **Replace “Move nothing and delete nothing.” with** “Preserve the files in place during triage and make changes only as part of verified recovery directed by the BTS lead or Eric.”

## What I ran

Read the patch and relevant source, checked it with `git apply --check`, used the round-one evidence, and ran one offline `.venv/bin/python` probe for failure after saving, with the real CLI and receipt and a simulated permissions change.

## What I did not run

No patch application, network, pytest, or production commands. No tracked files changed.

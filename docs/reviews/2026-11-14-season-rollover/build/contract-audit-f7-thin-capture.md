# Contract (draft 1, NOT approved to build): F7, a live-forward capture of a slate with fewer than ten scored batters

Status: written 2026-11-18 by the fresh session after three BLOCKs (`STOP-AUDIT-F4F8.md`). It is for a contract check, then a decision by the lead and Eric between **building this** and **leaving the capture as on `main`**. Nothing from branch `sb-audit-f4f8` is reused. Tier C (the verifier decides whether capture evidence is accepted).

## The defect on `main`
`scripts/live_forward_capture_once.py` exports the top `--top-n` (10) scored batters per variant and verifies with `--expected-top-n 10`; `verify_candidate_artifact_pair` requires exactly that many rows per date. On a slate with nine scored batters (Opening Night, one game, one lineup posted; ruling R18) the export writes nine rows, verification fails (`failed_verify`), and every later run answers `failed_verify_existing`: a `CRITICAL` capture alarm for that date for good, and no official evidence for the day. The resolve step (`scripts/live_forward_resolve_once.py`) has the same row rule.

## What changed from the three blocked attempts
The blocked branch recorded the lower count INSIDE the manifest the verifier checks, so anything that rewrote the artifact directory could also authorize itself. Here **the manifest and the parquets are byte-for-byte what `main` writes, and the verifier's rule is unchanged unless a separate file says otherwise**: a *thin receipt*, written once at export, outside the artifact directory, named by the SHA-256 of the manifest it authorizes.

## Definitions
- **Scored count** of a variant: rows of its prediction frame with a non-null `p_game_hit`, counted BEFORE `predictions_to_ranked_profiles`. **E** = `min(top_n, scored count)`.
- **Receipt directory**: a new option `--thin-receipt-dir` on `export-live-candidate-artifacts` and `verify-candidate-artifacts`. The two wrappers pass `<artifact root>` with `_thin_receipts` appended to its last path component (a sibling of the artifact root, never inside a date directory). No option: no receipt is written or read.
- **Receipt**: `<receipt dir>/<date>.<manifest sha256>.json`, created with exclusive create (never overwritten, never rewritten), holding `schema` (`bts_thin_capture_receipt_v1`), `date`, `candidate_name`, `git_commit`, `top_n`, `effective_top_n` (E), `manifest_sha256`, and `profile_sha256` for the production and the candidate parquet.
- **Export order:** both parquets, then the receipt, then `manifest.json` last (its bytes are fixed before the receipt is written). A receipt is written only when the option is given AND both variants have the same E AND `1 <= E < top_n`.
- **A receipt is in force** for an artifact directory only when: the file for the SHA-256 of the manifest bytes just read exists and is a regular file holding a JSON object; `schema` is exact; `effective_top_n` and `top_n` are exact `int` (not `bool`) with `1 <= effective_top_n < top_n`; `top_n` equals the manifest's `top_n` and `--expected-top-n`; `date`, `candidate_name` and `git_commit` equal the manifest's; `manifest_sha256` equals the name and the bytes; both `profile_sha256` values equal the files on disk. A file that exists and fails any test adds a FAILED check `thin_receipt` (the verification fails; it never raises). No file: no check is added.
- **The one changed rule:** the rows-per-date floor is `--expected-top-n`, as on `main`, unless a receipt is in force; then it is the receipt's `effective_top_n` and a passing check `thin_receipt` records it. For a `live_forward_resolved` directory the receipt is looked up for the SOURCE directory (new option `--thin-source-dir`, passed by the resolve wrapper) and must be in force there; the resolved manifest's `date`, `candidate_name`, `git_commit` and `top_n` must equal the receipt's.

## The table: inputs that must still be rejected, or still alarm (expected ten unless said)
| # | Input | Required outcome (same as `main` unless marked NEW) |
|---|---|---|
| R1 | Ten-row capture, a parquet truncated afterwards, manifest intact | row-count and rows-per-date checks fail; `failed_verify_existing`; `CRITICAL` |
| R2 | **Ten-row capture rewritten to nine rows in both parquets, with `row_counts` (and any other key, an `effective_top_n` included) rewritten in the manifest** (the third review's finding) | the rewritten manifest has a new hash, so no receipt; floor ten; fails |
| R3 | Nine-row capture WITH a receipt, a parquet truncated or any row altered afterwards | NEW check `thin_receipt` fails (profile hash); fails |
| R4 | Nine-row capture with a receipt, rewritten to eight rows with the manifest rewritten to match | new hash, no receipt; floor ten; fails |
| R5 | R4, and the old receipt copied to the new manifest's name | `manifest_sha256` inside differs from the name; `thin_receipt` fails |
| R6 | Rows lost in conversion: ten scored, a profile of nine (either variant or both) | NEW: the export raises before any file is written (profile length differs from E); no receipt, no manifest; `failed_export`, then `failed_partial_artifact_dir`; `CRITICAL` |
| R7 | The variants' scored counts differ (nine and eight) | no receipt; files as on `main`; verification fails |
| R8 | No scored row, or no `p_game_hit` column | the export raises, as on `main` |
| R9 | Legacy captures (made before this change; no receipt; any extra manifest keys) | ten rows pass and nine rows fail, exactly as on `main` |
| R10 | A receipt for another date, candidate, commit or `top_n`; `effective_top_n` of `true`, `9.0`, `"9"`, `0`, `-1`, `10`; not an object; unreadable; a directory or pipe at the path | `thin_receipt` fails; never raises; never blocks |
| R11 | The receipt deleted after a thin capture | floor ten; fails (as `main` for that capture) |
| R12 | Any caller that does not pass the option (direct CLI, the inventory, the surface export) | byte-identical reports to `main` |
| R13 | A resolved copy of a ten-row capture rewritten to nine rows | no receipt in force for the source; fails |
| R14 | A legacy manifest whose `run_kind` is not a string | failed checks, never an exception (as `main`) |

Must be ACCEPTED (the reason for the change): **A1** nine scored of nine, `top_n` ten: nine rows per variant, a receipt with E nine, `exported_verified`, then `existing_verified`, no alarm. **A2** twelve prediction rows of which three have no score: as A1. **A3** `top_n` five against nine scored: five rows, no receipt, as `main`. **A4** ten or more scored: manifest and parquets byte-identical to `main` for the same inputs and clock, no receipt. **A5** the resolved copy of A1 verifies through the resolve wrapper. **A6** a snapshot-drift recapture of a thin day (`refresh_stale_artifact`) writes its own receipt and verifies after the swap.

## Limits stated now
1. Something that rewrites the artifact directory AND creates a matching receipt in the receipt directory is not detected. Only the export writes receipts.
2. On a full day, replacing ten rows with ten different rows is not detected (as on `main`; on a thin day the receipt's hashes catch it).
3. A crash between the receipt and the manifest leaves a partial directory and an unused receipt: the day fails closed as any failed export does on `main`. The retry of a failed capture that the first attempt added is NOT part of this.
4. The inventory and the surface export are not changed; whether a verified thin day is counted by them is for the check to establish (item 6 of the brief).

## Tests an author must deliver
One test per row R1 to R14 and A1 to A6 through the real export, verifier and wrappers (subprocess transports captured); each R row RED against a build that takes the floor from anything inside the artifact directory; a mutant ledger (drop each clause of "in force" in turn; receipt written after the manifest; receipt written when E equals `top_n`; the lookup by a name other than the hash of the bytes read).

## The alternative: leave as on `main`
Accept one permanently alarming capture date per thin slate (expected on Opening Night 2027-03-24) and no official evidence for that day; add a line to checklist row B-opening. Cost if wrong: one day of research evidence per thin slate; no effect on any pick.

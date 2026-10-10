# State at the closing of bts-lead3's session (2026-10-10, ~12:50 EDT)

Written by the lead bts-lead3 under the herdr manager's closing protocol (Eric's words at 12:33 EDT, prompt log
`claude-shared/memory/conversation-logs/2026-10-10-mac.md` l.608: "drive both eor and bts to graceful session closings
and then do the same to this session"). This directory is a docs-only record; nothing here was reviewed, run or merged.

## The branch at closing
- Tip before this commit: `3019635` (note revision 3; d1 round 1 archived). The branch is pushed; nothing of it is on main.
- **The trust-boundary note revision 3** (`docs/sota_audit/2026-10-10-c2-framing-2026-trust-boundaries.md` at `3019635`,
  480 lines, sha256 `b29b81cf5c5370cff02e50a20c8f3ecd1a36f847b64c557548f0b83b270978ed`) is **UNREVIEWED**. It was
  committed before the manager's direction for revision 3 arrived and does not implement that direction: the wrapper
  still passes `--extra-charges <path>` and the off-launcher account still lives under the test's `OUT_ROOT`. The
  direction (the manager, 2026-10-10 ~00:5x EDT): no caller-selected account file at all; one fixed location C1 itself
  knows (beside `compute_ledger.tsv` under C1's data root); steps append only through a C1-provided append entry point
  or an equivalent C1-owned writer; the receipt carries the whole account's hash and length at decision, and the
  payload checks the account is still that prefix plus later appends; C1 counts a decided-but-unfinished launch as
  reserved in its own ledger view until TERMINAL and RECONCILED exist; no path, file or total chosen by the caller.
- **The code** is unchanged since `e0ff07c` (round 4). The last fast suite on the branch: `e0ff07c`, 4545 passed, 0 failed.

## The design reviews of the note
| round | reviewer | note revision | verdict | report |
|---|---|---|---|---|
| 1 | f2 (round 2, anchored) | r1 `abd2477` (sha `c54063e7…`) | BLOCK (not counted) | `docs/audit/2026-10-09-c2-framing-2026-code-codex-f2r2.md` sha `5c95d835…` |
| 2 | d1 (fresh, round 1) | r2 `f4edff3` (sha `b659f045…`) | BLOCK, D1–D7 (not counted) | `docs/audit/2026-10-09-c2-framing-2026-code-codex-d1.md` sha `5edcbd34…` |
| 3 | d1 round 2 | r3 `3019635` (sha `b29b81cf…`) | **NEVER RAN** | brief `../review-d1/prompt-d1-round2.md` sha `e7c46366…`, written 00:56, never handed over, no clear requested |

The manager's bound: d1's round 2 would have been the third design review of the note; a BLOCK there stops the design
phase for this lead's session. The session closed before it ran.

## Files here
- `allowance-row-draft.md` — register row `C2-framing-2026-allowance`, grammar-checked, cap 237.6901 (Eric's "100 cpu
  hours" + the prior effective shared total 137.6901). NOT written on main; must ride a suite-verified push.
- `cap-commit-patch.py` — the `CAP_H` → 237.6901 cap commit (ledger.py + test_c1_cycle + test_screen). NOT applied;
  it belongs after a merge-gate SIGN, with its own short review.
- `x37-row-draft.md` and `c2-framing-2026-source-inventory.draft.json` — X-37 and the source inventory, drafts; nothing
  published; no 2026 file has been opened or hashed.
- `note-row-*.md` — NOTE rows for main (the round-3 ledger, the push slip, round 4, review f2, review d1) in the lead's
  words; none written. Their `<result pending>` placeholders: suite at `d73693d` 4519/0; suite at `e0ff07c` 4545/0;
  d1 verdict BLOCK with report sha `5edcbd349a4d71d87aba2d9adc6b7b0346ac1fa36570d811a3b85ca804ad965a`.
- `box-status-read-20261010-0038.txt` — the read-only box status read of 00:37:57 EDT that met condition (c).
- `note-r2-commit.txt`, `note-r3-commit.txt` — the note's commit and sha records.
- `suite-summaries.txt` — the first and last lines and sha256 of the four full-suite logs kept in the results directory.

The raw records (RED/GREEN outputs, suite logs, mutation logs, review briefs and reports) stay in
`~/projects/bts-c2-framing-results/2026-test/` (not in git).

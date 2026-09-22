# 2026 Season Wrap — execution index and completion matrix

**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md` (approved by Eric 2026-09-22). **Register:** `docs/audit/2026-09-22-exposure-register.md`. Every row below ends in exactly one of: **completed** (+ evidence link), **unavailable** (+ reason), **deferred** (+ reason). Rows are appended as work lands; nothing is pre-marked.

| Item | Deliverable | Status | Evidence |
|---|---|---|---|
| W0.1 profile backup | `mdp_estpa_run` on box + manifests + restic + restore test | **completed 9/22** (146 files; restic `a2781532…`; empty-dir restore matches) | `docs/audit/2026-09-22-w0-preservation.md` |
| W0.1b pinned diagnostic | `confirm_mdp_policy_replay.py` output hash, Mac == box | completed 9/22 | same memo (sha256 `bdbb2a09…`) |
| W0.2a preliminary checkpoint | staged `prelim-20260922` + manifest + restic snapshot | **completed 9/22** (5,042 files, 125 MB; restic `a2781532…`; restore matches) | same memo |
| W0.3 exposure register | `docs/audit/2026-09-22-exposure-register.md` §A | completed 9/22 | file |
| W0.4 protocol inventory | same file §B (P-01…P-06 quoted) | completed 9/22 | file |
| W0.5 D3 / D5 / D8 recorded | same file §C | completed 9/22 | file |
| W0.6 bounded grab (~9/28) | runbook/wrapper (Codex-reviewed) + fixture rehearsal + execution | **Codex SIGN WITH EDITS at `05c8c0d`, edits applied in `1fdd460` (narrow verification pending)** — 2 design rounds + 7 code rounds; 58 offline tests = the rehearsal; **no board-page ceiling (Eric 9/22)**; profiles = D5 300 for this pass (full-field campaign = W0.6b); execution 9/28 after 08:00 ET | `docs/audit/2026-09-22-final-grab-runbook.md`; early cohort E `docs/audit/2026-09-22-early-cohort-2026-05-01.json`; C-01 |
| **W0.6b full-field profile campaign** (Eric 9/22: "grab all the picks all the public profiles (i assume 95000+)") | resumable multi-day job: design → Codex → TDD build → observed short production segment → extend | design r1 (`.codex-review/season-wrap/w06b-profile-campaign-design-codex.md`): **BLOCK ×5** — durable halt that survives crashes/failed writes (status recorded at headers, halt persisted before DM, ambiguous in-flight = quarantine); single writer + persistent limits (process lock, ET-day totals, count logins, 401/redirect/challenge = stop); SQLite receipts + immutable per-attempt raw + atomic batch manifests + process-death tests; frozen board id universe as denominator (imports validated, not "already fetched"); honest coverage (rounds lookups archived, no_history ≠ privacy, observed-history estimands, schema versioned with user_id). Answers: 3–8 s is unvalidated; 9k/day = 30× retired volume; start with a short OBSERVED production segment, no completion-date promise; seeded rank-stratified order; ≤1 ledger-reserved retry for classified transient failures only. **Plain statement relayed to owner: would not run unattended with the current restart contract; MLB ToU §1(xi) prohibits automated collection regardless of pacing; prefer an authorized bulk export if one exists.** Next: design v2 | brief `w06b-profile-campaign-design.md` |
| W0.7 final snapshot | staged final + tag-isolated restic + restore test | not started (after grab + grading cutoff) | |
| W0.8 research capture (D8) | `--capture-research-on-skip` deployed `f1b07e9` | completed 9/14; verified 9/19–9/21 | plan W0.8; `bts_index` memory |
| W1.1 ledger | parquet + builder + tests + reconciliation table | not started | |
| W1.4a DD tripwire read (P-01) | memo | **completed 9/22** (deviation recorded: snapshot reconstructed; trigger not fired, −4.3 pp at n=80) | `docs/audit/2026-09-22-dd-tripwire-formal-read.md` + `.json`; reader `scripts/audit/dd_tripwire_formal_read.py` |
| W1.2 bridge A–D | memo | not started | |
| W1.3 distinguishing tests | in bridge memo | not started | |
| W1.4b due reads (P-02…P-05) | one memo each | **P-04 (#16) inventoried 9/22 → INSUFFICIENT** (113/114 eligible < 120, stream frozen at 9/18); P-02 skip shadow = `insufficient_n` (15 resolved < 30, cannot grow); P-03 context shadow + P-05 tail audit deferred to the W0.7 final snapshot (season ends 9/27) | register X-11, X-07 |
| W1.5 incident register | memo | not started | |
| W1.6 corrections index + README | `docs/audit/2026-09-corrections-index.md`, README | index started 9/22 (C-01 parser bug, C-02 README) | file |
| W2.1 final-leader case series | memo | not started (needs W0.6) | |
| W2.2 as-of cohort comparison | memo | not started | |
| W2.3 MLB forecast benchmark | memo (gated) | not started | |
| W2.4 #87 mechanism mining | result memo under full protocol | not started | |
| W3 literature refresh | memo + tracker section | not started | |
| Decision memo (D1, D2, D4, D6, D7) | `docs/audit/<date>-2027-decisions.md` | not started | |
| W4 selected candidates | registrations / results / dispositions | not started (none selected) | |
| 2027 season-start checklist | `docs/ops/2027-season-start.md` | not started | |

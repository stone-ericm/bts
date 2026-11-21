# Second check of the D2 contract (version 2)

Same rules as your first brief (`.codex-review/wr-check-d2/brief.md`), including: no pytest, no wrapper runs, and while a `victoria3` process exists every probe is small, `nice -n 19`, seconds each. Probes in `.codex-review/wr-check-d2/r2/`.

This clone is at `1b3ab205`. The contract `docs/reviews/2026-11-18-whole-range/kickoff-fix-deploy-v6.md` is now version 2. It adopts your design finding differently from your proposal: there is no separate staged installation of jobs; the deploy owner itself switches the crontab and the two unit fragments in a new phase `install-jobs`, under the hold, after the stop, drain, reset, HEAD check and locked sync, rendering from the now-target checkout with `--no-sync`; the root installer keeps the tool, drop-in and locks and, on a box with no record, installs a provisioning fence (a `failed` hold at step `provisioned`). Your replacement texts are adopted in the section "Adopted from contract check 1", with the session's edits marked [edit]. Your first report is archived at `docs/reviews/2026-11-18-whole-range/contract-check-d2/contract-check.md`.

Answer the contract's "Questions the contract check must answer (second check)", with probes where a sequence is involved, and check every [edit] for whether it changes the meaning of your text in a way that makes something wrong. Classify MUST FIX / NOTE as before.

Deliverable: `.codex-review/wr-check-d2/contract-check-r2.md`, under 110 lines, verdict first (BUILD FROM THIS / BUILD WITH THESE EDITS with replacement texts / NOT YET with the smallest design change). End with what you ran and did not run. When complete print `WRCHKD2R2-<first 8 hex characters of the sha256 of contract-check-r2.md>` on its own line and `DONE`.

# Confirmation round: the probable-pitcher cache write (the void-reply half was withdrawn)

You are a fresh reviewer. **Assume the author is wrong.** Round 1 of this change set ended in BLOCK on its second half (a void result reply parsed by the feed regeneration made that tool's reply pairing worse). That half has been REMOVED: `src/bts/state/regenerate.py` is byte-identical to the base again. What remains is the cache write only.

This checkout is an isolated copy at `fb1acd4c`. Base on `main`: `df454725`. Confirm first: `git diff df454725 HEAD --stat` touches exactly `src/bts/features/compute.py` and `tests/test_leads_small_fixes.py`, and `git diff df454725 HEAD -- src/bts/state/` is empty. Change no tracked file; probes and your report go in `.codex-review/leads-r2/`. No network.

## The claim
`_build_probable_pitcher_lookup` (`src/bts/features/compute.py`) wrote `data/models/probable_pitcher_lookup.json` in place. It now writes a sibling temporary file (`.probable_pitcher_lookup.json.<pid>.tmp`) and renames it over the cache with `os.replace`, removing the temporary file on any failure that unwinds; a cache that exists but cannot be read or parsed prints one `WARNING: unreadable probable-pitcher cache …` line to stderr and is rebuilt from the raw feeds present, as before (before: silently). Returned lookups and cache bytes for the same inputs are unchanged.

Known and accepted (round 1 notes, not findings): a SIGKILL mid-write leaves an abandoned temporary file; two THREADS of one process share a temp name (production callers are separate processes); two processes can still lose each other's new entries (as before); a cache path that is a symlink is replaced by a regular file; the file mode follows the umask; the models directory must be writable; a readable cache with a scalar entry value is still accepted; no `fsync`.

## Establish, by running the real code
1. On synthetic raw directories, base (`git archive df454725 src`) versus this commit: identical returned lookup and identical cache bytes for a first build, an incremental build, nothing new, and no raw directory.
2. A failure during the write (`OSError`, `KeyboardInterrupt`, `SystemExit`, at the temp write and at the rename): the published cache is the complete old file or the complete new one, and no temporary file remains. On the base the same failure publishes a truncated file.
3. An unreadable cache (truncated JSON, invalid UTF-8, a JSON array): one stderr line, nothing on stdout, the lookup rebuilt from raw, the cache rewritten; a readable cache: silent, and not rewritten when nothing is new.
4. One new question: is there any caller or test in the repository for which the change from "silently continue" to "print a warning" or from "write in place" to "replace" changes an outcome that matters to a pick: grep for the cache path and the function, and for anything that captures stderr around feature computation and treats output as failure (the scheduler, `predict-json`, the orchestrator, cron wrappers, health checks).
5. The new tests: which fail on the base source; mutate a scratch copy (write in place; no cleanup; no warning; swallow the write error) and say whether each is caught.
6. Anything else in the remaining diff you consider wrong.

Run pytest through `~/bts-runs/yield_run.sh .codex-review/leads-r2/<name>.log <command…>`, one at a time (a `nice` warning is not an exit); small probe scripts may run directly; create files with your edit tool, never a shell heredoc; do not edit a script after asking to run it; no pattern kills. Do not run the whole fast suite (the lead's gate does).

Deliverable: `.codex-review/leads-r2/report.md`, under 60 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings`, `Notes`, one line per numbered item, `What I ran`. When complete, print `LEADSR2-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.

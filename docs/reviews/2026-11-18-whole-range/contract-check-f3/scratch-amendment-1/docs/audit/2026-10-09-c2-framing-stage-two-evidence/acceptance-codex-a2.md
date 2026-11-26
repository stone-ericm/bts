## Verdict

**SIGN.**

Accepted-commit: 94fd50432ac12a9c30bd91795e131b4a863d4f08

All four round-a2 checks hold. The a1 correction was applied unchanged, the index contains exactly the two authorized additions, and the evidence and run copies are unchanged. This closes the acceptance corrections; it approves no production change, deploy or further run.

## Checks

1. **Change scope — passed.** From the checkout root, I ran git rev-parse HEAD and git status --porcelain --untracked-files=no before checking the changes: HEAD was 94fd50432ac12a9c30bd91795e131b4a863d4f08 and the tracked tree was clean. git diff --name-only from fe2ae5c77d8ba70ea7c8fd630d07b35191c3b854 to that commit listed exactly docs/sota_audit/2026-10-09-result-c2-framing-stage-two.md and docs/sota_audit/2026-10-06-c2-cycle-index.md. git diff --summary showed no file-mode or other structural changes.

2. **Verbatim note correction — passed.** I hashed report-a1.md and obtained its specified SHA256, 066d739def52dea38b861780150932df0fb609bd2db76f8e451f3071a5662afe. I extracted its sole unified diff from the Corrections section and compared it with git diff --no-ext-diff --no-textconv --unified=3 for the note, ignoring only file headers and trailing hunk-context labels. All three hunk ranges and all context/removed/added lines matched: three removed lines and four added lines, at the same positions. I also replayed the prescribed hunks against the old note in memory, verifying each complete preimage; the result equaled both the new commit's note blob and the working-tree note byte for byte. No patch was applied during this round.

3. **Exactly two index additions — passed.** I compared the old and new index blobs line by line and checked the current file against the new blob. Both have 48 lines; only line 27, row (e), differs. Every existing row byte remains, with exactly two clauses inserted before its closing delimiter. Clause (a) distinguishes the note's guard receipt, 47,667.343496 seconds (printed 47,667.3), from the index's journal figure, 47,667.357411 seconds (printed 47,667.4), and says no number changes. Clause (b) records the a1 checker gap as OPEN for a later cycle with its own review, not now: zip without a length check, unchecked per-seed P@1 delta maps, and secondary metrics not reconciled with scorecards or the aggregate. It cites the a1 report hash and says the script remains unchanged. The insertion equals those two clauses exactly; no other index text or existing number changed. This check confirms the appended attribution, without re-auditing the receipts.

4. **Evidence and run copies unchanged — passed.** git diff --name-only for docs/audit/2026-10-09-c2-framing-stage-two-evidence/ was empty. I enumerated its seven tracked files, including tables.py, and compared each old-commit blob, new-commit blob and current file: all were byte-identical, with no extra file. I hashed all 150 supplied run files and compared the complete path/digest map with the supplied runs.sha256 and the accepted lists at the a1 commit: stage one's 45 entries plus stage two's 105. All three maps were identical, with no missing, added or changed run file. The a1 report itself still had its specified hash.

## What was run

I read prompt-a2.md first. All checks ran from /Users/eric/projects/bts-c2-framing-s2-acceptance, using the prescribed PYTHONPATH=., UV_CACHE_DIR=/tmp/uv-cache, TZ=America/New_York and uv run --offline python -B command form for Python. The Python checks invoked read-only git rev-parse, status, diff, show and ls-tree commands, parsed the a1 correction, replayed it in memory, compared index/evidence bytes, and computed SHA256 hashes. No result, table, rule or disposition was recomputed; tables.py was not executed, and the numerical questions settled in a1 were not re-opened.

No memory file or outside project corpus was consulted. There were no data/ reads, secrets or runtime-configuration reads, network/box/SSH/gh calls, escalation requests, tracked edits, or writes to the run copies. Only report-a2.md was written; no scratch directory was needed or created. Before completion I rechecked the required report sections and single Accepted-commit line, the unchanged a1 report hash, final HEAD and clean tracked status. I then used shasum -a 256 on the final report for the completion marker.

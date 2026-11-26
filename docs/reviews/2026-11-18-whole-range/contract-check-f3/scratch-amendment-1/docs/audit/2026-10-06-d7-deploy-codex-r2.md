## Verdict

SIGN — for deploying `f882411d939d1b8eb0433beee1514170a17f689a` with the planned box steps and the explicit P1-1 owner-approval gate. The seven final-candidate recertification runs now satisfy the retained runner-and-reviewer rule. No further code or plan edit is required by this follow-up. The five blank-line removals remain subject to Eric's approval before cron installation; this SIGN does not grant that approval.

## Findings

**P2-1 closed: the final-candidate evidence is supplied and checks out.** HEAD is `f84e5bb845f076a8d84eb25d15c7090773c3cc63`. Its entire diff from the deploy candidate is confined to `docs/audit/2026-10-06-d7-recert/`; production code, tests and dependency files are unchanged. The required labels match the seven listed in `docs/ops/reconcile-receipt-v1.md` under Re-certification, with no missing or duplicate run.

The new summary binds the full candidate SHA above, frozen tool pin `f453283d0a0ec39a939b22c6500b5fb36f5452c0`, and observer SHA256 `fc1b38da7bbec42e634156750def91f78c93d3e5e7b69783d2aff9d740f0933a`. I recomputed the observer hash from the frozen Git object and checked that the incident-register tooling in this checkout has no diff from that pin. The driver asserts the external tooling checkout's pin and cleanliness, checks the runner's import location, creates the owned worktree at the candidate, and sets each effective spec's baseline to that same candidate. Its defence execution uses the unchanged `defence.current_defence`; wrapper changes also simplify label selection and remove replay/resume handling. The timing fields are diagnostics, not certificate decisions.

I read every acceptance object and mutation patch. Independently, `.codex-review/d7-deploy/verify-r2-evidence.py` checked:

- Every raw spec hash against both the unchanged spec file and the original `results-f453283` summary; every effective spec equals the original with only its baseline changed, excluding the runner's derived assertion-location fields.
- Every acceptance, patch and stage-event hash against the supplied records. Applying each retained patch in memory to the candidate's actual source produces exactly the declared mutation. The mutation bodies match the earlier patches, and their anchors are at the recorded candidate line numbers.
- All 35 stage logs: green, green without observation, mutant, mutant without observation, and restored for each run. Their source-tree digests match independently reconstructed candidate or mutant bytes. Collected test-file hashes and imported BTS module hashes also match those bytes. Inventories agree across stages; setup and teardown pass; green/restored calls pass; declared killing nodes fail by assertions at their frozen test lines. Observer-on/off states, failure frames and normalized failure-message hashes agree.
- All 12 killing-node certificate objects, recomputed from the retained mutant events using the byte-unchanged frozen certificate evaluator. Every result is `ok: true`, with no reasons. I also compared the linked events with the original evidence while retaining exact witness values, categories, node identities and source qualnames; only execution-specific paths, locations and identifiers were excluded from that comparison.

The patch, fixture, killing-frame and positive-witness checks support these decisions:

| Run | Killing nodes | Retained wrong behavior under mutation | Runner / supplied reviewer / this review |
|---|---:|---|---|
| `D-I043-1` | 2 | Postponed primary or DD game returns `locked=true`, `game_started_or_final`; regeneration assertion fails | accepted / accept / accept |
| `D-I043-4` | 3 | Postponed, Cancelled and Canceled candidates return eligible; selection keeps `Voided Top` instead of `Fresh Pick` | accepted / accept / accept |
| `I-0813-a` | 2 | Warmup, including statusCode PW without detailed text, returns a started lock | accepted / accept / accept |
| `D-I043-2` | 2 | Postponed primary slot returns `hit` instead of `void`, including live-feed Preview | accepted / accept / accept |
| `D-I043-3` | 1 | Polling cap returns `unresolved` for an already-resolved day | accepted / accept / accept |
| `D-I063-2` | 1 | Four newer settled fixture picks cause a currentness-refusal alert and exit 2 instead of snapshot persistence | accepted / accept / accept |
| `I-0811-b` | 1 | Synthetic transient auth failure emits cookie-recapture advice | accepted / accept / accept |

The new `REVIEW-RUNS.md` contains explicit **accept** rows for all seven, satisfying the original task56 header's requirement for runner `accepted` **and** reviewer `accept`. These are the existing bounded synthetic contracts: component mocks remain disclosed, and `D-I043-3` still covers only the polling-cap clause. The clock-dependent cap test fixes `_now_et`; the two void-scoring tests use `cap_hour_et=10`, making their clock-cap predicate impossible. The other retained bad decisions use fixed statuses, fixture dates and injected failures. I found no new observer-cost-dependent killing decision in these closures. Equal normalized messages alone were not treated as semantic or observer-invisibility proof.

The verification output is retained in `.codex-review/d7-deploy/verified-r2-evidence.json`. I reviewed author-produced execution records; I did not rerun the certification worktree or inspect its destroyed execution environment. This closes the supplied-evidence gate within the frozen certificate model and does not establish installed production behavior.

**P1-1 resolved in the plan; operational approval remains pending.** The author adopts round 1's complete-crontab comparison text and expressly sends the five blank-line removals back to Eric through the herdr manager before installation. That additional approval condition is essential: until approval is recorded, Eric's original exact-two-change ruling still governs, and the expected five blank removals must stop installation after inspection. Following approval, compare the complete proposal with the backup, permit only the specified 07:40 addition, commented-checker removal and approved blank removals, then verify the installed crontab against that same complete expectation. Foreign commands, comments, assignments, URLs and arguments remain subject to exact comparison. The gate cannot pass merely because this review signs the deploy.

**Regression evidence and scope.** Round 1's independent sandbox run remains 3899 passed, 8 failed, 7 skipped and 22 xfailed, with the eight failures attributable to four denied C1 pre-exec cases, three denied AF_UNIX binds and one denied HTTP bind. The prompt reports an author-run outside-sandbox result at the candidate of 3907 passed, zero failed, 7 skipped and 22 xfailed. That is consistent with the eight environment failures passing outside the sandbox; it is author testimony, not my measured result. I did not repeat the full suite for a documentation-only delta. This round's independent evidence verifier passed.

This is a follow-up to the unchanged round-1 integration review, not a fresh blind review; its disclosed method limitations remain in that report. No new memory lookup occurred. This round read only in-checkout sources and supplied evidence, performed no network, SSH, escalation or live operation, and changed only review-owned artifacts. Final tracked status is clean and `git diff --check` passes. No push, deploy, real cron installation or production configuration change was performed.

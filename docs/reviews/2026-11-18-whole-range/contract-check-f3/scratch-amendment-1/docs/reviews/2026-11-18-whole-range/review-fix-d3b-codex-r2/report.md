# D3 and D3b round-two review

## Verdict

**BLOCK** at 428cfa93. The original blocking inputs pass. Two adjacent paths still certify legacy stop evidence without proving it.

## Blocking findings

1. **Legacy stop uncertainty disappears across containment invocations.** Fail reset before phase-2 legacy cleanup. Containment stops a loaded legacy unit, then sees not-found while the recorded /legacy-orphan still says populated 1. It writes an unverified stop. Matching recovery now sees blank not-found twice, has no legacy mark, and prints and records “Scheduler stopped: verified” and “containment re-checked” without reading that group. Killing the first owner after legacy stop and before record rewrite produces the same false verified from a surviving begun record. With a phase-2 mark, matching recovery correctly refuses, but same-boot re-run 123/2 and distinct run 708/1 again certify the surviving populated group while legacy.123.1 still exists. Persist unresolved legacy evidence before either stop path and consult it across same-boot attempts before certifying absence. Clear it only after checking saved groups or on reboot. Preserve the attempt ownership checks. `.github/workflows/deploy.yml:472,509–525,591–600,738–745,1244,1281–1297,1363–1372`. [Same attempt](legacy-repeat.log) · [Re-run and next run](legacy-cross-1.log)
2. **The loaded proof can use two incompatible manager samples.** In either containment caller a legacy fragment stays present and a known group is empty. The first post-stop query reports loaded, active and PID 4242. At the next query the unit is not-found, inactive and PID 0. The loaded gate uses the first query while the stopped check omits LoadState from the second. Both scripts print and store verified even though neither sample proves a loaded stopped unit and the fragment prevents absence. Require loaded in the same sample that supplies the stopped state, PID and group. A scratch check of the added property denied the mixed sample and kept the stable loaded control green. `.github/workflows/deploy.yml:454–467,479–490,509–527,1226–1239,1251–1262,1281–1299`. [Counterexample](legacy-mixed_proof.log) · [Fix control](legacy-mixed_proof-same-sample.log)

## Notes

- The revised paging sentence in INCIDENT §6 promises a POST too broadly. Missing, different-owner and committed records get no POST although they lack a recorded 2xx. Qualify the sentence with “when recovery runs containment.” Say the POST can be repeated, since the 2xx is the response. [Trace](operator-audit.log)
- The history probes keep the same legacy cgroup populated during the same boot. The simulated reboot clears both the orphan and /run. Delayed recovery of the older owner still leaves the newer attempt's services alone.

## 1. Original blocking inputs

| Finding | Original trigger and outcome |
|---|---|
| Codex 1 | Dot-path POST on deploy reset failure and matching recovery now rejected. Dot-only and escaped dot cases also reject. Ordinary POSTs use both required curl options. [Replay](replay-urls.log) |
| Codex 2a and Carbon B1 | Populated group with not-found, also first blank then populated, now read and unverified. Repeat recovery omits re-checked on the populated path. [Codex](replay-own_groups.log) · [Carbon](replay-carbon-groups.log) |
| Codex 2b and Carbon B3 | Not-found with a retained regular fragment or dangling symlink fails the loaded proof. Both callers report unverified. [Codex](replay-own_groups.log) · [Carbon](replay-carbon-dangling.log) |
| Carbon B2 | Phase-2 removal and reload, then a canary failure, with either populated or unknown group. The marked attempt stays unverified on deploy and matching recovery. [Replay](replay-carbon-groups.log) |
| Codex 3 and Carbon N2 | Original blocked-final-output model now launches one TERM 1/grace 1 print and ends at 279 s. Unblocked repeat prints all five lines with that same call. [Replay](table-final_print.log) |

## 2. Must still catch

| Case | Original input rerun | Outcome |
|---|---|---|
| K1 | First/all handler clock reads fail in both scripts | **PASS** stop, marker, POST and contained record |
| K2 | Wait, query, enable, stamp and restart signals at modeled 790/795/805 s with armed read 126 | **PASS** one containment, no post-begun handler read |
| K3 | Populated prior group then blank and always blank, both scopes | **PASS** unverified |
| K4 | Brace/range endpoints, valid dashboard and POST | **PASS** no glob POST, valid calls use globoff |
| K5 | Stage/rename 17, 124, 137; read-back 124/137; successful rename with wrapper status 17 | **PASS** preserved status and containment, exact confirmation passes |
| K6 | A5 and both B8a cutovers | **PASS** edits follow stopped/disabled check |
| K7 | Zero-exit canary with NUL in accepted prefix | **PASS** query failure and no accepted line |
| K8 | Foreign run, 301, denied lock/mark helpers, real held inode | **PASS** owner fenced, no redirects, waiter proceeds only after release |
| K9 | Four signals before begin stage/rename with all earlier record kinds | **PASS** not-started and byte-identical prior state |
| K10 | Four signals after actual rename; first fresh read 126 while beginning | **PASS** one containment |
| K11 | Loaded populated group becomes not-found, both callers | **PASS** individual containments check and refuse; later recovery can lose this evidence, finding 1 |
| K12 | Empty known group, missing group, absent throughout, pending job, failed check, dangling fragment | **PASS** original controls; adjacent split state proof in finding 2 |
| K13 | Slow proof in fallback clock model and old cutoff control | **PASS** pgrep runs with TERM 10 s; old cutoff omits it |
| K14 | Early/capped marker and late endpoint in both scripts | **PASS** marker by 240, POST by 270, modeled late valid curl TERM 15 s |
| K15 | Successful commit rename with read 126 | **PASS** logs 126, contains |
| K16 | Distinct before/after cgroups with both population directions and empty control | **PASS** populated group prevents verification; both empty groups checked |
| K17 | Foreign beginning, exact or mismatched committing read, same-ID beginning forgeries | **PASS** success only for exact committing match |
| K18 | Earlier verified page, stale stored stop/marker, fresh failures and failed rewrite | **PASS** no second POST; re-checked requires fresh stop and marker in the original cases |
| K19 | Identical shared proof, promotion and print locations | **PASS** only prove sets and emits these results |
| K20 | Restore instruction and both custom-directory saver examples | **PASS** checked deploy activates; examples name ET season and picks dir |
| Appended | Commit survives read 126 and failed containment rewrite | **PASS** recovery reports what record says and directs §4 verification |

[Table replay](required_replay.py)

## 3. Changes and ordinary-state controls

- A successful loaded-scheduler deploy writes an empty 0600 mark before the first legacy mutation. Injected mark creation failure contains before phase-2 legacy cleanup. TERM during that stop leaves the mark. Once another owner takes over, delayed older recovery changes no services. After simulated reboot, fresh absence verifies. [Mark order](legacy-boundary.log) · [Attempt order](legacy-cross-1.log)
- A marked, physically empty orphan stays unverified after unload, as R4 directs. An unmarked absent scope verifies with a successful blank read. A failed group query on the same empty scope causes a stop attempt and unverified. [Empty marked case](legacy-cross-0.log) · [Blank/error control](legacy-blank_controls.log)
- UUID and project/slug spellings still POST. Percent-encoding an unreserved hyphen yields the same decoded slug, but the new explicit R1 restriction rejects that spelling in both callers. [URL replay](replay-urls.log)
- The final-print trace fits its reserve. The added mark also uses part of the existing 800 s deploy deadline. With a modeled 2 s commit staging call, the pre-mark revision starts it at 790 s and passes. The 4 s mark moves staging to 794 s and reduces its TERM cap to 1 s, so the candidate contains on timeout. The normal-speed success control passes. [Phase timeline](legacy-phase-budget.log)

## 4. Operator sentences

| Text | Worst literal outcome |
|---|---|
| INCIDENT §4 first confirm stop and boot link disabled | Hold precedes restore and file replacement. Independent starts remain within L8. |
| INCIDENT §4 then install restic, set credentials, drill and restore selected snapshot | The restore replaces pick, receipt and saver files. The reordered instructions require the hold before any command. |
| INCIDENT §6 on recovery whenever no verified 2xx is recorded | An unowned, missing or committed record sends no POST. Relying on it can delay a hand check. |
| INCIDENT §6 earlier 2xx with failed rewrite | Authorized containment can repeat the POST if its response was never recorded. Other recovery outcomes can skip containment entirely. |
| INCIDENT §6 2xx is the POST response | A provider response can occur without notifying Eric. |
| INCIDENT §6 check provider history and Eric's receipt | Both checks are read-only and prevent treating HTTP evidence as delivery. |
| CLAUDE L10 says the record committed | A surviving committed record can coexist with a stopped scheduler. This wording describes the record. |
| B8c verify old cron backup, stop both services, hold scheduler link through normal boot | Protects old cron from loss and prevents an old local boot start. The separate persistent timer can catch up. |
| B8c surrounding approved research deploy, PASS/HEAD, cron then A7/B8a | Live activation and cron installation remain ordered behind checking the current SHA. |

## 5. Mutations

| Scratch mutation | Outcome in targeted existing tests |
|---|---|
| All 17 prior reviewer edits | Detected, including the longer print grace that survived round one |
| Drop the mark check in the first absence sample | **Survived.** Direct probe showed the legacy stop skipped. |
| Drop the mark check in the proof | **Survived.** Direct probe showed a marked not-found result with an empty group verifies. |
| Write the mark after the stop | **Survived.** Direct probe showed the phase-2 stop precedes the failing mark write. |
| Drop path-as-is, allow percent escapes, omit the loaded gate | All three detected |
| Ignore pre-stop group error, new group, stop status, boot link, or rounding | All five detected |

25 of 28 edits detected. The three survivors are test-coverage notes. [Mutation log](mutations.log) · [Direct controls for the survivors](legacy-mark-checks.log)

## What I ran

Local diff, archived probe inputs, direct workflow shell shims, Bash elapsed-time models and redirected advisory locks. Wrapped focused pytest collected 521 tests, including slow cases and the revised B8a assertion. 520 passed. The existing web saver socket-wiring test failed at a sandbox-denied loopback bind. The targeted mutation wrapper finished separately in six minutes. [Focused log](focused.log) · [Slot completion](PYTEST-DONE)

## What I did not run

Full fast suite, network, SSH/Actions, provider delivery, physical reboot or real systemd/cgroupfs/cron.

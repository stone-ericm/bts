# F2 round-two review

## Verdict

**BLOCK** on `3b7d5a18` against `a4e8278d`. Both findings below reproduced through the real code on temporary files.

## Blocking findings

1. **The mismatch receipt can overwrite WARN attention state and cost the next slate its digest.** Use a regular digest path `health_state/digest.json` and set the supported `warn_attention_state` override to `health_state/digest.json.dated_used_mismatch.json`. On July 15, 2027, present dated `used`, flag `active` and the first day of `calibration_drift`. The candidate writes attention state, sends the mismatch, then replaces the same regular file with its receipt. On July 16, use dated `active` and repeat `calibration_drift`. The base remembers day one and sends the repeated WARN. The candidate restarts the streak at one and sends nothing. At `src/bts/health/runner.py:349`, check whether the derived receipt names the same file as the configured attention state. If it does, dispatch without storing the receipt. Both runs used `probe_delivery.py` with the same overrides. [Base](probe-collision-base.log) · [Candidate](probe-collision-candidate.log)
2. **B8a tells the operator to retain a wrong flag and can cost the next slate a pick (R64).** On August 19, 2027, MLB shows an earned unused saver. An erroneous dated `used` row and an `active` flag generate the F2 DM. Before the recipient reads it, another operator trusts the row and marks the flag used in the dashboard. The recipient checks MLB and runs the exact status command, which now shows both records as `used`. The new B8a sentence explicitly says the flag is right and to leave it. At August 20's accepted automatic streak 9 with the shipped policy and hit probabilities .79/.80, the retained flag yields `skip`. The verified `active` flag yields `single`. `probe_operator.py` exercised the real dashboard transition, status, automatic contest read and MDP policy. In `docs/ops/2027-season-start.md:31`, replace the final conditional with “If MLB shows the saver has not been used, compare the current flag and any still-effective dated observation independently with MLB. Leave the flag as it is only if it agrees with MLB. If it differs, have Eric verify the current ET season and prior entries and reconcile the flag before the next automatic pick. For a wrong dated observation still in effect, use B8a’s prior-entry and pick-timing checks before correcting it.” [Operator replay](probe-operator.log)

## Edits

The B8a replacement is part of finding 2. No separate wording edits.

## Notes

- The inherited attention parser can reject a persisted mismatch row with `streak: "bad"` before either dispatch, as documented in Amendment 2 item 5. The replay reproduced it.
- The old Carbon B1 replay used a live coverage path containing `/oom/`. The base itself then sent the inherited OOM alert. The replay also encountered two sandbox-denied Unix socket binds. The new comparison used identical neutral paths on both revisions and verified the real coverage and runner results.
- The sandbox declined the requested priority change. Commands still ran with their timeouts.
- One standalone test retry started while a previous timed retry was still running. They overlapped for less than a second.

## 1. Scope and base comparison

Reviewed the two specified diffs. Round one touched two production files and the two authorized test files. Round two updated the same production files, the two B8a additions, the F2 tests and the adopted Carbon regression. The source, tests and operations docs at the review head matched `3b7d5a18`. No tracked files changed during review. The independent `git archive a4e8278d src` replay printed `bts.__file__` on both sides. All **540** pairs matched in raw alert order and content, captured DMs and status bytes. The 12 scheduler cases add actual health-enabled and health-disabled paths with configured, absent and empty recipients, in both live and historical modes. [Paired replay](probe-parity.log)

## 2. Comparison and round-one closure

`probe_comparison.py` ran **1,882** paired cases. The new WARN appeared only for in-effect dated `used` and a current-season `active` or `not_earned` flag. Every earlier alert matched the base, including coverage before the new comparison and the full 0–400 sweep of dated `not_earned` with a FIFO flag. Mode-000, overlong and untraversable links, stale and boolean season, invalid UTF-8/JSON, list, directory, FIFO, verified integer and recursion limits all completed. The parity grid covered expired, another-date, legacy and invalid-v3 observations. December 31 after UTC rollover, January 1 ET and C1's prior-slate clock read the expected season. Replacing either input after midnight changed the result as documented. [Comparison replay](probe-comparison.log)

| Round-one item | Result on round two |
| --- | --- |
| Codex B1 and Carbon B1, flag stat errors | Closed. The original Codex comparison completed 714 checks. Carbon’s historical stat cases passed. The original live comparison had the inherited OOM path issue above. The independent replay and the adopted regression preserved the collected alert in both modes. |
| Codex B2, symlink and hard-link receipts | Closed. The original runner probe stopped at its old assertion when the symlink case preserved the flag. The new replay confirmed both aliases and the two non-regular receipt cases sent without replacing the entry or changing saver bytes. |
| Codex B3, no receipt basename | Closed. The new runner replay used `/`, sent the existing digest first, and stored the acknowledged mismatch at the default path. |
| Codex B4, dashboard Undo between verification steps | Closed for the original sequence when the operator follows the new coordination and immediate refresh. The original text replay omits the new refresh and still invokes the stale command. The new replay included the refresh and stopped. Finding 2 covers a separate sequence with the wrong flag already visible at the fresh check. |
| Codex edit, picks-directory guidance | Closed. The revised live DM explicitly directs the operator to use the scheduler’s picks directory in B8a. The B8a heading identifies D. |
| Carbon N1 and Codex decoder note | Closed. The 200,000-level input raised `RecursionError`, and dropping the handler failed an actual test. |
| Carbon N2, flag read before dated `not_earned` warning | Closed for an unguarded read. The scratch check below distinguishes it from reordering two disjoint guarded branches. |
| Codex FIFO test note | Closed. The test arms each mode separately and raises an exception the reader cannot swallow. The independent comparison also bounded each read separately. |

## 3. Delivery

`probe_delivery.py` exercised `run_all_checks`, the real attention policy, and both dispatches. A 1,000-grapheme existing digest and its receipt matched the control byte for byte and went first. Separate same-date dedup worked in both incident orders and both modes. Consecutive processed dates advanced the label. The label preceded the single historical pointer. Either transport raising still allowed the other DM. A blank ID left the mismatch eligible and preserved the legacy digest's behavior. The custom and default receipt paths worked except for the overlap in finding 1. [Delivery replay](probe-delivery.log)

## 4. Operator text and B8a

`probe_operator.py` rendered both messages with captured delivery and walked the status, dashboard and saver commands offline. The original wrong-row/correct-flag case preserved the flag. A corrected same-day observation silenced the WARN. Updating the flag and MLB before the required recheck stopped a stale write. A verified used-saver repair the next morning changed later automatic reads for either flag and preserved the automatic pick already saved and the expired observation. The historical pointer led to inspection only. A January status read left the new flag intact. Finding 2 is the additional ordinary dashboard sequence. [Rendered messages and commands](probe-operator.log)

## 5. Tests and scratch checks

Ran all **66** files from the requested search, including F2, plus **14** from the contract index. The completed runs yielded **5,528 passed, 12 skipped and 1 xfailed**. One case in `test_web_saver.py` stopped at the sandbox's loopback bind. Two timed batches were split. A standalone replay showed the older `test_wr_e_texts.py` hitting an unmocked schedule fallback and waiting on retries. All 65 cases passed when the review harness supplied an offline HTTP failure at that boundary. [Search index](test-index.log) · [Additional files](extra-test-index.log) · [Offline replay](tests-08-offline-text.log) · [Socket test](tests-11.log)

The sole changed base test file is `tests/test_contest_dated_cs4a.py`. Its five added lines affect exactly the 28 authorized `used` combinations and assert one bounded read. The existing no-read trap and alert assertions remain. [Review diff](review.diff)

`probe_scratch.py` passed 786 control tests. The suite caught all 12 behavior-changing checks covering the missing key, normal flag reader, wall-clock year, WARN on a used flag, merged digest, mismatch first, shared digest status, missing recursion and OS error guards, premature unguarded flag read, missing acknowledgement and missing default receipt. The additional guarded reorder passed all 786 cases because the `used` and `not_earned` branches are disjoint. The unmodified Carbon scratch replay detected its first five changes, including its extra FIFO timeout, then stopped on an assertion that matched the old dispatch source. The new scratch runs exercised all remaining variants against round two. [Scratch driver](probe_scratch.py) · [Original Carbon run](round1-carbon-scratch.log)

## 6. Standing rows

The comparison and receipt cases covered malformed and non-regular input, ET midnight and January rollover. An interrupt after the second transport returned and before the receipt left the existing digest receipt and account bytes intact. The next invocation retried only the mismatch and then deduplicated it. Interrupting the first transport stopped the second until rerun. Thrown and unconfirmed sends had the receipt and retry behavior described above. [Delivery replay](probe-delivery.log)

## 7. Other changes

`git diff --check` was clean. No further findings in the changed code or text.

ran: both diffs, all five original round-one probe scripts, independent paired comparisons and EOD and operator replays, 80 requested or indexed test files, and 13 scratch changes plus the control; did not run: the full fast suite, live MLB or Bluesky calls, or the socket assertions beyond the denied bind.

# Change set H, round 2

Branch `fix/wr-h`. Round-2 commit `84805852`. The starting HEAD was `b8b89b37`. The difference from the round-1 build at `813935bc` was review documentation only.

## Tests first

| Item | Test | Initial result |
|---|---|---|
| R1 | `test_r1_malformed_paired_base_forces_single` and `test_r1_scheduler_delivers_single_for_paired_scalar` | All 12 malformed resolver cases returned double on the round-1 source. The scheduler also saved and sent a double. Both healthy controls passed. |
| R2 | `test_r2_fresh_selection_uses_fixed_horizon` | Passed on the round-1 source. Temporarily substituting the base horizon produced double with one effective day instead of single with four. |
| R3 | `test_r3_fresh_selection_dispatch_precedes_loader` | Passed on the round-1 source. Temporarily moving the loader ahead of dispatch failed on the first recorded call, even though the loader exception still degraded to single. |
| R4 | `test_r4_longest_walk_flags_incomplete_policy_fields` | Missing effective days, source, and artifacts went unreported on the round-1 source. The non-MDP source control already passed. |

Both scratch changes were reverted immediately. The 16 requested focused suites passed across three sequential bounded runs (109 + 168 + 254 = 531 tests). The environment rejected the niceness adjustment. Each run continued under the configured timeout. The whole fast suite and shared wrapper were left for their existing owners.

## Archived-base comparison

`test_legacy_parity_against_archived_base` extracted `git archive 0a2bec10 src` and asserted the imported `bts.__file__` for both trees. Every decision field and direct bar matched over 640,451 cases, including all 1,511 guard cells. Raw days covered 0–271 and effective days covered 0–180. The test uses the existing targeted grids and took about 15 seconds including the rest of its batch.

Matching SHA-256 `5ca5e0fe6fec880363ef2cee5925c81750485594c59c8a97621377d19973f35d`.

## Season walk

The full walk exited 1 on the 612 existing warnings. All 4,599 original rows, all warnings, and the year-boundary checks matched G. The 219 added rows yielded 207 singles (23 preseason), doubles on September 24–26, and nine post-finale legacy skips. All 210 plays had valid `chosen_on` and the expected selected slots.

The fresh JSON matched the decompressed `walk-h.json.gz` byte for byte. The command output also matched the transcript in `walk-h.txt`. Both saved H results were retained. The JSON SHA-256 is `1bd04d1eef0ba76efda4566171a4fb40fba8b2b1e411c8e37c832fe684b96e72`.

## Contract notes

The unchanged loader converts the serialized horizon to `int`. R1 therefore supplies `True` explicitly in the decoded decision context while retaining the exact re-bound tail and short table. The scheduler scalar test obtains its malformed decoded table directly from the paired files. The contract lists no R5 findings.

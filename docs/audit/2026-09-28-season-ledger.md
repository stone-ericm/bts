# W1.1 season ledger, Phase 1 — build record (2026-09-28)

**Exposure:** register row **X-19**, predeclared and pushed at `df3347e` before any real data was compiled. This memo reports only what X-19 lists: occurrence, row-kind, commit, history, entry, match and outcome-status counts; the local-vs-contest disagreement count (the C-03 check); and the per-rule recipe totals. It reports **no rates, hit percentages or model comparisons**. Any analysis of the ledger's outcomes is a new read with its own row.

**Authority:** spec `docs/superpowers/specs/2026-09-28-season-ledger-design.md` (v4) · plan `docs/superpowers/plans/2026-09-28-season-ledger-phase1.md` (rev 4, Task 13).

## 1. Identity

| Item | Value |
|---|---|
| Code | `df3347eb525570aecb328a306fb4d23f8c4ce82d` (the X-19 commit; the ledger code is identical to `54cf9b0`, reviewed by a fresh reviewer and by Codex code r1–r3, SIGN) |
| Builder | `season-ledger-phase1/3` · Python 3.12.6 · pyarrow 23.0.1 (the box's production venv) |
| Environment lock | `54a0327bcd0389c17ad6573f5570d8ceddaccc636fe673d20322cc6156726c97` (production `uv.lock`) |
| Evidence bundle | `data/hetzner_results/season_2026_ledger_evidence/v1` · manifest sha256 `6ebb09536366358d8fec2e22da3f4746a5e7e1d3ee45e63643949b10d1814a63` · acquired 2026-09-29T02:27:33Z (22:27 ET 9/28) from the frozen W0.7 snapshot `final-20260928` |
| Rules fingerprint | `5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f` — as predeclared in X-19 |
| Accepted build | `data/validation/season_2026_ledger/df3347eb525570aecb328a306fb4d23f8c4ce82d-20260929T022806Z-compile2-dd430abe/` — `ACCEPTED.json` written 2026-09-29T02:29:08Z after the twin build matched byte for byte (six files) |
| Backup | restic archive snapshot `92ecdc868e83087b41c019bac5d3b3188a9aa32d673bbdef17063a77ae95572b` — the backed-up manifest is identical to the sealed one and all 3,572 bundle members are present |

`data/validation/` is in no backup set; the build is reproducible byte for byte from bundle v1 and code `df3347e` (two independent compiles matched).

## 2. Runs (all through `run.sh`, each a transient `systemd-run --user` unit; nothing deployed, nothing restarted)

| Step | Run id | Result |
|---|---|---|
| Acquire | `20260929T022404Z-acquire-3758d7e8` | `sealed …/manifest.json`, `EXIT=0` |
| Compile twice, compare, accept | `20260929T022806Z-compile2-dd430abe` | both compiles passed every source check and output check; `accepted …: 6 identical files; rules fingerprint as predeclared`, `EXIT=0` |
| Back up and verify | `20260929T022938Z-backup-e7698934` | archive backup ok (9.86 MB added); `snapshot 92ecdc86…: manifest identical; all 3572 bundle members present`, `EXIT=0` |

**Failed attempts:** none.

## 3. Input coverage

- **Manifest:** 3,572 entries — picks tree 869; static captures 2,514 (173 rounds, 2,335 units, 2 players, 4 files of the 9/27 grab); schedules 187; logs 2.
- **Missing inputs:** none. All 187 MLB schedule fetches succeeded; all 187 schedules are complete.
- **Occurrences:** emitted 332,478 · excluded 300 · quarantined 3.

| Exclusion reason | Count |
|---|---|
| appledouble_resource_fork | 14 |
| corroboration_only (the two logs) | 2 |
| documentation | 1 |
| no_records (readable empty containers) | 5 |
| not_used_phase1 (grab squads) | 1 |
| runtime_marker | 1 |
| shadow_model_out_of_scope | 146 |
| skip_policy_shadow_out_of_scope | 16 |
| slate_binding_out_of_scope | 105 |
| state_snapshot_not_a_record | 9 |
| unrecognized_path | 0 |

- **Quarantine:** 3 × `slot_missing_identity_or_result`, the three contest-ledger lines with a null `playerId` known from the structure probe. Their rounds recur in neighbouring lines.
- **Type-mismatch rows:** none, in any source kind.

| Disposition | Count |
|---|---|
| canonical_decision | 92 |
| canonical_selection | 274 |
| not_selected | 5 |
| day_evidence | 167 |
| history_evidence | 1,509 |
| contest_evidence | 117,901 |
| lookup | 212,529 |
| reported_attempt | 1 |

## 4. Ledger rows

| Row kind | Count |
|---|---|
| selection | 274 |
| skip_day | 25 |
| unfinalized_day | 3 |
| unobserved_day | 7 |
| contest_only | 12 |

Selections by status:

| Field | Counts |
|---|---|
| finalization | decision 123 · pick_file_only 151 · unresolved 0 |
| commit_status | committed_evidenced 262 · unconfirmed 12 · conflicted 0 |
| history_status | known_incomplete 159 · unknown 115 · complete 0 (never inferred) |
| entry_status | confirmed 230 · unknown 44 |
| game_eligibility | unknown 274 |

Outcome status (selections and contest-only rows): graded 231 · match_ambiguous 10 · unknown 39 · unmapped 6.

## 5. Contest matching

| Match | Count | Reasons |
|---|---|---|
| evidenced | 107 | unit_capture 106 · unit_capture_no_local_selection 1 |
| inferred | 124 | pick_time_team_single_scheduled_game 124 |
| ambiguous | 5 | selection_game_pk_unrecorded 2 (the 3/29–3/30 files) · team_schedule_not_unique 3 |
| unmapped | 6 | no_unit_capture_contest_only 6 |

No `player_unknown`, `team_schedule_missing` or `team_schedule_incomplete`. Only `evidenced` and `inferred` matches transfer a grade.

## 6. The C-03 check

`local_vs_contest_disagreement`: **true 2** · false 175 · not comparable 97 (no comparable slot-level values on both sides).

The W1.5/C-03 register already names two true hits that local grading flipped to misses (5/10, 8/20). This count is consistent with that; the rows themselves were not read here.

## 7. Recipe reconciliation (historical membership unverified)

| Rule | Files | Primary | Legs | Window | Primaries | Legs | Published | Fit |
|---|---|---|---|---|---|---|---|---|
| S1 | F1 | G1 | G1 | 3/29→9/10 | 142 | 113 | 141/82 | none |
| S2 | F1 | G1 | G2 | 3/29→9/10 | 142 | 82 | 141/82 | legs_only |
| S3 | F1 | G2 | G1 | 3/29→9/10 | 111 | 113 | 141/82 | none |
| S4 | F1 | G2 | G2 | 3/29→9/10 | 111 | 82 | 141/82 | legs_only |
| S5 | F2 | G1 | G1 | 3/29→9/10 | 250 | 207 | 141/82 | none |
| S6 | F2 | G1 | G2 | 3/29→9/10 | 250 | 152 | 141/82 | none |
| S7 | F2 | G2 | G1 | 3/29→9/10 | 194 | 207 | 141/82 | none |
| S8 | F2 | G2 | G2 | 3/29→9/10 | 194 | 152 | 141/82 | none |
| T1 | F1 | G1 | G1 | →9/13 | 145 | 116 | 191/157 | none |
| T2 | F1 | G1 | G1 | →9/14 | 146 | 117 | 191/157 | none |
| T3 | F1 | G2 | G2 | →9/13 | 114 | 85 | 191/157 | none |
| T4 | F1 | G2 | G2 | →9/14 | 115 | 86 | 191/157 | none |
| T5 | F1 | G3 | G3 | →9/13 | 145 | 116 | 191/157 | none |
| T6 | F1 | G3 | G3 | →9/14 | 146 | 117 | 191/157 | none |
| T7 | F1 | G4 | G4 | →9/13 | 151 | 118 | 191/157 | none |
| T8 | F1 | G4 | G4 | →9/14 | 152 | 119 | 191/157 | none |
| T9 | F2 | G1 | G1 | →9/13 | 256 | 212 | 191/157 | none |
| T10 | F2 | G1 | G1 | →9/14 | 258 | 214 | 191/157 | none |
| T11 | F2 | G2 | G2 | →9/13 | 200 | 157 | 191/157 | legs_only |
| T12 | F2 | G2 | G2 | →9/14 | 202 | 159 | 191/157 | none |
| T13 | F2 | G3 | G3 | →9/13 | 256 | 212 | 191/157 | none |
| T14 | F2 | G3 | G3 | →9/14 | 258 | 214 | 191/157 | none |
| T15 | F2 | G4 | G4 | →9/13 | 264 | 216 | 191/157 | none |
| T16 | F2 | G4 | G4 | →9/14 | 266 | 218 | 191/157 | none |
| T17 | F3 | G1 | G1 | →9/13 | 259 | 215 | 191/157 | none |
| T18 | F3 | G1 | G1 | →9/14 | 261 | 217 | 191/157 | none |
| T19 | F3 | G2 | G2 | →9/13 | 201 | 158 | 191/157 | none |
| T20 | F3 | G2 | G2 | →9/14 | 203 | 160 | 191/157 | none |
| T21 | F3 | G3 | G3 | →9/13 | 259 | 215 | 191/157 | none |
| T22 | F3 | G3 | G3 | →9/14 | 261 | 217 | 191/157 | none |
| T23 | F3 | G4 | G4 | →9/13 | 322 | 268 | 191/157 | none |
| T24 | F3 | G4 | G4 | →9/14 | 324 | 270 | 191/157 | none |

**Labels (spec §8, I9):**
- **9/11 scorecard: `hypothesis`.** A rerun of prose is always a hypothesis. No pre-declared rule reproduces both 141 and 82. S2 and S4 reproduce the legs only; S1/S2's 142 primaries is the nearest total.
- **9/14 naive tally: `unrecoverable`.** No pre-declared rule reproduces both 191 and 157; T11 reproduces the legs only.

The rules were frozen before any count, and the fingerprint matched. No rule may be added or tuned now that these totals are known.

## 8. Known limits

- **Phase 1 scope:** production stream only. Shadow v1/v2, the skip-policy shadow, D8 research, cached-feed grading, MLB's current record, recipe epochs and slate binding are later increments (spec §12).
- **Streak saver:** `saver_available_before` is always null. No evidenced initial state exists; the single saver transition row is reported as an attempt (`written`).
- **Eligibility:** evidence counts only from BTS unit captures timed before lock, and captures exist only from 2026-07-04. Refusal archives exist nowhere, so all 274 selections read `unknown`.
- **History:** `known_incomplete` (159) marks a lineup-evolution version whose content no retained record holds. `complete` is never inferred.
- **Local grading (spec §12, for W1.5):** production's `grade_pick_in_feed` returns `miss` for a walks-only or did-not-play batter, and for any pre-suspension plate appearance, where BTS calls these a Pass. Local slot results are recorded as written and are never authoritative.

## 9. Reproduce

```bash
# on the box, from ~/projects/bts, with the code of df3347e (git archive → /tmp/ledger_code_<sha>):
.venv/bin/python /tmp/ledger_code_<sha>/scripts/audit/build_season_ledger.py compile \
  --bundle data/hetzner_results/season_2026_ledger_evidence/v1 --out <new directory> --code-sha <sha>
```

The six outputs match the accepted build byte for byte. The bundle is restorable from restic snapshot `92ecdc86…`; restic stores paths as `/data/hetzner_results/season_2026_ledger_evidence/v1/...`.

"""Mutation check for the 2026 framing test's code (the lead, 2026-10-09; revised for code review round 2).

What it shows: for each listed rule, the tests aimed at it fail when that rule's source span is broken. It does not
show that every rule of the design has a test, that the tests were written first, or anything about the stubbed
integration boundaries (review c1, Part 2). Each mutant replaces exactly one source span in a scratch worktree at the
given commit, runs only the tests aimed at it, records KILLED (pytest rc 1, with the first failing test and assertion),
SURVIVED (rc 0) or ERROR (any other rc), restores the file, and checks the scratch tree is clean before the next
mutant. The script exits nonzero if any mutant survives, errors, has a stale anchor, or leaves the tree dirty.

Usage (from a checkout of the commit under test):
    python3 docs/audit/2026-10-09-c2-framing-2026-evidence/mutation_check.py <commit> <scratch-worktree-path> <ledger.tsv>
"""
import subprocess
import sys
from pathlib import Path

F26 = "scripts/audit/c2_framing/f26.py"
BB = "src/bts/simulate/backtest_blend.py"
RULES = "tests/scripts/c2_framing/test_f26_rules.py"
RUN = "tests/scripts/c2_framing/test_f26_run.py"
HOOK = "tests/scripts/c2_framing/test_f26_hook.py"

MUTANTS = [
    ("M1 allowance source not Eric", F26, 'if not m or cells[3].split()[:1] != ["Eric"]:\n        return None\n    cap, budget',
     'if not m:\n        return None\n    cap, budget', RUN, "allowance"),
    ("M2 allowance cap not the launcher's", F26, 'if allow["cap"] != ledger.CAP_H:', "if False:", RUN, "allowance"),
    ("M3 seed order skipped (round 4: the loop also covers the current seed)", F26,
     "    for s in SEEDS[:SEEDS.index(seed) + 1]:", "    for s in SEEDS[SEEDS.index(seed):SEEDS.index(seed) + 1]:", RUN,
     "one_at_a_time"),
    ("M4 run's window refusal removed", F26, "        refuse_inside_the_window()                 # the last check",
     "        pass                                        # the last check", RUN, "refuses_inside_the_window"),
    ("M5 first-unit stop removed", F26, 'if len(units) == 1 and cpu > allow["first_unit_stop"] * 3600:', "if False:",
     RUN, "expensive"),
    ("M6 manifest calendar unchecked", F26, '"calendar": man.get("calendar") == calendar,', '"calendar": True,', RUN,
     "inconsistent_records or another_calendar"),
    ("M7 rank-1 calendar unchecked", F26, "if sorted(dates) != calendar or len(set(dates)) != len(dates):",
     "if False:", RUN, "damaged_evidence or aggregate_rederives"),
    ("M8 as-of admits the same date", F26, 'side="left"))', 'side="right"))', RULES, "doubleheader or framing_by"),
    ("M9 fielding side swapped", F26, 'np.where(out["is_home"].astype(bool).to_numpy(), "away", "home")',
     'np.where(out["is_home"].astype(bool).to_numpy(), "home", "away")', RULES, "posted_arm or projected_arm"),
    ("M10 projection admits same-day games", F26, "if g[0][0] < key[1]]", "if g[0][0] <= key[1]]", RULES, "same_day"),
    ("M11 unknowns dropped before the window", F26, "if g[0][0] < key[1]]",
     "if g[0][0] < key[1] and g[1] is not None]", RULES, "window_is_taken_first"),
    ("M12 L = 0 passes", F26, "L > 0 and positive_seeds", "L >= 0 and positive_seeds", RULES, "exactly_zero"),
    ("M13 any C entry counts", F26, 'and ap[0]["code"] == "2":', 'and any(e["code"] == "2" for e in ap):', RULES,
     "moved_to_catcher"),
    ("M14 pins row binds other pins", F26, "if m.group(2) != S.pins_digest(pins):", "if False:", RUN, "inputs_row"),
    ("M15 resumed flag unchecked", F26, "if not pd.api.types.is_bool_dtype(flag) or bool(flag.isna().any()):",
     "if False:", RULES, "resumed_flag"),
    ("M16 hook result ignored", BB, "            day_data = transformed\n", "            pass\n", HOOK,
     "override_reaches"),
    ("M17 strict mode ignored", BB, "                if strict_predict:\n                    raise",
     "                if False:\n                    raise", HOOK, "strict"),
    ("M18 hook shape unchecked", BB, "            if not (isinstance(transformed, pd.DataFrame)",
     "            if False and not (isinstance(transformed, pd.DataFrame)", HOOK, "shape"),
    ("M19 preparation scans unpinned feeds", F26, "        for pk in games[s]:\n",
     "        for pk in games[s] + sorted(int(q.stem) for q in (raw_dir / str(s)).glob('*.json') "
     "if int(q.stem) not in games[s]):\n", RUN, "no_unpinned"),
    ("M20 run not strict", F26, "predict_day_transform=transform, strict_predict=True)",
     "predict_day_transform=transform, strict_predict=False)", RUN, "end_to_end"),
    ("M21 units.json not reconciled", F26, "    if S._canon(units_file) != S._canon(units):", "    if False:", RUN,
     "units_file_that_differs"),
    ("M22 seed count unchecked", F26, "x.shape[0] != len(SEEDS)", "x.shape[0] < 1", RULES, "anything_but_ten"),
    ("M24 prep row may predate X-37", F26, "    if A.row_cells(earlier, row_id):", "    if False:", RUN,
     "preparation_read or inputs_row"),
    ("M25 strict day scores unchecked", BB, "                if strict_predict:\n                    _strict_scores(",
     "                if False:\n                    _strict_scores(", HOOK, "missing_scores or missing_a_row"),
    # round 2: review c1 findings 1-7
    ("M26 F1 reliever scores unchecked", BB, "    if strict:\n        _strict_scores(p_reliever",
     "    if False:\n        _strict_scores(p_reliever", HOOK, "reliever_score or wrong_length"),
    ("M27 F1 score index unchecked", BB, "        if not scores.index.equals(index):", "        if False:", HOOK,
     "missing_a_row"),
    ("M28 F1 score length unchecked (defense in depth: still refused downstream; killed by message)", BB, "        if values.shape != (len(index),):", "        if False:", HOOK,
     "wrong_length or array_of_scores"),
    ("M29 F2 label unchecked", F26, "        if int(h) != t[1]:", "        if False:", RUN, "coherently_wrong"),
    ("M30 F2 batter-game date unchecked", F26, "        if t is None or t[0] != d:", "        if t is None:", RUN,
     "coherently_wrong or another_date"),
    ("M31 F2 basis unchecked", F26, 'not bool((part["p_game_hit_basis"] == S.BASIS).all())', "False", RUN,
     "coherently_wrong"),
    ("M32 F2 ten-row cap unchecked", F26, ".size() > 10).any()):", ".size() > 99).any()):", RUN, "coherently_wrong or eleventh"),
    ("M33 F2 duplicate batter-game allowed", F26, "        if (d, b, g) in seen:", "        if False:", RUN,
     "coherently_wrong or within_ten_rows"),
    ("M34 F3 catcher not re-derived", F26, "        if rc != cid:", "        if False:", RUN, "rederives_the_catcher"),
    ("M35 F3 side-games not re-derived", F26, "    if {k: int(r.n_pa) for k, r in got.items()} != truth.side_games:",
     "    if False:", RUN, "rederives_the_catcher or moved_to_another_date"),
    ("M36 F3 counts unchecked", F26, '    if recorded.get("counts") != {"side_games": side_games, "pa_rows": pa_rows}:',
     "    if False:", RUN, "count_says_otherwise or recorded_counts or counts_edited_in_both"),
    ("M37 F3 identified ids unchecked", F26, '    if recorded.get("identified_ids") != sorted(ids):', "    if False:",
     RUN, "recorded_counts or ids_edited_in_both"),
    ("M38 F3 empty posted coverage allowed", F26, '    if arm == "A_posted" and side_games["identified"] == 0:',
     "    if False:", RUN, "count_says_otherwise"),
    ("M39 F3 values not recomputed", F26, "            if not _same(float(r.value), want):", "            if False:",
     RUN, "forged_expectation"),
    ("M41 F4 pre-2019 catcher ids allowed", F26,
     '    if bool((early & df["fielding_catcher_id"].notna()).any()):', "    if False:", RULES + " " + RUN,
     "before_2019 or pre_2019"),
    ("M42 F5 prepare's row gate removed", F26, "    if problem:\n        raise SystemExit(f\"refusing: {problem}\")\n"
     "    inv, problem = source_inventory(A.REPO, adm[\"exposure_commit\"])",
     "    if False:\n        raise SystemExit(f\"refusing: {problem}\")\n"
     "    inv, problem = source_inventory(A.REPO, adm[\"exposure_commit\"])", RUN,
     "preparation_function or prepare_command"),
    ("M43 R2-1 the inventory's paths unchecked", F26, "        if Path(given).resolve() != Path(inv[key]).resolve():",
     "        if False:", RUN, "but_the_inventorys"),
    ("M44 F5 historical pins unchecked", F26, "    return f\"the historical pins {bad} are not the screen's\" if bad else None",
     "    return None", RUN, "historical_pins"),
    ("M45 F5 source set unchecked", F26, "    if len(got) != len(set(got)) or set(got) != want:", "    if False:", RUN,
     "source_manifest or other_games_with_consistent"),
    ("M46 F5 table coverage unchecked", F26,
     '    return None if got == want else "the starter-proxy table does not cover', '    return None if True else "the',
     RUN, "cover_exactly"),
    ("M47 F6 launcher receipt unchecked", F26,
     "    problem = launcher_problems(d, seed, man, res, C1_DIR if c1_dir is None else c1_dir, allow[\"budget\"])",
     "    problem = None", RUN, "clean_launcher_receipt or clean_receipt"),
    ("M48 F6 nonzero rc accepted", F26, '    if problems or not (t.get("result") == "exit" and t.get("rc") == 0):',
     '    if problems or not (t.get("result") == "exit"):', RUN, "clean_launcher_receipt"),
    ("M49 F6 CPU bound unchecked", F26, 'if not (_finite_nonneg(total) and total <= t["cpu_seconds"]',
     'if not (True', RUN, "clean_launcher_receipt"),
    ("M50 F6 already-run seed launched", F26, "        if root.is_dir() and any(p.is_dir() for p in root.iterdir()):\n"
     "            raise SystemExit(f\"refusing: seed {seed} already has a run",
     "        if False:\n            raise SystemExit(f\"refusing: seed {seed} already has a run", RUN,
     "already_has_a_run"),
    ("M51 F7 batters' positions optional", F26, "_positions_ok(ap, required=batted)", "_positions_ok(ap, required=False)",
     RULES, "malformed_non_starting"),
    ("M52 F7 non-starters' ids unchecked (its bench-record case is accepted under it)", F26, "        if not _pos_int(pid):\n            bad_player = True\n"
     "            continue\n        batted", "        if False:\n            bad_player = True\n            continue\n"
     "        batted", RULES, "malformed_non_starting"),
    ("M53 F7 PA game ids coerced", F26, "    if not (pd.api.types.is_integer_dtype(col) and not pd.api.types.is_bool_dtype(col)):",
     "    if False:", RULES, "exact_positive"),
    ("M54 F7 table records unchecked", F26, "        problem = _record_problem(r)", "        problem = None", RULES,
     "inconsistent_table"),
    ("M55 F7 table sides unchecked", F26, '        if sorted(r["fielding_side"] for r in rs) != ["away", "home"]:',
     "        if False:", RULES, "inconsistent_table or two_away_records"),
    # arithmetic the review listed (Part 2, item 4)
    ("M56 five rates become four", F26, "MIN_RATES = 5 ", "MIN_RATES = 4 ", RULES, "five_nonmissing or framing_by"),
    ("M57 a tenth slot", F26, 'SLOTS = frozenset(f"{k}00" for k in range(1, 10))',
     'SLOTS = frozenset(f"{k}00" for k in range(1, 11))', RULES, "canonical_slot"),
    ("M58 tie goes to the earlier start", F26, "max((c for c in count if count[c] == top), key=lambda c: latest[c])",
     "min((c for c in count if count[c] == top), key=lambda c: latest[c])", RULES, "tie"),
    ("M59 another generator seed", F26, "BOOTSTRAP_SEED = 20261009", "BOOTSTRAP_SEED = 20261010", RULES,
     "disposition_quantities or block_resampling or continuous_deltas"),
    ("M60 another quantile method", F26, 'np.quantile(means, 0.1, method="linear")', 'np.quantile(means, 0.1, method="lower")',
     RULES, "disposition_quantities or block_resampling or continuous_deltas"),
    ("M61 five seeds suffice", F26, "SEEDS_POSITIVE_MIN = 6", "SEEDS_POSITIVE_MIN = 5", RULES, "six_positive"),
    ("M62 zero seeds count positive", F26, "positive_seeds = int((d > 0).sum())", "positive_seeds = int((d >= 0).sum())",
     RULES, "zero_seed"),
    ("M63 non-finite deltas allowed", F26, " or not bool(np.isfinite(x).all()):", ":", RULES, "non_finite"),
    ("M64 block resampling with another generator seed", F26,
     "    starts = np.random.default_rng(BOOTSTRAP_SEED).integers(", "    starts = np.random.default_rng(1).integers(",
     RULES, "block_resampling"),
    ("M65 featureless 2026 rows not stopped early", F26, "    if featureless:\n", "    if False:\n", RUN,
     "every_model_feature_missing"),
    # review c2: R2-1, R2-2 and the nonblocking items
    ("M66 the practical threshold made strict", F26, "    if m >= PRACTICAL_MIN and L > 0", "    if m > PRACTICAL_MIN and L > 0",
     RULES, "exactly_the_practical"),
    ("M67 disagreement flag in one direction", F26, '"dependence_disagreement": (L > 0) != (L_block > 0),',
     '"dependence_disagreement": L > 0 and not L_block > 0,', RULES, "both_directions"),
    ("M68 R2-1 inventory sha unchecked", F26, "    if S._sha(b) != cited:", "    if False:", RUN,
     "source_inventory_must_be"),
    ("M69 R2-1 inventory publication unchecked", F26,
     "    if A._show_bytes(repo, f\"{exposure_commit}^:{INVENTORY_REL}\") is not None:", "    if False:", RUN,
     "source_inventory_must_be"),
    ("M70 R2-1 extraction fields unchecked", F26, '    if inv["extraction"] != list(EXTRACTION_FIELDS):', "    if False:",
     RUN, "source_inventory_must_be"),
    ("M71 R2-1 selection rule unchecked", F26, '    if inv["selection"] != SELECTION:', "    if False:", RUN,
     "source_inventory_must_be"),
    ("M72 R2-1 inventory fields unchecked", F26, "    if not (isinstance(inv, dict) and set(inv) == INVENTORY_KEYS",
     "    if False and not (isinstance(inv, dict) and set(inv) == INVENTORY_KEYS", RUN, "source_inventory_must_be"),
    ("M73 R2-2 retained values not compared with the pinned expectation", F26,
     "        if not _same(v, want):\n            return f\"{arm}: side-game {(day, pk, side)} carries value",
     "        if False:\n            return f\"{arm}: side-game {(day, pk, side)} carries value", RUN,
     "fabricated_catcher_value or cannot_be_faked"),
    ("M74 R2-2 expect step ungated", F26, "    if problem:\n        raise SystemExit(f\"refusing: {problem}\")\n"
     "    inv, problem = source_inventory(A.REPO, xc)", "    if False:\n        raise SystemExit(f\"refusing: {problem}\")\n"
     "    inv, problem = source_inventory(A.REPO, xc)", RUN, "expect_step"),
    # round 3: the fresh review f1's findings and the field inventory (one mutant per new guard)
    ("M75 F1 rank order unchecked", F26, "        if bool((pr[1:] > pr[:-1]).any()):", "        if False:", RUN,
     "rank_swap"),
    ("M76 F1 exact ties refused", F26, "        if bool((pr[1:] > pr[:-1]).any()):", "        if bool((pr[1:] >= pr[:-1]).any()):",
     RUN, "exact_ties"),
    ("M77 F1 ties not counted", F26, "        adjacent += int(eq.sum())", "        adjacent += 0", RUN, "exact_ties"),
    ("M78 F2 lgb_params unchecked", F26, "    if lgb != reviewed_lgb_params():", "    if False:", RUN,
     "wrong_model_recipe"),
    ("M79 F2 only the two flags checked (the old check)", F26, "    if lgb != reviewed_lgb_params():",
     '    if not (lgb.get("deterministic") is True and lgb.get("force_row_wise") is True):', RUN, "wrong_model_recipe"),
    ("M80 F2 blend configurations unchecked", F26, '    if man.get("blend_configs") != recorded_blend_configs():',
     "    if False:", RUN, "wrong_blend_configuration"),
    ("M81 F2 the A arms recorded with the baseline columns", F26,
     'for c in S.blend_configs("baseline" if arm == "baseline" else "A")]', 'for c in S.blend_configs("baseline")]', RUN,
     "records_the_reviewed_blend"),
    ("M82 F3 manifest allowance unchecked at validation", F26, '    if man.get("allowance") != allow:', "    if False:",
     RUN, "allowance_must_equal"),
    ("M83 F3 first-unit stop unchecked at validation", F26,
     '    if units[0]["cpu_s"] > allow["first_unit_stop"] * 3600:', "    if False:", RUN, "over_the_first_unit_stop"),
    ("M84 F3 PENDING's declared budget unchecked", F26, 'p.get("declared_cpu_hours") == budget', "True", RUN,
     "c1s_own_rules and PENDING"),
    ("M85 F3 PENDING's limit unchecked", F26, 'p.get("limit_cpu_seconds") == int(budget * 3600)', "True", RUN,
     "c1s_own_rules and PENDING"),
    ("M86 F3 C1's terminal rules not applied", F26,
     '    problems = L.terminal_problems(t, p, unit) if isinstance(t, dict) else ["no receipt"]',
     '    problems = [] if isinstance(t, dict) else ["no receipt"]', RUN, "c1s_own_rules and TERMINAL"),
    ("M87 F3 RECONCILED compared loosely (the old check)", F26,
     '    if r != {"unit": unit, "result": "exit", "cpu_seconds": t["cpu_seconds"], "rc": 0, "problems": []}:',
     '    if not (isinstance(r, dict) and r.get("problems") == [] and r.get("cpu_seconds") == t["cpu_seconds"]):',
     RUN, "c1s_own_rules and RECONCILED"),
    ("M88 F3 effective total unchecked (round 4: the gate is launch's)", F26,
     "        problem = effective_total_problem(allow, out_root, C1_DIR)", "        problem = None",
     RUN, "effective_total"),
    ("M89 F3 off-launcher CPU left out of the effective total", F26,
     '        off_h = sum(r["cpu_s"] for r in off_launcher_rows(out_root)) / 3600', "        off_h = 0.0", RUN,
     "effective_total or malformed_off_launcher"),
    ("M90 F3 the budget not reserved in full", F26, '    if effective + allow["budget"] > allow["cap"]:',
     '    if effective > allow["cap"]:', RUN, "effective_total"),
    ("M91 F3 the launch wrapper's CPU not recorded before the gate", F26,
     '        record_off_launcher(out_root, "launch", charged_s, seed=seed)      # the wrapper\'s own charge, before the gate',
     "        pass", RUN, "effective_total"),
    ("M92 F3 the aggregate's CPU not frozen into its row", F26,
     "    aggregate_cpu = step.freeze()                                  # this row is appended on exit with this value",
     "    aggregate_cpu = 0.0", RUN, "reports_the_ledger"),
    ("M93 F4 expect ignores the prepared exposure commit", F26, '    if prepared.get("exposure_commit") != xc:',
     "    if False:", RUN, "unbound_preparation and exposure"),
    ("M94 F4 expect ignores the cited inventory sha", F26,
     '    if prepared.get("inventory_sha256") != cited_inventory_sha256(A.REPO, xc):', "    if False:", RUN,
     "unbound_preparation and inventory0"),
    ("M95 F4 expect's caller PA directory unchecked", F26,
     '    if data_dir.resolve() != Path(inv["pa_dir"]).resolve():', "    if False:", RUN,
     "caller_pa_directory"),
    ("M96 F4 expect's historical pins unchecked", F26, "    problem = historical_pins_problem(pins, screen_pins())",
     "    problem = None", RUN, "unbound_preparation and screen"),
    ("M97 F4 exclusive creation replaced by a replacing write", F26, "        os.link(tmp, path)",
     "        os.replace(tmp, path)", RUN, "exclusive_write_never_replaces"),
    ("M98 F4 prepare's record not written", F26,
     "    exclusive_write(out_dir / PREPARED_NAME, canonical(rec))   # the preparation's record; the prepared row binds it",
     "    pass", RUN, "durable_record_of_its_pins"),
    ("M99 F5 resumed_flag_2026 unchecked", F26, '    if man.get("resumed_flag_2026") != truth.resumed_flag_2026:',
     "    if False:", RUN, "declared_totals and resumed0"),
    ("M100 F5 the aggregate's resumed rows unchecked", F26,
     '        if by[s]["manifest"].get("resumed_portion_rows") != resumed_rows:', "        if False:", RUN,
     "every_seasons_resumed"),
    ("M101 F5 features_cpu_s unchecked", F26, '    if not (_finite_nonneg(fc) and fc <= res["total_cpu_s"]):',
     "    if False:", RUN, "declared_totals and features_cpu"),
    ("M102 F5 wall_s unchecked", F26,
     '    if not all(_finite_nonneg(u.get("wall_s")) for u in res.get("units") or []):', "    if False:", RUN,
     "negative_wall_time"),
    # round 4: review f1 round 2, B1–B3 (one mutant per new guard)
    ("M103 B1 expect ignores the prepared row", F26,
     "    problem = prepared_row_problem(A.REPO, register, xc, S._sha(prepared_bytes))\n    if problem:\n"
     "        raise SystemExit(f\"refusing: {problem}\")\n    if prepared.get(\"exposure_commit\") != xc:",
     "    problem = None\n    if problem:\n        raise SystemExit(f\"refusing: {problem}\")\n"
     "    if prepared.get(\"exposure_commit\") != xc:", RUN, "requires_the_prepared_row or replaced_table"),
    ("M104 B1 the run ignores the preparation chain", F26,
     "    problem = preparation_chain_problem(A.REPO, register, adm[\"exposure_commit\"], inputs_dir, pins)\n"
     "    if problem:\n        raise SystemExit(f\"refusing: {problem}\")",
     "    problem = None\n    if problem:\n        raise SystemExit(f\"refusing: {problem}\")", RUN,
     "require_the_preparation_chain"),
    ("M105 B1 the aggregate ignores the preparation chain", F26,
     "    problem = preparation_chain_problem(repo, register, adm[\"exposure_commit\"], inputs_dir, pins)\n"
     "    if problem:\n        raise RunInvalid(problem)",
     "    problem = None\n    if problem:\n        raise RunInvalid(problem)", RUN, "require_the_preparation_chain"),
    ("M106 B1 EXPECTED's pins not compared with the admission's", F26, 'and rec.get("pins") == pins', "and True", RUN,
     "require_the_preparation_chain and expected_pins"),
    ("M107 B1 EXPECTED's prepared sha not compared", F26,
     'and rec.get("prepared_sha256") == S._sha(prepared_bytes)', "and True", RUN,
     "require_the_preparation_chain and expected_prepared"),
    ("M108 B1 the recorded output namespace unchecked", F26, '    if prepared.get("out_dir") != str(inputs.resolve()):',
     "    if False:", RUN, "directories_or_cpu and namespace"),
    ("M109 B2 a record without the seed row accepted", F26,
     '    if not (rows and rows[0].get("step") == PRIOR_STEP and isinstance(rows[0].get("source"), str)):',
     "    if False:", RUN, "malformed_off_launcher"),
    ("M110 B2 a failed step not marked failed", F26,
     '            record_off_launcher(self.out_root, self.step, cpu, **({"failed": True} if exc_type else {}), **self.extra)',
     '            record_off_launcher(self.out_root, self.step, cpu, **self.extra)', RUN, "failed_step"),
    ("M111 B2 the launcher process not charged", F26,
     '    record_off_launcher(out_root, "launcher-process", _children_cpu_s() - children0, seed=seed, rc=rc)',
     "    pass", RUN, "launcher_process or effective_total"),
    ("M112 B3 other invocations ignored at the seed gate", F26,
     "        for o in other_units(c1, s, register_text, exclude=exclude):\n            if not o[\"acknowledged\"]:",
     "        for o in ():\n            if not o[\"acknowledged\"]:", RUN, "blocks_progress or current_seed"),
    ("M113 B3 anyone's acknowledgement counts", F26,
     '    return bool(m and m.group(2) == unit and cells[3].split()[:1] == ["Eric"])',
     "    return bool(m and m.group(2) == unit)", RUN, "acknowledgement_row_grammar or blocks_progress"),
    ("M114 B3 the aggregate ignores unacknowledged invocations", F26,
     "    for s in SEEDS:\n        for o in by[s][\"other_units\"]:\n            if not o[\"acknowledged\"]:",
     "    for s in ():\n        for o in by[s][\"other_units\"]:\n            if not o[\"acknowledged\"]:", RUN,
     "aggregate_reports_every_invocation"),
]


def sh(*args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def main(commit: str, scratch: str, ledger: str) -> int:
    scratch_p = Path(scratch)
    sh("git", "worktree", "add", "--detach", str(scratch_p), commit)
    try:
        sh("env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "sync", "--extra", "model", "--offline", "-q", cwd=scratch_p)
        rows = ["mutant\tfile\ttests\tselector\tresult\tpytest_rc\ttree_after_restore\tfirst_failure"]
        for label, rel, old, new, tests, selector in MUTANTS:
            path = scratch_p / rel
            original = path.read_bytes()
            text = original.decode()
            if text.count(old) != 1:
                rows.append(f"{label}\t{rel}\t{tests}\t{selector}\tSTALE ANCHOR ({text.count(old)} matches)\t-\t-\t")
                continue
            try:
                path.write_text(text.replace(old, new))
                # A same-size substitution written within the same second as a cached compile can be served from a
                # stale .pyc (the cache checks only size and whole-second mtime), so caches are purged and no
                # bytecode is written: every import compiles the mutated source.
                for cache in scratch_p.rglob("__pycache__"):
                    sh("rm", "-rf", str(cache))
                r = subprocess.run(["env", "PYTHONDONTWRITEBYTECODE=1", "UV_CACHE_DIR=/tmp/uv-cache",
                                    "TZ=America/New_York", "uv", "run",
                                    "--offline", "pytest", "-q", "-p", "no:cacheprovider", "-x", *tests.split(),
                                    "-k", selector], cwd=scratch_p, capture_output=True, text=True)
                result = "KILLED" if r.returncode == 1 else ("SURVIVED" if r.returncode == 0 else f"ERROR rc {r.returncode}")
                failed = next((ln for ln in r.stdout.splitlines() if ln.startswith("FAILED ")), "")
                assertion = next((ln.strip() for ln in r.stdout.splitlines() if ln.startswith("E ")), "")
                detail = f"{failed[7:]} | {assertion}"[:300] if result == "KILLED" else ""
            finally:
                path.write_bytes(original)
            clean = sh("git", "status", "--porcelain", "--untracked-files=no", cwd=scratch_p).stdout.strip()
            rows.append(f"{label}\t{rel}\t{tests}\t{selector}\t{result}\t{r.returncode}\t"
                        f"{'clean' if not clean else 'DIRTY'}\t{detail}")
            print(rows[-1], flush=True)
        rows.append(f"# commit {commit}")
        Path(ledger).write_text("\n".join(rows) + "\n")
    finally:
        sh("git", "worktree", "remove", "--force", str(scratch_p), check=False)
    bad = [r for r in rows[1:-1] if "\tKILLED\t" not in r or "\tclean\t" not in r]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))

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
    ("M3 seed order skipped", F26, "for s in SEEDS[:SEEDS.index(seed)]:", "for s in ():", RUN, "one_at_a_time"),
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
    ("M28 F1 score length unchecked", BB, "        if values.shape != (len(index),):", "        if False:", HOOK,
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
     "    if False:", RUN, "rederives_the_catcher"),
    ("M36 F3 counts unchecked", F26, '    if recorded.get("counts") != {"side_games": side_games, "pa_rows": pa_rows}:',
     "    if False:", RUN, "count_says_otherwise or recorded_counts or counts_edited_in_both"),
    ("M37 F3 identified ids unchecked", F26, '    if recorded.get("identified_ids") != sorted(ids):', "    if False:",
     RUN, "recorded_counts or ids_edited_in_both"),
    ("M38 F3 empty posted coverage allowed", F26, '    if arm == "A_posted" and side_games["identified"] == 0:',
     "    if False:", RUN, "count_says_otherwise"),
    ("M39 F3 values not recomputed", F26, "            if not _same(float(r.value), want):", "            if False:",
     RUN, "recomputes_every"),
    ("M40 F3 evidence agreement unchecked", F26,
     "        if any(not by[s][\"evidence\"][a].equals(ev) for s in SEEDS[1:]):", "        if False:", RUN,
     "evidence_differs"),
    ("M41 F4 pre-2019 catcher ids allowed", F26,
     '    if bool((early & df["fielding_catcher_id"].notna()).any()):', "    if False:", RULES + " " + RUN,
     "before_2019 or pre_2019"),
    ("M42 F5 prepare's row gate removed", F26, "    problem = prep_row_problem(A.REPO, (A.REPO / REGISTER_REL).read_text(), "
     "adm[\"exposure_commit\"])\n    if problem:\n        raise SystemExit",
     "    problem = None\n    if problem:\n        raise SystemExit", RUN, "preparation_function or prepare_command"),
    ("M43 F5 declared paths unchecked", F26, "        if Path(given).resolve() != Path(declared).resolve():",
     "        if False:", RUN, "declared_ones"),
    ("M44 F5 historical pins unchecked", F26, "    return f\"the historical pins {bad} are not the screen's\" if bad else None",
     "    return None", RUN, "historical_pins"),
    ("M45 F5 source set unchecked", F26, "    if len(got) != len(set(got)) or set(got) != want:", "    if False:", RUN,
     "source_manifest or other_games_with_consistent"),
    ("M46 F5 table coverage unchecked", F26,
     '    return None if got == want else "the starter-proxy table does not cover', '    return None if True else "the',
     RUN, "cover_exactly"),
    ("M47 F6 launcher receipt unchecked", F26,
     "    problem = launcher_problems(d, seed, man, res, C1_DIR if c1_dir is None else c1_dir)", "    problem = None",
     RUN, "clean_launcher_receipt or clean_receipt"),
    ("M48 F6 nonzero rc accepted", F26, 'and type(t.get("rc")) is int\n            and t["rc"] == 0',
     'and type(t.get("rc")) is int\n            and True', RUN, "clean_launcher_receipt"),
    ("M49 F6 CPU bound unchecked", F26, 'if not (_finite_nonneg(total) and total <= t["cpu_seconds"]',
     'if not (True', RUN, "clean_launcher_receipt"),
    ("M50 F6 already-run seed launched", F26, "    if root.is_dir() and any(p.is_dir() for p in root.iterdir()):\n"
     "        raise SystemExit(f\"refusing: seed {seed} already has a run",
     "    if False:\n        raise SystemExit(f\"refusing: seed {seed} already has a run", RUN, "already_has_a_run"),
    ("M51 F7 batters' positions optional", F26, "_positions_ok(ap, required=batted)", "_positions_ok(ap, required=False)",
     RULES, "malformed_non_starting"),
    ("M52 F7 non-starters' ids unchecked", F26, "        if not _pos_int(pid):\n            bad_player = True\n"
     "            continue\n        batted", "        if False:\n            bad_player = True\n            continue\n"
     "        batted", RULES, "malformed_non_starting"),
    ("M53 F7 PA game ids coerced", F26, "    if not (pd.api.types.is_integer_dtype(col) and not pd.api.types.is_bool_dtype(col)):",
     "    if False:", RULES, "exact_positive"),
    ("M54 F7 table records unchecked", F26, "        problem = _record_problem(r)", "        problem = None", RULES,
     "inconsistent_table"),
    ("M55 F7 table sides unchecked", F26, '        if sorted(r["fielding_side"] for r in rs) != ["away", "home"]:',
     "        if False:", RULES, "inconsistent_table"),
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
    ("M64 block resampling without the reset", F26,
     "    starts = np.random.default_rng(BOOTSTRAP_SEED).integers(", "    starts = np.random.default_rng(1).integers(",
     RULES, "block_resampling"),
    ("M65 featureless 2026 rows not stopped early", F26, "    if featureless:\n", "    if False:\n", RUN,
     "every_model_feature_missing"),
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

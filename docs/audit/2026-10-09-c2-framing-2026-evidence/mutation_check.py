"""Mutation check for the 2026 framing test's code (the lead, 2026-10-09). The run half of
`scripts/audit/c2_framing/f26.py` was written before its tests; this shows each registered rule's tests fail when that
rule is broken. Each mutant replaces exactly one source span in a scratch worktree at the given commit, runs only the
tests aimed at it, and records KILLED (the tests failed) or SURVIVED. The source is restored after every mutant.

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
    ("M13 any C entry counts", F26, 'if ap[0]["code"] == "2":', 'if any(e["code"] == "2" for e in ap):', RULES,
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
    ("M21 posted identification unchecked", F26,
     'if not (isinstance(posted.get("identified"), int) and posted["identified"] > 0):', "if False:", RUN,
     "inconsistent_records"),
    ("M22 seed count unchecked", F26, "x.shape[0] != len(SEEDS)", "x.shape[0] < 1", RULES, "anything_but_ten"),
    ("M23 preparation row unchecked", F26, "        if problem:                                           # before any 2026",
     "        if False:                                             # before any 2026", RUN, "prepare_command"),
    ("M24 prep row may predate X-37", F26, "    if A.row_cells(earlier, row_id):", "    if False:", RUN,
     "preparation_read or inputs_row"),
    ("M25 strict mode accepts missing scores", BB, "if strict_predict and not np.isfinite(", "if False and not np.isfinite(",
     HOOK, "missing_scores"),
]


def sh(*args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def main(commit: str, scratch: str, ledger: str) -> int:
    scratch_p = Path(scratch)
    sh("git", "worktree", "add", "--detach", str(scratch_p), commit)
    try:
        sh("env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "sync", "--extra", "model", "--offline", "-q", cwd=scratch_p)
        rows = ["mutant\tfile\ttests\tselector\tresult\tpytest_rc"]
        for label, rel, old, new, tests, selector in MUTANTS:
            path = scratch_p / rel
            original = path.read_bytes()
            text = original.decode()
            if text.count(old) != 1:
                rows.append(f"{label}\t{rel}\t{tests}\t{selector}\tSTALE ANCHOR ({text.count(old)} matches)\t-")
                continue
            try:
                path.write_text(text.replace(old, new))
                r = subprocess.run(["env", "UV_CACHE_DIR=/tmp/uv-cache", "TZ=America/New_York", "uv", "run",
                                    "--offline", "pytest", "-q", "-p", "no:cacheprovider", "-x", tests, "-k", selector],
                                   cwd=scratch_p, capture_output=True, text=True)
                result = "KILLED" if r.returncode == 1 else ("SURVIVED" if r.returncode == 0 else f"ERROR rc {r.returncode}")
                rows.append(f"{label}\t{rel}\t{tests}\t{selector}\t{result}\t{r.returncode}")
            finally:
                path.write_bytes(original)
            print(rows[-1], flush=True)
        clean = sh("git", "status", "--porcelain", "--untracked-files=no", cwd=scratch_p).stdout.strip()
        rows.append(f"# commit {commit}; scratch tracked tree after restore: {'clean' if not clean else clean}")
        Path(ledger).write_text("\n".join(rows) + "\n")
    finally:
        sh("git", "worktree", "remove", "--force", str(scratch_p), check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))

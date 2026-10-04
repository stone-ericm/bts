"""Merge shard outputs into one mutation_sweep.txt in the canonical mutant order.

    python3 merge_shards.py <commit> <sweep-script> <out-txt> <shard-out> [<shard-out> ...]

Each shard line is ``LABEL: VERDICT | rc=N | last output line | [killing nodes]`` (mutation_sweep.main).
Refuses (exit 1) when a mutant is missing, duplicated, or any shard did not end with DONE + exit=0.
"""
import ast, re, sys
from collections import Counter
from pathlib import Path

commit, script, out_path, *shards = sys.argv[1:]
src = Path(script).read_text()
M = ast.literal_eval("[" + re.search(r"^M = \[\n(.*?)^\]", src, re.S | re.M).group(1) + "]")
order = [lab for *_, lab in M]
lines, seen, head, problems = {}, Counter(), [], []
for i, sh in enumerate(shards):
    name = chr(ord("A") + i)
    text = Path(sh).read_text().splitlines()
    if text[-2:] != ["DONE", "exit=0"]:
        problems.append(f"shard {name} did not finish cleanly: {text[-2:]}")
    for l in text:
        if l.startswith("BASELINE"):
            head.append(f"# shard {name} {l[:120]}")
            continue
        for lab in order:
            if l.startswith(lab + ":"):
                lines[lab] = l
                seen[lab] += 1
missing = [lab for lab in order if lab not in lines]
dups = [lab for lab, n in seen.items() if n > 1]
verdicts = Counter(lines[lab][len(lab) + 2:].split(" |")[0] for lab in order if lab in lines)   # a title may hold ": "
out = [f"# strict mutation sweep at {commit} ({len(order)} mutants; whole suite per mutant; JUnit-XML classification;"
       " every killing node listed)",
       f"# run in {len(shards)} parallel shards, each in its own worktree at the same commit with its own clean baseline",
       *head,
       f"# verdicts: {dict(sorted(verdicts.items()))}; missing: {missing}; duplicated: {dups}", ""]
out += [lines[lab] for lab in order if lab in lines]
Path(out_path).write_text("\n".join(out) + "\n")
print(f"verdicts {dict(verdicts)} missing {missing} duplicated {dups} problems {problems}")
sys.exit(1 if (missing or dups or problems) else 0)

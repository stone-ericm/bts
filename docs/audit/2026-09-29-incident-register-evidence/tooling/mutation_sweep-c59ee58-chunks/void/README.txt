shard-<i>-a.exit-unrecorded.void (2026-10-03): chunk a of each shard ran to DONE with 16 KILLED over a clean 462-node
baseline, but run_rest.sh was edited (the .meta lines added) while the four runs were in progress. Bash reads a script
incrementally, so after mutation_sweep.py finished each wrapper hit a syntax error (exit 2) before appending its
"exit=$?" line. mutation_sweep.main prints DONE only immediately before `return 0`, but the exit code was not recorded,
so these outputs are voided and the same 64 labels are rerun with the unchanged, fixed wrapper as chunk a again.

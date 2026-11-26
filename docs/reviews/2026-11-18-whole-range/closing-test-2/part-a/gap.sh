#!/bin/zsh
# Closing test RE-RUN, part a: the four slice harnesses on FINAL2 (d0bdcdba) in this clean clone.
# Harnesses are copies of docs/reviews/2026-11-18-whole-range/slice-*/ placed at .codex-review/<slice>/,
# their original location relative to the repo root (their path constants assume it), with
# `git archive 619abda7 src scripts` as slice-2's base/. One harness change (as in the first run):
# slice-1-carbon/rehearsal_v3_harness.py also pins bts.health.runner.datetime (Carbon's C1 note 2).
# Invocations follow the first run's corrected forms (gap2.sh): slice-2 day matrices through
# offline_fixtures.day_matrix(); the statuses phase after startup/prepare, with their logs beside it, under BASE;
# slice-3 mutate.py with the 33 cases whose anchors exist (fresh-identity, fresh-policy, fallback-reload,
# fallback-fail have no anchor on FINAL2, as on FINAL). Each step: own log, exit code recorded, bounded by timeout.
cd /Users/eric/projects/bts-work/wr2-harness || exit 9
OUT=$PWD/closing-a/out2; mkdir -p $OUT
H=$PWD/.codex-review
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
PY=$PWD/.venv/bin/python
step() { local name=$1 secs=$2; shift 2; echo "$(date '+%F %T') start $name" >> $OUT/steps.txt
  timeout $secs "$@" > $OUT/$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $name rc=$rc" >> $OUT/steps.txt; }
HEADPP=$PWD/src
# gap: the steps that could not run in the first window because I had not placed the BASE archives
# (.codex-review/slice-1-codex/base, slice-1-carbon/basesha, slice-3-codex/base) and because cron_probe's
# shims did not cover FINAL2's flock+timeout watchdog line (harness-change-cron.diff).
step s1-rehearsal 3600 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_rehearsal.py --logs $OUT/s1-rehearsal-cases
step s1-edges 2400 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_edges.py
cp $H/slice-1-codex/opening_edges.json $OUT/s1-edges.json 2>/dev/null
step s1-edges-fixed 2400 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_edges.py --expect-fixed
cp $H/slice-1-codex/opening_edges.json $OUT/s1-edges-fixed.json 2>/dev/null
step s1-ops 1200 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_ops.py
cp $H/slice-1-codex/opening_ops.json $OUT/s1-ops.json 2>/dev/null
step c1-carbon 2400 env PYTHONPATH=$HEADPP:$H/slice-1-carbon $PY -m pytest -p no:cacheprovider -q $H/slice-1-carbon/test_slice1_rehearsal_final.py $H/slice-1-carbon/test_slice1_probes_v3.py -rA
step s2-cron 2400 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/cron_probe.py
echo "$(date '+%F %T') ALL STEPS DONE" >> $OUT/steps.txt

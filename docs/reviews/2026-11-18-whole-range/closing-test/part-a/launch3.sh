#!/bin/zsh
# One hold window for runs that are READY now (lead's slot rule): first the blind reviewer's 30 queued
# mutation runs, then Carbon slice 3's two queued runs (scripts read and approved by bts-wr-fresh8), then part a. Set the backtest hold at 12:40 ET
# (about 10 minutes before the current backtest's expected end), wait until no wrapped job runs, run each
# command through the wrapper one at a time, remove the hold the moment the last one ends.
L=/Users/eric/projects/bts-work/wr-closing-harness/closing-a
BLIND=/Users/eric/projects/bts-work/wr-closing-blind
HOLD=$HOME/bts-runs/backtest-hold
log() { echo "$(date '+%F %T') $*" >> $L/launch.txt; }
mine=0
trap '[[ $mine = 1 ]] && rm -f $HOLD && log "hold removed (trap)"' EXIT INT TERM
until [[ $(date +%H%M) -ge 1240 ]]; do sleep 20; done
if [[ -e $HOLD ]]; then log "hold already present (not mine); not touching it"; else touch $HOLD; mine=1; log "hold set"; fi
while pgrep -f "bts-runs/yield_run.sh" >/dev/null; do sleep 5; done
log "slot free"
cd $BLIND
for target in $(cut -d' ' -f1 $L/blind-targets.txt); do
  $HOME/bts-runs/yield_run.sh .codex-review/whole-range/mutation-$target.log env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/whole-range/mutations.py $target
  log "blind mutation $target rc=$?"
done
echo "SLOT RUNS DONE $(date '+%F %T')" > $BLIND/.codex-review/whole-range/slot-runs-done.txt
log "blind slot runs done"
cd /Users/eric/projects/bts-work/wr-closing-carbon3
$HOME/bts-runs/yield_run.sh .carbon-review/whole-range/slot-run-1.log .venv/bin/python .carbon-review/whole-range/run_mutations.py
log "carbon3 run_mutations rc=$?"
$HOME/bts-runs/yield_run.sh .carbon-review/whole-range/slot-run-2.log .venv/bin/python -m pytest -q .carbon-review/whole-range/test_slice3_interactions.py .carbon-review/whole-range/test_slice3_clocks_and_years.py .carbon-review/whole-range/test_slice3_fail_closed.py
log "carbon3 pytest rc=$?"
echo "SLOT RUNS DONE $(date '+%F %T')" > .carbon-review/whole-range/slot-runs-done.txt
$HOME/bts-runs/yield_run.sh $L/wrapped.log zsh $L/run_all.sh
log "wrapped part a exited rc=$?"
if [[ $mine = 1 ]]; then rm -f $HOLD; mine=0; log "hold removed"; fi
echo PARTA-FINISHED

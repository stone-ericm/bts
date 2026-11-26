#!/bin/zsh
# One hold window for part a: set the backtest hold (nothing consumes it now; kept per the kickoff), wait until
# no wrapped job runs, run run_all.sh as ONE wrapped job, remove the hold the moment it ends.
L=/Users/eric/projects/bts-work/wr2-harness/closing-a
HOLD=$HOME/bts-runs/backtest-hold
log() { echo "$(date '+%F %T') $*" >> $L/launch.txt; }
mine=0
trap '[[ $mine = 1 ]] && rm -f $HOLD && log "hold removed (trap)"' EXIT INT TERM
if [[ -e $HOLD ]]; then log "hold already present (not mine); not touching it"; else touch $HOLD; mine=1; log "hold set"; fi
while pgrep -f "bts-runs/yield_run.sh" >/dev/null; do sleep 5; done
log "slot free"
BTS_YIELD_SECONDS=10 $HOME/bts-runs/yield_run.sh $L/wrapped.log zsh $L/run_all.sh
log "wrapped part a exited rc=$?"
if [[ $mine = 1 ]]; then rm -f $HOLD; mine=0; log "hold removed"; fi
echo PARTA-FINISHED

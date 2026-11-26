#!/bin/zsh
# Second hold window: part a's gap steps, then part d's walk jobs, each one wrapped job, one after the other.
L=/Users/eric/projects/bts-work/wr2-harness/closing-a
HOLD=$HOME/bts-runs/backtest-hold
log() { echo "$(date '+%F %T') $*" >> $L/launch.txt; }
mine=0
trap '[[ $mine = 1 ]] && rm -f $HOLD && log "hold removed (trap)"' EXIT INT TERM
if [[ -e $HOLD ]]; then log "w2: hold already present (not mine); not touching it"; else touch $HOLD; mine=1; log "w2: hold set"; fi
while pgrep -f "bts-runs/yield_run.sh" >/dev/null; do sleep 5; done
log "w2: slot free"
BTS_YIELD_SECONDS=10 $HOME/bts-runs/yield_run.sh $L/wrapped-gap.log zsh $L/gap.sh
log "w2: gap exited rc=$?"
BTS_YIELD_SECONDS=10 $HOME/bts-runs/yield_run.sh $L/wrapped-walk.log zsh /Users/eric/projects/bts-work/wr2-walk/walk_jobs.sh
log "w2: walk jobs exited rc=$?"
if [[ $mine = 1 ]]; then rm -f $HOLD; mine=0; log "w2: hold removed"; fi
echo W2-FINISHED

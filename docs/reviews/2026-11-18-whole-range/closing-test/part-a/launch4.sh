#!/bin/zsh
# Second hold window: set the backtest hold at 15:20 ET, wait until no wrapped job runs, run gap2.sh as ONE
# wrapped job, remove the hold the moment it ends.
L=/Users/eric/projects/bts-work/wr-closing-harness/closing-a
HOLD=$HOME/bts-runs/backtest-hold
log() { echo "$(date '+%F %T') $*" >> $L/launch.txt; }
mine=0
trap '[[ $mine = 1 ]] && rm -f $HOLD && log "hold removed (trap)"' EXIT INT TERM
until [[ $(date +%H%M) -ge 1520 ]]; do sleep 20; done
if [[ -e $HOLD ]]; then log "gap2: hold already present (not mine); not touching it"; else touch $HOLD; mine=1; log "gap2: hold set"; fi
while pgrep -f "bts-runs/yield_run.sh" >/dev/null; do sleep 5; done
log "gap2: slot free"
BTS_YIELD_SECONDS=10 $HOME/bts-runs/yield_run.sh $L/wrapped2.log zsh $L/gap2.sh
log "gap2: wrapped exited rc=$?"
if [[ $mine = 1 ]]; then rm -f $HOLD; mine=0; log "gap2: hold removed"; fi
echo GAP2-FINISHED

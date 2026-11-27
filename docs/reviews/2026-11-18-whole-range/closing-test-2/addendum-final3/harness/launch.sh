#!/bin/zsh
L=/Users/eric/bts-runs/fresh15/closing3
HOLD=$HOME/bts-runs/backtest-hold
log() { echo "$(date '+%F %T') $*" >> $L/launch.txt; }
while pgrep -f '^(/bin/)?zsh /Users/eric/bts-runs/yield_run\.sh' >/dev/null; do sleep 5; done
touch $HOLD; log "hold set; slot free"
BTS_YIELD_SECONDS=10 $HOME/bts-runs/yield_run.sh $L/wrapped.log zsh $L/all.sh
log "wrapped job exited rc=$?"
rm -f $HOLD; log "hold removed"
echo CLOSING3-FINISHED >> $L/launch.txt

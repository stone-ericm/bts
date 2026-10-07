#!/bin/bash
# One load-phase re-run attempt N (row C2-2a-cost-r4-load). Nothing else of the author's runs meanwhile.
# Attempts 2 and 3 use the discarded warm-up (Eric's row C2-2a-cost-r4-warmup); attempt 1 ran without it.
set -u
N=$1
C=/Users/eric/projects/bts-c2-2a; B=/Users/eric/projects/bts-c2-2a-golden; I=/Users/eric/projects/c2-2a-bench-inputs
E=$C/docs/audit/2026-10-06-c2-2a-evidence/cost; cd $C
export UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York
snap() {
  echo "## $1: $(date '+%Y-%m-%d %H:%M:%S %Z')"; memory_pressure 2>/dev/null | tail -1
  vm_stat | egrep "Pages free|Pages stored in compressor|Pages occupied by compressor|Swapins|Swapouts"
  top -l 1 -o cmprs -n 6 -stats pid,command,cmprs,mem 2>/dev/null | tail -7
}
{ echo "# load attempt $N at candidate $(git rev-parse HEAD) (src as e36e038) vs baseline f882411"; snap before; } > $E/load_r4_a$N.conditions.txt
MODE=$([ "$N" -ge 2 ] && echo warmup-run || echo "")
uv run python $E/phases_r4_load.py $MODE $B $C $I $E/load_r4_a$N.jsonl > $E/load_r4_a$N.log 2>&1; echo "exit $?" >> $E/load_r4_a$N.log
snap after >> $E/load_r4_a$N.conditions.txt
uv run python $E/phases_r4_load.py summarise $E/load_r4_a$N.jsonl > $E/load_r4_a$N.summary.json 2>&1
python3 -c "
import json; s=json.load(open('$E/load_r4_a$N.summary.json'))['load']
import pathlib; w = pathlib.Path('$E/load_r4_a$N.warmup.json')
print('warmup (discarded) peak', round(json.loads(w.read_text())['rss_peak']/1e6) if w.exists() else None, 'MB')
print('attempt $N:', s['verdict'], 'control max', round(s['control_max_abs']/1e6,1), 'MB; pairs', [round(x/1e6,1) for x in s['pairs']], 'controls', [round(x/1e6,1) for x in s['control_pairs']])"

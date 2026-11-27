#!/bin/zsh
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
step() { local out=$1 name=$2 secs=$3; shift 3; echo "$(date '+%F %T') start $name" >> $out/steps.txt
  timeout $secs "$@" > $out/$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $name rc=$rc" >> $out/steps.txt; }
cd /Users/eric/projects/bts-work/wr3-harness || exit 9
O3=$PWD/closing-a/out3; mkdir -p $O3
step $O3 s3-refusals-cap401 2400 env PYTHONPATH=$PWD/src .venv/bin/python .codex-review/slice-3-codex/refusals_cap401.py
H=$PWD/.codex-review
step $O3 s1-ops-final2 1200 env PYTHONPATH=$PWD/src .venv/bin/python $H/slice-1-codex/opening_ops_final2.py
step $O3 c1-carbon-ea 2400 env PYTHONPATH=$PWD/src:$H/slice-1-carbon .venv/bin/python -m pytest -p no:cacheprovider -q $H/slice-1-carbon/test_slice1_rehearsal_final_ea.py -rA
echo "$(date '+%F %T') ALL STEPS DONE" >> $O3/steps.txt

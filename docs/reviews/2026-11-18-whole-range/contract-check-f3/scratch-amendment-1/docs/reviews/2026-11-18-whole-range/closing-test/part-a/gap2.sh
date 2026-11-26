#!/bin/zsh
# Second short hold window (closing test): runs that are READY now, one wrapped job.
#  1. slice-2 day matrices, invoked through offline_fixtures.day_matrix() as slice 2 ran them (part a's
#     first attempt called the module as a script, which has no main block, so it ran nothing);
#  2. slice-2 deploy "statuses" phase under BASE, with the startup/prepare outputs it reads placed beside it;
#  3. slice-3 mutation sample, the 33 cases whose source anchors still exist on FINAL (4 anchors removed by W2/C2);
#  4. Carbon slice 3's two side-by-side tests for its B1 (stale skip after a dated correction) and B2
#     (second fallback cascade after a 5a refusal), to reproduce them.
cd /Users/eric/projects/bts-work/wr-closing-harness || exit 9
OUT=$PWD/closing-a/out2; mkdir -p $OUT
H=$PWD/.codex-review
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
PY=$PWD/.venv/bin/python
HEADPP=$PWD/src
B2=$H/slice-2-codex/base/src
step() { local name=$1 secs=$2; shift 2; echo "$(date '+%F %T') start $name" >> $OUT/steps.txt
  timeout $secs "$@" > $OUT/$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $name rc=$rc" >> $OUT/steps.txt; }
M="import sys; sys.argv=['day_matrix']+sys.argv[1:]; import offline_fixtures as f; f.day_matrix()"
cd $H/slice-2-codex
step s2-matrix-head 2400 env PYTHONPATH=$HEADPP:$H/slice-2-codex $PY -c "$M"
step s2-matrix-policy-head 2400 env PYTHONPATH=$HEADPP:$H/slice-2-codex $PY -c "$M" --production
step s2-matrix-policy-head-inclusive 1200 env PYTHONPATH=$HEADPP:$H/slice-2-codex $PY -c "$M" --production --inclusive-control --case last_date
step s2-matrix-policy-base-bridge 2400 env PYTHONPATH=$B2:$H/slice-2-codex $PY -c "$M" --production --bridge
cp $PWD/../../closing-a/out/s2-deploy-startup.log deploy-startup.log
cp $PWD/../../closing-a/out/s2-deploy-prepare.log deploy-prepare.log
step s2-deploy-statuses 1200 env PYTHONPATH=$B2 $PY $H/slice-2-codex/deploy_probe.py statuses
cd /Users/eric/projects/bts-work/wr-closing-harness
step s3-mutate 6000 env PYTHONPATH=$HEADPP $PY $H/slice-3-codex/mutate.py explicit-required snapshot-copy optional-refused dated-identity automatic-match exact-int dated-canonical saver-consistency non-dated-identity automatic-replacement cli-warmup-refusal cli-catchup-refusal orchestrate-catchup-refusal orchestrate-warmup-refusal preview-stamp run-stamp cascade-stamp cascade-reuse-stamp earliest-refusal-copy dated-clears-reading dm-identifier public-identifier age-14h future-five-min recorded-at-forwarded startup-threshold startup-scoreable startup-delivered morning-dedup-date morning-cross-date void-slate morning-dedup-date-control morning-warn-date-control
cd /Users/eric/projects/bts-work/wr-closing-carbon3
step c3-carbon-b1-b2 1800 .venv/bin/python -m pytest -p no:cacheprovider -q -rA ".carbon-review/whole-range/test_slice3_base_and_edges.py::test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms" ".carbon-review/whole-range/test_slice3_base_and_edges.py::test_basesha_vs_head_step4_to_step5_double_cascade"
echo "$(date '+%F %T') ALL STEPS DONE" >> $OUT/steps.txt

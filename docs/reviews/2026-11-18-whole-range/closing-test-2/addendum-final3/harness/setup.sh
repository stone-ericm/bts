#!/bin/zsh
# bts-wr-fresh15: harness clone of FINAL3 (main 240324ab) for R72's renewed closing evidence.
set -e
SRC=/Users/eric/projects/bts-work/land15
W=/Users/eric/projects/bts-work/wr3-harness
CT=$SRC/docs/reviews/2026-11-18-whole-range
git -C $SRC bundle create /Users/eric/bts-runs/bundles/final3-240324ab.bundle main
rm -rf $W; git clone -q -b main /Users/eric/bts-runs/bundles/final3-240324ab.bundle $W
cd $W; git log --oneline -1
echo ".codex-review/" >> .git/info/exclude; echo "closing-a/" >> .git/info/exclude
UV_CACHE_DIR=$HOME/bts-runs/uv-cache uv sync --extra model -q
mkdir -p .codex-review
for s in slice-1-codex slice-1-carbon slice-2-codex slice-3-codex; do cp -R $CT/$s .codex-review/$s; done
for d in slice-1-codex/base slice-2-codex/base slice-3-codex/base slice-1-carbon/basesha; do
  mkdir -p .codex-review/$d; git -C $SRC archive 619abda7 src scripts | tar -x -C .codex-review/$d; done
P=$CT/closing-test-2/part-a
patch -s .codex-review/slice-2-codex/cron_probe.py $P/harness-change-cron.diff
patch -s .codex-review/slice-1-carbon/rehearsal_v3_harness.py $P/harness-change-rehearsal-runner-datetime.diff
cp $P/opening_ops_final2.py .codex-review/slice-1-codex/
cp $P/refusals_cap401.py .codex-review/slice-3-codex/
cp $P/test_slice1_rehearsal_final_ea.py .codex-review/slice-1-carbon/
echo SETUP-DONE

#!/bin/bash
# #48: copy klone's detection cache into makelab2's, never overwriting an entry that
# makelab2 already has (--ignore-existing). Every 20 min until klone has no mv48 jobs
# left, then one final pass. Two hops through WSL; both ssh masters are reused.
set -u
L=$HOME/mv48_sync/model_cache
mkdir -p "$L"
SSH="ssh -o BatchMode=yes"
pass() {
  rsync -a -e "$SSH" klone:/gscratch/scrubbed/jfroehli/mv48/model_cache/ "$L/" &&
  rsync -a --ignore-existing -e "$SSH" "$L/" makelab2:/homes/gws/jonf/mv48/model_cache/ &&
  echo "$(date -Is) synced: local $(find "$L" -type f | wc -l) files"
}
while true; do
  pass
  n=$($SSH klone 'squeue -h -u jfroehli -n mv48-molmo,mv48-qwen8b,mv48-open | wc -l')
  echo "$(date -Is) klone mv48 jobs left: $n"
  [ "$n" = "0" ] && break
  sleep 1200
done
pass
echo SYNC_DONE

#!/bin/bash
# copy each arm best.pth right after the listed epochs finish (the tagger recipe keeps only best.pth)
cd /homes/gws/jonf/nobackup/tagbench86
for E in 4 9 19 49; do
  for A in control pano cell; do
    until grep -q "\"epoch\": $E," train_$A.log; do sleep 20; done
    cp train_$A/best.pth train_$A/best_after_ep$E.pth
    echo "$(date -u +%FT%TZ) $A ep$E $(grep "\"epoch\": $E," train_$A.log)"
  done
done

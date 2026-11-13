#!/bin/bash
# crosswalk_s6 (docs/crosswalk_followups_285.md, section A4), on klone:
#   build   unpack the new panos into root_v6_full and root_v6_half beside root_v4 and verify the label digests
#   submit  a 20-step smoke run, then (only if it succeeds) three seeds per dose and the root_v4 step control
# Usage: bash s6_train.sh build && bash s6_train.sh submit
set -eu
W=/gscratch/scrubbed/$USER/crosswalk274
C=/gscratch/makelab/jonf/crosswalk274
REPO=/gscratch/scrubbed/$USER/RampNet274
TAR_SHA=29fe307bee34347265f8ab193ba2948385488a5a68106e5156e6bb7064dc7465
declare -A DIGEST=([full]=9b420ac04286 [half]=b1aa92b87461) TRAIN=([full]=5999 [half]=5146) STEPS=([full]=1397 [half]=1198)

digest() {   # the manifest's rule: sha256 over sorted lines '<split>/<id>.json <sha256 of the file>\n'
    python3 - "$1" <<'EOF'
import hashlib, os, sys
h = hashlib.sha256()
for s in ("train", "val", "test"):
    for fn in sorted(os.listdir(os.path.join(sys.argv[1], s))):
        if fn.endswith(".json"):
            with open(os.path.join(sys.argv[1], s, fn), "rb") as f:
                h.update(f"{s}/{fn} {hashlib.sha256(f.read()).hexdigest()}\n".encode())
print(h.hexdigest()[:12])
EOF
}

case "${1:?build or submit}" in
build)
    cd "$W"
    echo "$TAR_SHA  root_v6_new.tar" | sha256sum -c -
    for d in full half; do
        [ -e "root_v6_$d" ] && { echo "FATAL: $W/root_v6_$d exists" >&2; exit 1; }
        cp -al root_v4 "root_v6_$d"
    done
    tar xf root_v6_new.tar -C root_v6_full
    awk '{print "train/"$1".jpg"; print "train/"$1".json"}' root_v6_half_ids.txt > root_v6_half_list.txt
    tar xf root_v6_new.tar -C root_v6_half -T root_v6_half_list.txt
    for d in full half; do
        n=$(ls "root_v6_$d/train" | grep -c '\.jpg$'); j=$(ls "root_v6_$d/train" | grep -c '\.json$'); g=$(digest "root_v6_$d")
        echo "root_v6_$d: train $n jpg / $j json (want ${TRAIN[$d]}), val $(ls root_v6_$d/val | grep -c '\.jpg$'), digest $g (want ${DIGEST[$d]})"
        [ "$n" = "${TRAIN[$d]}" ] && [ "$j" = "${TRAIN[$d]}" ] && [ "$g" = "${DIGEST[$d]}" ] || { echo "FATAL: root_v6_$d mismatch" >&2; exit 1; }
    done
    ;;
submit)
    cd "$REPO"
    mkdir -p logs
    export RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 EXPECT_VAL=1029 TRIM_EPOCH=1
    T=stage_two/run_finetune_crosswalk_274.slurm
    run() {  # name data_root expect_train steps seed [sbatch args]: its own run and copy dirs, which must not exist
        local name=$1 root=$2 n=$3 steps=$4 seed=$5; shift 5
        local copy=$C/$name
        [ "$steps" = 20 ] && copy=$W/${name}_copy     # the smoke run's weights are not kept on durable storage
        for p in "$W/$name" "$copy"; do
            [ -e "$p" ] && { echo "FATAL: $p exists; a job there would resume another run's checkpoint" >&2; exit 1; }
        done
        DATA_ROOT=$root RUNDIR=$W/$name COPY_DEST=$copy SEED=$seed STEPS=$steps EXPECT_TRAIN=$n \
            sbatch --parsable --job-name="cw285_$name" "$@" $T
    }
    if [ "${RERUN:-}" ]; then   # RERUN="v6full_s1 v4steps1397_s1 ...": those runs again from step 0, no smoke run.
        for name in $RERUN; do  # A failed run's directory must be renamed aside first (run refuses an existing one).
            seed=${name##*_s}
            case $name in
                v6full_s*|v6half_s*) d=${name#v6}; d=${d%_s*}; echo "$name $(run "$name" "$W/root_v6_$d" "${TRAIN[$d]}" "${STEPS[$d]}" "$seed")" ;;
                v4steps${STEPS[full]}_s1) echo "$name $(run "$name" "$W/root_v4" 4294 "${STEPS[full]}" 1)" ;;
                *) echo "FATAL: unknown run $name" >&2; exit 1 ;;
            esac
        done
        squeue -u "$USER" -o "%.10i %.22j %.2t %.8M %R"
        exit 0
    fi
    smoke=$(run smoke20_v6 "$W/root_v6_full" "${TRAIN[full]}" 20 1)
    echo "smoke $smoke"
    for d in full half; do
        for seed in 1 2 3; do
            echo "v6${d}_s$seed $(run "v6${d}_s$seed" "$W/root_v6_$d" "${TRAIN[$d]}" "${STEPS[$d]}" "$seed" --dependency=afterok:$smoke)"
        done
    done
    echo "step control v4steps${STEPS[full]}_s1 $(run "v4steps${STEPS[full]}_s1" "$W/root_v4" 4294 "${STEPS[full]}" 1 --dependency=afterok:$smoke)"
    squeue -u "$USER" -o "%.10i %.22j %.2t %.8M %R"
    ;;
esac

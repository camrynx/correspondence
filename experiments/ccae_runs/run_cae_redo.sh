#!/usr/bin/env bash
# 2026-10-07: redo the French cAE Poor Man's UTD + DUSTED runs, and train/evaluate the
# Xitsonga classic-UTD cAE (wordsUTD.list; not DUSTED, not Poor Man). WavLM layer 21, test split.
#
#     bash experiments/ccae_runs/run_cae_redo.sh
#
# Same stage()/done-marker/log conventions as run_all.sh (shared state/ and logs/ dirs).

set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STATE=experiments/ccae_runs/state
LOGS=experiments/ccae_runs/logs
mkdir -p "$STATE" "$LOGS"
LAYER=21

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [[ -f $STATE/$name.done ]]; then echo "[skip] $name"; return 0; fi
    echo "$name" > $STATE/CURRENT
    echo "[$(date '+%F %T')] START $name" | tee -a $LOGS/pipeline.log
    echo "\$ $*" > $LOGS/$name.log
    if "$@" >> $LOGS/$name.log 2>&1; then
        touch $STATE/$name.done
        echo "[$(date '+%F %T')] DONE  $name" | tee -a $LOGS/pipeline.log
    else
        touch $STATE/$name.failed
        echo "[$(date '+%F %T')] FAIL  $name (see $LOGS/$name.log)" | tee -a $LOGS/pipeline.log
        rm -f $STATE/CURRENT
        exit 1
    fi
}

# The GPU may be shared: evaluation stages wait for >= GATE_MB free on two checks 60 s apart.
GATE_MB=${GATE_MB:-5500}
gpu_gate() {
    local ok=0
    while (( ok < 2 )); do
        local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if (( free >= GATE_MB )); then ok=$((ok+1)); else ok=0; fi
        echo "[$(date '+%F %T')] free=${free} MiB (need ${GATE_MB}) ok=${ok}/2" >> $LOGS/gpu_gate.log
        (( ok < 2 )) && sleep 60
    done
}
gated() {
    [[ -f $STATE/$1.done ]] && { echo "[skip] $1"; return 0; }
    echo "gpu_gate (before $1)" > $STATE/CURRENT
    gpu_gate
    stage "$@"
}

# ---------- 0) back up what this run overwrites (copy, never move) ----------
backup() {
    local B=experiments/ccae_runs/backup_before_cae_redo_2026-10-07
    mkdir -p $B
    for f in models/ZS17French_WavLM/cAE_PoorManUTD_layer_21.pt models/ZS17French_WavLM/cAE_DUSTED_layer_21.pt \
             models/Xitsonga_WavLM/cAE_UTD_layer_21.pt; do
        mkdir -p $B/$(dirname $f) && cp -a $f $B/$f
    done
    for d in results/ap/ZS17French_WavLMPoorManUTD/Normalised/test.samediff_cAE \
             results/ap/ZS17French_WavLMDUSTED/Normalised/test.samediff_cAE \
             results/abx/ZS17French_WavLMPoorManUTD/Normalised/test.abx_cAE \
             results/abx/ZS17French_WavLMPoorManUTD/Normalised/test.abx_cAE_layer21 \
             results/abx/ZS17French_WavLMDUSTED/Normalised/test.abx_cAE_layer21 \
             results/abx/Xitsonga_WavLMUTD/Normalised/test.abx_cAE_layer21; do
        [[ -e $d ]] && mkdir -p $B/$(dirname $d) && cp -a $d $B/$d
    done
    find $B -type f | sort
}
stage backup_before_cae_redo backup

eval_cae() {  # eval_cae <tag> <feature set> <item file>
    local tag=$1 set_=$2 items=$3
    gated ap_${tag}_cAE      $PY evaluation/evaluate_ap.py data/Features/$set_/Normalised/test.samediff_cAE/layer_$LAYER
    gated abxfeat_${tag}_cAE $PY evaluation/extract_abx_phone_features.py $set_ --subset test --layer $LAYER
    gated abx_${tag}_cAE     $PY evaluation/score_abx.py data/Features/$set_/Normalised/test.abx_cAE_layer$LAYER $items
}
FR_ITEMS=data/Information/ZS17French/abx_items/test.item
XI_ITEMS=data/Information/Buckeye_tsonga/abx_items/test.item

# ---------- 1) French cAE: Poor Man's UTD, then DUSTED ----------
stage train_zs17french_poorman_cAE_redo $PY training/train_cae.py --features wavlm --corpus french --pairs poorman
eval_cae zs17french_poorman_redo ZS17French_WavLMPoorManUTD $FR_ITEMS

stage train_zs17french_dusted_cAE_redo  $PY training/train_cae.py --features wavlm --corpus french --pairs dusted
eval_cae zs17french_dusted_redo  ZS17French_WavLMDUSTED $FR_ITEMS

# ---------- 2) Xitsonga classic-UTD cAE ----------
XU=Xitsonga_WavLMUTD
stage extract_xitsonga_utd $PY features/extract_ssl.py --model wavlm --corpus xitsonga --pairs utd
stage norm_xitsonga_utd    $PY features/normalise.py --features wavlm --corpus xitsonga --pairs utd --layer $LAYER
check_utd() {
    $PY experiments/ccae_runs/check_alignments.py data/Alignments/$XU/dev.samediff.alignments/layer_$LAYER  data/Features/$XU/Normalised/dev.samediff/layer_$LAYER &&
    $PY experiments/ccae_runs/check_alignments.py data/Alignments/$XU/test.samediff.alignments/layer_$LAYER data/Features/$XU/Normalised/test.samediff/layer_$LAYER
}
# The 2026-09-16 alignments predate wordsUTD.list (2026-09-17). Keep them if they still match the
# re-extracted features; otherwise regenerate (the generator prompts for cleanup -> answer y) and re-check.
if [[ ! -f $STATE/check_xitsonga_utd.done ]] && ! check_utd > $LOGS/check_xitsonga_utd_first.log 2>&1; then
    echo "[$(date '+%F %T')] NOTE  UTD alignments stale -> regenerating" | tee -a $LOGS/pipeline.log
    stage align_xitsonga_utd $PY training/align.py --features wavlm --corpus xitsonga --pairs utd --n-cpus 8
fi
stage check_xitsonga_utd check_utd
stage train_xitsonga_utd_cAE $PY training/train_cae.py --features wavlm --corpus xitsonga --pairs utd
eval_cae xitsonga_utd $XU $XI_ITEMS

rm -f $STATE/CURRENT
echo "[$(date '+%F %T')] CAE REDO RUN DONE" | tee -a $LOGS/pipeline.log

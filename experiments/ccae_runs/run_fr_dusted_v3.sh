#!/usr/bin/env bash
# 2026-10-07 14:07: French DUSTED list replaced again (19,366 pairs / 38,732 terms, written 14:06:43).
# Re-extract -> normalise -> regenerate alignments -> check -> train cAE -> AP -> ABX.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STATE=experiments/ccae_runs/state; LOGS=experiments/ccae_runs/logs; LAYER=21
FR=ZS17French_WavLMDUSTED
source <(sed -n '/^stage() {/,/^}/p' experiments/ccae_runs/run_cae_redo.sh)

backup() {  # this list's predecessor (3,256 terms): cAE checkpoint + results; copy, never move
    local B=experiments/ccae_runs/backup_before_fr_dusted_v3
    mkdir -p $B/Models/ZS17French_WavLM && cp -a models/ZS17French_WavLM/cAE_DUSTED_layer_21.pt $B/Models/ZS17French_WavLM/
    for d in results/ap/$FR/Normalised/test.samediff_cAE results/abx/$FR/Normalised/test.abx_cAE_layer21; do
        mkdir -p $B/$(dirname $d) && cp -a $d $B/$d
    done
    find $B -type f | sort
}
stage backup_fr_dusted_v3     backup
stage extract_fr_dusted_v3    $PY features/extract_ssl.py --model wavlm --corpus french --pairs dusted
stage norm_fr_dusted_v3       $PY features/normalise.py --features wavlm --corpus french --pairs dusted --layer $LAYER
stage align_fr_dusted_v3      $PY training/align.py --features wavlm --corpus french --pairs dusted --n-cpus 8
stage check_fr_dusted_v3      $PY experiments/ccae_runs/check_alignments.py data/Alignments/$FR/train.samediff.alignments/layer_$LAYER data/Features/$FR/Normalised/train.samediff/layer_$LAYER
stage train_fr_dusted_v3_cAE  $PY training/train_cae.py --features wavlm --corpus french --pairs dusted
stage ap_fr_dusted_v3_cAE     $PY evaluation/evaluate_ap.py data/Features/$FR/Normalised/test.samediff_cAE/layer_$LAYER
stage abxfeat_fr_dusted_v3_cAE $PY evaluation/extract_abx_phone_features.py $FR --subset test --layer $LAYER
stage abx_fr_dusted_v3_cAE    $PY evaluation/score_abx.py data/Features/$FR/Normalised/test.abx_cAE_layer$LAYER data/Information/ZS17French/abx_items/test.item
rm -f $STATE/CURRENT
echo "[$(date '+%F %T')] FR DUSTED V3 RUN DONE" | tee -a $LOGS/pipeline.log

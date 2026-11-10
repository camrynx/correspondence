#!/usr/bin/env bash
# Fills the ccAE results table (gold / DUSTED / Poor Man's UTD) for English (Buckeye),
# Xitsonga and French (ZS17French), WavLM Large layer 21. The ccAE is ALWAYS randomly
# initialised (--init random -> ccAErandinit_* checkpoints and output dirs).
#
# Every stage writes a done-marker to experiments/ccae_runs/state/, so re-running this script resumes
# where it stopped. Per-stage logs are in experiments/ccae_runs/logs/. Monitor with:
#     python experiments/ccae_runs/status.py          (or: watch -n 30 python experiments/ccae_runs/status.py)

set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STATE=experiments/ccae_runs/state
LOGS=experiments/ccae_runs/logs
mkdir -p "$STATE" "$LOGS"
LAYER=21
PREFIX=ccAErandinit

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

# ---------- 0) wait for the shared GPU ----------
# Waits until no other user has a run_all.sh running AND no other user's process has been
# on the GPU for 10 consecutive minutes (checked every 60 s).
wait_for_gpu() {
    local quiet=0
    while (( quiet < 10 )); do
        local others
        others=$( { pgrep -f run_all.sh -a | awk '{print $1}';
                    nvidia-smi --query-compute-apps=pid --format=csv,noheader; } 2>/dev/null \
                  | xargs -r ps -o user= -p 2>/dev/null | grep -vx "$USER" | sort -u | tr '\n' ' ')
        if [[ -n $others ]]; then
            quiet=0; echo "[$(date '+%F %T')] GPU busy (users: $others) - waiting"
        else
            quiet=$((quiet + 1)); echo "[$(date '+%F %T')] GPU free for $quiet/10 min"
        fi
        sleep 60
    done
}
if [[ ! -f $STATE/wait_gpu.done ]]; then
    echo wait_gpu > $STATE/CURRENT
    echo "[$(date '+%F %T')] START wait_gpu" | tee -a $LOGS/pipeline.log
    wait_for_gpu > $LOGS/wait_gpu.log 2>&1
    touch $STATE/wait_gpu.done
    echo "[$(date '+%F %T')] DONE  wait_gpu" | tee -a $LOGS/pipeline.log
fi

# ---------- per-corpus settings ----------
declare -A BASE=( [buckeye]=Buckeye_WavLM [xitsonga]=Xitsonga_WavLM [zs17french]=ZS17French_WavLM )
declare -A TRAIN=( [buckeye]=devpart1 [xitsonga]=dev [zs17french]=train )
declare -A EVAL=( [buckeye]=devpart2 [xitsonga]=test [zs17french]=test )
declare -A ITEMS=( [buckeye]=data/Information/Buckeye/abx_items/devpart2.item
                   [xitsonga]=data/Information/Buckeye_tsonga/abx_items/test.item
                   [zs17french]=data/Information/ZS17French/abx_items/test.item )
declare -A SUFFIX=( [gold]="" [dusted]=DUSTED [poorman]=PoorManUTD )

check_alignments() {  # check that the alignments a training run will use match the features
    local corpus=$1 b=${BASE[$1]} tr=${TRAIN[$1]}
    stage check_${corpus}_gold    $PY experiments/ccae_runs/check_alignments.py data/Alignments/$b/$tr.samediff/layer_$LAYER                       data/Features/$b/Normalised/$tr.samediff/layer_$LAYER
    stage check_${corpus}_dusted  $PY experiments/ccae_runs/check_alignments.py data/Alignments/${b}DUSTED/$tr.samediff.alignments/layer_$LAYER     data/Features/${b}DUSTED/Normalised/$tr.samediff/layer_$LAYER
    stage check_${corpus}_poorman $PY experiments/ccae_runs/check_alignments.py data/Alignments/${b}PoorManUTD/$tr.samediff.alignments/layer_$LAYER data/Features/$b/Normalised/$tr.samediff/layer_$LAYER
}

train_and_eval() {  # train_and_eval <corpus> <pairs>
    local corpus=$1 pairs=$2
    local set=${BASE[$corpus]}${SUFFIX[$pairs]} split=${EVAL[$corpus]}
    local tag=${corpus}_${pairs}_${PREFIX}
    stage train_$tag   $PY training/train_ccae.py --corpus $corpus --pairs $pairs --init random --layer $LAYER
    stage ap_$tag      $PY evaluation/evaluate_ap.py data/Features/$set/Normalised/$split.samediff_$PREFIX/layer_$LAYER
    stage abxfeat_$tag $PY evaluation/extract_abx_phone_features.py $set --subset $split --ccae --randinit --layer $LAYER
    stage abx_$tag     $PY evaluation/score_abx.py data/Features/$set/Normalised/$split.abx_${PREFIX}_layer$LAYER ${ITEMS[$corpus]}
}

# ---------- 1) French: gold features are current; DUSTED list changed -> redo DUSTED ----------
stage extract_zs17french_dusted $PY features/extract_ssl.py --model wavlm --corpus french --pairs dusted
stage norm_zs17french_dusted    $PY features/normalise.py --features wavlm --corpus french --pairs dusted --layer $LAYER
stage align_zs17french_dusted   $PY training/align.py --features wavlm --corpus french --pairs dusted --n-cpus 8
check_alignments zs17french
for p in gold dusted poorman; do train_and_eval zs17french $p; done

# ---------- 2) English (Buckeye): re-extract + normalise (DUSTED list unchanged since alignments) ----------
stage extract_buckeye_gold    $PY features/extract_ssl.py --model wavlm --corpus buckeye
stage norm_buckeye_gold       $PY features/normalise.py --features wavlm --corpus buckeye --layer $LAYER
stage extract_buckeye_dusted  $PY features/extract_ssl.py --model wavlm --corpus buckeye --pairs dusted
stage norm_buckeye_dusted     $PY features/normalise.py --features wavlm --corpus buckeye --pairs dusted --layer $LAYER
check_alignments buckeye
for p in gold dusted poorman; do train_and_eval buckeye $p; done

# ---------- 3) Xitsonga: re-extract + normalise; DUSTED list changed -> redo DUSTED alignments ----------
stage extract_xitsonga_gold   $PY features/extract_ssl.py --model wavlm --corpus xitsonga
stage norm_xitsonga_gold      $PY features/normalise.py --features wavlm --corpus xitsonga --layer $LAYER
stage extract_xitsonga_dusted $PY features/extract_ssl.py --model wavlm --corpus xitsonga --pairs dusted
stage norm_xitsonga_dusted    $PY features/normalise.py --features wavlm --corpus xitsonga --pairs dusted --layer $LAYER
# This generator asks "Clean up intermediate files (y/n)?" per split via input(); answer y
# (what the ZS17French generator does automatically) since the pipeline has no stdin.
stage align_xitsonga_dusted   $PY training/align.py --features wavlm --corpus xitsonga --pairs dusted --n-cpus 8
check_alignments xitsonga
for p in gold dusted poorman; do train_and_eval xitsonga $p; done

rm -f $STATE/CURRENT
echo "[$(date '+%F %T')] ALL DONE" | tee -a $LOGS/pipeline.log

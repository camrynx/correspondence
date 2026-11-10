#!/usr/bin/env bash
# Every command behind the paper's results tables, from feature extraction to scores.
#
#     bash scripts/reproduce_paper.sh            # everything (days of GPU time)
#
# Run it from the repository root, in the conda environment from environment.yml, after
# scripts/setup_speech_dtw.sh. It expects the corpora and lists described in the README
# under data/. Sections can also be run by hand; the order matters:
# features -> normalisation -> alignments -> training -> evaluation.
#
# French has no classic UTD word list, so it has no UTD rows.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

CORPORA="buckeye xitsonga french"
pairs_for() {  # word-pair sets available for a corpus
    if [ "$1" = french ]; then echo "gold dusted"; else echo "gold utd dusted"; fi
}

# ============================================================================
# 1) Preprocessing - windows.list (one forward pass per window, <= 90 s)
# ============================================================================
# data/Information/<corpus>/ must already hold utterances.list, words.list,
# wordsUTD.list (not French), UTDWordsDUSTED.txt, Splits/ and abx_items/ (see README).
for info in Buckeye Buckeye_tsonga ZS17French; do
    [ -f data/Information/$info/windows.list ] ||
        $PY preprocessing/make_windows.py data/Information/$info/utterances.list data/Information/$info/windows.list --max-window 90
done

# ============================================================================
# 2) Features (GPU for WavLM/HuBERT). MFCCs are speaker-normalised at extraction;
#    WavLM (layer 21) and HuBERT (layer 24) are normalised with training-split stats.
# ============================================================================
for corpus in $CORPORA; do
    for pairs in $(pairs_for $corpus); do
        $PY features/extract_mfcc.py --corpus $corpus --pairs $pairs
        for model in wavlm hubert; do
            $PY features/extract_ssl.py --model $model --corpus $corpus --pairs $pairs
            $PY features/normalise.py --features $model --corpus $corpus --pairs $pairs
        done
    done
done

# ============================================================================
# 3) Alignments - word pairs via DTW (CPU); Poor Man's UTD via nearest-neighbour
#    search over the gold features (uses the GPU if there is one)
# ============================================================================
for features in mfcc wavlm hubert; do
    for corpus in $CORPORA; do
        for pairs in $(pairs_for $corpus) poorman; do
            $PY training/align.py --features $features --corpus $corpus --pairs $pairs --n-cpus 8
        done
    done
done

# ============================================================================
# 4) Training (GPU) - each run also encodes every split's gold words for evaluation
# ============================================================================
for features in mfcc wavlm hubert; do
    for corpus in $CORPORA; do
        for pairs in $(pairs_for $corpus) poorman; do
            $PY training/train_cae.py  --features $features --corpus $corpus --pairs $pairs   # MSE cAE
            $PY training/train_ccae.py --features $features --corpus $corpus --pairs $pairs   # InfoNCE cAE, random init
        done
    done
done

# ============================================================================
# 5) Evaluation (GPU) - same-different AP and triphone ABX
# ============================================================================
# English is reported on zs (the held-out speakers), Xitsonga and French on test.

# Baselines: the input features themselves, no cAE
for spec in "Buckeye zs Buckeye" "Xitsonga test Buckeye_tsonga" "ZS17French test ZS17French"; do
    set -- $spec; prefix=$1; subset=$2; info=$3
    items=data/Information/$info/abx_items/$subset.item
    $PY evaluation/evaluate_ap.py data/Features/${prefix}_MFCCs/$subset.samediff
    $PY evaluation/extract_abx_phone_features.py ${prefix}_MFCCs --subset $subset --raw
    $PY evaluation/score_abx.py data/Features/${prefix}_MFCCs/$subset.abx $items
    for fl in WavLM:21 HuBERT:24; do
        F=${fl%%:*}; L=${fl##*:}
        $PY evaluation/evaluate_ap.py data/Features/${prefix}_$F/Normalised/$subset.samediff/layer_$L
        $PY evaluation/extract_abx_phone_features.py ${prefix}_$F --subset $subset --raw --layer $L
        $PY evaluation/score_abx.py data/Features/${prefix}_$F/Normalised/$subset.abx_layer$L $items
    done
done

# Trained models: evaluate.py runs AP + ABX for every flag combination
$PY evaluate.py --buckeye  --mfcc --wavlm --hubert --gold --utd --poorman --dusted --cae --ccae --subset zs
$PY evaluate.py --xitsonga --mfcc --wavlm --hubert --gold --utd --poorman --dusted --cae --ccae
$PY evaluate.py --french   --mfcc --wavlm --hubert --gold       --poorman --dusted --cae --ccae

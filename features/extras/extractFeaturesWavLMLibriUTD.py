"""
Extract WavLM features for LibriSpeech DUSTED terms. There is only a DUSTED list, over
train-clean-5, so --dusted is required.

    python features/extras/extractFeaturesWavLMLibriUTD.py --dusted
"""

import argparse
import glob
from os import path
import os
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

# ========= CONFIG =========
librispeech_datadir = "data/Datasets/LibriSpeech"  # holds train-clean-5/ and dev-clean/
librispeechSplits_dir = "data/Information/LibriSpeech/Splits"

utterances_list_fn = "data/Information/LibriSpeech/utterances.list"

parser = argparse.ArgumentParser(description="Extract WavLM features for LibriSpeech DUSTED UTD word segments.")
parser.add_argument(
    "--dusted", action="store_true",
    help="Use the DUSTED UTD word list. Currently REQUIRED - there is no "
    "legacy UTD word list for LibriSpeech (no wordsUTD.list exists).",
    )
args = parser.parse_args()

if not args.dusted:
    raise SystemExit(
        "Error: --dusted is required for LibriSpeech - there is no legacy "
        "UTD word list (data/Information/LibriSpeech/wordsUTD.list does "
        "not exist), only UTDWordsDUSTED.txt."
    )

USE_DUSTED = args.dusted

_DEFAULT_FEATURE_SET = "LibriSpeech_WavLMDUSTED"
words_list_fn = "data/Information/LibriSpeech/UTDWordsDUSTED.txt"

WAVLM_MODEL_NAME = os.environ.get("SD_MODEL", "microsoft/wavlm-large")
FEATURE_SET = os.environ.get("SD_FEATURE_SET", _DEFAULT_FEATURE_SET)
feature_dir = f"data/Features/{FEATURE_SET}/NotNormalised"

WAVLM_SAMPLE_RATE = 16000
WAVLM_FRAME_RATE = 50
WAVLM_HOP = WAVLM_SAMPLE_RATE // WAVLM_FRAME_RATE  # 320 samples ~= 20ms

# pad 40 samples each side (window 400, hop 320) so frames are centred.
PAD_TO_HOP = True
CENTER_PAD = True
WAVLM_CENTER_PAD = 40

# UTD/DUSTED terms are discovered, not forced-aligned, so the gold-word
# criteria (>= 5 characters, >= 0.5 s) don't apply - labels are cluster IDs
# like "PT100". Only sub-frame terms are dropped.
MIN_WORD_DURATION = 0.02  # shorter than one WavLM frame (20ms)

# DUSTED was only mined over train-clean-5 - see module docstring.
SUBSETS = ["train-clean-5"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(WAVLM_MODEL_NAME)

# Which layers to save; 0=embedding, 1-24=transformer layers. Must be an
# ITERABLE: a single layer needs a trailing comma, e.g. (21,). Matches LAYER
# in training/*_WavLMLibriUTD.py.
LAYERS = (21,)


# ========= HELPERS =========

def loadSpeakerList(subset):
    fn = path.join(librispeechSplits_dir, f"librispeech_{subset}_speakers.list")
    with open(fn, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ========= READ FILES =========

def load_utterances(fn):
    """file_label -> list of (seg_id, start, end). See extractFeaturesWavLMLibri.py."""
    utts = {}
    with open(fn) as f:
        for line in f:
            seg_id, start, end = line.strip().split()
            file_label = seg_id.split("_")[1]
            utts.setdefault(file_label, []).append((seg_id, float(start), float(end)))
    return utts


def load_words(fn):
    """file_label -> [(label, start, end)]."""
    words = {}
    n_brief = 0
    with open(fn) as f:
        for line in f:
            label, file_label, start, end = line.strip().split()
            if round(float(end) - float(start), 3) < MIN_WORD_DURATION:
                n_brief += 1
                continue
            words.setdefault(file_label, []).append((label, float(start), float(end)))

    print(
        f"Loaded {sum(len(v) for v in words.values())} words from {fn} "
        f"({n_brief} under {MIN_WORD_DURATION}s skipped)"
        )
    return words


def load_signal(wav_fn):
    """Read a flac as a mono 16 kHz torch tensor."""
    signal, sr = sf.read(wav_fn, dtype="float32")
    signal = torch.from_numpy(signal)
    if signal.ndim == 2:
        signal = signal.mean(-1)
    if sr != WAVLM_SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, WAVLM_SAMPLE_RATE)
    return signal


def wavlm_forward(chunk):
    """Pad to the hop, centre-pad, layer-norm, forward."""
    wav = chunk.float()
    if PAD_TO_HOP and (r := wav.numel() % WAVLM_HOP):
        wav = F.pad(wav, (0, WAVLM_HOP - r))
    if CENTER_PAD:
        wav = F.pad(wav, (WAVLM_CENTER_PAD, WAVLM_CENTER_PAD))

    x = F.layer_norm(wav.to(device), wav.shape).unsqueeze(0)
    with torch.no_grad():
        hs = model(x, output_hidden_states=True).hidden_states

    feats_by_layer = []
    for L in LAYERS:
        feats = hs[L].squeeze(0).cpu().numpy()
        assert feats.ndim == 2 and feats.shape[-1] == 1024, feats.shape
        feats_by_layer.append(feats)

    del x, hs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats_by_layer


# ========= CORE =========

def extract_subset(speakers, subset, utterances, words):

    output_dir = path.join(feature_dir, subset + ".samediff")
    os.makedirs(output_dir, exist_ok=True)

    subset_datadir = path.join(librispeech_datadir, subset)

    print(f"\nProcessing {subset}...")

    n_words = 0
    dropped_words = []

    for speaker in tqdm(speakers, desc="Speakers"):

        for wav_fn in sorted(glob.glob(path.join(subset_datadir, speaker, "*", "*.flac"))):

            filename = path.splitext(path.basename(wav_fn))[0]
            if filename not in utterances:
                continue

            signal = load_signal(wav_fn)

            recording_words_by_layer = [{} for _ in LAYERS]
            matched_words = set()

            for (seg_id, start_t, end_t) in utterances[filename]:

                chunk = signal[int(start_t * WAVLM_SAMPLE_RATE):int(end_t * WAVLM_SAMPLE_RATE)]
                if len(chunk) == 0:
                    continue

                feats_by_layer = wavlm_forward(chunk)

                # Use round(), not floor/ceil - see extractFeaturesWavLMUTD.py.
                utt_start_frame = round(start_t * WAVLM_FRAME_RATE)
                utt_end_frame = round(end_t * WAVLM_FRAME_RATE) + 1

                for (label, w_start, w_end) in words.get(filename, []):

                    word_start = round(w_start * WAVLM_FRAME_RATE)
                    word_end = round(w_end * WAVLM_FRAME_RATE) + 1

                    # One frame of slack at each edge (term times are on a 10 ms grid).
                    if not (word_start >= utt_start_frame - 1 and word_end <= utt_end_frame + 1):
                        continue

                    offset_start = max(0, word_start - utt_start_frame)
                    offset_end = min(len(feats_by_layer[0]), word_end - utt_start_frame)
                    if offset_end <= offset_start:
                        continue

                    word_key = f"{label}_{seg_id}_{word_start:06d}-{word_end:06d}"
                    for layer_idx, feats in enumerate(feats_by_layer):
                        recording_words_by_layer[layer_idx][word_key] = feats[offset_start:offset_end]
                    matched_words.add((label, w_start, w_end))
                    n_words += 1

            for (label, w_start, w_end) in words.get(filename, []):
                if (label, w_start, w_end) not in matched_words:
                    dropped_words.append((label, filename, w_start, w_end))

            for L, recording_words in zip(LAYERS, recording_words_by_layer):
                if not recording_words:
                    continue
                layer_dir = path.join(output_dir, f"layer_{L:02d}")
                os.makedirs(layer_dir, exist_ok=True)
                np.savez_compressed(path.join(layer_dir, filename + ".npz"), **recording_words)

            del signal

    dropped_words_fn = path.join(output_dir, "dropped_words.txt")
    with open(dropped_words_fn, "w") as f:
        for label, filename, w_start, w_end in dropped_words:
            f.write("{} {} {}-{}\n".format(label, filename, w_start, w_end))

    print(
        "Wrote", n_words, "word segments for", subset,
        "(", len(dropped_words), "words skipped, see", dropped_words_fn, ")"
        )


# ========= MAIN =========

def main():

    print("Loading WavLM...")
    model.to(device)
    model.eval()
    print("feat_extract_norm:", model.config.feat_extract_norm, "| layers:", LAYERS)

    print("Reading lists...")
    utterances = load_utterances(utterances_list_fn)
    words = load_words(words_list_fn)

    for subset in SUBSETS:
        extract_subset(loadSpeakerList(subset), subset, utterances, words)


if __name__ == "__main__":
    main()

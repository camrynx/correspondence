"""
Extract SpidR-Adapt features for Buckeye words, laid out like the WavLM/HuBERT features.

    python features/extras/extractFeaturesSpidR.py
    SD_LAYERS=12 SD_FEATURE_SET=Buckeye_SpidR_reptile_sl python features/extras/extractFeaturesSpidR.py
    SD_MODEL=/path/to/checkpoint.pt SD_FEATURE_SET=Buckeye_SpidR_adapted python features/extras/extractFeaturesSpidR.py
"""

import glob
from os import path
import os
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import math

from spidr_adapt.models import build_model

# ========= CONFIG =========
buckeye_datadir = "data/Datasets/Buckeye"
buckeyeSplits_dir = "data/Information/Buckeye/Splits"

# Extraction windows (preprocessing/make_windows.py): one forward pass per window, words sliced out of its features.
windows_list_fn = "data/Information/Buckeye/windows.list"
words_list_fn = "data/Information/Buckeye/words.list"

SPIDR_CHECKPOINT = os.environ.get(
    "SD_MODEL", "models/Spidr-Adapted/Spidr_Final_Buckeye.pt"
)
SPIDR_MODEL_TYPE = os.environ.get("SD_SPIDR_MODEL_TYPE", "spidr_reset")
FEATURE_SET = os.environ.get("SD_FEATURE_SET", "Buckeye_SpidR")
feature_dir = f"data/Features/{FEATURE_SET}/NotNormalised"

SPIDR_SAMPLE_RATE = 16000
SPIDR_FRAME_RATE = 50
SPIDR_HOP = SPIDR_SAMPLE_RATE // SPIDR_FRAME_RATE  # 320 samples ~= 20ms

# Pad 40 samples each side (window 400, hop 320) so frames are centred.
PAD_TO_HOP = True
CENTER_PAD = True
SPIDR_CENTER_PAD = 40

# Word selection as in Carlin et al. (2011): "word examples at least 0.5 sec in length
# and five characters as text"
MIN_WORD_DURATION = 0.5
MIN_WORD_CHARS = 5

SUBSETS = ["devpart1", "devpart2", "zs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(model_type=SPIDR_MODEL_TYPE, checkpoint=SPIDR_CHECKPOINT)

# get_intermediate_outputs() has one entry per transformer layer and no embedding
# entry, so layer N (1-indexed, as for HuBERT/WavLM) is list[N - 1]. All 12 layers by
# default; SD_LAYERS picks some, e.g. "8" or "6,9,12".
LAYERS = tuple(int(x) for x in os.environ.get("SD_LAYERS", ",".join(str(i) for i in range(1, 13))).split(","))


# ========= HELPERS =========

def uttlabel_to_uttkey(file_label):
    """e.g. s0201a -> s02_01a. This names the output file per RECORDING."""
    return file_label[0:3] + "_" + file_label[3:]


def loadSpeakerList(subset):
    fn = path.join(buckeyeSplits_dir, f"buckeye_{subset}_speakers.list")
    with open(fn, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ========= READ FILES =========

def load_utterances(fn):
    """file_label -> list of (seg_id, start, end)."""
    utts = {}
    with open(fn) as f:
        for line in f:
            seg_id, start, end = line.strip().split()
            file_label = seg_id.split("_")[1]  # e.g. s0701a
            utts.setdefault(file_label, []).append((seg_id, float(start), float(end)))
    return utts


def load_words(fn):
    """file_label -> [(label, start, end)], skipping <unk> and short words."""
    words = {}
    n_unk = n_short = n_brief = 0
    with open(fn) as f:
        for line in f:
            label, file_label, start, end = line.strip().split()
            if label == "<unk>":
                n_unk += 1
                continue
            if len(label) < MIN_WORD_CHARS:
                n_short += 1
                continue
            if round(float(end) - float(start), 3) < MIN_WORD_DURATION:
                n_brief += 1
                continue
            words.setdefault(file_label, []).append((label, float(start), float(end)))

    print(
        f"Loaded {sum(len(v) for v in words.values())} words from {fn} "
        f"({n_unk} <unk>, {n_short} under {MIN_WORD_CHARS} chars, "
        f"{n_brief} under {MIN_WORD_DURATION}s skipped)"
        )
    return words


def load_signal(wav_fn):
    """Read a wav as a mono 16 kHz torch tensor (soundfile, like the Libri script)."""
    signal, sr = sf.read(wav_fn, dtype="float32")
    signal = torch.from_numpy(signal)
    if signal.ndim == 2:
        signal = signal.mean(-1)
    if sr != SPIDR_SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SPIDR_SAMPLE_RATE)
    return signal


def spidr_forward(chunk):
    """Pad to the hop, centre-pad, layer-norm (as SpidR-Adapt's training does), forward.

    Returns one [frames, MODEL_DIMENSIONS] array per entry in LAYERS.
    """
    wav = chunk.float()
    if PAD_TO_HOP and (r := wav.numel() % SPIDR_HOP):
        wav = F.pad(wav, (0, SPIDR_HOP - r))
    if CENTER_PAD:
        wav = F.pad(wav, (SPIDR_CENTER_PAD, SPIDR_CENTER_PAD))

    x = F.layer_norm(wav.to(device), wav.shape).unsqueeze(0)
    with torch.no_grad():
        outputs = model.get_intermediate_outputs(x)  # list[i] = output of transformer layer i+1

    feats_by_layer = []
    for L in LAYERS:
        feats = outputs[L - 1].squeeze(0).cpu().numpy()
        assert feats.ndim == 2 and feats.shape[-1] == MODEL_DIMENSIONS, feats.shape
        feats_by_layer.append(feats)

    del x, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats_by_layer


# ========= CORE =========

def extract_subset(speakers, subset, utterances, words):

    output_dir = path.join(feature_dir, subset + ".samediff")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {subset}...")

    n_words = 0
    dropped_words = []

    for speaker in tqdm(speakers, desc="Speakers"):

        # Buckeye nests audio one level deep: <speaker>/<file_label>.wav
        for wav_fn in sorted(glob.glob(path.join(buckeye_datadir, speaker, "*.wav"))):

            filename = path.splitext(path.basename(wav_fn))[0]
            if filename not in utterances:
                continue

            signal = load_signal(wav_fn)

            recording_words_by_layer = [{} for _ in LAYERS]
            matched_words = set()

            for (seg_id, start_t, end_t) in utterances[filename]:

                chunk = signal[int(start_t * SPIDR_SAMPLE_RATE):int(end_t * SPIDR_SAMPLE_RATE)]
                if len(chunk) == 0:
                    continue

                # === SpidR, one forward pass per extraction window ===
                feats_by_layer = spidr_forward(chunk)

                # === slice out each word by frame offset ===
                utt_start_frame = int(math.floor(start_t * SPIDR_FRAME_RATE))
                utt_end_frame = int(math.ceil(end_t * SPIDR_FRAME_RATE))

                for (label, w_start, w_end) in words.get(filename, []):

                    word_start = int(math.floor(w_start * SPIDR_FRAME_RATE))
                    word_end = int(math.ceil(w_end * SPIDR_FRAME_RATE))

                    # word must fall entirely within this segment, not just
                    # start inside it
                    if not (word_start >= utt_start_frame and word_end <= utt_end_frame):
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

            # one .npz per RECORDING per layer (e.g. layer_12/s02_01a.npz)
            utt_key = uttlabel_to_uttkey(filename)
            for L, recording_words in zip(LAYERS, recording_words_by_layer):
                if not recording_words:
                    continue
                layer_dir = path.join(output_dir, f"layer_{L:02d}")
                os.makedirs(layer_dir, exist_ok=True)
                np.savez_compressed(path.join(layer_dir, utt_key + ".npz"), **recording_words)

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
    global MODEL_DIMENSIONS

    print("Loading SpidR-Adapt...")
    model.to(device)
    model.eval()

    # Probe MODEL_DIMENSIONS from the model itself rather than hardcoding -
    # SpidR has no HF-style .config.hidden_size to read.
    with torch.no_grad():
        probe = model.get_intermediate_outputs(torch.zeros(1, SPIDR_HOP * 4, device=device))
    MODEL_DIMENSIONS = probe[0].shape[-1]

    print(f"checkpoint: {SPIDR_CHECKPOINT} | model_type: {SPIDR_MODEL_TYPE} "
          f"| dims: {MODEL_DIMENSIONS} | layers: {LAYERS}")

    print("Reading lists...")
    utterances = load_utterances(windows_list_fn)
    words = load_words(words_list_fn)

    for subset in SUBSETS:
        extract_subset(loadSpeakerList(subset), subset, utterances, words)


if __name__ == "__main__":
    main()

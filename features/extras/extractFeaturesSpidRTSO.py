"""
Extract SpidR-Adapt features for Xitsonga words (see extractFeaturesSpidR.py).

    python features/extras/extractFeaturesSpidRTSO.py
    SD_MODEL=/path/to/checkpoint.pt SD_FEATURE_SET=Xitsonga_SpidR_adapted python features/extras/extractFeaturesSpidRTSO.py
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
xitsonga_datadir = "data/Datasets/Buckeye-Xitsonga"
xitsongaSplits_dir = "data/Information/Buckeye_tsonga/Splits"

# Extraction windows (preprocessing/make_windows.py): one forward pass per window, words sliced out of its features.
windows_list_fn = "data/Information/Buckeye_tsonga/windows.list"
words_list_fn = "data/Information/Buckeye_tsonga/words.list"

SPIDR_CHECKPOINT = os.environ.get(
    "SD_MODEL", "models/Spidr-Adapted/Spidr_Final_Xitsonga.pt"
)
SPIDR_MODEL_TYPE = os.environ.get("SD_SPIDR_MODEL_TYPE", "spidr_reset")
FEATURE_SET = os.environ.get("SD_FEATURE_SET", "Xitsonga_SpidR")
feature_dir = f"data/Features/{FEATURE_SET}/NotNormalised"

SPIDR_SAMPLE_RATE = 16000
SPIDR_FRAME_RATE = 50
SPIDR_HOP = SPIDR_SAMPLE_RATE // SPIDR_FRAME_RATE  # 320 samples ~= 20ms

PAD_TO_HOP = True
CENTER_PAD = True
SPIDR_CENTER_PAD = 40

# Word selection as in Carlin et al. (2011): "word examples at least 0.5 sec in length
# and five characters as text"
MIN_WORD_DURATION = 0.5
MIN_WORD_CHARS = 5

SUBSETS = ["dev", "test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(model_type=SPIDR_MODEL_TYPE, checkpoint=SPIDR_CHECKPOINT)

# See extractFeaturesSpidR.py for why this defaults to every layer instead of
# one. Override with SD_LAYERS, e.g. "SD_LAYERS=8" or "SD_LAYERS=6,9,12".
LAYERS = tuple(int(x) for x in os.environ.get("SD_LAYERS", ",".join(str(i) for i in range(1, 13))).split(","))


# ========= HELPERS =========

def loadSpeakerList(subset):
    """subset is "dev" or "test". The split files list speakers as
    "nchlt-tso-001m"; we only need the "001m" part to match get_speaker()
    in save_utterances_and_words_xitsonga.py."""
    fn = path.join(xitsongaSplits_dir, f"tsong{subset}_speakers.list")
    with open(fn, "r") as f:
        return [line.strip().split("-")[-1] for line in f if line.strip()]


# ========= READ FILES =========

def load_utterances(fn):
    """file_label -> [(seg_id, start, end)]; seg_id is <speaker>_<file_label>_<seg_idx>."""
    utts = {}
    with open(fn) as f:
        for line in f:
            seg_id, start, end = line.strip().split()
            _, rest = seg_id.split("_", 1)
            file_label, _ = rest.rsplit("_", 1)
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
    """As in extractFeaturesSpidR.py: one [frames, MODEL_DIMENSIONS] array per layer."""
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

        # Xitsonga keeps every wav in one flat directory, one per recording
        wav_pattern = path.join(xitsonga_datadir, f"nchlt_tso_{speaker}_*.wav")

        for wav_fn in sorted(glob.glob(wav_pattern)):

            filename = path.splitext(path.basename(wav_fn))[0]
            if filename not in utterances:
                continue

            signal = load_signal(wav_fn)

            recording_words_by_layer = [{} for _ in LAYERS]
            matched_words = set()

            # Each Xitsonga recording is one utterance: one forward pass per file.
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

            # one .npz per RECORDING per layer (each wav file IS one
            # recording in the Xitsonga corpus), e.g.
            # layer_12/nchlt_tso_001m_0007.npz
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
    global MODEL_DIMENSIONS

    print("Loading SpidR-Adapt...")
    model.to(device)
    model.eval()

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

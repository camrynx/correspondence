"""
Extract WavLM features for Buckeye phone tokens (phones.list from
preprocessing/save_phones_buckeye.py), the same way as for words. Only devpart1 and
devpart2 (the features are only used for training pairs) and one layer (--layer, default 21).

    python features/extras/extractFeaturesWavLMPhone.py --layer 21
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
import math

# ========= CONFIG =========
parser = argparse.ArgumentParser(description="Extract WavLM features for Buckeye phone tokens.")
parser.add_argument(
    "--layer", type=int, default=21,
    help="which WavLM hidden_states layer to extract (default: %(default)s, matching "
         "the layer the cAE/ccAE are trained on)",
    )
args = parser.parse_args()

buckeye_datadir = "data/Datasets/Buckeye"
buckeyeSplits_dir = "data/Information/Buckeye/Splits"

# Extraction windows (preprocessing/make_windows.py): one forward pass per window, words sliced out of its features.
windows_list_fn = "data/Information/Buckeye/windows.list"
phones_list_fn = "data/Information/Buckeye/phones.list"

WAVLM_MODEL_NAME = os.environ.get("SD_MODEL", "microsoft/wavlm-large")
FEATURE_SET = os.environ.get("SD_FEATURE_SET", "Buckeye_WavLMPhone")
feature_dir = f"data/Features/{FEATURE_SET}/NotNormalised"

MODEL_DIMENSIONS = 1024

WAVLM_SAMPLE_RATE = 16000
WAVLM_FRAME_RATE = 50
WAVLM_HOP = WAVLM_SAMPLE_RATE // WAVLM_FRAME_RATE  # 320 samples ~= 20ms

# pad 40 samples each side (window 400, hop 320) so frames are centred.
PAD_TO_HOP = True
CENTER_PAD = True
WAVLM_CENTER_PAD = 40

# Safety net only - phones.list has already had this filter applied by
# preprocessing/save_phones_buckeye.py. 0.06s == 3 frames at 50 Hz.
MIN_PHONE_DURATION = 0.06

# Phone features are only ever used to build training frame pairs; evaluation runs on
# the gold word features. zs is therefore not extracted.
SUBSETS = ["devpart1", "devpart2"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(WAVLM_MODEL_NAME)

LAYERS = (args.layer,)


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


def load_phones(fn):
    """file_label -> [(label, start, end)] (phones.list is already filtered)."""
    phones = {}
    n_brief = 0
    with open(fn) as f:
        for line in f:
            label, file_label, start, end = line.strip().split()
            if round(float(end) - float(start), 3) < MIN_PHONE_DURATION:
                n_brief += 1
                continue
            phones.setdefault(file_label, []).append((label, float(start), float(end)))

    print(
        f"Loaded {sum(len(v) for v in phones.values())} phones from {fn} "
        f"({n_brief} under {MIN_PHONE_DURATION}s skipped)"
        )
    return phones


def load_signal(wav_fn):
    """Read a wav as a mono 16 kHz torch tensor (soundfile, like the Libri script)."""
    signal, sr = sf.read(wav_fn, dtype="float32")
    signal = torch.from_numpy(signal)
    if signal.ndim == 2:
        signal = signal.mean(-1)
    if sr != WAVLM_SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, WAVLM_SAMPLE_RATE)
    return signal


def wavlm_forward(chunk):
    """Pad to the hop, centre-pad, layer-norm, forward; one [frames, 1024] array per layer."""
    wav = chunk.float()
    if PAD_TO_HOP and (r := wav.numel() % WAVLM_HOP):
        wav = F.pad(wav, (0, WAVLM_HOP - r))
    if CENTER_PAD:
        wav = F.pad(wav, (WAVLM_CENTER_PAD, WAVLM_CENTER_PAD))

    x = F.layer_norm(wav.to(device), wav.shape).unsqueeze(0)
    with torch.no_grad():
        hs = model(x, output_hidden_states=True).hidden_states  # [0]=embed, [L]=layer L

    feats_by_layer = []
    for L in LAYERS:
        feats = hs[L].squeeze(0).cpu().numpy()
        assert feats.ndim == 2 and feats.shape[-1] == MODEL_DIMENSIONS, feats.shape
        feats_by_layer.append(feats)

    del x, hs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats_by_layer


# ========= CORE =========

def extract_subset(speakers, subset, utterances, phones):

    output_dir = path.join(feature_dir, subset + ".samediff")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {subset}...")

    n_phones = 0
    dropped_phones = []

    for speaker in tqdm(speakers, desc="Speakers"):

        # Buckeye nests audio one level deep: <speaker>/<file_label>.wav
        for wav_fn in sorted(glob.glob(path.join(buckeye_datadir, speaker, "*.wav"))):

            filename = path.splitext(path.basename(wav_fn))[0]
            if filename not in utterances:
                continue

            signal = load_signal(wav_fn)

            recording_phones_by_layer = [{} for _ in LAYERS]
            matched_phones = set()

            for (seg_id, start_t, end_t) in utterances[filename]:

                chunk = signal[int(start_t * WAVLM_SAMPLE_RATE):int(end_t * WAVLM_SAMPLE_RATE)]
                if len(chunk) == 0:
                    continue

                # === WavLM, one forward pass per extraction window ===
                feats_by_layer = wavlm_forward(chunk)

                # === slice out each phone by frame offset ===
                utt_start_frame = int(math.floor(start_t * WAVLM_FRAME_RATE))
                utt_end_frame = int(math.ceil(end_t * WAVLM_FRAME_RATE))

                for (label, p_start, p_end) in phones.get(filename, []):

                    phone_start = int(math.floor(p_start * WAVLM_FRAME_RATE))
                    phone_end = int(math.ceil(p_end * WAVLM_FRAME_RATE))

                    # phone must fall entirely within this segment, not just
                    # start inside it
                    if not (phone_start >= utt_start_frame and phone_end <= utt_end_frame):
                        continue

                    offset_start = max(0, phone_start - utt_start_frame)
                    offset_end = min(len(feats_by_layer[0]), phone_end - utt_start_frame)
                    if offset_end <= offset_start:
                        continue

                    phone_key = f"{label}_{seg_id}_{phone_start:06d}-{phone_end:06d}"
                    for layer_idx, feats in enumerate(feats_by_layer):
                        recording_phones_by_layer[layer_idx][phone_key] = feats[offset_start:offset_end]
                    matched_phones.add((label, p_start, p_end))
                    n_phones += 1

            for (label, p_start, p_end) in phones.get(filename, []):
                if (label, p_start, p_end) not in matched_phones:
                    dropped_phones.append((label, filename, p_start, p_end))

            # one .npz per RECORDING per layer (e.g. layer_21/s02_01a.npz)
            utt_key = uttlabel_to_uttkey(filename)
            for L, recording_phones in zip(LAYERS, recording_phones_by_layer):
                if not recording_phones:
                    continue
                layer_dir = path.join(output_dir, f"layer_{L:02d}")
                os.makedirs(layer_dir, exist_ok=True)
                np.savez_compressed(path.join(layer_dir, utt_key + ".npz"), **recording_phones)

            del signal

    dropped_phones_fn = path.join(output_dir, "dropped_phones.txt")
    with open(dropped_phones_fn, "w") as f:
        for label, filename, p_start, p_end in dropped_phones:
            f.write("{} {} {}-{}\n".format(label, filename, p_start, p_end))

    print(
        "Wrote", n_phones, "phone segments for", subset,
        "(", len(dropped_phones), "phones skipped, see", dropped_phones_fn, ")"
        )


# ========= MAIN =========

def main():

    print("Loading WavLM...")
    model.to(device)
    model.eval()
    print("feat_extract_norm:", model.config.feat_extract_norm, "| layers:", LAYERS)

    print("Reading lists...")
    utterances = load_utterances(windows_list_fn)
    phones = load_phones(phones_list_fn)

    for subset in SUBSETS:
        extract_subset(loadSpeakerList(subset), subset, utterances, phones)


if __name__ == "__main__":
    main()

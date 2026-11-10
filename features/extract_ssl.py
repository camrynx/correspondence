"""
Extract WavLM / HuBERT features for word tokens.

    python features/extract_ssl.py --model wavlm --corpus buckeye
    python features/extract_ssl.py --model hubert --corpus xitsonga --pairs utd
    python features/extract_ssl.py --model wavlm --corpus buckeye --layers 12 21

One forward pass per window in windows.list (input padded to the hop, centre-padded by
40 samples and layer-normalised), then each token inside the window is sliced out.
Writes data/Features/<Corpus>_<Model>[UTD|DUSTED]/NotNormalised/<split>.samediff/layer_NN/;
run normalise.py afterwards.

Gold words (words.list) skip <unk>, words under 5 characters and under 0.5 s (the French
list is filtered already). Discovered terms (wordsUTD.list, UTDWordsDUSTED.txt) only skip
sub-frame terms; for Buckeye and Xitsonga their boundaries are rounded to frames with one
frame of slack at window edges, since their times are on a 10 ms grid.

SD_MODEL and SD_FEATURE_SET override the model and the output set name.
"""

import argparse
import glob
import math
import os
from os import path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoModel

SAMPLE_RATE = 16000
FRAME_RATE = 50
HOP = SAMPLE_RATE // FRAME_RATE  # 320 samples = 20 ms
CENTER_PAD = 40  # samples each side (window 400, hop 320), so frames are centred
MODEL_DIMENSIONS = 1024

MODELS = {
    "wavlm": dict(name="microsoft/wavlm-large", tag="WavLM", layer=21),
    "hubert": dict(name="facebook/hubert-large-ll60k", tag="HuBERT", layer=24),
}
CORPORA = {
    "buckeye": dict(prefix="Buckeye", info="Buckeye", data="data/Datasets/Buckeye",
                    splits=("devpart1", "devpart2", "zs"), speakers_file="buckeye_{}_speakers.list"),
    "xitsonga": dict(prefix="Xitsonga", info="Buckeye_tsonga", data="data/Datasets/Buckeye-Xitsonga",
                     splits=("dev", "test"), speakers_file="tsong{}_speakers.list"),
    "french": dict(prefix="ZS17French", info="ZS17French", data="data/Datasets/FrenchDatasets/french/train",
                   splits=("train", "test"), speakers_file="zs17french{}_speakers.list"),
}
WORD_LISTS = {"gold": "words.list", "utd": "wordsUTD.list", "dusted": "UTDWordsDUSTED.txt"}

# Gold-word selection (Carlin et al., 2011): at least 0.5 s and five characters.
MIN_WORD_DURATION = 0.5
MIN_WORD_CHARS = 5
MIN_TERM_DURATION = 0.02  # discovered terms: one frame


def parse_args():
    parser = argparse.ArgumentParser(description="Extract WavLM/HuBERT word features.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--corpus", choices=[*CORPORA, "zs17french"], required=True,
                        help="'zs17french' is an alias for 'french'")
    parser.add_argument("--pairs", choices=WORD_LISTS, default="gold",
                        help="which word list to extract tokens for (default: %(default)s)")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="only extract these splits (default: all of the corpus's splits)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="hidden states to save, 0 = CNN output, 1-24 = transformer layers "
                             "(default: 21 for WavLM, 24 for HuBERT)")
    args = parser.parse_args()
    if args.corpus == "zs17french":
        args.corpus = "french"
    if args.corpus == "french" and args.pairs == "utd":
        parser.error("there is no classic UTD word list for French; use --pairs dusted")
    args.layers = tuple(args.layers or (MODELS[args.model]["layer"],))
    if args.splits and not set(args.splits) <= set(CORPORA[args.corpus]["splits"]):
        parser.error(f"--splits must be among {CORPORA[args.corpus]['splits']}")
    return args


# ========= lists =========

def load_speakers(corpus, subset):
    """Speaker IDs for a split. Xitsonga lists them as "nchlt-tso-001m"; keys use "001m"."""
    c = CORPORA[corpus]
    with open(path.join("data/Information", c["info"], "Splits", c["speakers_file"].format(subset))) as f:
        speakers = [line.strip() for line in f if line.strip()]
    return [s.split("-")[-1] for s in speakers] if corpus == "xitsonga" else speakers


def load_windows(corpus, fn):
    """recording -> [(seg_id, start, end)] (seconds)."""
    utts = {}
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_id, start, end = line.split()
            if corpus == "buckeye":     # s07_s0701a_000 -> s0701a
                rec = seg_id.split("_")[1]
            elif corpus == "xitsonga":  # 001m_nchlt_tso_001m_0007_000 -> nchlt_tso_001m_0007
                rec = seg_id.split("_", 1)[1].rsplit("_", 1)[0]
            else:                       # F01R_000 -> F01R (one recording per speaker)
                rec = seg_id.rsplit("_", 1)[0]
            utts.setdefault(rec, []).append((seg_id, float(start), float(end)))
    return utts


def load_words(corpus, pairs, fn):
    """recording -> [(label, start, end)] (seconds), filtered as described in the docstring."""
    discovered = pairs != "gold"
    if corpus == "french" and discovered and not path.exists(fn):
        raise SystemExit(f"Error: discovered-term list not found at {fn}\n"
                         f"Lines should be '<clusterID> <file> <start> <end>', e.g. 'PT10 F01_R 12.5600 13.1400'.")
    words, skipped = {}, {"<unk>": 0, "short": 0, "brief": 0}
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, rec, start, end = line.split()
            # Durations are rounded to the millisecond so that floating-point subtraction
            # (16.0495 - 15.5495 == 0.4999999999999982) doesn't drop boundary words.
            duration = round(float(end) - float(start), 3)
            if discovered:
                if duration < MIN_TERM_DURATION:
                    skipped["brief"] += 1
                    continue
                if corpus == "xitsonga":
                    rec = rec.replace("-", "_")  # the UTD lists write nchlt-tso-127f-0126
                elif corpus == "french":
                    rec = rec.replace("_", "")   # accept F01_R as well as F01R
            elif corpus != "french":  # the French gold list is filtered at preprocessing
                if label == "<unk>":
                    skipped["<unk>"] += 1
                    continue
                if len(label) < MIN_WORD_CHARS:
                    skipped["short"] += 1
                    continue
                if duration < MIN_WORD_DURATION:
                    skipped["brief"] += 1
                    continue
            words.setdefault(rec, []).append((label, float(start), float(end)))
    print(f"Loaded {sum(len(v) for v in words.values())} words from {fn} (skipped: {skipped})")
    return words


# ========= audio and model =========

def load_signal(wav_fn):
    """A whole wav as a mono 16 kHz tensor."""
    signal, sr = sf.read(wav_fn, dtype="float32")
    signal = torch.from_numpy(signal)
    if signal.ndim == 2:
        signal = signal.mean(-1)
    if sr != SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SAMPLE_RATE)
    return signal


def load_chunk(wav_fn, start_t, end_t):
    """[start_t, end_t) of a wav as a mono 16 kHz tensor, read with a seek (for long files)."""
    with sf.SoundFile(wav_fn) as f:
        sr = f.samplerate
        start, end = int(start_t * sr), min(int(end_t * sr), len(f))
        if end <= start:
            return torch.zeros(0)
        f.seek(start)
        chunk = torch.from_numpy(f.read(frames=end - start, dtype="float32"))
    if chunk.ndim == 2:
        chunk = chunk.mean(-1)
    if sr != SAMPLE_RATE:
        chunk = torchaudio.functional.resample(chunk, sr, SAMPLE_RATE)
    return chunk


def forward(model, device, chunk, layers):
    """Pad to the hop, centre-pad, layer-norm, run the model; one [frames, 1024] array per layer."""
    wav = chunk.float()
    if r := wav.numel() % HOP:
        wav = F.pad(wav, (0, HOP - r))
    wav = F.pad(wav, (CENTER_PAD, CENTER_PAD))
    x = F.layer_norm(wav.to(device), wav.shape).unsqueeze(0)
    with torch.no_grad():
        hidden_states = model(x, output_hidden_states=True).hidden_states  # [0] = CNN output
    feats = [hidden_states[L].squeeze(0).cpu().numpy() for L in layers]
    for f in feats:
        assert f.ndim == 2 and f.shape[-1] == MODEL_DIMENSIONS, f.shape
    del x, hidden_states
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats


# ========= extraction =========

def recordings(corpus, speaker, utterances):
    """(recording id, wav path, output file stem) for every recording of a speaker."""
    data = CORPORA[corpus]["data"]
    if corpus == "french":  # one file per speaker: F01R -> F01_R.wav
        if speaker in utterances:
            wav_fn = path.join(data, speaker[:-1] + "_" + speaker[-1] + ".wav")
            if not path.exists(wav_fn):
                raise SystemExit(f"Error: no audio for {speaker} at {wav_fn}")
            yield speaker, wav_fn, speaker
        return
    pattern = (path.join(data, speaker, "*.wav") if corpus == "buckeye"
               else path.join(data, f"nchlt_tso_{speaker}_*.wav"))
    for wav_fn in sorted(glob.glob(pattern)):
        rec = path.splitext(path.basename(wav_fn))[0]
        if rec in utterances:
            # Buckeye output files are named s02_01a for recording s0201a.
            yield rec, wav_fn, (rec[0:3] + "_" + rec[3:] if corpus == "buckeye" else rec)


def extract_subset(model, device, args, speakers, subset, utterances, words, feature_dir):
    output_dir = path.join(feature_dir, subset + ".samediff")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing {subset}...")

    # Frame boundaries: see the module docstring.
    rounded = args.pairs != "gold" and args.corpus != "french"
    slack = 1 if rounded else 0

    def span(start_t, end_t):
        if rounded:
            return round(start_t * FRAME_RATE), round(end_t * FRAME_RATE) + 1
        return int(math.floor(start_t * FRAME_RATE)), int(math.ceil(end_t * FRAME_RATE))

    n_words, dropped = 0, []
    for speaker in tqdm(speakers, desc="Speakers"):
        for rec, wav_fn, out_stem in recordings(args.corpus, speaker, utterances):
            if args.corpus == "french":
                read = lambda s, e: load_chunk(wav_fn, s, e)  # noqa: E731
            else:
                signal = load_signal(wav_fn)
                read = lambda s, e: signal[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]  # noqa: E731

            by_layer = [{} for _ in args.layers]
            matched = set()
            for seg_id, start_t, end_t in utterances[rec]:
                chunk = read(start_t, end_t)
                if len(chunk) == 0:
                    continue
                feats = forward(model, device, chunk, args.layers)
                utt_start, utt_end = span(start_t, end_t)

                for label, w_start, w_end in words.get(rec, []):
                    word_start, word_end = span(w_start, w_end)
                    # the word must fall entirely within this window
                    if not (word_start >= utt_start - slack and word_end <= utt_end + slack):
                        continue
                    offset_start = max(0, word_start - utt_start)
                    offset_end = min(len(feats[0]), word_end - utt_start)
                    if offset_end <= offset_start:
                        continue
                    key = f"{label}_{seg_id}_{word_start:06d}-{word_end:06d}"
                    for i, layer_feats in enumerate(feats):
                        by_layer[i][key] = layer_feats[offset_start:offset_end]
                    matched.add((label, w_start, w_end))
                    n_words += 1

            dropped += [(label, rec, s, e) for label, s, e in words.get(rec, []) if (label, s, e) not in matched]
            for L, rec_words in zip(args.layers, by_layer):
                if rec_words:
                    layer_dir = path.join(output_dir, f"layer_{L:02d}")
                    os.makedirs(layer_dir, exist_ok=True)
                    np.savez_compressed(path.join(layer_dir, out_stem + ".npz"), **rec_words)

    dropped_fn = path.join(output_dir, "dropped_words.txt")
    with open(dropped_fn, "w") as f:
        for label, rec, s, e in dropped:
            f.write(f"{label} {rec} {s}-{e}\n")
    print(f"Wrote {n_words} word segments for {subset} ({len(dropped)} words skipped, see {dropped_fn})")


def main():
    args = parse_args()
    m, c = MODELS[args.model], CORPORA[args.corpus]
    info = path.join("data/Information", c["info"])
    suffix = {"gold": "", "utd": "UTD", "dusted": "DUSTED"}[args.pairs]
    feature_set = os.environ.get("SD_FEATURE_SET", f"{c['prefix']}_{m['tag']}{suffix}")
    feature_dir = f"data/Features/{feature_set}/NotNormalised"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(os.environ.get("SD_MODEL", m["name"])).to(device).eval()
    print("feat_extract_norm:", model.config.feat_extract_norm, "| layers:", args.layers)

    utterances = load_windows(args.corpus, path.join(info, "windows.list"))
    words = load_words(args.corpus, args.pairs, path.join(info, WORD_LISTS[args.pairs]))
    for subset in args.splits or c["splits"]:
        extract_subset(model, device, args, load_speakers(args.corpus, subset), subset, utterances, words, feature_dir)


if __name__ == "__main__":
    main()

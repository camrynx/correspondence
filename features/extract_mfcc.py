"""
Extract MFCC features for word tokens: 13 MFCCs with deltas and double deltas, computed
over whole recordings and normalised per speaker.

    python features/extract_mfcc.py --corpus buckeye
    python features/extract_mfcc.py --corpus french --pairs dusted

Tokens must lie inside one window of windows.list; the rest go to dropped_words.txt.
Writes data/Features/<Corpus>_MFCCs[UTD|DUSTED]/<split>.samediff/. For the long French
recordings only the aligned prefix is read.
"""

import argparse
import glob
import os
import shutil
from os import path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

FRAME_RATE = 100

CORPORA = {
    "buckeye": dict(prefix="Buckeye", info="Buckeye", data="data/Datasets/Buckeye",
                    splits=("devpart1", "devpart2", "zs"), speakers_file="buckeye_{}_speakers.list"),
    "xitsonga": dict(prefix="Xitsonga", info="Buckeye_tsonga", data="data/Datasets/Buckeye-Xitsonga",
                     splits=("dev", "test"), speakers_file="tsong{}_speakers.list"),
    "french": dict(prefix="ZS17French", info="ZS17French", data="data/Datasets/FrenchDatasets/french/train",
                   splits=("train", "test"), speakers_file="zs17french{}_speakers.list"),
}
WORD_LISTS = {"gold": "words.list", "utd": "wordsUTD.list", "dusted": "UTDWordsDUSTED.txt"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-speaker-normalised MFCC word features.")
    parser.add_argument("--corpus", choices=[*CORPORA, "zs17french"], required=True,
                        help="'zs17french' is an alias for 'french'")
    parser.add_argument("--pairs", choices=WORD_LISTS, default="gold",
                        help="which word list to extract tokens for (default: %(default)s)")
    args = parser.parse_args()
    if args.corpus == "zs17french":
        args.corpus = "french"
    if args.corpus == "french" and args.pairs == "utd":
        parser.error("there is no classic UTD word list for French; use --pairs dusted")
    return args


def load_speakers(corpus, subset):
    """Speaker IDs for a split. Xitsonga lists them as "nchlt-tso-001m"; keys use "001m"."""
    c = CORPORA[corpus]
    fn = path.join("data/Information", c["info"], "Splits", c["speakers_file"].format(subset))
    with open(fn) as f:
        speakers = [line.strip() for line in f if line.strip()]
    return [s.split("-")[-1] for s in speakers] if corpus == "xitsonga" else speakers


def to_frames(start, end):
    """Seconds -> (start_frame, end_frame), end excluded."""
    return int(round(float(start) * FRAME_RATE)), int(round(float(end) * FRAME_RATE)) + 1


def uttlabel_to_uttkey(file_label):
    """Buckeye recording label to feature key, e.g. s2301a -> s23_01a."""
    return file_label[0:3] + "_" + file_label[3:]


def recording_key(corpus, file_label):
    """Key that ties a recording's windows, words and features together."""
    return uttlabel_to_uttkey(file_label) if corpus == "buckeye" else file_label


def read_windows(corpus, list_fn):
    """recording key -> [(start_frame, end_frame)]. Lines: <seg_id> <start> <end>."""
    windows = {}
    with open(list_fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_id, start, end = line.split()
            if corpus == "buckeye":    # s07_s0701a_000
                key = recording_key(corpus, seg_id.split("_")[1])
            elif corpus == "xitsonga":  # 001m_nchlt_tso_001m_0007_000
                key = seg_id.split("_", 1)[1].rsplit("_", 1)[0]
            else:                       # F01R_000 (one recording per speaker)
                key = seg_id.rsplit("_", 1)[0]
            windows.setdefault(key, []).append(to_frames(start, end))
    return windows


def read_words(corpus, pairs, list_fn):
    """recording key -> [(label, start_frame, end_frame)]. Lines: <label> <file> <start> <end>."""
    if corpus == "french" and pairs != "gold" and not path.exists(list_fn):
        raise SystemExit(f"Error: discovered-term list not found at {list_fn}\n"
                         f"Lines should be '<clusterID> <file> <start> <end>', e.g. 'PT10 F01_R 12.5600 13.1400'.")
    words, n_brief = {}, 0
    with open(list_fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, file_label, start, end = line.split()
            if corpus == "french":
                if pairs != "gold":
                    # External term lists: drop sub-frame terms, and accept "F01_R" or "F01R".
                    if round(float(end) - float(start), 3) < 0.01:
                        n_brief += 1
                        continue
                    file_label = file_label.replace("_", "")
                key = file_label
            elif corpus == "xitsonga":
                # The Xitsonga UTD lists write the recording with hyphens.
                key = file_label.replace("-", "_") if pairs != "gold" else file_label
            else:
                key = recording_key(corpus, file_label)
            words.setdefault(key, []).append((label, *to_frames(start, end)))
    n = sum(len(v) for v in words.values())
    print(f"Loaded {n} words from {list_fn}" + (f" ({n_brief} under 0.01s skipped)" if n_brief else ""))
    return words


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def mfcc_features(signal, sample_rate):
    """(n_frames, 39): 13 MFCCs, deltas and double deltas, 25 ms window, 10 ms hop."""
    signal = preemphasis(signal, coeff=0.97)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=13, n_mels=24,
        n_fft=int(np.floor(0.025 * sample_rate)),
        hop_length=int(np.floor(0.01 * sample_rate)), fmin=64, fmax=8000,
    )
    return np.hstack([mfcc.T, librosa.feature.delta(mfcc).T, librosa.feature.delta(mfcc, order=2).T])


def mfcc_for_wavs(wav_fns):
    """{file stem: MFCC features} for whole wav files, at their native sample rate."""
    feats = {}
    for wav_fn in tqdm(wav_fns):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        feats[path.splitext(path.basename(wav_fn))[0]] = mfcc_features(signal, sample_rate)
    return feats


def mfcc_prefix(wav_fn, n_seconds):
    """MFCC features over only the first n_seconds of a wav (for the long French recordings)."""
    with sf.SoundFile(wav_fn) as f:
        sample_rate = f.samplerate
        signal = f.read(frames=min(int(n_seconds * sample_rate), len(f)), dtype="float32")
    if signal.ndim == 2:
        signal = signal.mean(-1)
    return mfcc_features(signal, sample_rate)


def normalise_recording(features):
    """Mean and variance normalisation of one recording (French: one recording per speaker)."""
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)


LOAD_MARGIN_SECONDS = 1.0  # French: read this far past the last window


def speaker_mvn(feat_dict, speaker_of):
    """Per-speaker mean and variance normalisation (statistics over all of a speaker's recordings)."""
    per_speaker = {}
    for key in sorted(feat_dict):
        per_speaker.setdefault(speaker_of[key], []).append(feat_dict[key])
    stats = {}
    for speaker, arrays in per_speaker.items():
        features = np.vstack(arrays)
        stats[speaker] = (np.mean(features, axis=0), np.std(features, axis=0))
    return {key: (feat_dict[key] - stats[speaker_of[key]][0]) / stats[speaker_of[key]][1]
            for key in tqdm(sorted(feat_dict), desc="Speaker MVN")}


def write_dropped(output_dir, dropped):
    fn = path.join(output_dir, "dropped_words.txt")
    with open(fn, "w") as f:
        for label, rec, start, end in dropped:
            f.write(f"{label} {rec} {start}-{end}\n")
    return fn


def extract_subset(corpus, pairs, speakers, windows, words, output_dir):
    """Buckeye / Xitsonga: whole recordings, words matched to the window they fall in."""
    os.makedirs(output_dir, exist_ok=True)
    data = CORPORA[corpus]["data"]

    feat_dict, speaker_of = {}, {}
    for speaker in speakers:
        # Buckeye: <speaker>/<recording>.wav; Xitsonga: flat nchlt_tso_<speaker>_<idx>.wav
        pattern = (path.join(data, speaker, "*.wav") if corpus == "buckeye"
                   else path.join(data, f"nchlt_tso_{speaker}_*.wav"))
        for stem, feats in mfcc_for_wavs(sorted(glob.glob(pattern))).items():
            key = speaker + "_" + stem[3:] if corpus == "buckeye" else stem  # s2301a -> s23_01a
            feat_dict[key] = feats
            speaker_of[key] = speaker
    feat_dict = speaker_mvn(feat_dict, speaker_of)

    # Xitsonga term times are on a 10 ms grid: allow one frame of slack, clipped to the window.
    slack = 1 if corpus == "xitsonga" and pairs != "gold" else 0

    n_words, dropped = 0, []
    for key in tqdm(sorted(k for k in windows if k in feat_dict), desc="Slicing words"):
        rec_words = {}
        for label, w_start, w_end in words.get(key, []):
            for seg_start, seg_end in windows[key]:
                if w_start >= seg_start - slack and w_end <= seg_end + slack:
                    s, e = (max(seg_start, w_start), min(seg_end, w_end)) if slack else (w_start, w_end)
                    if e <= s:
                        continue
                    rec_words[f"{label}_{key}_{w_start:06d}-{w_end:06d}"] = feat_dict[key][s:e]
                    n_words += 1
                    break
            else:
                dropped.append((label, key, w_start, w_end))
                print(f"Warning: word {label} in utterance {key} ({w_start}-{w_end}) "
                      f"did not fall entirely inside any segment - skipped")
        if rec_words:
            np.savez_compressed(path.join(output_dir, key + ".npz"), **rec_words)
    fn = write_dropped(output_dir, dropped)
    print(f"Wrote {n_words} word segments ({len(dropped)} words skipped, see {fn})")


def extract_subset_french(speakers, windows, words, output_dir):
    """French: one long recording per speaker; read only its aligned prefix."""
    os.makedirs(output_dir, exist_ok=True)
    data = CORPORA["french"]["data"]
    n_words, dropped = 0, []
    for speaker in tqdm(speakers, desc="Speakers"):
        segments = windows.get(speaker)
        if not segments:
            continue
        wav_fn = path.join(data, speaker[:-1] + "_" + speaker[-1] + ".wav")  # F01R -> F01_R.wav
        if not path.exists(wav_fn):
            raise SystemExit(f"Error: no audio for {speaker} at {wav_fn}")

        n_seconds = max(end for _, end in segments) / FRAME_RATE + LOAD_MARGIN_SECONDS
        features = normalise_recording(mfcc_prefix(wav_fn, n_seconds))

        speaker_words = {}
        for label, w_start, w_end in words.get(speaker, []):
            inside = any(w_start >= s and w_end <= e for s, e in segments)
            if not inside or w_end > len(features):
                dropped.append((label, speaker, w_start, w_end))
                continue
            speaker_words[f"{label}_{speaker}_{w_start:06d}-{w_end:06d}"] = features[w_start:w_end]
            n_words += 1
        if speaker_words:
            np.savez_compressed(path.join(output_dir, speaker + ".npz"), **speaker_words)
    fn = write_dropped(output_dir, dropped)
    print(f"Wrote {n_words} word segments ({len(dropped)} words skipped, see {fn})")


def combine_into_all(input_dirs, output_dir):
    """Copy every split's .npz files into one all.samediff directory."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Combining subsets into {output_dir}")
    for input_dir in input_dirs:
        for npz_file in sorted(glob.glob(path.join(input_dir, "*.npz"))):
            dest = path.join(output_dir, path.basename(npz_file))
            if path.exists(dest):
                print(f"  Warning: {path.basename(dest)} already exists in {output_dir}, overwriting.")
            shutil.copy2(npz_file, dest)


def main():
    args = parse_args()
    c = CORPORA[args.corpus]
    info = path.join("data/Information", c["info"])
    suffix = {"gold": "", "utd": "UTD", "dusted": "DUSTED"}[args.pairs]
    feature_dir = path.join("data/Features", os.environ.get("SD_FEATURE_SET", f"{c['prefix']}_MFCCs{suffix}"))

    windows = read_windows(args.corpus, path.join(info, "windows.list"))
    words = read_words(args.corpus, args.pairs, path.join(info, WORD_LISTS[args.pairs]))

    for subset in c["splits"]:
        print(f"\nProcessing {subset}...")
        output_dir = path.join(feature_dir, subset + ".samediff")
        speakers = load_speakers(args.corpus, subset)
        if args.corpus == "french":
            extract_subset_french(speakers, windows, words, output_dir)
        else:
            extract_subset(args.corpus, args.pairs, speakers, windows, words, output_dir)

    if args.pairs != "gold" and args.corpus != "french":
        combine_into_all([path.join(feature_dir, f"{s}.samediff") for s in c["splits"]],
                         path.join(feature_dir, "all.samediff"))


if __name__ == "__main__":
    main()

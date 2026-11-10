"""
Frame alignments for cAE training.

    python training/align.py --features wavlm --corpus buckeye --pairs gold --n-cpus 8
    python training/align.py --features hubert --corpus xitsonga --pairs poorman

gold / utd / dusted: every pair of tokens with the same label is aligned with DTW
(speech_dtw), for the training and validation splits.

poorman: every training-split frame is paired with its nearest (cosine) frame from a
different speaker, or with --allow-same-speaker from any other token. Uses the GPU if
available.

Output: data/Alignments/<set>/<split>.samediff[.alignments][/layer_NN]/, pickled chunks of
label -> [{"pair": (key_1, key_2), "frame_pairs": int32 (n, 2)}] plus metadata.pkl.
"""

from __future__ import annotations

import argparse
import pickle
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CALC_DTW_PATHS = REPO_ROOT / "third_party" / "speech_dtw" / "utils" / "calculate_dtw_paths.py"

FEATURES = {
    "mfcc": dict(tag="MFCCs", layer=None),
    "wavlm": dict(tag="WavLM", layer=21),
    "hubert": dict(tag="HuBERT", layer=24),
}

# Token keys are <label>_..._<start>-<end> and labels can contain underscores, so the
# speaker is matched from the right. MFCC and SSL keys differ.
CORPORA = {
    "buckeye": dict(
        prefix="Buckeye", info="Buckeye", train="devpart1", val="devpart2", all=("devpart1", "devpart2", "zs"),
        speakers_file="buckeye_{}_speakers.list", speaker_from_line=lambda s: s,
        key_re={
            "ssl": r"^(?P<label>.+)_(?P<speaker>s\d{2})_(?P<recording>s\d{4}[ab])_(?P<word_idx>\d+)_(?P<fr_start>\d+)-(?P<fr_end>\d+)$",
            "mfcc": r"^(?P<label>.+)_(?P<speaker>s\d{2})_(?P<recording>\d{2}[ab])_(?P<fr_start>\d+)-(?P<fr_end>\d+)$",
        },
    ),
    "xitsonga": dict(
        # The split files list speakers as "nchlt-tso-001m"; keys use just "001m".
        prefix="Xitsonga", info="Buckeye_tsonga", train="dev", val="test", all=("dev", "test"),
        speakers_file="tsong{}_speakers.list", speaker_from_line=lambda s: s.split("-")[-1],
        key_re={
            "ssl": r"^(?P<label>.+)_(?P<speaker>\d{3}[mf])_nchlt_tso_(?P=speaker)_(?P<file_idx>\d+)_(?P<seg_idx>\d+)_(?P<fr_start>\d+)-(?P<fr_end>\d+)$",
            "mfcc": r"^(?P<label>.+)_nchlt_tso_(?P<speaker>\d{3}[mf])_(?P<file_idx>\d+)_(?P<fr_start>\d+)-(?P<fr_end>\d+)$",
        },
    ),
    "french": dict(
        prefix="ZS17French", info="ZS17French", train="train", val="test", all=("train", "test"),
        speakers_file="zs17french{}_speakers.list", speaker_from_line=lambda s: s,
        key_re={
            "ssl": r"^(?P<label>.+)_(?P<speaker>[FM]\d{2}[NOR])_(?P<seg_idx>\d+)_(?P<fr_start>\d+)-(?P<fr_end>\d+)$",
            "mfcc": r"^(?P<label>.+)_(?P<speaker>[FM]\d{2}[NOR])_(?P<fr_start>\d+)-(?P<fr_end>\d+)$",
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frame alignments for cAE training.")
    parser.add_argument("--features", choices=FEATURES, required=True)
    parser.add_argument("--corpus", choices=[*CORPORA, "zs17french"], required=True,
                        help="'zs17french' is an alias for 'french'")
    parser.add_argument("--pairs", choices=["gold", "utd", "dusted", "poorman"], default="gold",
                        help="(default: %(default)s)")
    parser.add_argument("--layer", type=int, default=None,
                        help="WavLM/HuBERT layer (default: 21 for WavLM, 24 for HuBERT; ignored for MFCCs)")

    dtw = parser.add_argument_group("word pairs (gold / utd / dusted)")
    dtw.add_argument("--n-cpus", type=int, default=8, help="parallel DTW jobs (default: %(default)s)")
    dtw.add_argument("--keep-work", action="store_true",
                     help="keep the intermediate pair lists / per-job features / DTW paths")

    pm = parser.add_argument_group("Poor Man's UTD (poorman)")
    pm.add_argument("--splits", nargs="+", default=None,
                    help="splits to pool for the nearest-neighbour search (default: the training split only)")
    pm.add_argument("--allow-same-speaker", action="store_true",
                    help="allow neighbours from the same speaker (any other word token); "
                         "written to the <set>PoorManUTDAnySpk alignment set")
    pm.add_argument("--features-dir", type=Path, default=None,
                    help="override: directory holding one <split>.samediff[/layer_NN] folder per split")
    pm.add_argument("--splits-dir", type=Path, default=None, help="override: directory of speaker lists")
    pm.add_argument("--output-dir", type=Path, default=None, help="override: output directory")
    pm.add_argument("--chunk-size", type=int, default=500, help="entries per output chunk (default: %(default)s)")
    pm.add_argument("--block-size", type=int, default=1024, help="query frames per search batch (default: %(default)s)")
    pm.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")

    args = parser.parse_args()
    if args.corpus == "zs17french":
        args.corpus = "french"
    if args.corpus == "french" and args.pairs == "utd":
        parser.error("there is no classic UTD word list for French; use --pairs dusted")
    if args.pairs != "poorman" and (args.allow_same_speaker or args.splits):
        parser.error("--allow-same-speaker / --splits only apply to --pairs poorman")
    feat = FEATURES[args.features]
    args.layer = None if feat["layer"] is None else (args.layer or feat["layer"])
    return args


def layer_dir(layer: int | None) -> tuple[str, ...]:
    return () if layer is None else (f"layer_{layer:02d}",)


# --- DTW word pairs ---

def load_npz_keys(features_dir: Path) -> list[str]:
    """Sorted token keys from every .npz file in features_dir."""
    keys = []
    for npz_file in sorted(features_dir.glob("*.npz")):
        with np.load(npz_file, allow_pickle=False) as archive:
            keys.extend(archive.files)
    return sorted(keys)


def derive_label(key: str) -> str:
    """Word label: everything before the first underscore."""
    return key.split("_", 1)[0]


def create_same_word_pairs(keys: list[str]) -> tuple[list[str], dict[str, str]]:
    """All unique same-label pairs, as "key_1 key_2" strings, and pair -> label."""
    by_label: dict[str, list[str]] = {}
    for key in keys:
        by_label.setdefault(derive_label(key), []).append(key)

    pairs, pair_label = [], {}
    for label, label_keys in sorted(by_label.items()):
        for i in range(len(label_keys) - 1):
            for j in range(i + 1, len(label_keys)):
                pair = f"{label_keys[i]} {label_keys[j]}"
                pairs.append(pair)
                pair_label[pair] = label
    return pairs, pair_label


def split_lines(lines: list[str], n_files: int, output_dir: Path) -> None:
    """Write lines into pairs.1.list ... pairs.<n_files>.list, in order."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    per_file = int(ceil(len(lines) / n_files))
    for i in range(n_files):
        chunk = lines[i * per_file:(i + 1) * per_file]
        if chunk or i == 0:
            (output_dir / f"pairs.{i + 1}.list").write_text("".join(line + "\n" for line in chunk))


def build_job_features(features_dir: Path, pairs_split_dir: Path, n_jobs: int, out_dir: Path) -> dict[int, Path]:
    """One small .npz per job holding only the tokens that job's pairs reference, so the
    parallel workers don't each load the whole (possibly huge) features directory."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    key_jobs: dict[str, list[int]] = {}
    for job in range(1, n_jobs + 1):
        fn = pairs_split_dir / f"pairs.{job}.list"
        if fn.exists():
            for line in fn.read_text().splitlines():
                for key in line.split():
                    jobs = key_jobs.setdefault(key, [])
                    if job not in jobs:
                        jobs.append(job)

    job_arrays: dict[int, dict] = {job: {} for job in range(1, n_jobs + 1)}
    for npz_file in sorted(features_dir.glob("*.npz")):
        with np.load(npz_file, allow_pickle=False) as archive:
            for key in archive.files:
                for job in key_jobs.get(key, ()):
                    job_arrays[job][key] = archive[key]

    job_fns = {}
    for job, arrays in job_arrays.items():
        job_fns[job] = out_dir / f"features.{job}.npz"
        np.savez(job_fns[job], **arrays)
    return job_fns


def run_dtw_jobs(n_jobs: int, job_features: dict[int, Path], pairs_split_dir: Path, paths_split_dir: Path) -> None:
    if paths_split_dir.exists():
        shutil.rmtree(paths_split_dir)
    paths_split_dir.mkdir(parents=True)

    def run(job: int) -> None:
        cmd = [sys.executable, str(CALC_DTW_PATHS), "--input_fmt", "npz",
               str(pairs_split_dir / f"pairs.{job}.list"), str(job_features[job]),
               str(paths_split_dir / f"paths.{job}.pkl")]
        with (paths_split_dir / f"paths.{job}.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=log)

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for future in as_completed([pool.submit(run, job) for job in range(1, n_jobs + 1)]):
            future.result()


def save_chunked(output_dir: Path, entries: dict, chunk_size: int) -> None:
    """entries (label -> alignment list) into alignments.chunk_NNNN.pkl files of
    chunk_size labels each, in sorted label order, plus metadata.pkl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(entries)
    n_chunks = (len(labels) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        with open(output_dir / f"alignments.chunk_{i:04d}.pkl", "wb") as f:
            pickle.dump({label: entries[label] for label in labels[i * chunk_size:(i + 1) * chunk_size]}, f)
    metadata = {
        "total_words": len(entries),  # "words" even for Poor Man's entries, for compatibility
        "total_chunks": n_chunks,
        "chunk_size": chunk_size,
        "words_per_chunk": [len(labels[i * chunk_size:(i + 1) * chunk_size]) for i in range(n_chunks)],
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"  Saved {len(entries)} labels in {n_chunks} chunks to {output_dir}")


def dtw_align(features_dir: Path, output_dir: Path, n_cpus: int, keep_work: bool) -> None:
    if not features_dir.is_dir():
        raise SystemExit(f"Error: {features_dir} does not exist or is not a directory")
    work_dir = output_dir.parent / "work" / features_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)

    keys = load_npz_keys(features_dir)
    pairs, pair_label = create_same_word_pairs(keys)
    print(f"  {len(keys)} tokens, {len(pairs)} same-label pairs")
    if not pairs:
        print("  Warning: no same-label pairs found", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    pairs_split_dir, paths_split_dir = work_dir / "pairs_split", work_dir / "paths_split"
    split_lines(pairs, n_cpus, pairs_split_dir)
    n_jobs = len(list(pairs_split_dir.glob("pairs.*.list")))
    job_features = build_job_features(features_dir, pairs_split_dir, n_jobs, work_dir / "features_split")
    print(f"  Running DTW in {n_jobs} jobs...")
    run_dtw_jobs(n_jobs, job_features, pairs_split_dir, paths_split_dir)

    paths = []
    for job in range(1, n_jobs + 1):
        with open(paths_split_dir / f"paths.{job}.pkl", "rb") as f:
            paths.extend(pickle.load(f))

    entries: dict[str, list] = {}
    for pair, path in zip(pairs, paths):
        key_1, key_2 = pair.split()
        entries.setdefault(pair_label[pair], []).append(
            {"pair": (key_1, key_2), "frame_pairs": np.array(path, dtype=np.int32)})
    save_chunked(output_dir, entries, chunk_size=100)

    if not keep_work:
        shutil.rmtree(work_dir)
        if not any(work_dir.parent.iterdir()):
            work_dir.parent.rmdir()


# --- Poor Man's UTD ---

def load_corpus(features_dir: Path, layer: int | None, split_speakers: dict[str, set[str]], key_re: re.Pattern) -> dict:
    """Every frame of every token whose speaker is in its split's speaker list, pooled."""
    seg_keys, seg_lens, seg_speaker, frames = [], [], [], []
    n_bad_key = n_bad_speaker = 0
    for split, speakers in split_speakers.items():
        split_dir = Path(features_dir, f"{split}.samediff", *layer_dir(layer))
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing features split dir: {split_dir}")
        for npz_file in sorted(split_dir.glob("*.npz")):
            with np.load(npz_file, allow_pickle=False) as archive:
                for key in archive.files:
                    match = key_re.match(key)
                    if not match:
                        n_bad_key += 1
                        continue
                    if match.group("speaker") not in speakers:
                        n_bad_speaker += 1
                        continue
                    arr = archive[key]
                    seg_keys.append(key)
                    seg_lens.append(arr.shape[0])
                    seg_speaker.append(match.group("speaker"))
                    frames.append(arr.astype(np.float32, copy=False))
    if n_bad_key:
        print(f"  Skipped {n_bad_key} token(s) with an unparseable key")
    if n_bad_speaker:
        print(f"  Skipped {n_bad_speaker} token(s) whose speaker isn't in its split's speaker list")
    if not seg_keys:
        raise RuntimeError("No tokens loaded - check the features and splits directories")

    speaker_code = {s: i for i, s in enumerate(sorted(set(seg_speaker)))}
    lens = np.array(seg_lens, dtype=np.int64)
    print(f"  Loaded {lens.sum()} frames from {len(seg_keys)} tokens, {len(speaker_code)} speakers")
    return {
        "frames": np.concatenate(frames, axis=0),
        "speaker_per_frame": np.repeat(np.array([speaker_code[s] for s in seg_speaker], dtype=np.int64), lens),
        "seg_keys": seg_keys,
        "seg_id_per_frame": np.repeat(np.arange(len(seg_keys), dtype=np.int64), lens),
        "local_idx_per_frame": np.concatenate([np.arange(n, dtype=np.int64) for n in seg_lens]),
    }


def nearest_neighbours(frames: np.ndarray, group_per_frame: np.ndarray, device: str, block_size: int):
    """For every frame, the cosine-nearest frame in a different group. Returns (index, similarity)."""
    import torch
    from tqdm import tqdm

    feats = torch.from_numpy(frames).to(device)
    feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-8)
    groups = torch.from_numpy(group_per_frame).to(device)
    n = frames.shape[0]
    nn_idx, nn_sim = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.float32)
    for start in tqdm(range(0, n, block_size), desc="NN search"):
        end = min(start + block_size, n)
        sims = feats[start:end] @ feats.T
        sims.masked_fill_(groups[start:end].unsqueeze(1) == groups.unsqueeze(0), float("-inf"))
        best_sim, best_idx = sims.max(dim=1)
        nn_idx[start:end] = best_idx.cpu().numpy()
        nn_sim[start:end] = best_sim.cpu().numpy()
    return nn_idx, nn_sim


def poorman_align(args: argparse.Namespace) -> None:
    import torch

    corpus, feat = CORPORA[args.corpus], FEATURES[args.features]
    is_mfcc = args.layer is None
    base = f"{corpus['prefix']}_{feat['tag']}"
    features_dir = args.features_dir or Path("data/Features", base, *(() if is_mfcc else ("Normalised",)))
    splits_dir = args.splits_dir or Path("data/Information", corpus["info"], "Splits")
    splits = args.splits or [corpus["train"]]
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        splits_name = "all" if sorted(splits) == sorted(corpus["all"]) else "+".join(splits)
        any_spk = "AnySpk" if args.allow_same_speaker else ""
        output_dir = Path("data/Alignments", f"{base}PoorManUTD{any_spk}",
                          f"{splits_name}.samediff.alignments", *layer_dir(args.layer))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_speakers = {}
    for split in splits:
        fn = splits_dir / corpus["speakers_file"].format(split)
        if not fn.is_file():
            raise FileNotFoundError(f"Missing speaker list: {fn}")
        split_speakers[split] = {corpus["speaker_from_line"](line.strip())
                                 for line in fn.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"  {split}: {len(split_speakers[split])} speakers")

    data = load_corpus(features_dir, args.layer, split_speakers,
                       re.compile(corpus["key_re"]["mfcc" if is_mfcc else "ssl"]))
    if args.allow_same_speaker:
        print("Searching for nearest neighbours in other word tokens (any speaker)...")
        groups = data["seg_id_per_frame"]
    else:
        print("Searching for cross-speaker nearest neighbours...")
        groups = data["speaker_per_frame"]
    nn_idx, nn_sim = nearest_neighbours(data["frames"], groups, device, args.block_size)
    print(f"  Mean cosine similarity: {nn_sim.mean():.4f}  (min: {nn_sim.min():.4f}, max: {nn_sim.max():.4f})")

    # One entry per frame: poormanutd_<frame index> -> [that frame and its neighbour].
    seg_keys, seg_id, local = data["seg_keys"], data["seg_id_per_frame"], data["local_idx_per_frame"]
    n_digits = len(str(len(seg_id) - 1))
    entries = {}
    for i in range(len(seg_id)):
        j = int(nn_idx[i])
        entries[f"poormanutd_{i:0{n_digits}d}"] = [{
            "pair": (seg_keys[seg_id[i]], seg_keys[seg_id[j]]),
            "frame_pairs": np.array([[int(local[i]), int(local[j])]], dtype=np.int32),
            "similarity": float(nn_sim[i]),
        }]
    save_chunked(output_dir, entries, args.chunk_size)


# ---------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.pairs == "poorman":
        poorman_align(args)
        return

    corpus, feat = CORPORA[args.corpus], FEATURES[args.features]
    is_mfcc = args.layer is None
    feature_set = f"{corpus['prefix']}_{feat['tag']}" + {"gold": "", "utd": "UTD", "dusted": "DUSTED"}[args.pairs]
    # The SSL UTD/DUSTED alignments live in "<split>.samediff.alignments", the rest in "<split>.samediff".
    align_name = "samediff.alignments" if args.pairs != "gold" and not is_mfcc else "samediff"
    for split in (corpus["train"], corpus["val"]):
        features_dir = Path("data/Features", feature_set, *(() if is_mfcc else ("Normalised",)),
                            f"{split}.samediff", *layer_dir(args.layer))
        output_dir = Path("data/Alignments", feature_set, f"{split}.{align_name}", *layer_dir(args.layer))
        print(f"\n=== {feature_set} {split}: {features_dir} -> {output_dir}")
        dtw_align(features_dir, output_dir, args.n_cpus, args.keep_work)
    print("\nAll alignments generated.")


if __name__ == "__main__":
    main()

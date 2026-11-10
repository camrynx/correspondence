"""
DTW alignments for LibriSpeech DUSTED same-term pairs (train-clean-5 only).

    python training/extras/generate_alignments_WavLMLibriUTD.py --n-cpus 8 --dusted
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from math import ceil
from pathlib import Path
import shutil
import subprocess
import sys
import pickle
import zipfile

import numpy as np
from numpy.lib import format as npy_format


REPO_ROOT = Path(__file__).resolve().parents[2]
DTW_UTILS_DIR = REPO_ROOT / "third_party" / "speech_dtw" / "utils"
CALC_DTW_PATHS = DTW_UTILS_DIR / "calculate_dtw_paths.py"

# Layer to align (a layer_NN/ under data/Features/LibriSpeech_WavLMDUSTED/Normalised/...).
LAYER = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed DTW alignments for LibriSpeech DUSTED UTD WavLM features."
    )
    parser.add_argument("--n-cpus", type=int, default=8, help="number of parallel DTW jobs (default: %(default)s)")
    parser.add_argument(
        "--dusted", action="store_true",
        help="Required - there is no legacy UTD word list for LibriSpeech, only DUSTED.",
        )
    return parser.parse_args()


def load_npz_keys(features_dir: Path) -> list[str]:
    keys = []
    if not features_dir.exists():
        print(f"Warning: Layer directory not found: {features_dir}")
        return keys
    for npz_file in sorted(features_dir.glob("*.npz")):
        with np.load(npz_file, allow_pickle=False) as archive:
            keys.extend(archive.files)
    return sorted(keys)


def derive_label(utterance_id: str) -> str:
    return utterance_id.split("_", 1)[0]


def create_same_word_pairs(utterance_ids: list[str]) -> tuple[list[str], dict]:
    label_to_utt_ids: dict[str, list[str]] = {}
    for utt_id in utterance_ids:
        label_to_utt_ids.setdefault(derive_label(utt_id), []).append(utt_id)

    pairs = []
    pair_metadata = {}
    for label, utt_ids in sorted(label_to_utt_ids.items()):
        if len(utt_ids) < 2:
            continue
        for i in range(len(utt_ids) - 1):
            for j in range(i + 1, len(utt_ids)):
                pair_str = f"{utt_ids[i]} {utt_ids[j]}"
                pairs.append(pair_str)
                pair_metadata[pair_str] = label
    return pairs, pair_metadata


def write_lines(output_fn: Path, lines: list[str]) -> None:
    with output_fn.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def split_file(input_fn: Path, n_files: int, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = input_fn.read_text().splitlines(keepends=True)
    if not lines:
        return

    n_lines_per_split = int(ceil(float(len(lines)) / n_files))
    basename = input_fn.stem
    extension = input_fn.suffix

    i_split = 1
    n_cur_lines = 0
    current_fn = output_dir / f"{basename}.{i_split}{extension}"
    current_file = current_fn.open("w", encoding="utf-8")

    try:
        for line in lines:
            current_file.write(line)
            n_cur_lines += 1
            if n_cur_lines == n_lines_per_split:
                current_file.close()
                if i_split == n_files:
                    return
                i_split += 1
                current_fn = output_dir / f"{basename}.{i_split}{extension}"
                current_file = current_fn.open("w", encoding="utf-8")
                n_cur_lines = 0
    finally:
        if not current_file.closed:
            current_file.close()


def run_dtw_job(job: int, features_fn: Path, pairs_split_dir: Path, paths_split_dir: Path) -> None:
    split_pair_fn = pairs_split_dir / f"pairs.{job}.list"
    split_paths_fn = paths_split_dir / f"paths.{job}.pkl"
    log_fn = paths_split_dir / f"paths.{job}.log"

    cmd = [
        sys.executable, str(CALC_DTW_PATHS),
        "--input_fmt", "npz",
        str(split_pair_fn), str(features_fn), str(split_paths_fn),
    ]
    with log_fn.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)


def collect_needed_utt_ids(pairs_split_dir: Path, n_cpus: int) -> dict[int, set[str]]:
    needed_by_job = {}
    for job in range(1, n_cpus + 1):
        ids: set[str] = set()
        split_pair_fn = pairs_split_dir / f"pairs.{job}.list"
        if split_pair_fn.exists():
            with split_pair_fn.open(encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        ids.update(parts)
        needed_by_job[job] = ids
    return needed_by_job


def build_job_feature_subsets(features_dir: Path, needed_by_job: dict[int, set[str]], subset_dir: Path) -> dict[int, Path]:
    """Write each job's feature arrays straight into its .npz, one source file at a time,
    so memory holds one source file's arrays rather than every job's."""
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    id_to_jobs: dict[str, list[int]] = {}
    for job, ids in needed_by_job.items():
        for utt_id in ids:
            id_to_jobs.setdefault(utt_id, []).append(job)

    job_paths = {job: subset_dir / f"features.{job}.npz" for job in needed_by_job}
    job_zips = {
        job: zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True)
        for job, path in job_paths.items()
    }

    try:
        for npz_file in sorted(features_dir.glob("*.npz")):
            with np.load(npz_file, allow_pickle=False) as archive:
                for utt_id in archive.files:
                    jobs = id_to_jobs.get(utt_id)
                    if not jobs:
                        continue
                    buf = io.BytesIO()
                    npy_format.write_array(buf, archive[utt_id], allow_pickle=False)
                    encoded = buf.getvalue()
                    arcname = f"{utt_id}.npy"
                    for job in jobs:
                        job_zips[job].writestr(arcname, encoded)
    finally:
        for zf in job_zips.values():
            zf.close()

    return job_paths


def launch_dtw_jobs(n_cpus: int, job_features_fns: dict[int, Path], pairs_split_dir: Path, paths_split_dir: Path) -> None:
    if paths_split_dir.exists():
        shutil.rmtree(paths_split_dir)
    paths_split_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        futures = [
            executor.submit(run_dtw_job, job, job_features_fns[job], pairs_split_dir, paths_split_dir)
            for job in range(1, n_cpus + 1)
        ]
        for future in as_completed(futures):
            future.result()


def load_pickled_paths(paths_split_dir: Path, n_cpus: int) -> list:
    all_paths = []
    for job in range(1, n_cpus + 1):
        with (paths_split_dir / f"paths.{job}.pkl").open("rb") as f:
            all_paths.extend(pickle.load(f))
    return all_paths


def aggregate_alignments_by_word(pairs: list[str], pair_metadata: dict, all_paths: list) -> dict:
    word_alignments: dict = {}
    for pair_str, path in zip(pairs, all_paths):
        utt_id_1, utt_id_2 = pair_str.split()
        word_label = pair_metadata[pair_str]
        word_alignments.setdefault(word_label, [])
        frame_pairs = np.array(path, dtype=np.int32)
        word_alignments[word_label].append({"pair": (utt_id_1, utt_id_2), "frame_pairs": frame_pairs})
    return word_alignments


def save_alignments_chunked(output_dir: Path, word_alignments: dict, chunk_size: int = 100) -> None:
    print(f"Saving alignments to chunked pickle files in {output_dir} (chunk size: {chunk_size} words)")
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_words = sorted(word_alignments.keys())
    n_chunks = (len(sorted_words) + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_words = sorted_words[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        chunk_data = {word: word_alignments[word] for word in chunk_words}
        with (output_dir / f"alignments.chunk_{chunk_idx:04d}.pkl").open("wb") as f:
            pickle.dump(chunk_data, f)

    metadata = {
        "total_words": len(word_alignments),
        "total_chunks": n_chunks,
        "chunk_size": chunk_size,
        "words_per_chunk": [len(sorted_words[i * chunk_size:(i + 1) * chunk_size]) for i in range(n_chunks)],
    }
    with (output_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {n_chunks} alignment chunks to {output_dir}")
    print(f"  Total words: {len(word_alignments)}")


def generate_alignments(features_dir: Path, n_cpus: int, output_dir: Path) -> Path:
    if not features_dir.is_dir():
        print(f"Error: {features_dir} does not exist or is not a directory", file=sys.stderr)
        raise SystemExit(1)

    basename = features_dir.name  # e.g. "layer_21"
    work_dir = output_dir.parent / "work" / basename
    work_dir.mkdir(parents=True, exist_ok=True)

    pairs_fn = work_dir / "pairs.list"
    pairs_split_dir = work_dir / "pairs_split"
    paths_split_dir = work_dir / "paths_split"

    print(f"\n  Loading word keys from {features_dir}...")
    utterance_ids = load_npz_keys(features_dir)
    print(f"  Found {len(utterance_ids)} word instances")

    print(f"  Creating same-word pairs...")
    pairs, pair_metadata = create_same_word_pairs(utterance_ids)
    print(f"  Created {len(pairs)} same-word pairs")

    if len(pairs) == 0:
        print(f"  Warning: No same-word pairs found!", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    write_lines(pairs_fn, pairs)
    split_file(pairs_fn, n_cpus, pairs_split_dir)
    print(f"  Split pairs into {n_cpus} jobs")

    print(f"  Building per-job feature subsets...")
    features_split_dir = work_dir / "features_split"
    needed_by_job = collect_needed_utt_ids(pairs_split_dir, n_cpus)
    job_features_fns = build_job_feature_subsets(features_dir, needed_by_job, features_split_dir)

    print(f"  Launching DTW alignment jobs...")
    launch_dtw_jobs(n_cpus, job_features_fns, pairs_split_dir, paths_split_dir)
    print(f"  DTW jobs completed")

    print(f"  Aggregating alignments by word...")
    all_paths = load_pickled_paths(paths_split_dir, n_cpus)
    word_alignments = aggregate_alignments_by_word(pairs, pair_metadata, all_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_alignments_chunked(output_dir, word_alignments, chunk_size=100)

    print(f"  Alignment generation complete for {features_dir}")
    print(f"  Total words with alignments: {len(word_alignments)}")

    reply = input(f"\n  Clean up intermediate files for {basename} (y/n)? ").strip().lower()
    if reply.startswith("y"):
        shutil.rmtree(work_dir)
        print(f"  Cleaned up intermediate files")

    return output_dir


def main() -> None:
    args = parse_args()

    if not args.dusted:
        raise SystemExit(
            "Error: --dusted is required for LibriSpeech - there is no legacy "
            "UTD word list, only UTDWordsDUSTED.txt."
        )

    feature_set = "LibriSpeech_WavLMDUSTED"
    _LAYER_SUBDIR = f"layer_{LAYER:02d}"
    subset = "train-clean-5"  # DUSTED only covers train-clean-5 - no validation split

    print("\n" + "=" * 60)
    print(f"Generating alignments for {subset} (word-level, layer {LAYER})")
    print("=" * 60)

    features_dir = Path(f"data/Features/{feature_set}/Normalised/{subset}.samediff/{_LAYER_SUBDIR}")
    output_dir = Path(f"data/Alignments/{feature_set}/{subset}.samediff.alignments/{_LAYER_SUBDIR}")

    try:
        generate_alignments(features_dir, args.n_cpus, output_dir)
        print("\n" + "=" * 60)
        print(f"Alignments done for {subset}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError generating alignments for {subset}: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

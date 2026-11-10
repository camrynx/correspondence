"""
Same-different DTW distances, step 1 of 2 (then samediff.py): build every token pair in a
features directory, split the pairs across jobs and run speech_dtw's
calculate_dtw_costs.py on each. A Python version of speech_dtw's run_calcdists.sh.
evaluate_ap.py does the whole evaluation in one step and is what the paper uses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1] #So the file path commands happen on the repo root
DTW_UTILS_DIR = REPO_ROOT / "third_party" / "speech_dtw" / "utils"
CALC_DTW_COSTS = DTW_UTILS_DIR / "calculate_dtw_costs.py" # Code we use to calculate DTW costs, which we call in run_dtw_job() below
FEATURES_ROOT = REPO_ROOT / "data" / "Features"
DISTANCES_ROOT = REPO_ROOT / "results" / "abx"


def distances_dir_for(features_dir: Path) -> Path:
    """results/abx/<features path relative to data/Features> (or just the basename)."""

    resolved = features_dir.resolve()
    try:
        subpath = resolved.relative_to(FEATURES_ROOT)
    except ValueError:
        subpath = Path(features_dir.name)
    return DISTANCES_ROOT / subpath


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare pair files and launch same-different DTW jobs."
    )
    parser.add_argument(
        "features_dir",
        help="Directory containing NPZ files with feature matrices (one file per utterance)",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=16,
        help="number of parallel DTW jobs to launch",
    )
    parser.add_argument(
        "--normalize-dims",
        action="store_true",
        help="Normalise each feature dimension to zero mean / unit variance (pooled "
        "over the whole feature set) before the DTW, as in Carlin et al. (2011), the "
        "same-different paper this evaluation is from. Forwarded to "
        "calculate_dtw_costs.py --normalize_dims. Every DTW worker loads the whole "
        "feature directory, so all workers derive the same statistics and their "
        "distances stay mutually comparable.",
    )
    return parser.parse_args()

#Example argument: python samediff.py --n-cpus 8 Data/Features/devpart2.dd.npz 
# or python samediff.py --n-cpus 8 Data/Features/devpart2.samediff.dd.npz (for filtered words)


def load_npz_keys(features_dir: Path) -> list[str]:
    """Return all feature keys from all NPZ files in the directory."""

    keys = []
    for npz_file in sorted(features_dir.glob("*.npz")):
        with np.load(npz_file, allow_pickle=False) as archive:
            keys.extend(archive.files)
    return sorted(keys)


def write_lines(output_fn: Path, lines: list[str]) -> None:
    """Writing one item per line, and start a new line."""

    with output_fn.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def ensure_utterance_ids(features_dir: Path, utterance_ids_fn: Path) -> None:
    """Write the feature archive keys to disk if the file is missing."""

    if utterance_ids_fn.is_file():
        return

    utterance_ids = load_npz_keys(features_dir)
    write_lines(utterance_ids_fn, utterance_ids)


def derive_label(utterance_id: str) -> str:
    """Extract the word label from a same-different token ID."""

    return utterance_id.split("_", 1)[0]


def write_labels(utterance_ids_fn: Path, labels_fn: Path) -> None:
    """Write one word label per utterance ID if the file is missing."""

    if labels_fn.is_file():
        return

    utterance_ids = [line.strip() for line in utterance_ids_fn.read_text().splitlines() if line.strip()]
    labels = [derive_label(utterance_id) for utterance_id in utterance_ids]
    write_lines(labels_fn, labels)


def ensure_pairs(utterance_ids_fn: Path, pairs_fn: Path) -> None:
    """All unique token pairs, in pdist order."""

    utterance_ids = [
        line.strip() for line in utterance_ids_fn.read_text().splitlines() if line.strip()
    ]
    pairs: list[str] = []
    for i in range(len(utterance_ids) - 1):
        for j in range(i + 1, len(utterance_ids)):
            pairs.append(f"{utterance_ids[i]} {utterance_ids[j]}")
    write_lines(pairs_fn, pairs)


def split_file(input_fn: Path, n_files: int, output_dir: Path) -> None:
    """Split a text file into numbered chunks while preserving order."""

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


def run_dtw_job(
    job: int,
    features_dir: Path,
    pairs_split_dir: Path,
    distances_split_dir: Path,
    normalize_dims: bool = False,
) -> None:
    """Run one DTW worker and capture its output in a job-specific log file."""

    split_pair_fn = pairs_split_dir / f"pairs.{job}.list"
    split_dist_fn = distances_split_dir / f"distances.{job}.dist"
    log_fn = distances_split_dir / f"distances.{job}.log"

    cmd = [
        sys.executable,
        str(CALC_DTW_COSTS),
        "--binary_dists",
        "--input_fmt",
        "npz_dir",
        *(["--normalize_dims"] if normalize_dims else []),
        str(split_pair_fn),
        str(features_dir),
        str(split_dist_fn),
    ]

    with log_fn.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)


def launch_dtw_jobs(
    n_cpus: int,
    features_dir: Path,
    pairs_split_dir: Path,
    distances_split_dir: Path,
    normalize_dims: bool = False,
) -> None:
    """Launch one DTW process per split and wait for all of them to finish."""

    if distances_split_dir.exists():
        shutil.rmtree(distances_split_dir)
    distances_split_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        futures = [
            executor.submit(
                run_dtw_job,
                job,
                features_dir,
                pairs_split_dir,
                distances_split_dir,
                normalize_dims,
            )
            for job in range(1, n_cpus + 1)
        ]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    """Run the full same-different distance-preparation pipeline."""

    start_time = time.perf_counter()

    args = parse_args()
    features_dir = Path(args.features_dir)

    if not features_dir.is_dir():
        print(f"Error: {features_dir} does not exist or is not a directory", file=sys.stderr)
        raise SystemExit(1)

    # Outputs (IDs, labels, pairs, distance splits, logs) go under results/abx/.
    distances_dir = distances_dir_for(features_dir)
    utterance_ids_fn = distances_dir / "utterance_ids.list"
    pairs_fn = distances_dir / "pairs.list"
    pairs_split_dir = distances_dir / "pairs_split"
    labels_fn = distances_dir / "labels.list"
    distances_split_dir = distances_dir / "distances_split"

    distances_dir.mkdir(parents=True, exist_ok=True)

    for stale_fn in (
        distances_dir / "distances.dist",
        distances_dir / "samediff_result.txt",
    ):
        if stale_fn.exists():
            stale_fn.unlink()

    # Step 1: build the token list and word labels used to enumerate pairs.
    ensure_utterance_ids(features_dir, utterance_ids_fn) #Writes the utterance IDs to a file if the file doesn't already exist
    write_labels(utterance_ids_fn, labels_fn) # Writes the word labels to a file if the file doesn't already exist, by deriving them from the utterance IDs

    # Step 2: generate all token pairs and split them into per-job chunks.
    ensure_pairs(utterance_ids_fn, pairs_fn) # Writes all unique token pairs to a file by reading the utterance IDs file
    split_file(pairs_fn, args.n_cpus, pairs_split_dir) # Splits the pairs file into one file per job and refreshes stale split files

    # Step 3: launch the DTW workers. Each worker reads one pair split and
    # writes its distances and logs to the split-specific output directory.
    launch_dtw_jobs(
        args.n_cpus, features_dir, pairs_split_dir, distances_split_dir, args.normalize_dims
    )

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time/(60**2):.6f} hours")


    print(f"Completed. Run python evaluation/samediff.py --n-cpus {args.n_cpus} {features_dir}")


if __name__ == "__main__":
    main()
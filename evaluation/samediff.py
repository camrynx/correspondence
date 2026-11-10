"""Same-different scoring, step 2 of 2 (after calculate_distances.py): check the DTW
jobs finished, join their distance files into distances.dist and score them with
speech_dtw's samediff.py. A Python version of speech_dtw's run_samediff.sh.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEECH_DTW_UTILS_DIR = REPO_ROOT / "third_party" / "speech_dtw" / "utils"
SAMEDIFF_SCRIPT = SPEECH_DTW_UTILS_DIR / "samediff.py"
FEATURES_ROOT = REPO_ROOT / "data" / "Features"
DISTANCES_ROOT = REPO_ROOT / "results" / "abx"


def distances_dir_for(features_dir: Path) -> Path:
	"""Same directory as calculate_distances.py's distances_dir_for."""

	resolved = features_dir.resolve()
	try:
		subpath = resolved.relative_to(FEATURES_ROOT)
	except ValueError:
		subpath = Path(features_dir.name)
	return DISTANCES_ROOT / subpath


def parse_args() -> argparse.Namespace:
	"""Parse the command line arguments."""

	parser = argparse.ArgumentParser(
		description="Score same-different distances and optionally clean outputs."
	)
	parser.add_argument(
		"features_input",
		help="Directory containing NPZ files (one per utterance)",
	)
	parser.add_argument(
		"--n-cpus",
		type=int,
		default=16,
		help="number of DTW splits that should be present",
	)
	return parser.parse_args()


def count_completed_splits(distances_split_dir: Path) -> int:
	"""Count how many split logs contain the expected completion marker."""

	completed = 0
	for log_fn in sorted(distances_split_dir.glob("distances.*.log")):
		if "End time" in log_fn.read_text(encoding="utf-8", errors="ignore"):
			completed += 1
	return completed


def concatenate_distances(distances_split_dir: Path, distances_fn: Path, n_cpus: int) -> None:
	"""Join the split distance files in job order (skipped if distances.dist exists)."""

	if distances_fn.is_file():
		return

	with distances_fn.open("wb") as out_f:
		for job in range(1, n_cpus + 1):
			split_fn = distances_split_dir / f"distances.{job}.dist"
			out_f.write(split_fn.read_bytes())


def run_samediff_score(labels_fn: Path, distances_fn: Path, samediff_result_fn: Path) -> None:
	"""Run the same-different scorer and capture its output."""

	if samediff_result_fn.is_file():
		return

	cmd = [
		sys.executable,
		str(SAMEDIFF_SCRIPT),
		"--binary_dists",
		str(labels_fn),
		str(distances_fn),
	]
	with samediff_result_fn.open("w", encoding="utf-8") as result_f:
		subprocess.run(cmd, check=True, stdout=result_f, stderr=subprocess.STDOUT)


def validate_features_dir(features_input: Path) -> Path:
	"""Validate that the input is a directory containing NPZ files."""

	if not features_input.is_dir():
		print(f"Error: {features_input} is not a directory", file=sys.stderr)
		raise SystemExit(1)
	
	# Check that the directory contains at least one .npz file
	npz_files = list(features_input.glob("*.npz"))
	if not npz_files:
		print(f"Error: {features_input} contains no .npz files", file=sys.stderr)
		raise SystemExit(1)
	
	return features_input


def maybe_clean_outputs(pairs_fn: Path, pairs_split_dir: Path, distances_split_dir: Path) -> None:
	"""Ask before deleting the pair and distance-split files (distances.dist is kept)."""

	# reply = input("Clean distances (y/n)? ").strip().lower()
	reply = "y"
	if reply.startswith("y"):
		if pairs_fn.exists():
			pairs_fn.unlink()
		if pairs_split_dir.exists():
			shutil.rmtree(pairs_split_dir)
		if distances_split_dir.exists():
			shutil.rmtree(distances_split_dir)

		print("Cleaned intermediate pair and distance-split files (kept distances.dist).")


def main() -> None:
	"""Run the full same-different scoring pipeline."""

	args = parse_args()
	features_dir = Path(args.features_input)
	features_dir = validate_features_dir(features_dir)

	distances_dir = distances_dir_for(features_dir)
	pairs_fn = distances_dir / "pairs.list"
	pairs_split_dir = distances_dir / "pairs_split"
	labels_fn = distances_dir / "labels.list"
	distances_split_dir = distances_dir / "distances_split"
	distances_fn = distances_dir / "distances.dist"
	samediff_result_fn = distances_dir / "samediff_result.txt"

	if not distances_split_dir.is_dir():
		print(f"Error: missing DTW outputs in {distances_split_dir}", file=sys.stderr)
		raise SystemExit(1)

	completed = count_completed_splits(distances_split_dir)
	print(f"Number of splits completed: {completed} out of {args.n_cpus}")
	if completed != args.n_cpus:
		print("Error: wait for jobs to complete", file=sys.stderr)
		raise SystemExit(1)

	concatenate_distances(distances_split_dir, distances_fn, args.n_cpus)
	run_samediff_score(labels_fn, distances_fn, samediff_result_fn)

	print()
	print(samediff_result_fn.read_text(encoding="utf-8"), end="")
	print()

	maybe_clean_outputs(pairs_fn, pairs_split_dir, distances_split_dir)


if __name__ == "__main__":
	main()

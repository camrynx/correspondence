"""
DTW alignments for same-phone pairs (Buckeye, from extractFeaturesWavLMPhone.py), in the
same format as training/align.py.

There are about 40 phone types over 230k devpart1 tokens, so all within-type pairs
(about 1.6 billion) is far too many. Pairs are sampled uniformly from that pool until the
number of frame pairs matches the word-level runs (426,334 for devpart1, 133,607 for
devpart2). Phones are only a few frames long, so DTW runs in-process.

    python training/extras/generate_alignments_WavLMPhone.py --seed 11 --layer 21
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import ceil
from pathlib import Path
import pickle
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "speech_dtw"))
from speech_dtw import _dtw  # noqa: E402

# Which extracted layer to align; must match a layer_NN/ subdirectory under
# data/Features/Buckeye_WavLMPhone/Normalised/<subset>.samediff/ (see
# features/extras/extractFeaturesWavLMPhone.py --layer and normalisePhone.py --layer).
LAYER = 21

FEATURE_SET = "Buckeye_WavLMPhone"

# Frame-pair budget per subset: the same as the gold word-pair runs, so the phone
# condition trains for the same epoch size. These are the "Total frame pairs indexed"
# figures the word-level ccAE run reported.
FRAME_PAIR_BUDGET = {
    "devpart1": 426_334,
    "devpart2": 133_607,
}

# How many candidate pairs to DTW before truncating to the budget. Mean DTW path length
# for 3-8 frame sequences is ~6.5, so budget/4.0 comfortably oversamples; the surplus is
# discarded by dropping whole pairs in random order, which keeps the sample unbiased.
OVERSAMPLE_FRAMES_PER_PAIR = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed DTW alignments for phone-level WavLM features.",
    )
    parser.add_argument(
        "--layer", type=int, default=LAYER,
        help="which extracted layer to align (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=11,
        help="random seed for pair sampling (default: %(default)s, matching training)",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=None,
        help="how many candidate pairs to sample and DTW before truncating to the frame-pair "
             "budget. Default: budget / 4.0, which oversamples by roughly 1.6x.",
    )
    parser.add_argument(
        "--frame-pairs", type=int, default=None,
        help="override the frame-pair budget for every subset (default: the word-level "
             "counts, 426334 for devpart1 and 133607 for devpart2)",
    )
    parser.add_argument(
        "--subsets", nargs="+", default=["devpart1", "devpart2"],
        help="which subsets to process (default: %(default)s)",
    )
    return parser.parse_args()


def load_features(features_dir: Path) -> dict[str, np.ndarray]:
    """Load every phone token's frames from the .npz files in features_dir."""
    if not features_dir.is_dir():
        raise FileNotFoundError(
            f"Features directory not found: {features_dir}\n"
            f"Please run:\n"
            f"  python features/extras/extractFeaturesWavLMPhone.py --layer {features_dir.name.split('_')[-1]}\n"
            f"  python features/extras/normalisePhone.py --layer {features_dir.name.split('_')[-1]}"
        )

    features: dict[str, np.ndarray] = {}
    npz_files = sorted(features_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {features_dir}")

    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=False) as archive:
            for key in archive.files:
                features[key] = archive[key].astype(np.float32)
    return features


def derive_label(utterance_id: str) -> str:
    """Extract the phone label from a token key (everything before the first underscore)."""
    return utterance_id.split("_", 1)[0]


def sample_same_phone_pairs(
    keys: list[str], n_pairs: int, rng: np.random.Generator
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Sample n_pairs distinct same-phone pairs uniformly from all such pairs: pick a
    phone type with probability proportional to n(n-1)/2, then a pair within it."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_label[derive_label(key)].append(key)

    labels = sorted(l for l, toks in by_label.items() if len(toks) >= 2)
    if not labels:
        raise ValueError("No phone type has at least 2 tokens - nothing to pair.")

    counts = np.array([len(by_label[l]) for l in labels], dtype=np.float64)
    pool = counts * (counts - 1) / 2.0
    total_pool = pool.sum()
    probs = pool / total_pool

    print(f"  {len(labels)} phone types, {int(counts.sum()):,} tokens, "
          f"{int(total_pool):,} possible pairs")
    if n_pairs > total_pool:
        raise ValueError(
            f"Asked for {n_pairs:,} pairs but only {int(total_pool):,} exist."
        )

    # Sample with a modest surplus and de-duplicate. At these ratios collisions are
    # negligible (~2 expected for 1e5 draws from 1.6e9), so one pass is enough, but we
    # loop until the target is met to stay correct for small feature sets.
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    label_index = {l: i for i, l in enumerate(labels)}

    while len(pairs) < n_pairs:
        n_draw = int(ceil((n_pairs - len(pairs)) * 1.05)) + 16
        drawn_labels = rng.choice(len(labels), size=n_draw, p=probs)
        for li in drawn_labels:
            toks = by_label[labels[li]]
            i, j = rng.integers(0, len(toks), size=2)
            if i == j:
                continue
            a, b = (toks[i], toks[j]) if i < j else (toks[j], toks[i])
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))
            if len(pairs) == n_pairs:
                break

    pair_labels = {}
    for a, b in pairs:
        pair_labels[f"{a} {b}"] = derive_label(a)
    return pairs, pair_labels


def dtw_pairs(
    pairs: list[tuple[str, str]], features: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """DTW path for each pair (speech_dtw's cosine multivariate_dtw)."""
    paths: list[np.ndarray] = []
    t0 = time.time()
    for idx, (a, b) in enumerate(pairs):
        feat_a = np.asarray(features[a], dtype=np.double, order="c")
        feat_b = np.asarray(features[b], dtype=np.double, order="c")
        path = _dtw.multivariate_dtw(feat_a, feat_b)[0]
        # speech_dtw returns the path from the end backwards; the word pipeline stores it
        # exactly as returned, so we do too - order is irrelevant, each entry is an
        # independent (frame_a, frame_b) training example.
        paths.append(np.array(path, dtype=np.int32))
        if (idx + 1) % 25000 == 0:
            print(f"    {idx + 1:,}/{len(pairs):,} pairs  ({time.time() - t0:.1f}s)")
    print(f"    {len(pairs):,} pairs DTW'd in {time.time() - t0:.1f}s")
    return paths


def truncate_to_budget(
    pairs: list[tuple[str, str]],
    paths: list[np.ndarray],
    budget: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, str]], list[np.ndarray], int]:
    """Keep whole pairs, in random order, until the frame-pair budget is reached."""
    order = rng.permutation(len(pairs))
    kept_pairs, kept_paths, total = [], [], 0
    for i in order:
        if total >= budget:
            break
        kept_pairs.append(pairs[i])
        kept_paths.append(paths[i])
        total += len(paths[i])
    return kept_pairs, kept_paths, total


def aggregate_by_label(
    pairs: list[tuple[str, str]], paths: list[np.ndarray]
) -> dict[str, list[dict]]:
    """Group alignments by phone label, in the format the training datasets expect."""
    alignments: dict[str, list[dict]] = defaultdict(list)
    for (a, b), frame_pairs in zip(pairs, paths):
        alignments[derive_label(a)].append({"pair": (a, b), "frame_pairs": frame_pairs})
    return dict(alignments)


def save_alignments_chunked(output_dir: Path, alignments: dict, chunk_size: int = 5) -> None:
    """Save chunked pickles (5 labels per chunk, since there are only ~40 labels)."""
    print(f"Saving alignments to chunked pickle files in {output_dir} (chunk size: {chunk_size} labels)")
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_labels = sorted(alignments.keys())
    n_chunks = (len(sorted_labels) + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_labels = sorted_labels[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        chunk_data = {label: alignments[label] for label in chunk_labels}
        with open(output_dir / f"alignments.chunk_{chunk_idx:04d}.pkl", "wb") as f:
            pickle.dump(chunk_data, f)

    metadata = {
        "total_words": len(alignments),
        "total_chunks": n_chunks,
        "chunk_size": chunk_size,
        "words_per_chunk": [
            len(sorted_labels[i * chunk_size:(i + 1) * chunk_size]) for i in range(n_chunks)
        ],
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {n_chunks} alignment chunks to {output_dir}")
    print(f"  Total phone labels: {len(alignments)}")


def main() -> None:
    args = parse_args()
    layer_subdir = f"layer_{args.layer:02d}"

    for subset in args.subsets:
        budget = args.frame_pairs or FRAME_PAIR_BUDGET.get(subset)
        if budget is None:
            raise ValueError(
                f"No frame-pair budget known for subset '{subset}'; pass --frame-pairs."
            )
        n_pairs = args.n_pairs or int(ceil(budget / OVERSAMPLE_FRAMES_PER_PAIR))

        print(f"\n{'=' * 70}\n{subset}  (budget {budget:,} frame pairs)\n{'=' * 70}")

        features_dir = (
            REPO_ROOT / "data/Features" / FEATURE_SET / "Normalised"
            / f"{subset}.samediff" / layer_subdir
        )
        print(f"Loading features from {features_dir}...")
        features = load_features(features_dir)
        print(f"  {len(features):,} phone tokens loaded")

        rng = np.random.default_rng(args.seed)
        print(f"Sampling {n_pairs:,} candidate pairs...")
        pairs, _ = sample_same_phone_pairs(sorted(features.keys()), n_pairs, rng)

        print("Computing DTW paths in process...")
        paths = dtw_pairs(pairs, features)
        sampled_frames = sum(len(p) for p in paths)
        print(f"  {sampled_frames:,} frame pairs from {len(pairs):,} pairs "
              f"(mean path {sampled_frames / len(pairs):.2f})")

        if sampled_frames < budget:
            suggested = int(ceil(n_pairs * budget / max(sampled_frames, 1) * 1.1))
            print(
                f"\nWARNING: only {sampled_frames:,} frame pairs, short of the "
                f"{budget:,} budget.\n"
                f"         Re-run with --n-pairs {suggested} to hit it.",
                file=sys.stderr,
            )

        pairs, paths, total = truncate_to_budget(pairs, paths, budget, rng)
        print(f"Truncated to {total:,} frame pairs from {len(pairs):,} pairs")

        alignments = aggregate_by_label(pairs, paths)
        per_label = sorted(
            ((label, sum(len(e["frame_pairs"]) for e in entries)) for label, entries in alignments.items()),
            key=lambda kv: -kv[1],
        )
        print("  frame pairs per phone (top 10): "
              + ", ".join(f"{l}={n:,}" for l, n in per_label[:10]))

        output_dir = (
            REPO_ROOT / "data/Alignments" / FEATURE_SET
            / f"{subset}.samediff" / layer_subdir
        )
        save_alignments_chunked(output_dir, alignments)

    print("\nDone.")


if __name__ == "__main__":
    main()

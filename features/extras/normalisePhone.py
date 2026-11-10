"""
Normalise Buckeye phone features with the Buckeye_WavLM word statistics (from
normalise.py), not their own, since models trained on phone pairs are evaluated on the
word features.

    python features/extras/normalisePhone.py --layer 21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_FEATURE_SET = "Buckeye_WavLM"
PHONE_FEATURE_SET = "Buckeye_WavLMPhone"

# Phone features exist only to build training frame pairs; evaluation runs on the gold
# word features, so there is no zs phone set (see extractFeaturesWavLMPhone.py).
SUBSETS = ["devpart1", "devpart2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalise Buckeye phone features with the base Buckeye_WavLM statistics.",
    )
    parser.add_argument(
        "--layer", type=int, default=21,
        help="which extracted layer to normalise (default: %(default)s)",
    )
    return parser.parse_args()


def normalise_folder(input_dir: Path, output_dir: Path, mean: np.ndarray, std: np.ndarray) -> None:
    """Apply (x - mean) / std to every array in every .npz under input_dir."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            f"Please run:\n"
            f"  python features/extras/extractFeaturesWavLMPhone.py"
        )

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    n_tokens = 0
    for npz_file in tqdm(npz_files, desc=f"Normalising {input_dir.parent.name}"):
        with np.load(npz_file, allow_pickle=False) as f:
            normalised = {}
            for key in f.files:
                arr = f[key].astype(np.float32)
                normalised[key] = (arr - mean) / std
                n_tokens += 1
        np.savez_compressed(output_dir / npz_file.name, **normalised)

    print(f"  wrote {len(npz_files)} files ({n_tokens:,} phone tokens) to {output_dir}")


def main() -> None:
    args = parse_args()
    layer_subdir = f"layer_{args.layer:02d}"

    stats_fn = (
        REPO_ROOT / "data/Features" / BASE_FEATURE_SET / "Normalised"
        / f"norm_stats_layer_{args.layer:02d}.pt"
    )
    if not stats_fn.is_file():
        raise FileNotFoundError(
            f"Base normalisation statistics not found: {stats_fn}\n"
            f"Please run:\n"
            f"  python features/normalise.py --layer {args.layer}"
        )

    stats = torch.load(stats_fn, map_location="cpu")
    mean = stats["mean"].numpy().astype(np.float32)
    std = stats["std"].numpy().astype(np.float32)
    print(f"Loaded base statistics from {stats_fn}  (dim {mean.shape[0]})")
    print("These were fit on devpart1 WORD tokens and are applied here unchanged, so the")
    print("phone training features match the word features Stage 3 will encode.")

    for subset in SUBSETS:
        input_dir = (
            REPO_ROOT / "data/Features" / PHONE_FEATURE_SET / "NotNormalised"
            / f"{subset}.samediff" / layer_subdir
        )
        output_dir = (
            REPO_ROOT / "data/Features" / PHONE_FEATURE_SET / "Normalised"
            / f"{subset}.samediff" / layer_subdir
        )
        normalise_folder(input_dir, output_dir, mean, std)

    print("\nDone.")


if __name__ == "__main__":
    main()

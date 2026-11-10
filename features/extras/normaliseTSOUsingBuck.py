"""
Normalise Xitsonga WavLM features with the Buckeye training statistics, so a
Buckeye-trained cAE sees inputs scaled the way it was trained (zero-shot transfer).

    python features/extras/normaliseTSOUsingBuck.py --layer 21
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# ----------------------------- CONFIGURE THESE -----------------------------

parser = argparse.ArgumentParser(
    description="Normalise Xitsonga WavLM features with the Buckeye stats."
    )
parser.add_argument(
    "--layer", type=int, default=21,
    help="Which extracted layer to normalise (default matches LAYERS in "
         "extractFeaturesWavLMTSO.py; must be the layer normalise.py used, "
         "since the Buckeye stats come from there).",
    )
args = parser.parse_args()

# The Buckeye stats must be for the same layer.
_LAYER_SUBDIR = f"layer_{args.layer:02d}"

# Stats computed from Buckeye devpart1, for this layer (see normalise.py)
NORM_STATS_PATH = Path(f"data/Features/Buckeye_WavLM/Normalised/norm_stats_{_LAYER_SUBDIR}.pt")

# Input directories (raw, unnormalised Xitsonga features)
DEV_DIR  = Path(f"data/Features/Xitsonga_WavLM/NotNormalised/dev.samediff/{_LAYER_SUBDIR}")
TEST_DIR = Path(f"data/Features/Xitsonga_WavLM/NotNormalised/test.samediff/{_LAYER_SUBDIR}")

# Output directories (normalised features)
DEV_OUT  = Path(f"data/Features/Xitsonga_WavLM/NormalisedFromBuckeye/dev.samediff/{_LAYER_SUBDIR}")
TEST_OUT = Path(f"data/Features/Xitsonga_WavLM/NormalisedFromBuckeye/test.samediff/{_LAYER_SUBDIR}")

# ---------------------------------------------------------------------------


def normalise_folder(input_dir: Path, output_dir: Path, mean: torch.Tensor, std: torch.Tensor):
    """(x - mean) / std for every file in input_dir, written to output_dir."""
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"  Warning: no .npz files found in {input_dir}, skipping.")
        return

    mean_np = mean.numpy().astype(np.float32)
    std_np  = std.numpy().astype(np.float32)

    print(f"Normalising {len(npz_files)} files from {input_dir}")
    print(f"         -> {output_dir}")

    for npz_file in tqdm(npz_files, desc=f"Normalising {input_dir.parent.name}"):
        with np.load(npz_file, allow_pickle=False) as f:
            normalised = {}
            for key in f.files:
                arr = f[key].astype(np.float32)
                normalised[key] = (arr - mean_np) / std_np

        out_path = output_dir / npz_file.name
        np.savez_compressed(out_path, **normalised)

    print(f"  Done. Saved to {output_dir}")


def main():
    print(f"Loading Buckeye normalisation stats from {NORM_STATS_PATH}")
    stats = torch.load(NORM_STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    normalise_folder(DEV_DIR, DEV_OUT, mean, std)
    print()
    normalise_folder(TEST_DIR, TEST_OUT, mean, std)

    print("\nDone.")


if __name__ == "__main__":
    main()

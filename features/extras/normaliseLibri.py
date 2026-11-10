"""
Normalise LibriSpeech WavLM features with train-clean-5 statistics (applied to
train-clean-5 and dev-clean).

    python features/extras/normaliseLibri.py [--layer N]
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# ----------------------------- CONFIGURE THESE -----------------------------

parser = argparse.ArgumentParser(description="Normalise LibriSpeech gold-word WavLM features.")
parser.add_argument(
    "--layer", type=int, default=21,
    help="Which extracted layer to normalise (default matches LAYERS in "
         "extractFeaturesWavLMLibri.py).",
    )
args = parser.parse_args()

_LAYER_SUBDIR = f"layer_{args.layer:02d}"

# Input directories (raw, unnormalised features)
TRAIN_DIR = Path(f"data/Features/LibriSpeech_WavLM/NotNormalised/train-clean-5.samediff/{_LAYER_SUBDIR}")
DEV_DIR   = Path(f"data/Features/LibriSpeech_WavLM/NotNormalised/dev-clean.samediff/{_LAYER_SUBDIR}")

# Output directories (normalised features)
TRAIN_OUT = Path(f"data/Features/LibriSpeech_WavLM/Normalised/train-clean-5.samediff/{_LAYER_SUBDIR}")
DEV_OUT   = Path(f"data/Features/LibriSpeech_WavLM/Normalised/dev-clean.samediff/{_LAYER_SUBDIR}")

# Where to save the normalisation stats (load this at inference time)
NORM_STATS_PATH = Path(f"data/Features/LibriSpeech_WavLM/Normalised/norm_stats_{_LAYER_SUBDIR}.pt")

# ---------------------------------------------------------------------------


def compute_stats(input_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std over all frames in input_dir (two passes)."""
    input_dir = Path(input_dir)
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    print(f"Computing normalisation stats from {len(npz_files)} files in {input_dir}...")

    total_frames = 0
    running_sum = None

    for npz_file in tqdm(npz_files, desc="Pass 1/2: computing mean"):
        with np.load(npz_file, allow_pickle=False) as f:
            for key in f.files:
                arr = f[key].astype(np.float64)  # (n_frames, feat_dim)
                if running_sum is None:
                    running_sum = arr.sum(axis=0)
                else:
                    running_sum += arr.sum(axis=0)
                total_frames += arr.shape[0]

    mean = running_sum / total_frames  # (feat_dim,)

    running_sq_diff = np.zeros_like(mean)

    for npz_file in tqdm(npz_files, desc="Pass 2/2: computing std"):
        with np.load(npz_file, allow_pickle=False) as f:
            for key in f.files:
                arr = f[key].astype(np.float64)
                running_sq_diff += ((arr - mean) ** 2).sum(axis=0)

    std = np.sqrt(running_sq_diff / total_frames) + 1e-8  # (feat_dim,)

    mean_t = torch.from_numpy(mean.astype(np.float32))
    std_t = torch.from_numpy(std.astype(np.float32))

    print(f"  Frames processed : {total_frames:,}")
    print(f"  Mean  - min: {mean_t.min():.4f}, max: {mean_t.max():.4f}, avg: {mean_t.mean():.4f}")
    print(f"  Std   - min: {std_t.min():.4f},  max: {std_t.max():.4f},  avg: {std_t.mean():.4f}")

    return mean_t, std_t


def normalise_folder(input_dir: Path, output_dir: Path, mean: torch.Tensor, std: torch.Tensor):
    """(x - mean) / std for every file in input_dir, written to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"  Warning: no .npz files found in {input_dir}, skipping.")
        return

    mean_np = mean.numpy().astype(np.float32)
    std_np = std.numpy().astype(np.float32)

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
    # Step 1: compute stats from training set only
    mean, std = compute_stats(TRAIN_DIR)

    # Step 2: save stats for use at inference time
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, NORM_STATS_PATH)
    print(f"\nSaved normalisation stats to {NORM_STATS_PATH}")

    # Step 3: normalise both splits with training stats
    print()
    normalise_folder(TRAIN_DIR, TRAIN_OUT, mean, std)
    print()
    normalise_folder(DEV_DIR, DEV_OUT, mean, std)

    print("\nDone.")
    print(f"At inference time, load stats with:")
    print(f"  stats = torch.load('{NORM_STATS_PATH}')")
    print(f"  features = (features - stats['mean'].to(device)) / stats['std'].to(device)")


if __name__ == "__main__":
    main()

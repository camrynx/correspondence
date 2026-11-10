"""
Normalise WavLM / HuBERT features with each dimension's mean and std over the training
split.

    python features/normalise.py --features wavlm --corpus buckeye
    python features/normalise.py --features hubert --corpus xitsonga --pairs dusted

Saves the stats to <set>/Normalised/norm_stats_layer_NN.pt and writes Normalised/ copies of
every split (plus all.samediff/ for Buckeye and Xitsonga UTD/DUSTED). MFCCs are normalised
at extraction instead.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FEATURES = {"wavlm": ("WavLM", 21), "hubert": ("HuBERT", 24)}
CORPORA = {
    "buckeye": ("Buckeye", ("devpart1", "devpart2", "zs")),
    "xitsonga": ("Xitsonga", ("dev", "test")),
    "french": ("ZS17French", ("train", "test")),  # first split = training split
}


def compute_stats(input_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std over every frame in input_dir, in two streaming passes."""
    npz_files = sorted(Path(input_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    print(f"Computing normalisation stats from {len(npz_files)} files in {input_dir}...")

    total_frames, running_sum = 0, None
    for npz_file in tqdm(npz_files, desc="Pass 1/2: mean"):
        with np.load(npz_file, allow_pickle=False) as f:
            for key in f.files:
                arr = f[key].astype(np.float64)
                running_sum = arr.sum(axis=0) if running_sum is None else running_sum + arr.sum(axis=0)
                total_frames += arr.shape[0]
    mean = running_sum / total_frames

    running_sq_diff = np.zeros_like(mean)
    for npz_file in tqdm(npz_files, desc="Pass 2/2: std"):
        with np.load(npz_file, allow_pickle=False) as f:
            for key in f.files:
                running_sq_diff += ((f[key].astype(np.float64) - mean) ** 2).sum(axis=0)
    std = np.sqrt(running_sq_diff / total_frames) + 1e-8

    mean_t, std_t = torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))
    print(f"  Frames: {total_frames:,}  mean in [{mean_t.min():.4f}, {mean_t.max():.4f}]  "
          f"std in [{std_t.min():.4f}, {std_t.max():.4f}]")
    return mean_t, std_t


def normalise_folder(input_dir: Path, output_dir: Path, mean: torch.Tensor, std: torch.Tensor) -> None:
    """(x - mean) / std for every token in input_dir, written under the same file names."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"  Warning: no .npz files found in {input_dir}, skipping.")
        return
    mean_np, std_np = mean.numpy().astype(np.float32), std.numpy().astype(np.float32)
    print(f"Normalising {len(npz_files)} files: {input_dir} -> {output_dir}")
    for npz_file in tqdm(npz_files, desc=f"Normalising {input_dir.parent.name}"):
        with np.load(npz_file, allow_pickle=False) as f:
            normalised = {key: (f[key].astype(np.float32) - mean_np) / std_np for key in f.files}
        np.savez_compressed(output_dir / npz_file.name, **normalised)


def combine_into_all(input_dirs: list[Path], output_dir: Path) -> None:
    """Copy every split's normalised .npz files into one all.samediff directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Combining normalised splits into {output_dir}")
    for input_dir in input_dirs:
        for npz_file in sorted(input_dir.glob("*.npz")):
            dest = output_dir / npz_file.name
            if dest.exists():
                print(f"  Warning: {dest.name} already exists in {output_dir}, overwriting.")
            shutil.copy2(npz_file, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise WavLM/HuBERT features with training-split stats.")
    parser.add_argument("--features", choices=FEATURES, required=True)
    parser.add_argument("--corpus", choices=[*CORPORA, "zs17french"], required=True,
                        help="'zs17french' is an alias for 'french'")
    parser.add_argument("--pairs", choices=["gold", "utd", "dusted"], default="gold",
                        help="which extracted token set (default: %(default)s)")
    parser.add_argument("--layer", type=int, default=None,
                        help="layer to normalise (default: 21 for WavLM, 24 for HuBERT)")
    args = parser.parse_args()
    corpus = "french" if args.corpus == "zs17french" else args.corpus

    tag, default_layer = FEATURES[args.features]
    prefix, splits = CORPORA[corpus]
    layer_dir = f"layer_{(args.layer or default_layer):02d}"
    root = Path("data/Features") / (f"{prefix}_{tag}" + {"gold": "", "utd": "UTD", "dusted": "DUSTED"}[args.pairs])
    inputs = {s: root / "NotNormalised" / f"{s}.samediff" / layer_dir for s in splits}
    outputs = {s: root / "Normalised" / f"{s}.samediff" / layer_dir for s in splits}
    stats_path = root / "Normalised" / f"norm_stats_{layer_dir}.pt"

    mean, std = compute_stats(inputs[splits[0]])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"Saved normalisation stats to {stats_path}")

    for s in splits:
        normalise_folder(inputs[s], outputs[s], mean, std)
    if args.pairs != "gold" and corpus != "french":
        combine_into_all([outputs[s] for s in splits], root / "Normalised" / "all.samediff" / layer_dir)
    print("Done.")


if __name__ == "__main__":
    main()

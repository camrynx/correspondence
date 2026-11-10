"""Save the pre-activation output of every hidden layer of the gold Buckeye WavLM/HuBERT
cAE, not just the bottleneck.

Writes <split>.samediff_cAE/layer_NN/cAE_layer_K/*.npz for K = 1..9 (K = 8 is the
bottleneck, the same as train_cae.py's output).

    python training/extract_cae_all_layers.py --feature-type wavlm hubert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

SPLITS = ("devpart1", "devpart2", "zs")

# Same architecture as training/train_cae.py.
CONFIGS = {
    "wavlm": dict(
        feature_set="Buckeye_WavLM",
        layer=21,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
    ),
    "hubert": dict(
        feature_set="Buckeye_HuBERT",
        layer=24,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
    ),
}


class Autoencoder(nn.Module):
    """Same architecture as training/train_cae.py / train_cae.py's Autoencoder."""

    def __init__(self, input_dim: int, hidden_units: int, encoding_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_units), nn.Tanh()]
        for layer in range(num_layers - 1):
            if layer == num_layers - 3:
                layers += [nn.Linear(hidden_units, encoding_dim), nn.Tanh()]
            elif layer == num_layers - 2:
                layers += [nn.Linear(encoding_dim, hidden_units), nn.Tanh()]
            else:
                layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_cae(ckpt_fn: Path, config: dict) -> Autoencoder:
    model = Autoencoder(config["input_dim"], config["hidden_units"], config["encoding_dim"], config["num_layers"]).to(device)
    model.load_state_dict(torch.load(ckpt_fn, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def encode_all_layers(model: Autoencoder, x: torch.Tensor, num_layers: int) -> dict[int, torch.Tensor]:
    """One forward pass -> {layer K (1-indexed): pre-activation output of its Linear},
    stopping before the output layer."""
    modules = list(model.network)
    outputs: dict[int, torch.Tensor] = {}
    h = x
    i = 0
    for layer_idx in range(1, num_layers + 1):
        h = modules[i](h)  # Linear
        i += 1
        outputs[layer_idx] = h
        if i < len(modules) and isinstance(modules[i], (nn.Tanh, nn.ReLU)):
            h = modules[i](h)
            i += 1
    return outputs


def extract_split(model: Autoencoder, num_layers: int, input_dir: Path, output_root: Path, desc: str) -> None:
    """Encode every .npz in input_dir through every layer, writing to
    output_root/cAE_layer_{1..num_layers}/<same filename>.npz."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    layer_dirs = {}
    for layer_idx in range(1, num_layers + 1):
        d = output_root / f"cAE_layer_{layer_idx}"
        d.mkdir(parents=True, exist_ok=True)
        layer_dirs[layer_idx] = d

    for npz_file in tqdm(npz_files, desc=desc):
        with np.load(npz_file, allow_pickle=False) as f:
            per_layer_dicts: dict[int, dict[str, np.ndarray]] = {k: {} for k in range(1, num_layers + 1)}
            for utt_id in sorted(f.files):
                features = torch.from_numpy(f[utt_id]).float().to(device)
                layer_outputs = encode_all_layers(model, features, num_layers)
                for layer_idx, emb in layer_outputs.items():
                    per_layer_dicts[layer_idx][utt_id] = emb.cpu().numpy().astype(np.float32)

        for layer_idx, embeddings_dict in per_layer_dicts.items():
            np.savez_compressed(layer_dirs[layer_idx] / npz_file.name, **embeddings_dict)


def run_feature_type(feature_type: str) -> None:
    config = CONFIGS[feature_type]
    layer_subdir = f"layer_{config['layer']:02d}"

    ckpt_fn = REPO_ROOT / "models" / config["feature_set"] / f"cAE_{layer_subdir}.pt"
    if not ckpt_fn.is_file():
        raise FileNotFoundError(f"No gold checkpoint at {ckpt_fn}")

    print(f"\n=== {feature_type.upper()} ({config['feature_set']}, {layer_subdir}) ===")
    print(f"Loading cAE from {ckpt_fn}...")
    model = load_cae(ckpt_fn, config)

    features_root = REPO_ROOT / "data/Features" / config["feature_set"] / "Normalised"

    for split in SPLITS:
        input_dir = features_root / f"{split}.samediff" / layer_subdir
        if not input_dir.is_dir():
            print(f"Skipping split '{split}': {input_dir} not found")
            continue
        output_root = features_root / f"{split}.samediff_cAE" / layer_subdir
        extract_split(
            model, config["num_layers"], input_dir, output_root,
            desc=f"{feature_type} {split}: all layers",
        )
        print(f"Saved {config['num_layers']} per-layer directories under {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--feature-type", nargs="+", choices=sorted(CONFIGS), default=sorted(CONFIGS),
        help="Which Buckeye cAE(s) to extract from (default: both wavlm and hubert).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for feature_type in args.feature_type:
        run_feature_type(feature_type)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""Encode a directory of word features with a trained cAE checkpoint.

train_cae.py encodes its own corpus; this applies any checkpoint to any features, e.g. a
Buckeye cAE to Xitsonga. The architecture is read from the checkpoint. Output: one .npz
per input .npz with the same keys, holding the bottleneck output before its activation.

    python training/encode_with_cae.py \
        data/Features/Xitsonga_WavLM/NormalisedFromBuckeye/test.samediff/layer_21 \
        models/Buckeye_WavLM/cAE_layer_21.pt \
        data/Features/Xitsonga_WavLM/NormalisedFromBuckeye/test.samediff_cAE/layer_21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    """Same architecture as training/train_cae.py's Autoencoder."""

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


def architecture_from_state_dict(state_dict: dict) -> tuple[int, int, int, int]:
    """(input_dim, hidden_units, encoding_dim, num_layers) from the checkpoint's Linear
    weights; the narrowest hidden layer is the bottleneck."""

    weights = [
        state_dict[key]
        for key in sorted(state_dict, key=lambda k: int(k.split(".")[1]))
        if key.endswith(".weight")
    ]
    if not weights:
        raise SystemExit("Error: checkpoint contains no 'network.N.weight' tensors")

    input_dim = weights[0].shape[1]
    hidden_units = weights[0].shape[0]
    # Skip the output layer, which can be narrower than the bottleneck (MFCC: 39 < 100).
    encoding_dim = min(w.shape[0] for w in weights[:-1])
    num_layers = len(weights) - 1
    return input_dim, hidden_units, encoding_dim, num_layers


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Encode word-level .npz features through a trained cAE checkpoint."
    )
    parser.add_argument(
        "features_dir",
        type=Path,
        help="Directory of input .npz files, e.g. "
        "data/Features/Xitsonga_WavLM/NormalisedFromBuckeye/test.samediff/layer_21",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="cAE (or AE) checkpoint to encode with, e.g. models/Buckeye_WavLM/cAE_layer_21.pt",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Where to write the encoded .npz files (created if missing)",
    )
    parser.add_argument(
        "--cae-layer",
        type=int,
        default=None,
        help="Which hidden layer of the cAE to encode to, before its activation "
        "(1-indexed: 1 = first hidden Linear, num_layers-1 = the bottleneck). "
        "Matches extract_cae_all_layers.py's cAE_layer_K numbering. Defaults to "
        "the bottleneck, i.e. what train_cae.py's Stage 3 saves.",
    )
    return parser.parse_args()


def main() -> None:
    """Encode one features directory through one checkpoint."""

    args = parse_args()

    if not args.features_dir.is_dir():
        raise SystemExit(f"Error: features directory not found: {args.features_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Error: checkpoint not found: {args.checkpoint}")

    npz_files = sorted(args.features_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"Error: no .npz files in {args.features_dir}")

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    input_dim, hidden_units, encoding_dim, num_layers = architecture_from_state_dict(state_dict)
    print(
        f"Checkpoint {args.checkpoint}: input_dim={input_dim} hidden_units={hidden_units} "
        f"encoding_dim={encoding_dim} num_layers={num_layers}"
    )

    model = Autoencoder(input_dim, hidden_units, encoding_dim, num_layers).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Hidden layer K's pre-activation output is network[:2*K - 1]; K = num_layers - 1
    # is the bottleneck.
    if args.cae_layer is None:
        slice_index = 2 * num_layers - 3
    elif not 1 <= args.cae_layer <= num_layers:
        raise SystemExit(f"Error: --cae-layer must be between 1 and {num_layers}, got {args.cae_layer}")
    else:
        slice_index = 2 * args.cae_layer - 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_utterances = 0
    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc="Encoding"):
            embeddings_dict = {}
            with np.load(npz_file, allow_pickle=False) as f:
                for utt_id in sorted(f.files):
                    frames = f[utt_id]
                    if frames.shape[-1] != input_dim:
                        raise SystemExit(
                            f"Error: {npz_file.name}:{utt_id} has {frames.shape[-1]} dimensions "
                            f"but the checkpoint expects {input_dim} - wrong feature type or layer?"
                        )
                    x = torch.from_numpy(frames).float().to(device)
                    embeddings_dict[utt_id] = (
                        model.network[:slice_index](x).cpu().numpy().astype(np.float32)
                    )
            n_utterances += len(embeddings_dict)
            np.savez_compressed(args.output_dir / npz_file.name, **embeddings_dict)

    print(f"Wrote {n_utterances} encoded utterances across {len(npz_files)} files to {args.output_dir}")


if __name__ == "__main__":
    main()

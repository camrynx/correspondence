"""
Train a correspondence autoencoder (cAE) on aligned frame pairs.

    python training/train_cae.py --features wavlm --corpus buckeye --pairs gold
    python training/train_cae.py --features mfcc --corpus french --pairs dusted

Each frame is mapped to the frame it is aligned with (MSE loss), in both directions.
Validation uses the held-out split's alignments where there are any (not for DUSTED or
Poor Man's UTD, which only cover the training split). After training, every split's
gold word features are encoded through the bottleneck, whatever the training pairs, and
written to <set>/.../<split>.samediff_cAE for evaluate_ap.py.

Splits: buckeye devpart1 / devpart2 / zs, xitsonga dev / test, french train / test
(train / validation / test, with test also used for validation where there are only two).
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FEATURES = {
    # pair_order: layout of the frame-pair index, which decides the batches under the
    # seeded shuffle. The paper's MFCC word-pair models used "blocked" (an alignment's
    # forward pairs, then its reverse pairs), everything else "interleaved".
    "mfcc": dict(tag="MFCCs", dim=39, hidden_units=250, encoding_dim=100, lr=1e-3, layer=None),
    "wavlm": dict(tag="WavLM", dim=1024, hidden_units=2048, encoding_dim=128, lr=1e-5, layer=21),
    "hubert": dict(tag="HuBERT", dim=1024, hidden_units=2048, encoding_dim=128, lr=1e-5, layer=24),
}

CORPORA = {
    "buckeye": dict(prefix="Buckeye", train="devpart1", val="devpart2", encode=("devpart1", "devpart2", "zs")),
    "xitsonga": dict(prefix="Xitsonga", train="dev", val="test", encode=("dev", "test")),
    "french": dict(prefix="ZS17French", train="train", val="test", encode=("train", "test")),
}

PAIRS = ("gold", "utd", "dusted", "poorman")

N_EPOCHS = 30
BATCH_SIZE = 2048
NUM_HIDDEN_LAYERS = 9  # 7 hidden + the bottleneck and the layer after it
ACTIVATION = "Tanh"  # "Tanh" or "ReLU"
SEED = 11
# network[:BOTTLENECK_SLICE] is the encoder: up to and including the bottleneck Linear,
# but not its activation.
BOTTLENECK_SLICE = 2 * NUM_HIDDEN_LAYERS - 3  # == 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a correspondence autoencoder (cAE).")
    parser.add_argument("--features", choices=FEATURES, required=True)
    parser.add_argument("--corpus", choices=[*CORPORA, "zs17french"], required=True,
                        help="'zs17french' is an alias for 'french'")
    parser.add_argument("--pairs", choices=PAIRS, default="gold", help="training pairs (default: %(default)s)")
    parser.add_argument("--layer", type=int, default=None,
                        help="WavLM/HuBERT layer (default: 21 for WavLM, 24 for HuBERT; ignored for MFCCs)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="training epochs (default: %(default)s)")
    parser.add_argument("--allow-same-speaker", action="store_true",
                        help="--pairs poorman only: use the pairs from training/align.py "
                             "--allow-same-speaker (written as the PoorManUTDAnySpk set)")
    args = parser.parse_args()
    if args.corpus == "zs17french":
        args.corpus = "french"
    if args.corpus == "french" and args.pairs == "utd":
        parser.error("there is no classic UTD word list for French; use --pairs dusted")
    if args.allow_same_speaker and args.pairs != "poorman":
        parser.error("--allow-same-speaker only applies to --pairs poorman")
    return args


class Config:
    """Every path one (features, corpus, pairs) run reads or writes."""

    def __init__(self, features: str, corpus: str, pairs: str, layer: int | None, any_speaker: bool):
        feat, corp = FEATURES[features], CORPORA[corpus]
        self.features, self.corpus, self.pairs = features, corpus, pairs
        self.dim, self.hidden_units, self.encoding_dim, self.lr = (
            feat["dim"], feat["hidden_units"], feat["encoding_dim"], feat["lr"])
        self.is_mfcc = feat["layer"] is None
        self.layer = None if self.is_mfcc else (layer or feat["layer"])
        self.layer_dir = "" if self.is_mfcc else f"layer_{self.layer:02d}"
        self.pair_order = "blocked" if self.is_mfcc and pairs != "poorman" else "interleaved"

        base = f"{corp['prefix']}_{feat['tag']}"
        variant = {"gold": "", "utd": "UTD", "dusted": "DUSTED", "poorman": "PoorManUTD"}[pairs]
        if any_speaker:
            variant += "AnySpk"
        feature_set = base + variant
        train, val = corp["train"], corp["val"]
        # The SSL UTD/DUSTED/Poor Man's alignment generators write "<split>.samediff.alignments",
        # everything else "<split>.samediff".
        align_name = "samediff.alignments" if (pairs == "poorman" or (pairs in ("utd", "dusted") and not self.is_mfcc)) else "samediff"

        self.train_alignments = self._alignments(feature_set, f"{train}.{align_name}")
        if pairs == "poorman":
            # Poor Man's alignments index straight into the gold features.
            self.train_features = [self._features(base, f"{train}.samediff")]
        else:
            self.train_features = [self._features(feature_set, f"{train}.samediff")]

        if pairs in ("dusted", "poorman"):
            self.val_alignments = self.val_features = None  # mined over the training split only
        else:
            self.val_alignments = self._alignments(feature_set, f"{val}.{align_name}")
            self.val_features = self._features(feature_set, f"{val}.samediff")

        ckpt_name = "_".join(p for p in ("cAE", variant, self.layer_dir) if p) + ".pt"
        self.checkpoint = Path("models", base, ckpt_name)
        self.encode = [(self._features(base, f"{s}.samediff"), self._features(feature_set, f"{s}.samediff_cAE"))
                       for s in corp["encode"]]
        self.run_name = f"{feature_set}_{corpus}" + (f"_{self.layer_dir}" if self.layer_dir else "")

    def _features(self, feature_set: str, split_dir: str) -> Path:
        """data/Features/<set>/[Normalised/]<split_dir>[/layer_NN]"""
        return Path("data/Features", feature_set, *(() if self.is_mfcc else ("Normalised",)), split_dir, self.layer_dir)

    def _alignments(self, feature_set: str, split_dir: str) -> Path:
        """data/Alignments/<set>/<split_dir>[/layer_NN]"""
        return Path("data/Alignments", feature_set, split_dir, self.layer_dir)

    def align_command(self) -> str:
        extra = " --allow-same-speaker" if "AnySpk" in str(self.train_alignments) else ""
        return (f"python training/align.py --features {self.features} --corpus {self.corpus} "
                f"--pairs {self.pairs}{extra}")


class ChunkedAlignmentDataset(Dataset):
    """Aligned (input, target) frame pairs, both directions.

    Features and pair indices are kept in a few numpy arrays rather than Python objects,
    so forked DataLoader workers don't copy them.
    """

    def __init__(self, alignment_dir: Path, features_dirs: list[Path], pair_order: str = "interleaved"):
        assert pair_order in ("interleaved", "blocked"), pair_order
        alignment_dir = Path(alignment_dir)

        metadata_fn = alignment_dir / "metadata.pkl"
        if metadata_fn.is_file():
            with open(metadata_fn, "rb") as f:
                n_chunks = pickle.load(f)["total_chunks"]
        else:
            n_chunks = len(list(alignment_dir.glob("alignments.chunk_*.pkl")))
        chunk_fns = [alignment_dir / f"alignments.chunk_{i:04d}.pkl" for i in range(n_chunks)]
        chunk_fns = [fn for fn in chunk_fns if fn.is_file()]

        # Pass 1: which tokens the alignments reference.
        needed: set[str] = set()
        for fn in tqdm(chunk_fns, desc="Scanning chunks"):
            with open(fn, "rb") as f:
                for alignments in pickle.load(f).values():
                    for a in alignments:
                        needed.update(a["pair"])

        # Load the referenced tokens into one array (a later directory wins on a clash).
        arrays: dict[str, np.ndarray] = {}
        for features_dir in features_dirs:
            for npz_file in tqdm(sorted(Path(features_dir).glob("*.npz")), desc=f"Loading {Path(features_dir).name}"):
                with np.load(npz_file, allow_pickle=False) as f:
                    for key in f.files:
                        if key in needed:
                            arrays[key] = f[key].astype(np.float32)
        offset, chunks, total = {}, [], 0
        for key, arr in arrays.items():
            offset[key] = total
            chunks.append(arr)
            total += arr.shape[0]
        self.feature_data = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), np.float32)
        del arrays, chunks

        # Pass 2: (token, frame) pairs -> rows of feature_data. Pairs whose tokens are
        # missing from the features are skipped.
        index_in, index_out = [], []
        for fn in tqdm(chunk_fns, desc="Indexing chunks"):
            with open(fn, "rb") as f:
                chunk = pickle.load(f)
            for label in sorted(chunk):
                for a in chunk[label]:
                    u1, u2 = a["pair"]
                    if u1 not in offset or u2 not in offset:
                        continue
                    fp = np.asarray(a["frame_pairs"], dtype=np.int64).reshape(-1, 2)
                    r1, r2 = fp[:, 0] + offset[u1], fp[:, 1] + offset[u2]
                    if pair_order == "interleaved":
                        index_in.append(np.stack([r1, r2], axis=1).ravel())
                        index_out.append(np.stack([r2, r1], axis=1).ravel())
                    else:
                        index_in.append(np.concatenate([r1, r2]))
                        index_out.append(np.concatenate([r2, r1]))
        self.index_in = np.concatenate(index_in) if index_in else np.empty(0, np.int64)
        self.index_out = np.concatenate(index_out) if index_out else np.empty(0, np.int64)
        print(f"Total frame pairs indexed: {len(self)}")

    def __len__(self) -> int:
        return self.index_in.shape[0]

    def __getitem__(self, idx: int):
        return (torch.from_numpy(self.feature_data[self.index_in[idx]]),
                torch.from_numpy(self.feature_data[self.index_out[idx]]))


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, encoding_dim: int,
                 num_layers: int = NUM_HIDDEN_LAYERS, activation_function: str = ACTIVATION):
        super().__init__()

        def act():
            return nn.ReLU() if activation_function == "ReLU" else nn.Tanh()

        layers = [nn.Linear(input_dim, hidden_units), act()]
        for layer in range(num_layers - 1):
            if layer == num_layers - 3:  # down to the bottleneck
                layers += [nn.Linear(hidden_units, encoding_dim), nn.Tanh()]
            elif layer == num_layers - 2:  # back up
                layers += [nn.Linear(encoding_dim, hidden_units), act()]
            else:
                layers += [nn.Linear(hidden_units, hidden_units), act()]
        layers.append(nn.Linear(hidden_units, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def init_weights(module: nn.Module) -> None:
    """Xavier/Glorot uniform for Tanh, Kaiming/He uniform for ReLU; zero biases."""
    if isinstance(module, nn.Linear):
        if ACTIVATION == "Tanh":
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


def train_cae(model, train_dataset, validate_dataset, lr: float, device, n_epochs: int = N_EPOCHS, seed: int = SEED):
    wandb.define_metric("cAE/epoch")
    wandb.define_metric("cAE/train_loss", step_metric="cAE/epoch")
    if validate_dataset is not None:
        wandb.define_metric("cAE/validation_loss", step_metric="cAE/epoch")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = (DataLoader(validate_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
                  if validate_dataset is not None else None)

    def mean_loss(loader, n):
        return sum(criterion(model(x.to(device)), y.to(device)).item() * x.size(0) for x, y in loader) / n

    model.eval()
    print("Calculating initial losses...")
    with torch.no_grad():
        log = {"cAE/train_loss": mean_loss(train_loader, len(train_dataset)), "cAE/epoch": 0}
        if val_loader is not None:
            log["cAE/validation_loss"] = mean_loss(val_loader, len(validate_dataset))
    print(f"Epoch 0/{n_epochs} (pre-training)  " + "  ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "cAE/epoch"))
    wandb.log(log)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"cAE Epoch {epoch}/{n_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        log = {"cAE/train_loss": train_loss / len(train_dataset), "cAE/epoch": epoch}

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                log["cAE/validation_loss"] = mean_loss(val_loader, len(validate_dataset))

        print(f"Epoch {epoch}/{n_epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "cAE/epoch"))
        wandb.log(log)

    return model


def encode_split(model, input_dir: Path, output_dir: Path, device, name: str) -> None:
    """Encode every token in input_dir's .npz files through the bottleneck (before its
    activation) and write them under the same file names in output_dir."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    npz_files = sorted(input_dir.glob("*.npz")) if input_dir.is_dir() else []
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc=f"Encoding {name}"):
            with np.load(npz_file, allow_pickle=False) as f:
                encoded = {key: model.network[:BOTTLENECK_SLICE](torch.from_numpy(f[key]).float().to(device))
                           .cpu().numpy().astype(np.float32) for key in sorted(f.files)}
            np.savez_compressed(output_dir / npz_file.name, **encoded)
    print(f"Saved encoded features to {output_dir} ({len(npz_files)} files)")


def main() -> None:
    args = parse_args()
    cfg = Config(args.features, args.corpus, args.pairs, args.layer, args.allow_same_speaker)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project="cAE Training", name=cfg.run_name, config={
        "features": args.features, "corpus": args.corpus, "pairs": args.pairs, "layer": cfg.layer,
        "n_epochs_cAE": args.epochs, "lr_cAE": cfg.lr, "batch_size_cAE": BATCH_SIZE, "optimizer_cAE": "Adam",
        "hidden_units": cfg.hidden_units, "encoding_dim": cfg.encoding_dim,
        "num_hidden_layers": NUM_HIDDEN_LAYERS, "activation_function": ACTIVATION, "device": str(device),
    })

    print("Stage 1: Building the autoencoder...")
    model = Autoencoder(cfg.dim, cfg.hidden_units, cfg.encoding_dim).to(device)
    model.apply(init_weights)
    print(model)

    print("\nStage 2: Training the correspondence autoencoder...")
    if not cfg.train_alignments.is_dir():
        raise FileNotFoundError(f"Training alignments not found: {cfg.train_alignments}\n"
                                f"Generate them with:\n  {cfg.align_command()}")
    print(f"Training on {cfg.train_alignments}")
    train_dataset = ChunkedAlignmentDataset(cfg.train_alignments, cfg.train_features, cfg.pair_order)

    validate_dataset = None
    if cfg.val_alignments is None:
        print("No validation split: these pairs are mined over the training split only.")
    elif cfg.val_alignments.is_dir():
        print(f"Validating on {cfg.val_alignments}")
        validate_dataset = ChunkedAlignmentDataset(cfg.val_alignments, [cfg.val_features], cfg.pair_order)
    else:
        print(f"Warning: {cfg.val_alignments} not found. Training without validation loss.")

    model = train_cae(model, train_dataset, validate_dataset, cfg.lr, device, args.epochs)
    cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint)
    print(f"Saved trained cAE to {cfg.checkpoint}")

    print("\nStage 3: Encoding the gold word features through the bottleneck...")
    encoder = Autoencoder(cfg.dim, cfg.hidden_units, cfg.encoding_dim).to(device)
    encoder.load_state_dict(torch.load(cfg.checkpoint, map_location=device, weights_only=True))
    for input_dir, output_dir in cfg.encode:
        encode_split(encoder, input_dir, output_dir, device, input_dir.parent.name if cfg.layer_dir else input_dir.name)

    wandb.finish()


if __name__ == "__main__":
    main()

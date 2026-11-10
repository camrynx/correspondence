"""
Train a cAE on LibriSpeech DUSTED pairs (train-clean-5 only, no validation), then encode
the gold train-clean-5 and dev-clean word features.

    python training/extras/train_cAE_WavLMLibriUTD.py --dusted
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

#------------------------------------PARAMETER CHANGING------------------------------------

import random

parser = argparse.ArgumentParser(description="Train the LibriSpeech WavLM cAE on DUSTED UTD word pairs.")
parser.add_argument(
    "--dusted", action="store_true",
    help="Required - there is no legacy UTD word list for LibriSpeech, only DUSTED.",
    )
args = parser.parse_args()

if not args.dusted:
    raise SystemExit(
        "Error: --dusted is required for LibriSpeech - there is no legacy "
        "UTD word list, only UTDWordsDUSTED.txt."
    )

_FEATURE_SET = "LibriSpeech_WavLMDUSTED"
_MODEL_SUFFIX = "DUSTED"

#Training parameters
n_epochs_cAE = 30
lr_cAE = 1e-5
batch_size_cAE = 2048
optimizer_cAE = "Adam"  # Options: "Adam", "SGD", "Adagrad"

#Model parameters
hidden_units = 2048
encoding_dim = 128
num_hidden_layers = 7
num_hidden_layers += 2
activation_function = "Tanh"

indexforSaving = 2*num_hidden_layers - 3

# Must match LAYER in extractFeaturesWavLMLibriUTD.py / normaliseLibriUTD.py
# / generate_alignments_WavLMLibriUTD.py.
LAYER = 21
_LAYER_SUBDIR = f"layer_{LAYER:02d}"

#Data paths
cAE_alignment_train_path = Path(f"data/Alignments/{_FEATURE_SET}/train-clean-5.samediff.alignments/{_LAYER_SUBDIR}")  # Pre-generated DUSTED pairs (chunked pickles)
cAE_features_train_path = Path(f"data/Features/{_FEATURE_SET}/Normalised/train-clean-5.samediff/{_LAYER_SUBDIR}")  # DUSTED features, for extracting encoded frame pairs

#Model save paths - checkpoint lives under the base set, like every
#other UTD/DUSTED/PoorManUTD variant in this repo.
cAE_save_path = Path(f"models/LibriSpeech_WavLM/cAE_{_MODEL_SUFFIX}_{_LAYER_SUBDIR}.pt")
cAE_save_path.parent.mkdir(parents=True, exist_ok=True)

#Stage 3 always encodes the BASE gold-word features, never a UTD-specific extraction.
filtered_features_train_input = Path(f"data/Features/LibriSpeech_WavLM/Normalised/train-clean-5.samediff/{_LAYER_SUBDIR}")
filtered_features_dev_input = Path(f"data/Features/LibriSpeech_WavLM/Normalised/dev-clean.samediff/{_LAYER_SUBDIR}")
filtered_features_train_cae_path = Path(f"data/Features/{_FEATURE_SET}/Normalised/train-clean-5.samediff_cAE/{_LAYER_SUBDIR}")
filtered_features_dev_cae_path = Path(f"data/Features/{_FEATURE_SET}/Normalised/dev-clean.samediff_cAE/{_LAYER_SUBDIR}")

#Other
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
features_dim = 1024

print(f"Using device: {device}")

#------------------------------------FUNCTIONS------------------------------------

class ChunkedAlignmentDataset(Dataset):
    def __init__(self, alignment_dir: Path, features_dirs):
        self.alignment_dir = Path(alignment_dir)
        if isinstance(features_dirs, (str, Path)):
            features_dirs = [features_dirs]
        self.features_dirs = [Path(d) for d in features_dirs]

        print("Pre-loading features...")
        self.features: dict[str, np.ndarray] = {}
        for features_dir in self.features_dirs:
            for npz_file in sorted(features_dir.glob("*.npz")):
                with np.load(npz_file, allow_pickle=False) as f:
                    for key in f.files:
                        self.features[key] = f[key].astype(np.float32)

        print("Indexing alignments...")
        self.index: list[tuple[str, str, int, int]] = []

        metadata_fn = self.alignment_dir / "metadata.pkl"
        if metadata_fn.is_file():
            with open(metadata_fn, "rb") as f:
                meta = pickle.load(f)
            n_chunks = meta["total_chunks"]
        else:
            n_chunks = len(sorted(self.alignment_dir.glob("alignments.chunk_*.pkl")))

        for chunk_idx in tqdm(range(n_chunks), desc="Indexing chunks"):
            chunk_fn = self.alignment_dir / f"alignments.chunk_{chunk_idx:04d}.pkl"
            if not chunk_fn.is_file():
                continue
            with open(chunk_fn, "rb") as f:
                chunk_data = pickle.load(f)
            for word_label in sorted(chunk_data):
                for alignment_info in chunk_data[word_label]:
                    u1, u2 = alignment_info["pair"]
                    if u1 not in self.features or u2 not in self.features:
                        continue
                    for fi1, fi2 in alignment_info["frame_pairs"]:
                        self.index.append((u1, u2, int(fi1), int(fi2)))
                        self.index.append((u2, u1, int(fi2), int(fi1)))

        print(f"Total frame pairs indexed: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        u_in, u_tgt, fi_in, fi_tgt = self.index[idx]
        x = torch.from_numpy(self.features[u_in][fi_in])
        y = torch.from_numpy(self.features[u_tgt][fi_tgt])
        return x, y


def train_cae(model, train_dataset, validate_dataset=None, seed=11):
    wandb.define_metric("cAE/epoch")
    wandb.define_metric("cAE/train_loss", step_metric="cAE/epoch")
    if validate_dataset is not None:
        wandb.define_metric("cAE/validation_loss", step_metric="cAE/epoch")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_cAE)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_cAE, shuffle=True,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        validate_dataset, batch_size=batch_size_cAE, num_workers=4, pin_memory=True,
    ) if validate_dataset is not None else None

    model.eval()
    print("Calculating initial losses...")
    with torch.no_grad():
        init_train_loss = sum(
            criterion(model(x.to(device)), y.to(device)).item() * x.size(0)
            for x, y in train_loader
        ) / len(train_dataset)
        log_dict = {"cAE/train_loss": init_train_loss, "cAE/epoch": 0}
        msg = f"Epoch 0/{n_epochs_cAE} (pre-training)  train={init_train_loss:.4f}"
    print(msg)
    wandb.log(log_dict)

    for epoch in range(1, n_epochs_cAE + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"cAE Epoch {epoch}/{n_epochs_cAE}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        avg_train = train_loss / len(train_dataset)

        log_dict = {"cAE/train_loss": avg_train, "cAE/epoch": epoch}
        msg = f"Epoch {epoch}/{n_epochs_cAE}  train={avg_train:.4f}"

        print(msg)
        wandb.log(log_dict)

    return model


def check_alignments_exist() -> bool:
    train_exists = cAE_alignment_train_path.is_dir()
    print(f"Checking for pre-generated alignments...")
    print(f"  train-clean-5 DUSTED alignments (for training): {'found' if train_exists else 'NOT FOUND'}")

    if not train_exists:
        raise FileNotFoundError(
            f"train-clean-5 alignment directory not found: {cAE_alignment_train_path}\n"
            f"Please run:\n"
            f"  python training/extras/generate_alignments_WavLMLibriUTD.py --n-cpus 8 --dusted"
        )
    return train_exists


def save_filtered_features_from_npz(model, input_dir, output_dir, model_name="AE"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    print(f"Loading filtered features from {input_dir}... ({len(npz_files)} files)")

    model.eval()

    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc=f"Extracting {model_name} embeddings"):
            with np.load(npz_file, allow_pickle=False) as f:
                embeddings_dict = {}
                for utt_id in sorted(f.files):
                    features = torch.from_numpy(f[utt_id]).float().to(device)
                    embeddings = model.network[:indexforSaving](features)
                    embeddings_dict[utt_id] = embeddings.cpu().numpy().astype(np.float32)

                output_npz_fn = output_dir / npz_file.name
                np.savez_compressed(output_npz_fn, **embeddings_dict)

    print(f"Saved encoded embeddings to {output_dir} ({len(npz_files)} files)")

# ----------AUTOENCODER----------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=features_dim, hidden_units=hidden_units, encoding_dim=encoding_dim, num_layers=num_hidden_layers, activation_function=activation_function):
        super(Autoencoder, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        if activation_function == "ReLU":
            layers.append(nn.ReLU())
        elif activation_function == "Tanh":
            layers.append(nn.Tanh())

        for layer in range(num_layers - 1):
            if layer == num_layers-3:
                layers.append(nn.Linear(hidden_units, encoding_dim))
                layers.append(nn.Tanh())
            elif layer == num_layers-2:
                layers.append(nn.Linear(encoding_dim, hidden_units))
                if activation_function == "ReLU":
                    layers.append(nn.ReLU())
                elif activation_function == "Tanh":
                    layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(hidden_units, hidden_units))
                if activation_function == "ReLU":
                    layers.append(nn.ReLU())
                elif activation_function == "Tanh":
                    layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_units, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def init_weights_classic(module):
    if isinstance(module, nn.Linear):
        if activation_function == "Tanh":
            nn.init.xavier_uniform_(module.weight)
        elif activation_function == "ReLU":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


#-----------------------------------RUNNING THE PIPELINE-----------------------------------


def main():
    seed = 11
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(
        project="cAE Training",
        name=f"WavLMLibri-{_MODEL_SUFFIX}_{_LAYER_SUBDIR}",
        config={
            "n_epochs_cAE": n_epochs_cAE, "lr_cAE": lr_cAE, "batch_size_cAE": batch_size_cAE,
            "optimizer_cAE": optimizer_cAE, "hidden_units": hidden_units, "encoding_dim": encoding_dim,
            "num_hidden_layers": num_hidden_layers, "activation_function": activation_function,
            "device": str(device),
        }
    )

    print("Stage 1: Building the autoencoder...")
    cAE = Autoencoder().to(device)
    cAE.apply(init_weights_classic)
    print("Initialized autoencoder with the following architecture:")
    print(cAE)

    print("\nStage 2: Training the correspondence autoencoder...")
    check_alignments_exist()

    print("\nBuilding train-clean-5 DUSTED alignment dataset (training)...")
    train_dataset = ChunkedAlignmentDataset(cAE_alignment_train_path, cAE_features_train_path)
    print(f"Training data for cAE: {len(train_dataset)} frame pairs")
    print("\nNo validation split: DUSTED was only mined over train-clean-5.")

    cAE = train_cae(cAE, train_dataset, None, seed=seed)

    torch.save(cAE.state_dict(), cAE_save_path)
    print(f"Saved trained cAE to {cAE_save_path}")

    print("\nStage 3: Extracting and saving bottleneck layer embeddings (before activation)...")

    if not os.path.exists(cAE_save_path):
        print(f"Warning: cAE model not found at {cAE_save_path}")
    else:
        cAE_for_features = Autoencoder().to(device)
        cAE_for_features.load_state_dict(torch.load(cAE_save_path, map_location=device, weights_only=True))
        cAE_for_features.eval()

        print("\nExtracting cAE-encoded embeddings from the bottleneck layer (before activation)...")
        save_filtered_features_from_npz(cAE_for_features, filtered_features_train_input, filtered_features_train_cae_path, "cAE (train-clean-5)")
        save_filtered_features_from_npz(cAE_for_features, filtered_features_dev_input, filtered_features_dev_cae_path, "cAE (dev-clean)")

    print("\nStage 3 complete. Bottleneck layer embeddings saved successfully!")

    wandb.finish()

if __name__ == "__main__":
    main()

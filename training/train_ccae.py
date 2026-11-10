"""
Train a contrastive cAE (ccAE): the cAE's network and frame pairs, with an InfoNCE loss.

Both frames of an aligned pair go through the same encoder, and each bottleneck
embedding is trained to be closer to its partner than to the other frames in the batch
(excluding frames with the same label). The loss follows contrastive_loss in
christiaanjacobs/globalphone_awe_pytorch (Jacobs, Matusevych and Kamper, SLT 2021), made
symmetric and without its off-by-one; see info_nce(). Only the encoder is trained; the
decoder is kept so the checkpoint loads like a cAE checkpoint.

    python training/train_ccae.py --features wavlm  --corpus buckeye  --pairs dusted
    python training/train_ccae.py --features mfcc   --corpus xitsonga --pairs utd
    python training/train_ccae.py --features hubert --corpus french   --pairs poorman

The encoder starts from a random initialisation (--init random, the default and what
the paper reports) or from the trained cAE (--init cae). After training, every split's
gold words are encoded for evaluation, as in train_cae.py.
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

#------------------------------------PARAMETER CHANGING------------------------------------

import random

parser = argparse.ArgumentParser(
    description="Train a contrastive correspondence autoencoder (ccAE) on WavLM frame pairs.",
    )
parser.add_argument(
    "--corpus", choices=["buckeye", "xitsonga", "zs17french", "french"], default="buckeye",
    help="buckeye (train devpart1, validate devpart2), xitsonga (dev / test) or french "
         "(train / test); zs17french is an alias for french",
    )
parser.add_argument(
    "--pairs", choices=["gold", "dusted", "utd", "poorman", "phone"], required=True,
    help="gold, dusted, utd (not French) or poorman. 'phone' (phone pairs, negatives "
         "masked by phone label) is an oracle diagnostic, not a zero-resource setting.",
    )
parser.add_argument(
    "--init", choices=["cae", "random"], default="random",
    help="random (default) or start from the trained cAE (cae); random runs are "
         "tagged 'randinit'",
    )
parser.add_argument(
    "--tau", type=float, default=0.1,
    help="InfoNCE temperature (default: %(default)s)",
    )
parser.add_argument(
    "--tag", default="",
    help="alphanumeric tag added to the model name so sweep runs don't overwrite each "
         "other, e.g. --tag Tau005 (pass the same to extract_abx_phone_features.py --model-prefix)",
    )
parser.add_argument(
    "--batch-size", type=int, default=2048,
    help="frame pairs per batch; sets the number of negatives (default: %(default)s)",
    )
parser.add_argument(
    "--features", choices=["wavlm", "hubert", "mfcc"], default="wavlm",
    help="input features (default: %(default)s); each uses its cAE's architecture, "
         "learning rate and layer",
    )
parser.add_argument(
    "--lr", type=float, default=None,
    help="Adam learning rate (default: the cAE's, 1e-5 for WavLM/HuBERT, 1e-3 for MFCCs)",
    )
parser.add_argument(
    "--epochs", type=int, default=30,
    help="training epochs (default: %(default)s)",
    )
parser.add_argument(
    "--layer", type=int, default=None,
    help="layer (default: 21 for WavLM, 24 for HuBERT; ignored for MFCCs)",
    )
parser.add_argument(
    "--seed", type=int, default=11,
    help="random seed (default: %(default)s)",
    )
parser.add_argument(
    "--save-every", type=int, default=0,
    help="also save a checkpoint every N epochs; 0 = off (default: %(default)s)",
    )
parser.add_argument(
    "--diag-frames", type=int, default=8192,
    help="validation embeddings used for the collapse diagnostics (default: %(default)s)",
    )
args = parser.parse_args()
if args.corpus == "french":  # alias
    args.corpus = "zs17french"

# Model parameters, the same as train_cae.py (--init cae loads a cAE checkpoint into
# this network). 9 hidden layers, bottleneck Linear at network index 14.
_FEATURES = {
    "wavlm": dict(tag="WavLM", features_dim=1024, hidden_units=2048, encoding_dim=128, lr=1e-5, layer=21),
    "hubert": dict(tag="HuBERT", features_dim=1024, hidden_units=2048, encoding_dim=128, lr=1e-5, layer=24),
    "mfcc": dict(tag="MFCCs", features_dim=39, hidden_units=250, encoding_dim=100, lr=1e-3, layer=None),
}[args.features]
_IS_MFCC = args.features == "mfcc"

#Training parameters
n_epochs_ccAE = args.epochs
lr_ccAE = args.lr if args.lr is not None else _FEATURES["lr"]
batch_size_ccAE = args.batch_size
optimizer_ccAE = "Adam"  # Options: "Adam", "SGD", "Adagrad"
tau_ccAE = args.tau

hidden_units = _FEATURES["hidden_units"]
encoding_dim = _FEATURES["encoding_dim"]
num_hidden_layers = 7
num_hidden_layers += 2 # Add 2 more layers for the bottleneck and the decoder layer after the bottleneck
activation_function = "Tanh"  # Options: "ReLU", "Tanh"

# network[:BOTTLENECK_SLICE] is the encoder up to the bottleneck Linear, before its Tanh:
# the space the loss is applied to and the one that gets evaluated.
BOTTLENECK_SLICE = 2 * num_hidden_layers - 3  # == 15

# Xitsonga and French have no third split; test is both validation and test.
_CORPUS = {
    "buckeye": dict(
        base_feature_set="Buckeye_WavLM", models_dir="models/Buckeye_WavLM",
        train_split="devpart1", val_split="devpart2",
        encode_splits=("devpart1", "devpart2", "zs"),
    ),
    "xitsonga": dict(
        base_feature_set="Xitsonga_WavLM", models_dir="models/Xitsonga_WavLM",
        train_split="dev", val_split="test",
        encode_splits=("dev", "test"),
    ),
    "zs17french": dict(
        base_feature_set="ZS17French_WavLM", models_dir="models/ZS17French_WavLM",
        train_split="train", val_split="test",
        encode_splits=("train", "test"),
    ),
}[args.corpus]

# One entry per (corpus, pairs).
#   alignment_dirname: gold alignments live in <split>.samediff, UTD/DUSTED/Poor Man's in
#     <split>.samediff.alignments.
#   no_val: DUSTED terms were mined over the training split only, so there is nothing to
#     validate on.
#   features_from_base / val_from_base: Poor Man's alignments index into the gold
#     features, and validation uses the gold validation alignments.
#   mask_by: which negatives info_nce() masks. Word pairs mask frames with the same label.
#     Poor Man's entries have no label, so only exact copies of the anchor frame (common,
#     since one frame is often the nearest neighbour of many) are masked.
# "generator" and "trainer" are only used in error messages.
_PAIRINGS = {
    ("buckeye", "gold"): dict(
        suffix="", variant="", alignment_dirname="samediff",
    ),
    ("buckeye", "dusted"): dict(
        suffix="DUSTED", variant="DUSTED", alignment_dirname="samediff.alignments",
        no_val=True,  # DUSTED mined over the train split only, as for zs17french
    ),
    # Classic UTD terms span every split, so they have validation alignments.
    # Labels are cluster IDs ("PT100"). There is no French UTD list.
    ("buckeye", "utd"): dict(
        suffix="UTD", variant="UTD", alignment_dirname="samediff.alignments",
    ),
    ("buckeye", "poorman"): dict(
        suffix="PoorManUTD", variant="PoorManUTD", alignment_dirname="samediff.alignments",
        features_from_base=True, val_from_base=True, mask_by="frame",
    ),
    ("buckeye", "phone"): dict(
        suffix="Phone", variant="Phone", alignment_dirname="samediff",
        generator="python training/extras/generate_alignments_WavLMPhone.py",
        trainer="python training/extras/train_cAE_WavLMPhone.py",
    ),
    ("xitsonga", "gold"): dict(
        suffix="", variant="", alignment_dirname="samediff",
    ),
    ("xitsonga", "dusted"): dict(
        suffix="DUSTED", variant="DUSTED", alignment_dirname="samediff.alignments",
        no_val=True,  # DUSTED mined over the train split only, as for zs17french
    ),
    ("xitsonga", "utd"): dict(  # classic UTD, see ("buckeye", "utd")
        suffix="UTD", variant="UTD", alignment_dirname="samediff.alignments",
    ),
    ("xitsonga", "poorman"): dict(
        suffix="PoorManUTD", variant="PoorManUTD", alignment_dirname="samediff.alignments",
        features_from_base=True, val_from_base=True, mask_by="frame",
    ),
    ("zs17french", "gold"): dict(
        suffix="", variant="", alignment_dirname="samediff",
    ),
    # The French DUSTED terms come from the training split only: no validation.
    ("zs17french", "dusted"): dict(
        suffix="DUSTED", variant="DUSTED", alignment_dirname="samediff.alignments",
        no_val=True,
    ),
    ("zs17french", "poorman"): dict(
        suffix="PoorManUTD", variant="PoorManUTD", alignment_dirname="samediff.alignments",
        features_from_base=True, val_from_base=True, mask_by="frame",
    ),
}
if (args.corpus, args.pairs) not in _PAIRINGS:
    parser.error(
        f"--pairs {args.pairs} is not available for --corpus {args.corpus} "
        f"(phone pairs are built from Buckeye's english.phn only; there is no French UTD list)"
    )
_PAIRING = _PAIRINGS[(args.corpus, args.pairs)]
if args.features != "wavlm" and args.pairs == "phone":
    parser.error("--pairs phone is WavLM-only (there is no MFCC/HuBERT phone-pair generator)")
if args.pairs != "phone":
    # Commands that produce this run's prerequisites (only used in error messages).
    _corpus_arg = "french" if args.corpus == "zs17french" else args.corpus
    _PAIRING = {**_PAIRING,
                "generator": f"python training/align.py --features {args.features} --corpus {_corpus_arg} --pairs {args.pairs}",
                "trainer": f"python training/train_cae.py --features {args.features} --corpus {_corpus_arg} --pairs {args.pairs}"}

# _CORPUS names the WavLM sets; the others are the same with the feature tag swapped.
_BASE_FEATURE_SET = _CORPUS["base_feature_set"].replace("WavLM", _FEATURES["tag"])
_MODELS_DIR = _CORPUS["models_dir"].replace("WavLM", _FEATURES["tag"])
_TRAIN_SPLIT = _CORPUS["train_split"]
_VAL_SPLIT = _CORPUS["val_split"]
_ENCODE_SPLITS = _CORPUS["encode_splits"]

_FEATURE_SET = _BASE_FEATURE_SET + _PAIRING["suffix"]
_VARIANT_SUFFIX = _PAIRING["variant"]
_ALIGNMENT_DIRNAME = _PAIRING["alignment_dirname"]
if _IS_MFCC and args.pairs in ("utd", "dusted"):
    # align.py write "<subset>.samediff", unlike the WavLM ones.
    _ALIGNMENT_DIRNAME = "samediff"
_MASK_BY = _PAIRING.get("mask_by", "label")

# The init mode is part of the model prefix, so random and cAE-initialised runs never
# overwrite each other.
if not args.tag.isalnum() and args.tag:
    parser.error(
        f"--tag must be alphanumeric with no underscores (got {args.tag!r}); the evaluation "
        f"scripts recover the model variant by splitting checkpoint names on '_'."
    )
_MODEL_PREFIX = ("ccAE" if args.init == "cae" else "ccAErandinit") + args.tag

# Layer to train on; the checkpoint and outputs are tagged with it. MFCCs have no
# layers, so their paths and checkpoints have no layer_NN part.
LAYER = None if _IS_MFCC else (args.layer if args.layer is not None else _FEATURES["layer"])
_LAYER_SUBDIR = "" if _IS_MFCC else f"layer_{LAYER:02d}"


def _features_dir(feature_set: str, split_dir: str) -> Path:
    """data/Features/<set>/[Normalised/]<split_dir>[/layer_NN]"""
    return Path("data/Features", feature_set, *(() if _IS_MFCC else ("Normalised",)), split_dir, _LAYER_SUBDIR)


def _alignments_dir(feature_set: str, split_dir: str) -> Path:
    """data/Alignments/<set>/<split_dir>[/layer_NN]"""
    return Path("data/Alignments", feature_set, split_dir, _LAYER_SUBDIR)


#Data paths
ccAE_alignment_train_path = _alignments_dir(_FEATURE_SET, f"{_TRAIN_SPLIT}.{_ALIGNMENT_DIRNAME}")  # Pre-generated same-label alignments (chunked pickles)
_TRAIN_FEATURE_SET = _BASE_FEATURE_SET if _PAIRING.get("features_from_base") else _FEATURE_SET
ccAE_features_train_path = _features_dir(_TRAIN_FEATURE_SET, f"{_TRAIN_SPLIT}.samediff")  # Token features the frame pairs index into
if _PAIRING.get("val_from_base"):
    # See the note on _PAIRINGS: validate on the base gold split's alignments + features.
    ccAE_alignment_val_path = _alignments_dir(_BASE_FEATURE_SET, f"{_VAL_SPLIT}.samediff")
    ccAE_features_val_path = _features_dir(_BASE_FEATURE_SET, f"{_VAL_SPLIT}.samediff")
else:
    ccAE_alignment_val_path = _alignments_dir(_FEATURE_SET, f"{_VAL_SPLIT}.{_ALIGNMENT_DIRNAME}")  # Pre-generated same-label alignments for validation
    ccAE_features_val_path = _features_dir(_FEATURE_SET, f"{_VAL_SPLIT}.samediff")  # Token features the frame pairs index into

#Model load path (the trained cAE this ccAE is initialised from; unused with --init random)
_CAE_CKPT_NAME = "_".join(p for p in ("cAE", _VARIANT_SUFFIX, _LAYER_SUBDIR) if p) + ".pt"
cAE_load_path = Path(f"{_MODELS_DIR}/{_CAE_CKPT_NAME}")

#Model save path
_CCAE_CKPT_NAME = "_".join(p for p in (_MODEL_PREFIX, _VARIANT_SUFFIX, _LAYER_SUBDIR) if p) + ".pt"
ccAE_save_path = Path(f"{_MODELS_DIR}/{_CCAE_CKPT_NAME}")
ccAE_save_path.parent.mkdir(parents=True, exist_ok=True)

# Encoded outputs: always the gold features, whatever the training pairs, so every
# condition is scored on the same input.
STAGE3_SPLITS = [
    (
        split,
        _features_dir(_BASE_FEATURE_SET, f"{split}.samediff"),
        _features_dir(_FEATURE_SET, f"{split}.samediff_{_MODEL_PREFIX}"),
    )
    for split in _ENCODE_SPLITS
]

#Other
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
features_dim = _FEATURES["features_dim"]

print(f"Using device: {device}")

#------------------------------------FUNCTIONS------------------------------------

class ContrastiveAlignmentDataset(Dataset):
    """Aligned frame pairs, loaded lazily.

    Differs from train_cae.py's dataset in two ways: each item also carries a group id
    per side (its label, or for mask_by="frame" the frame's own identity) for masking
    negatives, and pairs are not duplicated in reverse, since the loss is symmetric.
    """
    def __init__(self, alignment_dir: Path, features_dir: Path, mask_by: str = "label"):
        assert mask_by in ("label", "frame"), mask_by
        self.alignment_dir = Path(alignment_dir)
        self.features_dir = Path(features_dir)
        self.mask_by = mask_by

        # Load all features once; each frame is stored once, not once per pair.
        print("Pre-loading features...")
        self.features: dict[str, np.ndarray] = {}
        for npz_file in sorted(self.features_dir.glob("*.npz")):
            with np.load(npz_file, allow_pickle=False) as f:
                for key in f.files:
                    self.features[key] = f[key].astype(np.float32)

        # Build a flat index: list of (utt_id_1, utt_id_2, frame_idx_1, frame_idx_2, group_x, group_y)
        # We store only indices/ids - no frame data is duplicated here.
        print("Indexing alignments...")
        self.index: list[tuple[str, str, int, int, int, int]] = []
        self.label_names: list[str] = []
        label_to_id: dict[str, int] = {}
        frame_to_id: dict[tuple[str, int], int] = {}

        def frame_id(u: str, fi: int) -> int:
            return frame_to_id.setdefault((u, fi), len(frame_to_id))

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
                label_id = label_to_id.get(word_label)
                if label_id is None:
                    label_id = len(self.label_names)
                    label_to_id[word_label] = label_id
                    self.label_names.append(word_label)
                for alignment_info in chunk_data[word_label]:
                    u1, u2 = alignment_info["pair"]
                    if u1 not in self.features or u2 not in self.features:
                        continue
                    for fi1, fi2 in alignment_info["frame_pairs"]:
                        fi1, fi2 = int(fi1), int(fi2)
                        if self.mask_by == "label":
                            groups = (label_id, label_id)
                        else:
                            groups = (frame_id(u1, fi1), frame_id(u2, fi2))
                        self.index.append((u1, u2, fi1, fi2, *groups))

        print(f"Total frame pairs indexed: {len(self.index)}")
        if self.mask_by == "label":
            print(f"Distinct word labels: {len(self.label_names)}")
        else:
            print(f"Distinct frames: {len(frame_to_id)} (negatives masked by frame identity, not word label)")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        u_in, u_tgt, fi_in, fi_tgt, group_x, group_y = self.index[idx]
        x = torch.from_numpy(self.features[u_in][fi_in])
        y = torch.from_numpy(self.features[u_tgt][fi_tgt])
        return x, y, group_x, group_y


def info_nce(z_x, z_y, group_x, group_y, tau, collect_stats=False):
    """Symmetric frame-level InfoNCE over a batch of aligned pairs.

    Args:
        z_x, z_y: [B, encoding_dim] bottleneck outputs for the two sides of each pair.
        group_x, group_y: [B] group ids (label, or frame identity for Poor Man's pairs).
        tau: temperature.
        collect_stats: also return positive/negative similarity statistics.

    Every one of the 2B embeddings is an anchor; its partner is the positive and every
    other embedding outside its group is a negative. Cosine similarity, temperature
    scaling, positive kept in the denominator, as in the reference implementation, with
    two changes: the reference skips the last pair of each batch, and only uses one
    direction. Masking by group is needed because, unlike the reference's word-level
    sampler, a batch of frames contains many frames of the same word.
    """
    B = z_x.size(0)
    z = F.normalize(torch.cat([z_x, z_y], dim=0), dim=1)   # [2B, D]
    sim = z @ z.t()                                        # [2B, 2B] cosine

    idx = torch.arange(2 * B, device=z.device)
    pos = torch.cat([idx[B:], idx[:B]])                    # x_i <-> y_i

    g = torch.cat([group_x, group_y], dim=0)
    block = g[:, None] == g[None, :]                       # same group (word label / frame)
    block[idx, idx] = True                                 # ...and self
    block[idx, pos] = False                                # ...but never the positive

    logits = (sim / tau).masked_fill(block, float("-inf"))
    loss = F.cross_entropy(logits, pos)

    stats = {}
    if collect_stats:
        with torch.no_grad():
            negatives = ~block
            negatives[idx, pos] = False                    # candidates minus the positive
            n_negatives = negatives.sum()
            stats = {
                "mean_pos_sim": sim[idx, pos].mean().item(),
                "mean_neg_sim": ((sim * negatives).sum() / n_negatives.clamp(min=1)).item(),
                "mean_effective_negatives": (n_negatives.float() / (2 * B)).item(),
            }

    return loss, stats


def embedding_diagnostics(model, loader, max_frames):
    """Collapse diagnostics: mean and variance of the bottleneck activations, effective
    rank of the singular value spectrum, and the top singular values. An effective rank
    falling towards 1 means the embeddings are collapsing.
    """
    model.eval()
    chunks, n = [], 0
    with torch.no_grad():
        for x, _y, _gx, _gy in loader:
            z = model.network[:BOTTLENECK_SLICE](x.to(device))
            chunks.append(z.float().cpu())
            n += z.size(0)
            if n >= max_frames:
                break

    Z = torch.cat(chunks, dim=0)[:max_frames]
    s = torch.linalg.svdvals(Z - Z.mean(dim=0, keepdim=True))
    p = s / s.sum().clamp(min=1e-12)
    effective_rank = torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())

    diagnostics = {
        "embed_mean": Z.mean().item(),
        "embed_var": Z.var().item(),
        "effective_rank": effective_rank.item(),
    }
    for i, value in enumerate(s[:10].tolist(), start=1):
        diagnostics[f"singular_value_{i:02d}"] = value
    return diagnostics


def run_epoch(model, loader, optimizer=None):
    """One pass over `loader`; trains if `optimizer` is given. Returns the mean loss and
    similarity statistics."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_items = 0.0, 0
    stat_totals: dict[str, float] = {}
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for x, y, group_x, group_y in loader:
            x, y = x.to(device), y.to(device)
            group_x, group_y = group_x.to(device), group_y.to(device)

            # Both frames go through the same encoder.
            z_x = model.network[:BOTTLENECK_SLICE](x)
            z_y = model.network[:BOTTLENECK_SLICE](y)

            loss, stats = info_nce(z_x, z_y, group_x, group_y, tau_ccAE, collect_stats=True)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
            n_batches += 1
            for key, value in stats.items():
                stat_totals[key] = stat_totals.get(key, 0.0) + value

    mean_stats = {key: value / max(n_batches, 1) for key, value in stat_totals.items()}
    return total_loss / max(total_items, 1), mean_stats


def train_ccae(model, train_dataset, validate_dataset=None, seed=11):
    wandb.define_metric("ccAE/epoch")
    for metric in (
        "ccAE/train_loss", "ccAE/validation_loss",
        # similarity diagnostics, per split
        "ccAE/train_mean_pos_sim", "ccAE/train_mean_neg_sim", "ccAE/train_mean_effective_negatives",
        "ccAE/val_mean_pos_sim", "ccAE/val_mean_neg_sim", "ccAE/val_mean_effective_negatives",
        # collapse diagnostics, always measured on the validation embeddings
        "ccAE/embed_mean", "ccAE/embed_var", "ccAE/effective_rank",
    ):
        wandb.define_metric(metric, step_metric="ccAE/epoch")

    # Only the encoder is optimised. The decoder stays in the checkpoint because the
    # loaders infer the bottleneck position from the number of Linear layers.
    optimizer = optim.Adam(model.network[:BOTTLENECK_SLICE].parameters(), lr=lr_ccAE)

    # To ensure reproducibility with DataLoader shuffling and multiple workers, we set a worker_init_fn
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_ccAE,
        shuffle=True,
        num_workers=4,       # parallel prefetch - tune to your CPU count
        pin_memory=True,     # faster CPU->GPU transfer
        worker_init_fn=seed_worker,
        generator=g
    )

    # Shuffle validation too: the index is in label order, so unshuffled batches would be
    # nearly all one label. g_val is reseeded each pass so every epoch sees the same batches.
    g_val = torch.Generator()
    val_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size_ccAE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_val,
    ) if validate_dataset is not None else None

    def validate():
        """Validation loss and similarity statistics (prefixed val_) plus the collapse
        diagnostics."""
        g_val.manual_seed(seed)
        loss, stats = run_epoch(model, val_loader)
        logs = {f"val_{k}": v for k, v in stats.items()}
        g_val.manual_seed(seed)
        logs.update(embedding_diagnostics(model, val_loader, args.diag_frames))
        return loss, logs

    print("Calculating initial losses...")
    init_train_loss, init_train_stats = run_epoch(model, train_loader)
    log_dict = {"ccAE/train_loss": init_train_loss, "ccAE/epoch": 0}
    log_dict.update({f"ccAE/train_{k}": v for k, v in init_train_stats.items()})
    msg = f"Epoch 0/{n_epochs_ccAE} (pre-training)  train={init_train_loss:.4f}"
    if val_loader is not None:
        init_val_loss, init_val_stats = validate()
        log_dict["ccAE/validation_loss"] = init_val_loss
        log_dict.update({f"ccAE/{k}": v for k, v in init_val_stats.items()})
        msg += f"  val={init_val_loss:.4f}  eff_rank={init_val_stats['effective_rank']:.1f}"
    print(msg)
    wandb.log(log_dict)

    for epoch in range(1, n_epochs_ccAE + 1):
        avg_train, train_stats = run_epoch(model, train_loader, optimizer=optimizer)

        log_dict = {"ccAE/train_loss": avg_train, "ccAE/epoch": epoch}
        log_dict.update({f"ccAE/train_{k}": v for k, v in train_stats.items()})
        msg = (
            f"Epoch {epoch}/{n_epochs_ccAE}  train={avg_train:.4f}"
            f"  pos_sim={train_stats['mean_pos_sim']:.3f}"
            f"  neg_sim={train_stats['mean_neg_sim']:.3f}"
            f"  negs/anchor={train_stats['mean_effective_negatives']:.0f}"
        )

        if val_loader is not None:
            avg_val, val_stats = validate()
            log_dict["ccAE/validation_loss"] = avg_val
            log_dict.update({f"ccAE/{k}": v for k, v in val_stats.items()})
            msg += (
                f"  val={avg_val:.4f}"
                f"  eff_rank={val_stats['effective_rank']:.1f}"
                f"  embed_var={val_stats['embed_var']:.3f}"
            )

        print(msg)
        wandb.log(log_dict)

        if args.save_every and epoch % args.save_every == 0 and epoch != n_epochs_ccAE:
            epoch_path = ccAE_save_path.with_name(f"{ccAE_save_path.stem}_epoch{epoch:02d}.pt")
            torch.save(model.state_dict(), epoch_path)
            print(f"  saved intermediate checkpoint to {epoch_path}")

    return model


def check_alignments_exist() -> bool:
    """Check the training alignment directory exists (and whether validation's does)."""
    train_exists = ccAE_alignment_train_path.is_dir()
    val_exists = not _PAIRING.get("no_val") and ccAE_alignment_val_path.is_dir()

    print(f"Checking for pre-generated alignments...")
    print(f"  {_TRAIN_SPLIT} alignments (for training): {'found' if train_exists else 'NOT FOUND'}")
    if _PAIRING.get("no_val"):
        print(f"  No validation split: DUSTED was only mined over {_TRAIN_SPLIT}.")
    else:
        print(f"  {_VAL_SPLIT} alignments (for validation): {'found' if val_exists else 'NOT FOUND'}")

    if not train_exists:
        raise FileNotFoundError(
            f"{_TRAIN_SPLIT} alignment directory not found: {ccAE_alignment_train_path}\n"
            f"Please run (to match --corpus {args.corpus} --pairs {args.pairs}):\n"
            f"  {_PAIRING['generator']}"
        )

    return train_exists, val_exists


def save_filtered_features_from_npz(model, input_dir, output_dir, model_name="ccAE"):
    """Encode every token in input_dir through the bottleneck (before its activation)
    and save the results under the same file names in output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all NPZ files in input directory
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    print(f"Loading filtered features from {input_dir}... ({len(npz_files)} files)")

    # Extract embeddings for all utterances
    model.eval()

    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc=f"Extracting {model_name} embeddings"):
            with np.load(npz_file, allow_pickle=False) as f:
                # Load each utterance from this file
                embeddings_dict = {}
                for utt_id in sorted(f.files):
                    # Load features
                    features = torch.from_numpy(f[utt_id]).float().to(device)

                    # network[:15]: up to the bottleneck Linear, before its Tanh.
                    embeddings = model.network[:BOTTLENECK_SLICE](features)

                    # Save to dict
                    embeddings_dict[utt_id] = embeddings.cpu().numpy().astype(np.float32)

                # Save this file's embeddings with the same name
                output_npz_fn = output_dir / npz_file.name
                np.savez_compressed(output_npz_fn, **embeddings_dict)

    print(f"Saved encoded embeddings to {output_dir} ({len(npz_files)} files)")


# ----------AUTOENCODER----------
# Same as train_cae.py's Autoencoder, so cAE checkpoints load into it.
class Autoencoder(nn.Module):
    def __init__(self, input_dim=features_dim, hidden_units=hidden_units, encoding_dim=encoding_dim, num_layers=num_hidden_layers, activation_function=activation_function):
        super(Autoencoder, self).__init__()

        layers = []

        #Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        if activation_function == "ReLU":
            layers.append(nn.ReLU())
        elif activation_function == "Tanh":
            layers.append(nn.Tanh())

        #Hidden layers
        for layer in range(num_layers - 1):
            if layer == 6: #the 6th layer goes down in dimensions
                layers.append(nn.Linear(hidden_units, encoding_dim))
                if activation_function == "ReLU":
                    layers.append(nn.ReLU())
                elif activation_function == "Tanh":
                    layers.append(nn.Tanh())
            elif layer == 7: #we need to go back up in dimensions
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

        #Output layer
        layers.append(nn.Linear(hidden_units, input_dim))

        #set up the network
        self.network  = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def init_weights_classic(module):
    """Xavier/Glorot for Tanh, Kaiming/He for ReLU; zero biases."""
    if isinstance(module, nn.Linear):
        if activation_function == "Tanh":
            nn.init.xavier_uniform_(module.weight)
        elif activation_function == "ReLU":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


#-----------------------------------RUNNING THE PIPELINE-----------------------------------


def main():
    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check the inputs BEFORE wandb.init, so a missing input doesn't leave an empty run
    # behind. (train_cae.py checks these inside Stages 1-2, after init.)
    train_exists, val_exists = check_alignments_exist()
    if args.init == "cae" and not cAE_load_path.is_file():
        raise FileNotFoundError(
            f"Trained cAE not found at {cAE_load_path}\n"
            f"Please run:\n"
            f"  {_PAIRING['trainer']}\n"
            f"or train the ccAE from scratch instead with --init random."
        )

    _CORPUS_TAG = _FEATURES["tag"] + {"buckeye": "", "xitsonga": "TSO", "zs17french": "ZS17French"}[args.corpus]
    run_name = "-".join(p for p in (_CORPUS_TAG, _MODEL_PREFIX, _VARIANT_SUFFIX) if p)

    # Initialize Weights & Biases
    wandb.init(
        project="cAE Training",
        name="_".join(p for p in (run_name, _LAYER_SUBDIR) if p),
        config={
            "model": "ccAE",
            "features": args.features,
            "corpus": args.corpus,
            "pairs": args.pairs,
            "init": args.init,
            "init_from": str(cAE_load_path) if args.init == "cae" else "random",
            "n_epochs_ccAE": n_epochs_ccAE,
            "lr_ccAE": lr_ccAE,
            "batch_size_ccAE": batch_size_ccAE,
            "optimizer_ccAE": optimizer_ccAE,
            "tau_ccAE": tau_ccAE,
            "negative_mask": "word_label" if _MASK_BY == "label" else "frame_identity",
            "hidden_units": hidden_units,
            "encoding_dim": encoding_dim,
            "num_hidden_layers": num_hidden_layers,
            "activation_function": activation_function,
            "layer": LAYER,
            "seed": seed,
            "device": str(device),
        }
    )

    #------STAGE 1------
    print("Stage 1: Building the autoencoder...")

    ccAE = Autoencoder().to(device)

    if args.init == "cae":
        # Existence was already checked in the pre-flight at the top of main().
        ccAE.load_state_dict(torch.load(cAE_load_path, map_location=device, weights_only=True))
        print(f"Initialised autoencoder from {cAE_load_path} with the following architecture:")
    else:
        # The decoder keeps its random initialisation (it is never trained).
        ccAE.apply(init_weights_classic)
        print("Randomly initialised autoencoder with the following architecture:")

    print(ccAE)

    #------STAGE 2------
    print("\nStage 2: Training the encoder with frame-level InfoNCE...")
    print(f"  pairs={args.pairs}  tau={tau_ccAE}  lr={lr_ccAE}  batch_size={batch_size_ccAE}  epochs={n_epochs_ccAE}")

    # Alignments were already located at the top of main().
    print(f"\nBuilding {_TRAIN_SPLIT} alignment dataset (training)...")
    train_dataset = ContrastiveAlignmentDataset(
        ccAE_alignment_train_path, ccAE_features_train_path, mask_by=_MASK_BY
    )
    print(f"Training data for ccAE: {len(train_dataset)} frame pairs")

    # Load validation data if available
    validate_dataset = None
    if _PAIRING.get("no_val"):
        print("\nNo validation split: training without validation loss.")
    elif val_exists:
        print(f"\nBuilding {_VAL_SPLIT} alignment dataset (validation)...")
        validate_dataset = ContrastiveAlignmentDataset(
            ccAE_alignment_val_path, ccAE_features_val_path
        )
    else:
        print(f"\nWarning: {_VAL_SPLIT} alignments not found. Training without validation loss.")

    # Train the ccAE
    ccAE = train_ccae(ccAE, train_dataset, validate_dataset, seed=seed)

    # Plain state_dict, as the loaders expect.
    torch.save(ccAE.state_dict(), ccAE_save_path)
    print(f"Saved trained ccAE to {ccAE_save_path}")

    #-----STAGE 3-----
    print(f"\nStage 3: Extracting and saving 8th layer embeddings ({encoding_dim}-dim, before activation)...")

    if not os.path.exists(ccAE_save_path):
        print(f"Warning: ccAE model not found at {ccAE_save_path}")
    else:
        ccAE_for_features = Autoencoder().to(device)
        ccAE_for_features.load_state_dict(torch.load(ccAE_save_path, map_location=device, weights_only=True))
        ccAE_for_features.eval()

        print("\nExtracting ccAE-encoded embeddings from 8th layer (before activation)...")
        for split, input_dir, output_dir in STAGE3_SPLITS:
            save_filtered_features_from_npz(
                ccAE_for_features, input_dir, output_dir, f"ccAE ({split})"
            )

    print("\nStage 3 complete. Bottleneck layer embeddings saved successfully!")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

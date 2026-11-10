"""Encode the segments of an ABX item file with a trained cAE, for fastabx.

For every utterance in the subset, computes the input features (MFCC, WavLM, HuBERT or
SpidR), normalises them as for training, runs them through the cAE up to the
bottleneck and saves one .npy file of frame-level embeddings per utterance
(e.g. s07_s0701a_000.npy). score_abx.py then scores them. Features are computed with the
same code as the word features (features/extract_mfcc.py, features/extract_ssl.py).

The feature set picks the checkpoint: Buckeye_WavLM uses models/Buckeye_WavLM/cAE_layer_21.pt,
Buckeye_WavLMDUSTED uses cAE_DUSTED_layer_21.pt, and so on. The input is always the gold
(base) features; only the model changes.

    python evaluation/extract_abx_phone_features.py Buckeye_MFCCs --subset zs
    python evaluation/extract_abx_phone_features.py Buckeye_WavLMDUSTED --subset zs
    python evaluation/extract_abx_phone_features.py Xitsonga_HuBERTUTD --subset test
    python evaluation/extract_abx_phone_features.py Buckeye_WavLMDUSTED --subset zs --ccae --randinit
    python evaluation/extract_abx_phone_features.py Buckeye_WavLM --subset zs --raw    # no cAE

    # Xitsonga encoded by the Buckeye-trained cAE
    python evaluation/extract_abx_phone_features.py Xitsonga_WavLM --subset test --cae-from Buckeye_WavLM

    # SpidR-Adapt features (no cAE): --raw and --layer are required
    python evaluation/extract_abx_phone_features.py Buckeye_SpidR_adapted --subset zs --raw --layer 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_EXTRACTION_DIR = REPO_ROOT / "features"


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Import a script by path (the directories aren't packages)."""

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Features come from the same code as the word features (features/extract_mfcc.py and
# features/extract_ssl.py), exposed under the names used below. Loaded lazily so MFCC
# runs don't need transformers. The LibriSpeech and SpidR extractors live in features/extras/.
_CORPUS_KEY = {"buckeye": "buckeye", "xitsonga": "xitsonga", "zs17french": "french"}
_mfcc_modules: dict[str, types.ModuleType] = {}


def _mfcc_module(corpus: str) -> types.ModuleType:
    """The MFCC routines for a corpus, from features/extract_mfcc.py."""

    if corpus not in _mfcc_modules:
        import glob

        m = _load_module("extract_mfcc", FEATURE_EXTRACTION_DIR / "extract_mfcc.py")
        data = m.CORPORA[_CORPUS_KEY[corpus]]["data"]
        ns = types.SimpleNamespace(
            buckeye_datadir=data, xitsonga_datadir=data, french_datadir=data,
            uttlabel_to_uttkey=m.uttlabel_to_uttkey,
            speaker_to_filelabel=lambda speaker: speaker[:-1] + "_" + speaker[-1],
            extractMFCCfromDirectory=lambda d: m.mfcc_for_wavs(sorted(glob.glob(str(Path(d) / "*.wav")))),
            extractMFCCfromPattern=lambda pattern: m.mfcc_for_wavs(sorted(glob.glob(pattern))),
            extract_mfcc=m.mfcc_prefix,
            LOAD_MARGIN_SECONDS=m.LOAD_MARGIN_SECONDS,
        )
        if corpus == "zs17french":
            ns.speaker_mvn = m.normalise_recording
        elif corpus == "buckeye":  # speaker = key prefix (s23_01a -> s23)
            ns.speaker_mvn = lambda fd: m.speaker_mvn(fd, {k: k.split("_")[0] for k in fd})
        else:
            ns.speaker_mvn = m.speaker_mvn
        _mfcc_modules[corpus] = ns
    return _mfcc_modules[corpus]


_EXTRAS_MODULE_FILES = {
    ("wavlm", "librispeech"): "extractFeaturesWavLMLibri.py",
    ("hubert", "librispeech"): "extractFeaturesHubertLibri.py",
    ("spidr", "buckeye"): "extractFeaturesSpidR.py",
    ("spidr", "xitsonga"): "extractFeaturesSpidRTSO.py",
}
_neural_modules: dict[tuple[str, str], types.ModuleType] = {}


def _neural_module(feature_type: str, corpus: str) -> types.ModuleType:
    """The WavLM/HuBERT extraction settings for a corpus, from features/extract_ssl.py
    (SD_MODEL overrides the model, as it does there)."""

    import os

    key = (feature_type, corpus)
    if key not in _neural_modules:
        if key in _EXTRAS_MODULE_FILES:
            fn = _EXTRAS_MODULE_FILES[key]
            _neural_modules[key] = _load_module(Path(fn).stem, FEATURE_EXTRACTION_DIR / "extras" / fn)
        else:
            m = _load_module("extract_ssl", FEATURE_EXTRACTION_DIR / "extract_ssl.py")
            model, data = m.MODELS[feature_type], m.CORPORA[_CORPUS_KEY[corpus]]["data"]
            prefix = "WAVLM" if feature_type == "wavlm" else "HUBERT"
            _neural_modules[key] = types.SimpleNamespace(**{
                "__name__": "extract_ssl",
                f"{prefix}_MODEL_NAME": os.environ.get("SD_MODEL", model["name"]),
                f"{prefix}_SAMPLE_RATE": m.SAMPLE_RATE, f"{prefix}_HOP": m.HOP,
                f"{prefix}_CENTER_PAD": m.CENTER_PAD, "PAD_TO_HOP": True, "CENTER_PAD": True,
                "LAYERS": (model["layer"],),
                "buckeye_datadir": data, "xitsonga_datadir": data, "french_datadir": data,
                "uttlabel_to_uttkey": lambda label: label[0:3] + "_" + label[3:],
                "speaker_to_filelabel": lambda speaker: speaker[:-1] + "_" + speaker[-1],
            })
    return _neural_modules[key]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- cAE ARCHITECTURE -----------------------------
# Same network as training/train_cae.py.

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


def load_cae(ckpt_fn: Path, input_dim: int, hidden_units: int, encoding_dim: int, num_layers: int):
    """Load a trained cAE and return (model, bottleneck_slice_index)."""

    model = Autoencoder(input_dim, hidden_units, encoding_dim, num_layers).to(device)
    model.load_state_dict(torch.load(ckpt_fn, map_location=device, weights_only=True))
    model.eval()
    bottleneck_index = 2 * num_layers - 3  # matches indexforSaving in train_cae.py
    return model, bottleneck_index


def cae_layer_slice_index(cae_layer: int, num_layers: int) -> int:
    """Slice of the network giving the pre-activation output of hidden layer `cae_layer` (1-indexed).

    Layer k's Linear is at index 2*(k-1); cae_layer == num_layers - 1 is the bottleneck.
    """

    if not 1 <= cae_layer <= num_layers:
        raise ValueError(f"--cae-layer must be between 1 and {num_layers}, got {cae_layer}")
    return 2 * cae_layer - 1


@torch.no_grad()
def encode(model: Autoencoder, bottleneck_index: int, frames: np.ndarray) -> np.ndarray:
    """Frame-level bottleneck embeddings (no pooling)."""

    x = torch.from_numpy(frames).float().to(device)
    embeddings = model.network[:bottleneck_index](x)
    return embeddings.cpu().numpy().astype(np.float32)


# ----------------------------- CONFIG PER COMBO -----------------------------

CONFIGS = {
    ("buckeye", "mfcc"): dict(
        frame_rate=100,
        input_dim=39, hidden_units=250, encoding_dim=100, num_layers=9,
        base_feature_set="Buckeye_MFCCs",
        models_dir=REPO_ROOT / "models/Buckeye_MFCCs",
        splits_dir=REPO_ROOT / "data/Information/Buckeye/Splits",
        speakers_pattern="buckeye_{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye/utterances.list",
        # MFCCs are speaker-normalised at extraction, so there is no Normalised/ level.
        normalised_subdir="",
    ),
    ("xitsonga", "mfcc"): dict(
        frame_rate=100,
        input_dim=39, hidden_units=250, encoding_dim=100, num_layers=9,
        base_feature_set="Xitsonga_MFCCs",
        models_dir=REPO_ROOT / "models/Xitsonga_MFCCs",
        splits_dir=REPO_ROOT / "data/Information/Buckeye_tsonga/Splits",
        speakers_pattern="tsong{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/utterances.list",
        normalised_subdir="",
    ),
    ("buckeye", "wavlm"): dict(
        const_prefix="WAVLM",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="Buckeye_WavLM",
        models_dir=REPO_ROOT / "models/Buckeye_WavLM",
        splits_dir=REPO_ROOT / "data/Information/Buckeye/Splits",
        speakers_pattern="buckeye_{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/Buckeye/windows.list",
        # Always the base set's stats: every variant is scored on the same input.
        norm_stats_fn=REPO_ROOT / "data/Features/Buckeye_WavLM/Normalised/norm_stats.pt",
        # Results go under Normalised/, next to the word-level results.
        normalised_subdir="Normalised",
    ),
    ("xitsonga", "wavlm"): dict(
        const_prefix="WAVLM",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="Xitsonga_WavLM",
        models_dir=REPO_ROOT / "models/Xitsonga_WavLM",
        normalised_subdir="Normalised",
        splits_dir=REPO_ROOT / "data/Information/Buckeye_tsonga/Splits",
        speakers_pattern="tsong{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/windows.list",
        norm_stats_fn=REPO_ROOT / "data/Features/Xitsonga_WavLM/Normalised/norm_stats.pt",
    ),
    # Same shape and cAE architecture as WavLM.
    ("buckeye", "hubert"): dict(
        const_prefix="HUBERT",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="Buckeye_HuBERT",
        models_dir=REPO_ROOT / "models/Buckeye_HuBERT",
        splits_dir=REPO_ROOT / "data/Information/Buckeye/Splits",
        speakers_pattern="buckeye_{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/Buckeye/windows.list",
        norm_stats_fn=REPO_ROOT / "data/Features/Buckeye_HuBERT/Normalised/norm_stats.pt",
        normalised_subdir="Normalised",
    ),
    ("xitsonga", "hubert"): dict(
        const_prefix="HUBERT",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="Xitsonga_HuBERT",
        models_dir=REPO_ROOT / "models/Xitsonga_HuBERT",
        normalised_subdir="Normalised",
        splits_dir=REPO_ROOT / "data/Information/Buckeye_tsonga/Splits",
        speakers_pattern="tsong{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/windows.list",
        norm_stats_fn=REPO_ROOT / "data/Features/Xitsonga_HuBERT/Normalised/norm_stats.pt",
    ),
    # hubert-base-ls960 (768-dim). No cAE or normalisation stats for this set, so it is
    # only used with --raw; the architecture values are placeholders.
    ("librispeech", "hubert"): dict(
        const_prefix="HUBERT",
        frame_rate=50,
        input_dim=768, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="LibriSpeech_HuBERT",
        models_dir=REPO_ROOT / "models/LibriSpeech_HuBERT",
        splits_dir=REPO_ROOT / "data/Information/LibriSpeech/Splits",
        speakers_pattern="librispeech_{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/LibriSpeech/utterances.list",
        normalise=False,
        normalised_subdir="NotNormalised",
    ),
    # WavLM on LibriSpeech (train-clean-5 / dev-clean).
    ("librispeech", "wavlm"): dict(
        const_prefix="WAVLM",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="LibriSpeech_WavLM",
        models_dir=REPO_ROOT / "models/LibriSpeech_WavLM",
        splits_dir=REPO_ROOT / "data/Information/LibriSpeech/Splits",
        speakers_pattern="librispeech_{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/LibriSpeech/utterances.list",
        # Always the BASE model's stats, regardless of variant - see the
        # ("buckeye", "wavlm") entry above for why.
        norm_stats_fn=REPO_ROOT / "data/Features/LibriSpeech_WavLM/Normalised/norm_stats.pt",
        normalised_subdir="Normalised",
    ),
    # SpidR-Adapt (features/extras/extractFeaturesSpidR.py): 768-dim, 50 Hz, not a HF
    # model, so it goes through extract_spidr(). Raw features only (no cAE, no
    # normalisation); the architecture values are placeholders.
    ("buckeye", "spidr"): dict(
        const_prefix="SPIDR",
        frame_rate=50,
        input_dim=768, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="Buckeye_SpidR_adapted",
        models_dir=REPO_ROOT / "models/Buckeye_SpidR_adapted",
        splits_dir=REPO_ROOT / "data/Information/Buckeye/Splits",
        speakers_pattern="buckeye_{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/Buckeye/windows.list",
        normalise=False,
        normalised_subdir="NotNormalised",
    ),
    ("xitsonga", "spidr"): dict(
        const_prefix="SPIDR",
        frame_rate=50,
        input_dim=768, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="Xitsonga_SpidR_adapted",
        models_dir=REPO_ROOT / "models/Xitsonga_SpidR_adapted",
        splits_dir=REPO_ROOT / "data/Information/Buckeye_tsonga/Splits",
        speakers_pattern="tsong{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/Buckeye_tsonga/windows.list",
        normalise=False,
        normalised_subdir="NotNormalised",
    ),
    # ZeroSpeech 2017 French: train = the 10 Relatives, test = the 17 usable Outsiders.
    # Speaker keys drop the underscore ("F01R").
    ("zs17french", "mfcc"): dict(
        frame_rate=100,
        input_dim=39, hidden_units=250, encoding_dim=100, num_layers=9,
        base_feature_set="ZS17French_MFCCs",
        models_dir=REPO_ROOT / "models/ZS17French_MFCCs",
        splits_dir=REPO_ROOT / "data/Information/ZS17French/Splits",
        speakers_pattern="zs17french{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/ZS17French/utterances.list",
        normalised_subdir="",
    ),
    ("zs17french", "wavlm"): dict(
        const_prefix="WAVLM",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="ZS17French_WavLM",
        models_dir=REPO_ROOT / "models/ZS17French_WavLM",
        normalised_subdir="Normalised",
        splits_dir=REPO_ROOT / "data/Information/ZS17French/Splits",
        speakers_pattern="zs17french{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/ZS17French/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/ZS17French/windows.list",
        norm_stats_fn=REPO_ROOT / "data/Features/ZS17French_WavLM/Normalised/norm_stats.pt",
    ),
    ("zs17french", "hubert"): dict(
        const_prefix="HUBERT",
        frame_rate=50,
        input_dim=1024, hidden_units=2048, encoding_dim=128, num_layers=9,
        base_feature_set="ZS17French_HuBERT",
        models_dir=REPO_ROOT / "models/ZS17French_HuBERT",
        normalised_subdir="Normalised",
        splits_dir=REPO_ROOT / "data/Information/ZS17French/Splits",
        speakers_pattern="zs17french{subset}_speakers.list",
        utterances_fn=REPO_ROOT / "data/Information/ZS17French/utterances.list",
        windows_fn=REPO_ROOT / "data/Information/ZS17French/windows.list",
        norm_stats_fn=REPO_ROOT / "data/Features/ZS17French_HuBERT/Normalised/norm_stats.pt",
    ),
}

BASE_FEATURE_SETS = {key: config["base_feature_set"] for key, config in CONFIGS.items()}


class UnknownFeatureSetError(ValueError):
    """Raised when a feature-set name doesn't match any known base set prefix."""


def resolve_feature_set(feature_set: str) -> tuple[str, str, str]:
    """Split a feature-set name or path into (corpus, feature_type, variant).

    "Buckeye_WavLMUTD" or "data/Features/Buckeye_WavLMUTD" -> ("buckeye", "wavlm", "UTD").
    """

    name = Path(feature_set).name
    matches = [
        (corpus, feature_type, prefix)
        for (corpus, feature_type), prefix in BASE_FEATURE_SETS.items()
        if name.startswith(prefix)
    ]
    if not matches:
        raise UnknownFeatureSetError(
            f"'{feature_set}' doesn't start with a known feature set "
            f"({sorted(BASE_FEATURE_SETS.values())}); expected e.g. "
            f"'Buckeye_MFCCs' or 'Buckeye_WavLMPoorManUTD'"
        )
    corpus, feature_type, prefix = max(matches, key=lambda m: len(m[2]))
    variant = name[len(prefix):]
    return corpus, feature_type, variant


def available_variants(models_dir: Path, model_prefix: str = "cAE") -> list[str]:
    """Checkpoint variants available for a base feature set ("" = the base model)."""

    variants = []
    if (models_dir / f"{model_prefix}.pt").is_file():
        variants.append("")
    for fn in sorted(models_dir.glob(f"{model_prefix}_*.pt")):
        variants.append(fn.stem[len(f"{model_prefix}_"):])
    return variants


def speaker_of_seg(seg_id: str) -> str:
    """First underscore-delimited field of a segment ID (matches build_abx_item_file.py)."""

    return seg_id.split("_", 1)[0]


def file_label_of_seg(corpus: str, seg_id: str) -> str:
    """Recording label embedded in a segment ID (matches build_abx_item_file.py)."""

    if corpus == "buckeye":
        return seg_id.split("_")[1]
    if corpus == "zs17french":
        # "F01R_000" -> "F01_R": one recording per speaker, named with the underscore.
        speaker = seg_id.rsplit("_", 1)[0]
        return speaker[:-1] + "_" + speaker[-1]
    # xitsonga: "001m_nchlt_tso_001m_0007_000" -> "nchlt_tso_001m_0007"
    # librispeech: "1272_1272-128104-0000_000" -> "1272-128104-0000"
    return seg_id.split("_", 1)[1].rsplit("_", 1)[0]


def load_utterances_by_file(utterances_fn: Path, corpus: str) -> dict[str, list[tuple[str, float, float]]]:
    """utterances.list as file_label -> [(seg_id, start, end), ...]."""

    by_file: dict[str, list[tuple[str, float, float]]] = {}
    with utterances_fn.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_id, start, end = line.split()
            file_label = file_label_of_seg(corpus, seg_id)
            by_file.setdefault(file_label, []).append((seg_id, float(start), float(end)))
    return by_file


def windows_with_utterances(
    utterances: list[tuple[str, float, float]],
    windows: list[tuple[str, float, float]] | None,
) -> list[tuple[tuple[str, float, float], list[tuple[str, float, float]]]]:
    """Pair each extraction window of a recording with the utterances inside it.

    With no windows.list, each utterance is its own window.
    """

    if windows is None:
        return [(utt, [utt]) for utt in utterances]

    paired = []
    claimed = set()
    for window in sorted(windows, key=lambda w: w[1]):
        inside = [u for u in utterances if u[1] >= window[1] - 1e-6 and u[2] <= window[2] + 1e-6]
        claimed.update(u[0] for u in inside)
        paired.append((window, inside))
    unclaimed = [u[0] for u in utterances if u[0] not in claimed]
    if unclaimed:
        raise SystemExit(
            f"Error: {len(unclaimed)} utterances lie in no window (e.g. {unclaimed[:3]}) - "
            f"windows.list is out of date; rerun preprocessing/make_windows.py"
        )
    return paired


def utterance_frames(
    feats: np.ndarray, window_start: float, seg_start: float, seg_end: float, sample_rate: int, hop: int
) -> np.ndarray:
    """One utterance's frames out of its window's features (frame 0 = utterance start).

    Returns as many frames as a forward pass over the utterance alone would, repeating
    the window's last frame if rounding runs past its end.
    """

    first = round((int(seg_start * sample_rate) - int(window_start * sample_rate)) / hop)
    n = math.ceil((int(seg_end * sample_rate) - int(seg_start * sample_rate)) / hop)
    frames = feats[first:first + n]
    if 0 < len(frames) < n:
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), axis=0)], axis=0)
    return frames


def available_subsets(splits_dir: Path, pattern: str) -> list[str]:
    """Subsets that have a speakers list (for the error message)."""

    prefix, suffix = pattern.split("{subset}")
    names = []
    for fn in sorted(splits_dir.glob(f"{prefix}*{suffix}")):
        if fn.name.startswith(prefix) and fn.name.endswith(suffix):
            names.append(fn.name[len(prefix): len(fn.name) - len(suffix)])
    return names


def load_subset_speakers(splits_dir: Path, pattern: str, subset: str, corpus: str) -> list[str]:
    """Load speakers for a subset, in the form the raw feature-extraction modules expect."""

    speakers_fn = splits_dir / pattern.format(subset=subset)
    if not speakers_fn.is_file():
        available = available_subsets(splits_dir, pattern)
        print(
            f"Error: no speakers list for subset '{subset}' at {speakers_fn}; "
            f"available subsets: {available}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with speakers_fn.open(encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    if corpus == "xitsonga":
        # Splits files list "nchlt-tso-001m"; wav filenames/seg_id speaker field use "001m".
        return [s.split("-")[-1] for s in raw]
    return raw


def resolve_layer(module, prefix: str, layer: int | None) -> int:
    """Layer to score: --layer if given, else the extractor's single default layer."""

    if layer is not None:
        return layer

    layer_const = f"{prefix}_LAYER"
    if hasattr(module, layer_const):
        return getattr(module, layer_const)

    layers = tuple(getattr(module, "LAYERS", ()))
    if len(layers) == 1:
        return layers[0]

    raise SystemExit(
        f"Error: {module.__name__} has no {layer_const} constant and its "
        f"LAYERS = {layers} does not name a single layer to default to "
        f"- pass --layer explicitly for this feature set"
    )


def norm_stats_path(config: dict, layer: int) -> Path:
    """norm_stats_layer_NN.pt next to config["norm_stats_fn"]."""

    return Path(config["norm_stats_fn"]).with_name(f"norm_stats_layer_{layer:02d}.pt")


def pad_tail(arr: np.ndarray, n: int = 2) -> np.ndarray:
    """Repeat the last frame n times.

    fastabx converts onsets and offsets to frames itself, and a triphone ending at the
    segment boundary can round to one frame past the end of the array.
    """

    if arr.shape[0] == 0:
        return arr
    return np.concatenate([arr, np.repeat(arr[-1:], n, axis=0)], axis=0)


# ------------------------------- MFCC PATH -----------------------------------

def extract_mfcc(corpus: str, subset: str, config: dict) -> dict[str, np.ndarray]:
    """{seg_id: MFCC array}, speaker-normalised as in extract_mfcc.py."""

    module = _mfcc_module(corpus)
    speakers = load_subset_speakers(config["splits_dir"], config["speakers_pattern"], subset, corpus)

    if corpus == "buckeye":
        feat_dict: dict[str, np.ndarray] = {}
        for speaker in speakers:
            speaker_dir = Path(module.buckeye_datadir) / speaker
            speaker_feat_dict = module.extractMFCCfromDirectory(str(speaker_dir))
            for wav_file, arr in speaker_feat_dict.items():
                feat_dict[speaker + "_" + wav_file[3:]] = arr
        print("Per speaker mean and variance normalisation:")
        feat_dict = module.speaker_mvn(feat_dict)

        def recording_key(file_label: str) -> str:
            return module.uttlabel_to_uttkey(file_label)

    elif corpus == "zs17french":
        # French reads only the aligned prefix of each (very long) recording.
        feat_dict = {}
        utterances_by_file_tmp = load_utterances_by_file(config["utterances_fn"], corpus)
        for speaker in speakers:
            file_label = module.speaker_to_filelabel(speaker)
            segs = utterances_by_file_tmp.get(file_label)
            if not segs:
                continue
            wav_fn = Path(module.french_datadir) / f"{file_label}.wav"
            last_end = max(end for _, _, end in segs)
            arr = module.extract_mfcc(
                str(wav_fn), last_end + module.LOAD_MARGIN_SECONDS
            )
            feat_dict[file_label] = module.speaker_mvn(arr)

        def recording_key(file_label: str) -> str:
            return file_label

    else:  # xitsonga
        feat_dict = {}
        speaker_of_key = {}
        for speaker in speakers:
            wav_pattern = str(Path(module.xitsonga_datadir) / f"nchlt_tso_{speaker}_*.wav")
            speaker_feat_dict = module.extractMFCCfromPattern(wav_pattern)
            for utt_key, arr in speaker_feat_dict.items():
                feat_dict[utt_key] = arr
                speaker_of_key[utt_key] = speaker
        print("Per speaker mean and variance normalisation:")
        feat_dict = module.speaker_mvn(feat_dict, speaker_of_key)

        def recording_key(file_label: str) -> str:
            return file_label

    # utterances.list, keyed by recording (file_label) -> [(seg_id, start, end), ...],
    # identical parsing to the WavLM extraction scripts' load_utterances.
    utterances_by_file = load_utterances_by_file(config["utterances_fn"], corpus)

    speakers_set = set(speakers)
    segment_features: dict[str, np.ndarray] = {}
    dropped = 0
    for file_label, segs in tqdm(utterances_by_file.items(), desc="Slicing MFCC segments"):
        key = recording_key(file_label)
        if key not in feat_dict:
            continue
        recording_arr = feat_dict[key]
        for seg_id, start_t, end_t in segs:
            if speaker_of_seg(seg_id) not in speakers_set:
                continue
            start_frame = int(round(start_t * 100))
            end_frame = int(round(end_t * 100)) + 1  # end excluded, matches extract_utterances()
            seg_arr = recording_arr[start_frame:end_frame]
            if seg_arr.shape[0] == 0:
                dropped += 1
                continue
            segment_features[seg_id] = pad_tail(seg_arr)

    if dropped:
        print(f"Warning: {dropped} segments had zero MFCC frames and were dropped")

    return segment_features


# --------------------------- WavLM/HuBERT PATH ---------------------------------

def extract_wavlm(
    corpus: str,
    feature_type: str,
    subset: str,
    config: dict,
    layer: int | None = None,
    whole_recording: bool = False,
) -> dict[str, np.ndarray]:
    """{key: WavLM/HuBERT features} for the utterances of a subset.

    One forward pass per extraction window, then each utterance's frames are sliced out
    (frame 0 = utterance start). With whole_recording=True, one pass per recording
    instead, keyed by recording (for item files with recording-relative times, e.g.
    the LibriSpeech phoneme task). `layer` defaults to the extractor's layer.
    """

    import glob
    import soundfile as sf
    import torchaudio

    module = _neural_module(feature_type, corpus)
    prefix = config["const_prefix"]
    model_name = getattr(module, f"{prefix}_MODEL_NAME")
    sample_rate = getattr(module, f"{prefix}_SAMPLE_RATE")
    layer = resolve_layer(module, prefix, layer)

    speakers = load_subset_speakers(config["splits_dir"], config["speakers_pattern"], subset, corpus)
    speakers_set = set(speakers)

    utterances_by_file = load_utterances_by_file(config["utterances_fn"], corpus)

    print(f"Loading {prefix.title()}...")
    if feature_type == "hubert":
        # As in the extractors:
        # both load HubertModel directly rather than the generic AutoModel.
        from transformers import HubertModel

        model = HubertModel.from_pretrained(model_name)
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    segment_features: dict[str, np.ndarray] = {}

    if corpus == "buckeye":
        wav_iter = [
            (speaker, wav_fn)
            for speaker in speakers
            for wav_fn in glob.glob(str(Path(module.buckeye_datadir) / speaker / "*.wav"))
        ]
    elif corpus == "librispeech":
        # LibriSpeech: <subset>/<speaker>/<chapter>/<utt_id>.flac
        wav_iter = [
            (speaker, wav_fn)
            for speaker in speakers
            for wav_fn in glob.glob(str(Path(module.librispeech_datadir) / subset / speaker / "*" / "*.flac"))
        ]
    elif corpus == "zs17french":
        # Flat wavs named by recording, one per speaker: F01R -> F01_R.wav.
        wav_iter = [
            (speaker, str(Path(module.french_datadir) / f"{module.speaker_to_filelabel(speaker)}.wav"))
            for speaker in speakers
        ]
    else:
        wav_iter = [
            (speaker, wav_fn)
            for speaker in speakers
            for wav_fn in glob.glob(str(Path(module.xitsonga_datadir) / f"nchlt_tso_{speaker}_*.wav"))
        ]

    # Every WavLM/HuBERT module here frames at 50 Hz (320-sample hop at 16 kHz).
    hop = getattr(module, f"{prefix}_HOP", sample_rate // 50)
    windows_by_file = (
        load_utterances_by_file(config["windows_fn"], corpus) if config.get("windows_fn") else None
    )

    def forward(chunk: torch.Tensor) -> np.ndarray:
        if hasattr(module, "processor"):
            input_values = module.processor(
                chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt"
            ).input_values.to(device)
            with torch.no_grad():
                outputs = model(input_values, output_hidden_states=True)
            feats = outputs.hidden_states[layer].squeeze(0).cpu().numpy()
            del input_values, outputs
            return feats
        # Pad to the hop, centre-pad and layer-norm, as in features/extract_ssl.py.
        center_pad = getattr(module, f"{prefix}_CENTER_PAD")
        wav = chunk.float()
        if module.PAD_TO_HOP and (r := wav.numel() % hop):
            wav = F.pad(wav, (0, hop - r))
        if module.CENTER_PAD:
            wav = F.pad(wav, (center_pad, center_pad))
        x = F.layer_norm(wav.to(device), wav.shape).unsqueeze(0)
        with torch.no_grad():
            outputs = model(x, output_hidden_states=True)
        feats = outputs.hidden_states[layer].squeeze(0).cpu().numpy()
        del x, outputs
        return feats

    desc = f"Extracting {prefix.title()} " + ("recordings" if whole_recording else "windows")
    for speaker, wav_fn in tqdm(wav_iter, desc=desc):
        filename = Path(wav_fn).stem
        if filename not in utterances_by_file:
            continue

        # Load as the extractors do: soundfile
        # (not librosa), mono-mix, torchaudio resample only if needed.
        signal, sr = sf.read(wav_fn, dtype="float32")
        signal = torch.from_numpy(signal)
        if signal.ndim == 2:
            signal = signal.mean(-1)
        if sr != sample_rate:
            signal = torchaudio.functional.resample(signal, sr, sample_rate)

        if whole_recording:
            # One pass over the whole recording, keyed by recording.
            if len(signal):
                segment_features[module.uttlabel_to_uttkey(filename)] = pad_tail(forward(signal))
        else:
            # One pass per window; utterances are sliced out of it.
            windows = windows_by_file.get(filename) if windows_by_file is not None else None
            for (_, win_start, win_end), utts in windows_with_utterances(utterances_by_file[filename], windows):
                utts = [u for u in utts if speaker_of_seg(u[0]) in speakers_set]
                chunk = signal[int(win_start * sample_rate):int(win_end * sample_rate)]
                if not utts or len(chunk) == 0:
                    continue
                feats = forward(chunk)
                for seg_id, seg_start, seg_end in utts:
                    segment_features[seg_id] = pad_tail(
                        utterance_frames(feats, win_start, seg_start, seg_end, sample_rate, hop)
                    )

        del signal
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return segment_features


# ----------------------------- SPIDR-ADAPT PATH ---------------------------------

def extract_spidr(
    corpus: str,
    subset: str,
    config: dict,
    layer: int | None = None,
) -> dict[str, np.ndarray]:
    """{seg_id: SpidR-Adapt features}, extracted like extract_wavlm().

    SpidR-Adapt isn't a HF model: get_intermediate_outputs() returns one entry per
    transformer layer with no embedding entry, so layer N is list[N - 1].
    """

    import glob
    import soundfile as sf
    import torchaudio
    from spidr_adapt.models import build_model

    module = _neural_module("spidr", corpus)
    prefix = config["const_prefix"]
    checkpoint = getattr(module, f"{prefix}_CHECKPOINT")
    model_type = getattr(module, f"{prefix}_MODEL_TYPE")
    sample_rate = getattr(module, f"{prefix}_SAMPLE_RATE")
    hop = getattr(module, f"{prefix}_HOP")
    center_pad = getattr(module, f"{prefix}_CENTER_PAD")
    layer = resolve_layer(module, prefix, layer)

    speakers = load_subset_speakers(config["splits_dir"], config["speakers_pattern"], subset, corpus)
    speakers_set = set(speakers)
    utterances_by_file = load_utterances_by_file(config["utterances_fn"], corpus)
    windows_by_file = (
        load_utterances_by_file(config["windows_fn"], corpus) if config.get("windows_fn") else None
    )

    print(f"Loading SpidR-Adapt (checkpoint={checkpoint}, model_type={model_type})...")
    model = build_model(model_type=model_type, checkpoint=checkpoint)
    model.to(device)
    model.eval()

    segment_features: dict[str, np.ndarray] = {}

    if corpus == "buckeye":
        wav_iter = [
            (speaker, wav_fn)
            for speaker in speakers
            for wav_fn in glob.glob(str(Path(module.buckeye_datadir) / speaker / "*.wav"))
        ]
    else:  # xitsonga
        wav_iter = [
            (speaker, wav_fn)
            for speaker in speakers
            for wav_fn in glob.glob(str(Path(module.xitsonga_datadir) / f"nchlt_tso_{speaker}_*.wav"))
        ]

    for speaker, wav_fn in tqdm(wav_iter, desc="Extracting SpidR-Adapt windows"):
        filename = Path(wav_fn).stem
        if filename not in utterances_by_file:
            continue

        # Load as extract_wavlm() does: soundfile, mono-mix,
        # torchaudio resample only if needed.
        signal, sr = sf.read(wav_fn, dtype="float32")
        signal = torch.from_numpy(signal)
        if signal.ndim == 2:
            signal = signal.mean(-1)
        if sr != sample_rate:
            signal = torchaudio.functional.resample(signal, sr, sample_rate)

        # One forward pass per extraction window, then each utterance's
        # frames sliced out of it - same as extract_wavlm().
        windows = windows_by_file.get(filename) if windows_by_file is not None else None
        for (_, win_start, win_end), utts in windows_with_utterances(utterances_by_file[filename], windows):
            utts = [u for u in utts if speaker_of_seg(u[0]) in speakers_set]
            chunk = signal[int(win_start * sample_rate):int(win_end * sample_rate)]
            if not utts or len(chunk) == 0:
                continue

            # Same pad-to-hop / centre-pad / raw F.layer_norm preprocessing
            # as spidr_forward() in extractFeaturesSpidR(TSO).py.
            wav = chunk.float()
            if module.PAD_TO_HOP and (r := wav.numel() % hop):
                wav = F.pad(wav, (0, hop - r))
            if module.CENTER_PAD:
                wav = F.pad(wav, (center_pad, center_pad))
            x = F.layer_norm(wav.to(device), wav.shape).unsqueeze(0)
            with torch.no_grad():
                outputs = model.get_intermediate_outputs(x)  # list[i] = output of transformer layer i+1
            feats = outputs[layer - 1].squeeze(0).cpu().numpy()
            del x, outputs

            for seg_id, seg_start, seg_end in utts:
                segment_features[seg_id] = pad_tail(
                    utterance_frames(feats, win_start, seg_start, seg_end, sample_rate, hop)
                )

        del signal
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return segment_features


# --------------------------------- MAIN ---------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract per-segment cAE features for the triphone ABX task."
    )
    parser.add_argument(
        "feature_set",
        help="Feature set directory or name, e.g. 'Buckeye_MFCCs', 'Buckeye_WavLMPoorManUTD', "
        "or 'data/Features/Buckeye_WavLMDUSTED' - a known base set "
        "(Buckeye_MFCCs/Buckeye_WavLM/Xitsonga_MFCCs/Xitsonga_WavLM) optionally "
        "followed by a model variant suffix (UTD/DUSTED/PoorManUTD/...).",
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="any subset with a Splits/*_speakers.list file, e.g. devpart1/devpart2/zs "
        "for buckeye, dev/test for xitsonga",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip the cAE entirely and save the raw (normalised) MFCC/WavLM features "
        "directly, as the baseline to compare cAE-encoded ABX scores against. Raw "
        "features don't depend on which cAE checkpoint would encode them, so this "
        "only accepts a base feature_set (no UTD/DUSTED/PoorManUTD/... suffix) - "
        "one raw run covers every variant's baseline.",
    )
    parser.add_argument(
        "--ae",
        action="store_true",
        help="use a plain autoencoder checkpoint (models/<Set>/AE.pt, same architecture) "
        "instead of the cAE; writes to <subset>.abx_AE",
    )
    parser.add_argument(
        "--ccae",
        action="store_true",
        help="Use the contrastive correspondence autoencoder (models/<Set>/ccAE.pt, "
        "trained by training/train_ccae.py - frame-level InfoNCE instead of "
        "the cAE's MSE correspondence loss) instead of the cAE. Same architecture and "
        "same bottleneck slice either way, only which checkpoint is loaded changes. "
        "Writes to '<subset>.abx_ccAE' instead of '<subset>.abx_cAE'.",
    )
    parser.add_argument(
        "--randinit",
        action="store_true",
        help="With --ccae, select the randomly-initialised ccAE "
        "(models/<Set>/ccAErandinit_*.pt, trained by train_ccae.py --init random) "
        "rather than the one fine-tuned from the cAE. Writes to "
        "'<subset>.abx_ccAErandinit'. Only valid with --ccae.",
    )
    parser.add_argument(
        "--model-prefix",
        default=None,
        metavar="PREFIX",
        help="Select the checkpoint by an explicit prefix instead of --ae/--ccae, for "
        "hyperparameter-sweep runs tagged by training/train_ccae.py --tag "
        "(e.g. --model-prefix ccAErandinitTau005 loads "
        "models/<Set>/ccAErandinitTau005_layer_NN.pt and writes to "
        "'<subset>.abx_ccAErandinitTau005'). Overrides --ccae/--ae; not valid with --raw.",
    )
    parser.add_argument(
        "--cae-from",
        default=None,
        metavar="SET",
        help="Encode this corpus's audio with ANOTHER feature set's cAE, for the "
        "cross-lingual transfer setup (e.g. --cae-from Buckeye_WavLM while "
        "extracting Xitsonga_WavLM: the Buckeye-trained cAE applied zero-shot to "
        "Xitsonga). Takes a feature-set name with an optional variant suffix "
        "(Buckeye_WavLMUTD selects models/Buckeye_WavLM/cAE_UTD_layer_NN.pt), and "
        "switches the WavLM/HuBERT normalisation stats to that set's too - the cAE "
        "has to see its input in the distribution it was trained on, which is the "
        "same reasoning as features/extras/normaliseTSOUsingBuck.py's for the "
        "word-level features. Must be the same feature type as the positional set "
        "(no encoding WavLM frames with an MFCC cAE), and the positional set must "
        "then be a base set, since the variant comes from here instead. Writes to "
        "'<subset>.abx_cAE_from<SET>' so it never collides with the same-corpus run.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="WavLM/HuBERT layer to extract, as an index into hidden_states "
        "(0 = feature-extractor output, i = output of transformer layer i). "
        "Overrides the layer the extraction module itself saves (its LAYERS "
        "tuple, e.g. (24,) in extract_ssl.py). Not valid for "
        "MFCC. Appends '_layerNN' to the output directory name so different "
        "layers' features/results don't collide.",
    )
    parser.add_argument(
        "--cae-layer",
        type=int,
        default=None,
        help="Which hidden layer of the cAE to encode to, before its activation "
        "(1-indexed: 1 = first hidden Linear, num_layers-1 = the bottleneck, "
        "num_layers = the last hidden layer before the reconstruction output). "
        "Matches training/extract_cae_all_layers.py's cAE_layer_K numbering. "
        "Defaults to the bottleneck (previous behaviour). Appends "
        "'_cAElayerK' to the output directory name. Not valid with --raw.",
    )
    parser.add_argument(
        "--whole-recording",
        action="store_true",
        help="Run one forward pass per whole recording (frame 0 = recording "
        "start) instead of one per utterances.list segment, and key/name "
        "output files by recording ID instead of seg_id. For the "
        "isolated-phoneme task, whose item file (e.g. "
        "data/Information/LibriSpeech/abx_items/phoneme-dev-clean.item) is "
        "keyed by whole-recording ID with recording-relative onsets/offsets, "
        "not by segment. Requires --raw (no cAE has been trained on "
        "whole-recording features) and doesn't apply to MFCC. Writes to "
        "'<subset>.abx_whole{layerNN}' instead of '<subset>.abx{layerNN}' so "
        "it never collides with the segment-keyed features.",
    )
    return parser.parse_args()


def main() -> None:
    """Extract and save per-segment features (cAE/AE-encoded, or raw with --raw) for one feature-set/subset."""

    args = parse_args()
    try:
        corpus, feature_type, variant = resolve_feature_set(args.feature_set)
    except UnknownFeatureSetError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    config = CONFIGS[(corpus, feature_type)]
    # The init mode is part of the model prefix (cAE / ccAE / ccAErandinit), as in train_ccae.py.
    if args.model_prefix:
        model_prefix = args.model_prefix
    elif args.ccae:
        model_prefix = "ccAErandinit" if args.randinit else "ccAE"
    elif args.ae:
        model_prefix = "AE"
    else:
        model_prefix = "cAE"

    # --cae-from takes the checkpoint and normalisation stats from another feature set;
    # the audio, splits and item file still come from the positional set.
    cae_config, cae_variant, cae_tag = config, variant, ""
    if args.cae_from:
        try:
            src_corpus, src_feature_type, cae_variant = resolve_feature_set(args.cae_from)
        except UnknownFeatureSetError as error:
            print(f"Error: --cae-from {error}", file=sys.stderr)
            raise SystemExit(1) from None
        cae_config = CONFIGS[(src_corpus, src_feature_type)]
        cae_tag = f"_from{Path(args.cae_from).name}"

        if src_feature_type != feature_type:
            print(
                f"Error: --cae-from {args.cae_from} is a {src_feature_type} feature set but "
                f"'{args.feature_set}' is {feature_type} - a cAE can only encode the feature "
                f"type it was trained on",
                file=sys.stderr,
            )
            raise SystemExit(1)

        if args.raw:
            print("Error: --raw and --cae-from are mutually exclusive (--raw skips the model entirely)", file=sys.stderr)
            raise SystemExit(1)

        if variant:
            print(
                f"Error: --cae-from supplies the model variant, so the positional set must be a "
                f"base set; pass '{config['base_feature_set']}' instead of '{args.feature_set}' "
                f"(and put the variant on --cae-from, e.g. --cae-from {args.cae_from}{variant})",
                file=sys.stderr,
            )
            raise SystemExit(1)

    if args.raw and variant:
        print(
            f"Error: --raw doesn't take a model variant (raw features are the same "
            f"regardless of which cAE would encode them); pass '{config['base_feature_set']}' "
            f"instead of '{args.feature_set}'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.raw and args.ae:
        print("Error: --raw and --ae are mutually exclusive (--raw skips the model entirely)", file=sys.stderr)
        raise SystemExit(1)

    if args.raw and args.ccae:
        print("Error: --raw and --ccae are mutually exclusive (--raw skips the model entirely)", file=sys.stderr)
        raise SystemExit(1)

    if args.ae and args.ccae:
        print("Error: --ae and --ccae are mutually exclusive (they select different checkpoints)", file=sys.stderr)
        raise SystemExit(1)

    if args.randinit and not args.ccae:
        print("Error: --randinit only applies to --ccae (only the ccAE has a random-init variant)", file=sys.stderr)
        raise SystemExit(1)

    if args.model_prefix and args.raw:
        print("Error: --model-prefix and --raw are mutually exclusive (--raw skips the model entirely)", file=sys.stderr)
        raise SystemExit(1)

    if args.model_prefix and not args.model_prefix.isalnum():
        print(
            f"Error: --model-prefix must be alphanumeric (got {args.model_prefix!r}); it is "
            f"joined to the variant and layer with '_' to form the checkpoint name",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.layer is not None and feature_type == "mfcc":
        print("Error: --layer only applies to WavLM/HuBERT feature sets, not MFCC", file=sys.stderr)
        raise SystemExit(1)

    if args.whole_recording and feature_type == "mfcc":
        print("Error: --whole-recording only applies to WavLM/HuBERT feature sets, not MFCC", file=sys.stderr)
        raise SystemExit(1)

    if args.whole_recording and not args.raw:
        print(
            "Error: --whole-recording requires --raw (no cAE has been trained on "
            "whole-recording features)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.cae_layer is not None and args.raw:
        print("Error: --cae-layer doesn't apply to --raw (there's no cAE to pick a layer of)", file=sys.stderr)
        raise SystemExit(1)

    if args.cae_layer is not None and not 1 <= args.cae_layer <= config["num_layers"]:
        print(
            f"Error: --cae-layer must be between 1 and {config['num_layers']} for "
            f"{config['base_feature_set']}, got {args.cae_layer}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    layer_suffix = f"_layer{args.layer:02d}" if args.layer is not None else ""
    cae_layer_suffix = f"_cAElayer{args.cae_layer}" if args.cae_layer is not None else ""

    # Checkpoints and normalisation stats are tagged with the layer (MFCC: no layer).
    effective_layer = None
    if feature_type != "mfcc":
        effective_layer = resolve_layer(
            _neural_module(feature_type, corpus), config["const_prefix"], args.layer
        )

    if not args.raw:
        if feature_type == "mfcc":
            ckpt_name = f"{model_prefix}_{cae_variant}.pt" if cae_variant else f"{model_prefix}.pt"
        else:
            layer_tag = f"_layer_{effective_layer:02d}"
            ckpt_name = f"{model_prefix}_{cae_variant}{layer_tag}.pt" if cae_variant else f"{model_prefix}{layer_tag}.pt"
        cae_ckpt = cae_config["models_dir"] / ckpt_name
        if not cae_ckpt.is_file():
            available = available_variants(cae_config["models_dir"], model_prefix)
            print(
                f"Error: no checkpoint at {cae_ckpt}; available {model_prefix} variants for "
                f"{cae_config['base_feature_set']}: {available or '(none found)'} "
                f"(pass e.g. '{cae_config['base_feature_set']}UTD' to select "
                f"{model_prefix}_UTD{'' if feature_type == 'mfcc' else f'_layer_{effective_layer:02d}'}.pt, "
                f"or --layer to target a different layer's checkpoint)",
                file=sys.stderr,
            )
            raise SystemExit(1)

    if feature_type == "mfcc":
        segment_features = extract_mfcc(corpus, args.subset, config)
    elif feature_type == "spidr":
        if args.whole_recording:
            print("Error: --whole-recording is not implemented for SpidR-Adapt", file=sys.stderr)
            raise SystemExit(1)
        segment_features = extract_spidr(corpus, args.subset, config, layer=args.layer)
    else:
        segment_features = extract_wavlm(
            corpus, feature_type, args.subset, config, layer=args.layer,
            whole_recording=args.whole_recording,
        )

        # With --cae-from these are the source set's stats, which is what the cAE was
        # trained on. Whole-recording features are used raw.
        if config.get("normalise", True) and not args.whole_recording:
            if "norm_stats_fn" not in cae_config:
                raise SystemExit(
                    f"Error: --cae-from {args.cae_from} has no normalisation stats, but "
                    f"{config['base_feature_set']} features are normalised - there is no "
                    f"consistent way to standardise the input for that cAE"
                )
            stats_fn = norm_stats_path(cae_config, effective_layer)
            if not stats_fn.is_file():
                raise SystemExit(
                    f"Error: no normalisation stats at {stats_fn} - run the matching "
                    f"features/normalise.py with --layer {effective_layer} first"
                )
            norm_stats = torch.load(stats_fn, map_location="cpu")
            mean = norm_stats["mean"].numpy().astype(np.float32)
            std = norm_stats["std"].numpy().astype(np.float32)
            segment_features = {
                seg_id: (arr.astype(np.float32) - mean) / std for seg_id, arr in segment_features.items()
            }

    if args.raw:
        whole_suffix = "_whole" if args.whole_recording else ""
        out_dir = (
            REPO_ROOT / "data/Features" / config["base_feature_set"]
            / config["normalised_subdir"] / f"{args.subset}.abx{whole_suffix}{layer_suffix}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, frames in tqdm(sorted(segment_features.items()), desc="Saving raw features"):
            np.save(out_dir / f"{key}.npy", frames.astype(np.float32))
        print(f"Wrote {len(segment_features)} feature files to {out_dir}")
        return

    print(f"Loaded {len(segment_features)} segments; loading {model_prefix} from {cae_ckpt}...")
    model, bottleneck_index = load_cae(
        cae_ckpt, config["input_dim"], config["hidden_units"],
        config["encoding_dim"], config["num_layers"],
    )
    slice_index = (
        cae_layer_slice_index(args.cae_layer, config["num_layers"])
        if args.cae_layer is not None
        else bottleneck_index
    )

    # A --cae-from run is written under the target set, tagged with the source.
    out_dir = (
        REPO_ROOT / "data/Features" / f"{config['base_feature_set']}{variant}"
        / config["normalised_subdir"]
        / f"{args.subset}.abx_{model_prefix}{cae_tag}{cae_layer_suffix}{layer_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for seg_id, frames in tqdm(sorted(segment_features.items()), desc=f"Encoding with {model_prefix}"):
        embeddings = encode(model, slice_index, frames)
        np.save(out_dir / f"{seg_id}.npy", embeddings)

    print(f"Wrote {len(segment_features)} segment feature files to {out_dir}")


if __name__ == "__main__":
    main()

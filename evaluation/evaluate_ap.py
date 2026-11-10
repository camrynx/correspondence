"""Same-different evaluation (AP, plus AP for same- and different-speaker pairs).

Takes a directory of .npz files of word tokens, DTW-scores every token pair with
Nicol Visser's samediff package and writes results.json, ap_result.txt and two plots
under results/ap/.

    python evaluation/evaluate_ap.py data/Features/Buckeye_HuBERT/Normalised/zs.samediff_cAE/layer_24
    python evaluation/evaluate_ap.py --backend cpu DIR

Each feature dimension is normalised over all frames before the DTW (Carlin et al.,
2011), as in calculate_distances.py --normalize-dims; --no-normalize-dims turns this
off. Results agree closely with calculate_distances.py + samediff.py, but not always to
the last digit: those store distances as float32 and don't group tied distances when
computing precision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

# This directory has its own samediff.py, which would shadow the samediff package.
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir in sys.path:
    sys.path.remove(_this_dir)

from samediff.evaluate import evaluate_results
from samediff.plots import plot_distance_histogram, plot_precision_vs_recall
from samediff.run import run_tasks
from samediff.setup import Tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_ROOT = REPO_ROOT / "data" / "Features"
OUT_ROOT = REPO_ROOT / "results" / "ap"


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Same-different AP evaluation via samediff (Nicol Visser).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "features_input",
        help="directory of .npz files of word tokens",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cuda",
        help="'cpu' (Rust) or 'cuda' (Triton); falls back to cpu if cuda is "
        "unavailable or fails (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean", "euclidean_squared"],
        default="cosine",
        help="DTW frame distance (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        help="pairs scored per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--normalize-dims",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalise each dimension over all frames before the DTW (default: on)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: the features path under results/ap/)",
    )
    return parser.parse_args()


def output_dir_for(features_dir: Path) -> Path:
    """results/ap/<features path relative to data/Features>."""

    resolved = features_dir.resolve()
    try:
        subpath = resolved.relative_to(FEATURES_ROOT)
    except ValueError:
        subpath = Path(features_dir.name)
    return OUT_ROOT / subpath


def validate_features_dir(features_input: Path) -> list[Path]:
    """Validate that the input is a directory containing NPZ files, return them sorted."""

    if not features_input.is_dir():
        raise SystemExit(f"Error: {features_input} is not a directory")

    npz_files = sorted(features_input.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"Error: {features_input} contains no .npz files")

    return npz_files


def parse_key(key: str) -> tuple[str, str | None]:
    """(word, speaker) from a token key.

    The word is everything before the first "_". The speaker is usually the next field
    ("actually_s18_s1801a_021_014163-014190" -> "s18"), but Xitsonga keys contain
    "nchlt_tso" and the speaker follows it.
    """

    parts = key.split("_")
    word = parts[0]
    for i in range(len(parts) - 2):
        if parts[i] == "nchlt" and parts[i + 1] == "tso" and i + 2 < len(parts):
            return word, parts[i + 2]
    speaker = parts[1] if len(parts) > 1 else None
    return word, speaker


def load_tokens(npz_files: list[Path]) -> tuple[list[dict], list[np.ndarray]]:
    """All token arrays in the .npz files, with their (word, speaker) rows."""

    rows: list[dict] = []
    features: list[np.ndarray] = []
    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=False) as archive:
            for key in archive.files:
                array = np.asarray(archive[key], dtype=np.float32)
                assert array.ndim == 2, (
                    f"Expected a (T, D) continuous feature array for {key!r} in "
                    f"{npz_file}, got shape {array.shape}. This script only "
                    "handles continuous (DTW) features; discrete unit "
                    "sequences aren't supported here."
                )
                word, speaker = parse_key(key)
                rows.append({"#word": word, "speaker": speaker})
                features.append(array)
    return rows, features


def normalize_dims(features: list[np.ndarray]) -> list[np.ndarray]:
    """Normalise each dimension to zero mean and unit variance over all frames."""

    dim = features[0].shape[1]
    n_frames = 0
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    for feats in features:
        x = feats.astype(np.float64)
        total += x.sum(axis=0)
        total_sq += (x**2).sum(axis=0)
        n_frames += x.shape[0]
    mean = total / n_frames
    std = np.sqrt(np.maximum(total_sq / n_frames - mean**2, 0))
    std[std == 0] = 1.0
    print(
        f"Normalizing feature dimensions (zero mean, unit variance) over "
        f"{n_frames} frames from {len(features)} tokens, {dim} dims"
    )
    return [
        ((feats.astype(np.float64) - mean) / std).astype(np.float32)
        for feats in features
    ]


def step_average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Step-wise AP: sum of (R_i - R_{i-1}) * P_i, as in sklearn.

    samediff only reports AP for all pairs; applying this to its same-speaker and
    different-speaker recall curves gives AP SWSP and AP SWDP from the same ranking.
    Returns NaN if the condition has no pairs.
    """

    recall = np.asarray(recall, dtype=np.float64)
    if len(recall) == 0 or np.isnan(recall).all():
        return math.nan
    precision = np.asarray(precision, dtype=np.float64)
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def resolve_backend(backend: str) -> str:
    """Use the CPU backend if PyTorch or a GPU is missing."""

    if backend != "cuda":
        return backend

    try:
        import torch
    except ImportError:
        print("PyTorch not available; falling back to --backend cpu", file=sys.stderr)
        return "cpu"

    if not torch.cuda.is_available():
        print("No CUDA-enabled GPU found; falling back to --backend cpu", file=sys.stderr)
        return "cpu"

    return "cuda"


def run_tasks_with_fallback(tasks: Tasks, backend: str, **kwargs):
    """run_tasks, retried on the CPU if the CUDA run fails (e.g. out of memory)."""

    backend = resolve_backend(backend)
    if backend != "cuda":
        return run_tasks(tasks, backend=backend, **kwargs)

    try:
        return run_tasks(tasks, backend="cuda", **kwargs)
    except Exception as exc:
        print(
            f"CUDA backend failed ({exc!r}); retrying on --backend cpu",
            file=sys.stderr,
        )
        return run_tasks(tasks, backend="cpu", **kwargs)


def build_tasks(rows: list[dict], features: list[np.ndarray]) -> Tasks:
    """samediff Tasks built in memory from tokens that are already sliced."""

    df = pl.DataFrame(rows).with_row_index("row_idx")
    return Tasks(df=df, features=features, kind="continuous")


def main() -> None:
    args = parse_args()
    features_dir = Path(args.features_input)
    npz_files = validate_features_dir(features_dir)

    out_dir = args.out_dir or output_dir_for(features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokens from {len(npz_files)} NPZ files in {features_dir}")
    rows, features = load_tokens(npz_files)
    print(f"Loaded {len(features)} tokens")

    if args.normalize_dims:
        features = normalize_dims(features)

    tasks = build_tasks(rows, features)

    n = len(features)
    print(f"Scoring {n * (n - 1) // 2} pairs (backend={args.backend}, metric={args.metric})")
    results = run_tasks_with_fallback(
        tasks,
        backend=args.backend,
        batch_size=args.batch_size,
        metric=args.metric,
        dur_normalize=True,
    )
    evaluated = evaluate_results(results)

    ap_swsp = step_average_precision(evaluated.precision, evaluated.recall_swsp)
    ap_swdp = step_average_precision(evaluated.precision, evaluated.recall_swdp)

    # Sanity check: our estimator should reproduce samediff's AP on all pairs.
    ap_sw_recomputed = step_average_precision(evaluated.precision, evaluated.recall_sw)
    if not math.isclose(ap_sw_recomputed, evaluated.average_precision, rel_tol=1e-9, abs_tol=1e-12):
        print(
            f"Warning: recomputed AP SW ({ap_sw_recomputed}) disagrees with samediff's "
            f"({evaluated.average_precision}); AP SWSP/SWDP use the recomputed estimator "
            f"and may not be comparable to AP SW",
            file=sys.stderr,
        )

    metrics = {
        "AP": evaluated.average_precision,
        "AP SWSP": ap_swsp,
        "AP SWDP": ap_swdp,
        "PRB SW": evaluated.prb_sw,
        "PRB SWSP": evaluated.prb_swsp,
        "PRB SWDP": evaluated.prb_swdp,
    }

    print()
    for name, value in metrics.items():
        print(f"{name}: {value}")

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(metrics, indent=4))
    print(f"\nSaved metrics to {json_path.resolve()}")

    result_txt_path = out_dir / "ap_result.txt"
    result_txt_path.write_text(
        "".join(f"{name}: {value}\n" for name, value in metrics.items())
    )

    hist_path = out_dir / "distance-histogram.png"
    fig_hist = plot_distance_histogram(results)
    fig_hist.savefig(hist_path, format="png")
    print(f"Saved distance histogram to {hist_path.resolve()}")

    pr_path = out_dir / "precision-recall.png"
    fig_pr = plot_precision_vs_recall(evaluated)
    fig_pr.savefig(pr_path, format="png")
    print(f"Saved precision-recall plot to {pr_path.resolve()}")


if __name__ == "__main__":
    main()

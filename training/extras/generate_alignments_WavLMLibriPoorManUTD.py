"""
Poor Man's UTD alignments for LibriSpeech: each train-clean-5 WavLM frame paired with its
nearest (cosine) frame from another speaker. Same method as training/align.py --pairs
poorman, for LibriSpeech keys ("apostle_1272_1272-128104-0000_000_000076-000108").

    python training/extras/generate_alignments_WavLMLibriPoorManUTD.py
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# <label>_<speaker>_<file_label>_<seg_idx>_<start>-<end>; labels can contain
# underscores, so match from the right.
KEY_RE = re.compile(
    r"^(?P<label>.+)_(?P<speaker>\d+)_(?P<recording>\d+-\d+-\d+)_"
    r"(?P<word_idx>\d+)_(?P<fr_start>\d+)-(?P<fr_end>\d+)$"
)

# Layer to search over (a layer_NN/ under --features-dir/<split>.samediff/).
LAYER = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LibriSpeech Poor Man's UTD nearest-neighbour frame alignments."
    )
    parser.add_argument(
        "--features-dir", type=Path, default=Path("data/Features/LibriSpeech_WavLM/Normalised"),
        help="directory containing one <split>.samediff/layer_NN subfolder per split "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("data/Information/LibriSpeech/Splits"),
        help="directory of librispeech_<split>_speakers.list files (default: %(default)s)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train-clean-5"],
        help="split names to pool together for the NN search (default: %(default)s -- "
        "the training split only, so the alignments used to train the cAE never see "
        "dev-clean frames)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="output directory for chunked alignment pickles "
        "(default: data/Alignments/LibriSpeech_WavLMPoorManUTD/<splits>.samediff.alignments/layer_NN, "
        "where <splits> is the joined split names, and NN is the LAYER constant near the top of this file)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="alignment entries per output pickle chunk (default: %(default)s)")
    parser.add_argument("--block-size", type=int, default=1024, help="query frames per NN-search batch (default: %(default)s)")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    return parser.parse_args()


def load_split_speakers(splits_dir: Path, splits: list[str]) -> dict[str, set[str]]:
    """Return {split_name: {speaker_id, ...}} from librispeech_<split>_speakers.list files."""
    split_speakers = {}
    for split in splits:
        fn = splits_dir / f"librispeech_{split}_speakers.list"
        if not fn.is_file():
            raise FileNotFoundError(f"Missing speaker list: {fn}")
        with fn.open(encoding="utf-8") as f:
            split_speakers[split] = {line.strip() for line in f if line.strip()}
    return split_speakers


def load_corpus(features_dir: Path, split_speakers: dict[str, set[str]]) -> dict:
    seg_keys: list[str] = []
    seg_frame_counts: list[int] = []
    seg_speaker: list[str] = []
    frame_arrays: list[np.ndarray] = []

    n_skipped_speaker = 0
    n_skipped_parse = 0

    for split, speakers in split_speakers.items():
        split_dir = features_dir / f"{split}.samediff" / f"layer_{LAYER:02d}"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing features split dir: {split_dir}")

        npz_files = sorted(split_dir.glob("*.npz"))
        for npz_file in tqdm(npz_files, desc=f"Loading {split}"):
            with np.load(npz_file, allow_pickle=False) as archive:
                for key in archive.files:
                    match = KEY_RE.match(key)
                    if not match:
                        n_skipped_parse += 1
                        continue

                    speaker = match.group("speaker")
                    if speaker not in speakers:
                        n_skipped_speaker += 1
                        continue

                    arr = archive[key]
                    seg_keys.append(key)
                    seg_frame_counts.append(arr.shape[0])
                    seg_speaker.append(speaker)
                    frame_arrays.append(arr.astype(np.float32, copy=False))

    if n_skipped_parse:
        print(f"  Skipped {n_skipped_parse} segment(s) with an unparseable key")
    if n_skipped_speaker:
        print(f"  Skipped {n_skipped_speaker} segment(s) whose speaker isn't in its split's speaker list")

    if not seg_keys:
        raise RuntimeError("No segments loaded -- check --features-dir / --splits-dir")

    frames = np.concatenate(frame_arrays, axis=0)

    speakers_sorted = sorted(set(seg_speaker))
    speaker_to_code = {spk: i for i, spk in enumerate(speakers_sorted)}
    seg_speaker_code = np.array([speaker_to_code[s] for s in seg_speaker], dtype=np.int64)

    seg_frame_counts_arr = np.array(seg_frame_counts, dtype=np.int64)
    speaker_per_frame = np.repeat(seg_speaker_code, seg_frame_counts_arr)
    seg_id_per_frame = np.repeat(np.arange(len(seg_keys), dtype=np.int64), seg_frame_counts_arr)
    local_idx_per_frame = np.concatenate([np.arange(c, dtype=np.int64) for c in seg_frame_counts])

    print(f"  Loaded {frames.shape[0]} frames from {len(seg_keys)} segments, {len(speakers_sorted)} speakers")

    return {
        "frames": frames,
        "speaker_per_frame": speaker_per_frame,
        "seg_keys": seg_keys,
        "seg_id_per_frame": seg_id_per_frame,
        "local_idx_per_frame": local_idx_per_frame,
    }


def find_nearest_neighbours(frames: np.ndarray, speaker_per_frame: np.ndarray, device: str, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = frames.shape[0]

    feats = torch.from_numpy(frames).to(device)
    feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-8)
    speakers = torch.from_numpy(speaker_per_frame).to(device)

    nn_idx = np.empty(n, dtype=np.int64)
    nn_sim = np.empty(n, dtype=np.float32)

    for start in tqdm(range(0, n, block_size), desc="NN search"):
        end = min(start + block_size, n)

        sims = feats[start:end] @ feats.T
        same_speaker = speakers[start:end].unsqueeze(1) == speakers.unsqueeze(0)
        sims.masked_fill_(same_speaker, float("-inf"))

        best_sim, best_idx = sims.max(dim=1)
        nn_idx[start:end] = best_idx.cpu().numpy()
        nn_sim[start:end] = best_sim.cpu().numpy()

    return nn_idx, nn_sim


def build_alignment_entries(seg_keys, seg_id_per_frame, local_idx_per_frame, nn_idx, nn_sim) -> dict:
    n = len(seg_id_per_frame)
    n_digits = len(str(n - 1))
    entries = {}

    for i in range(n):
        j = int(nn_idx[i])
        query_key = seg_keys[seg_id_per_frame[i]]
        target_key = seg_keys[seg_id_per_frame[j]]
        entry_id = f"poormanutd_{i:0{n_digits}d}"

        entries[entry_id] = [{
            "pair": (query_key, target_key),
            "frame_pairs": np.array([[int(local_idx_per_frame[i]), int(local_idx_per_frame[j])]], dtype=np.int32),
            "similarity": float(nn_sim[i]),
        }]

    return entries


def save_alignments_chunked(output_dir: Path, entries: dict, chunk_size: int) -> None:
    print(f"Saving alignments to chunked pickle files in {output_dir} (chunk size: {chunk_size})")
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_ids = sorted(entries.keys())
    n_chunks = (len(sorted_ids) + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_ids = sorted_ids[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        chunk_data = {entry_id: entries[entry_id] for entry_id in chunk_ids}
        with (output_dir / f"alignments.chunk_{chunk_idx:04d}.pkl").open("wb") as f:
            pickle.dump(chunk_data, f)

    metadata = {
        "total_words": len(entries),
        "total_chunks": n_chunks,
        "chunk_size": chunk_size,
        "words_per_chunk": [len(sorted_ids[i * chunk_size:(i + 1) * chunk_size]) for i in range(n_chunks)],
    }
    with (output_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {n_chunks} alignment chunks to {output_dir}")
    print(f"  Total entries: {len(entries)}")


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        splits_name = "+".join(args.splits)
        output_dir = Path(
            f"data/Alignments/LibriSpeech_WavLMPoorManUTD/{splits_name}.samediff.alignments/layer_{LAYER:02d}"
        )

    print("\nLoading split speaker lists...")
    split_speakers = load_split_speakers(args.splits_dir, args.splits)
    for split, speakers in split_speakers.items():
        print(f"  {split}: {len(speakers)} speakers")

    print("\nLoading WavLM features...")
    corpus = load_corpus(args.features_dir, split_speakers)

    print("\nSearching for cross-speaker nearest neighbours...")
    nn_idx, nn_sim = find_nearest_neighbours(corpus["frames"], corpus["speaker_per_frame"], device, args.block_size)
    print(f"  Mean cosine similarity: {nn_sim.mean():.4f}  (min: {nn_sim.min():.4f}, max: {nn_sim.max():.4f})")

    print("\nBuilding alignment entries...")
    entries = build_alignment_entries(
        corpus["seg_keys"], corpus["seg_id_per_frame"], corpus["local_idx_per_frame"], nn_idx, nn_sim
    )

    save_alignments_chunked(output_dir, entries, args.chunk_size)
    print("\nDone.")


if __name__ == "__main__":
    main()

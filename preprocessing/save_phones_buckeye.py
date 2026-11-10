"""
Build phones.list for Buckeye from english.phn, in the same format as words.list:

    <label> <file_label> <start> <end>
    ah s0701a 5.3955 5.5255

SIL/SPN are dropped, as are phones shorter than --min-duration (default 0.06 s, three
WavLM frames; keeps about 55% of Buckeye phones) and phones not inside an utterance.

    python preprocessing/save_phones_buckeye.py
    python preprocessing/save_phones_buckeye.py --min-duration 0.10
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PHONES_ALIGNMENT_FN = REPO_ROOT / "data/Information/Buckeye/english.phn"
UTTERANCES_LIST_FN = REPO_ROOT / "data/Information/Buckeye/utterances.list"
PHONES_LIST_FN = REPO_ROOT / "data/Information/Buckeye/phones.list"

# Not phones: the forced aligner's silence and unintelligible-speech markers.
NON_PHONES = {"SIL", "SPN", "sil", "spn"}

# 0.06 s == 3 frames at WavLM's 50 Hz.
DEFAULT_MIN_PHONE_DURATION = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phones.list from Buckeye phone-level forced alignments.",
    )
    parser.add_argument(
        "--min-duration", type=float, default=DEFAULT_MIN_PHONE_DURATION,
        help="drop phone tokens shorter than this many seconds "
             "(default: %(default)s = 3 frames at 50 Hz)",
    )
    parser.add_argument(
        "--phones-list-fn", type=Path, default=PHONES_LIST_FN,
        help="output path (default: %(default)s)",
    )
    return parser.parse_args()


def load_segments(fn: Path) -> dict[str, list[tuple[float, float]]]:
    """file_label -> list of (start, end) for each utterance segment."""
    segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with fn.open() as f:
        for line in f:
            seg_id, start, end = line.split()
            file_label = seg_id.split("_")[1]  # s07_s0701a_000 -> s0701a
            segments[file_label].append((float(start), float(end)))
    for file_label in segments:
        segments[file_label].sort()
    return segments


def inside_a_segment(spans: list[tuple[float, float]], start: float, end: float) -> bool:
    """True if [start, end] lies entirely within one of the segments."""
    return any(seg_start <= start and end <= seg_end for seg_start, seg_end in spans)


def main() -> None:
    args = parse_args()

    if not PHONES_ALIGNMENT_FN.is_file():
        raise FileNotFoundError(f"Phone alignments not found: {PHONES_ALIGNMENT_FN}")
    if not UTTERANCES_LIST_FN.is_file():
        raise FileNotFoundError(
            f"Utterance list not found: {UTTERANCES_LIST_FN}\n"
            f"Please run:\n"
            f"  python preprocessing/the utterance/word-list preprocessing"
        )

    segments = load_segments(UTTERANCES_LIST_FN)
    print(f"Loaded segments for {len(segments)} recordings from {UTTERANCES_LIST_FN}")

    kept: list[tuple[str, str, float, float]] = []
    counts = Counter()
    n_total = n_nonphone = n_brief = n_outside = 0

    with PHONES_ALIGNMENT_FN.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            file_label, start, end, label = parts
            n_total += 1

            if label in NON_PHONES:
                n_nonphone += 1
                continue

            start, end = float(start), float(end)
            # Round to the millisecond to avoid floating-point misses at the boundary.
            if round(end - start, 3) < args.min_duration:
                n_brief += 1
                continue

            if not inside_a_segment(segments.get(file_label, []), start, end):
                n_outside += 1
                continue

            kept.append((label, file_label, start, end))
            counts[label] += 1

    args.phones_list_fn.parent.mkdir(parents=True, exist_ok=True)
    with args.phones_list_fn.open("w", encoding="utf-8") as f:
        for label, file_label, start, end in kept:
            f.write(f"{label} {file_label} {start:.4f} {end:.4f}\n")

    print(f"\nRead {n_total:,} phone alignment lines")
    print(f"  dropped {n_nonphone:,} SIL/SPN")
    print(f"  dropped {n_brief:,} shorter than {args.min_duration}s")
    print(f"  dropped {n_outside:,} not inside any utterance segment")
    print(f"Wrote {len(kept):,} phone tokens ({len(counts)} types) to {args.phones_list_fn}")
    print("\nMost frequent types:")
    for label, n in counts.most_common(10):
        print(f"  {label:5} {n:7,}")


if __name__ == "__main__":
    main()

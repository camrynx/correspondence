"""
Convert a tree of Montreal Forced Aligner TextGrids (<speaker>/<chapter>/<utt_id>.TextGrid)
into a .wrd file for save_utterances_and_words_librispeech.py, one row per interval of
the "words" tier:

    <utt_id> <start> <end> <token>

Empty intervals become SIL, and spn/<unk> become <unk>.

    python preprocessing/textgrid_to_wrd.py train-clean-100-FA \
        data/Information/LibriSpeech/librispeech_train-clean-100.wrd

    # only utterances with audio in a given subset
    python preprocessing/textgrid_to_wrd.py train-clean-100-FA \
        data/Information/LibriSpeech/librispeech_train-clean-5.wrd \
        --filter-audio-dir data/Datasets/LibriSpeech/train-clean-5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


# Header of each IntervalTier, used to cut out the "words" tier.
TIER_RE = re.compile(
    r'item \[\d+\]:\s*class = "IntervalTier"\s*name = "([^"]*)"'
)

# One interval block: xmin / xmax / text, in that fixed order.
INTERVAL_RE = re.compile(
    r'xmin = ([\-0-9.]+)\s*xmax = ([\-0-9.]+)\s*text = "([^"]*)"'
)


def parse_words_tier(text: str) -> list[tuple[float, float, str]]:
    """Return [(xmin, xmax, token), ...] from a TextGrid's "words" tier, in
    file (time) order. Empty list if the file has no tier named "words"."""

    tiers = list(TIER_RE.finditer(text))
    for i, m in enumerate(tiers):
        if m.group(1) != "words":
            continue
        start = m.end()
        end = tiers[i + 1].start() if i + 1 < len(tiers) else len(text)
        segment = text[start:end]
        return [
            (float(xmin), float(xmax), token)
            for xmin, xmax, token in INTERVAL_RE.findall(segment)
        ]
    return []


def normalise_token(token: str) -> str:
    """Map MFA's silence/OOV markers onto this repo's .wrd convention."""
    if token == "":
        return "SIL"
    if token.lower() in ("spn", "<unk>"):
        return "<unk>"
    return token


def load_filter_ids(filter_audio_dir: Path) -> set[str]:
    """Utterance IDs (.flac stems) present under a LibriSpeech subset directory."""
    return {fn.stem for fn in filter_audio_dir.rglob("*.flac")}


def convert(root: Path, out_fn: Path, allowed_ids: set[str] | None) -> None:
    tg_files = sorted(root.rglob("*.TextGrid"))
    if not tg_files:
        raise SystemExit(f"Error: no .TextGrid files found under {root}")

    n_written = 0
    n_files_written = 0
    n_skipped_filter = 0
    n_skipped_no_words_tier = 0

    out_fn.parent.mkdir(parents=True, exist_ok=True)
    with out_fn.open("w", encoding="utf-8") as out:
        for tg_fn in tg_files:
            utt_id = tg_fn.stem

            if allowed_ids is not None and utt_id not in allowed_ids:
                n_skipped_filter += 1
                continue

            intervals = parse_words_tier(tg_fn.read_text(encoding="utf-8"))
            if not intervals:
                n_skipped_no_words_tier += 1
                continue

            for xmin, xmax, token in intervals:
                out.write(f"{utt_id} {xmin:.4f} {xmax:.4f} {normalise_token(token)}\n")
                n_written += 1
            n_files_written += 1

    print(
        f"Read {len(tg_files)} TextGrid file(s) under {root}\n"
        f"  wrote {n_written} rows from {n_files_written} utterances to {out_fn}\n"
        f"  ({n_skipped_filter} skipped by --filter-audio-dir, "
        f"{n_skipped_no_words_tier} skipped - no \"words\" tier found)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert a tree of MFA TextGrid files into a flat .wrd file.'
    )
    parser.add_argument("textgrid_dir", type=Path, help="Root directory to search for .TextGrid files")
    parser.add_argument("out_fn", type=Path, help="Path to write the combined .wrd file to")
    parser.add_argument(
        "--filter-audio-dir", type=Path, default=None,
        help="Only convert utterances whose .flac already exists somewhere under this "
        "directory (searched recursively) - e.g. a LibriSpeech subset directory such "
        "as data/Datasets/LibriSpeech/train-clean-5. Omit to convert every TextGrid "
        "found under textgrid_dir.",
    )
    args = parser.parse_args()

    allowed_ids = None
    if args.filter_audio_dir is not None:
        allowed_ids = load_filter_ids(args.filter_audio_dir)
        if not allowed_ids:
            raise SystemExit(f"Error: no .flac files found under {args.filter_audio_dir}")
        print(f"Filtering to {len(allowed_ids)} utterance(s) with audio under {args.filter_audio_dir}")

    convert(args.textgrid_dir, args.out_fn, allowed_ids)


if __name__ == "__main__":
    main()

"""Convert a segment-keyed ABX item file to whole recordings.

build_abx_item_file.py keys rows by utterance with utterance-relative times:

    1272_1272-128104-0000_000 0.0000 0.1800 IH M S 1272

This rewrites #file as the recording and adds the utterance's start time:

    1272-128104-0000 0.5000 0.6800 IH M S 1272

so the items can be scored on features from one forward pass per recording
(extract_abx_phone_features.py --raw --whole-recording). Utterances are short (median
about 1.6 s), so per-utterance features lose context; on same-different, whole-recording
extraction gave 0.926 AP against 0.628.

    python evaluation/rebase_item_file.py \
        data/Information/LibriSpeech/abx_items/triphone-dev-clean.item \
        data/Information/LibriSpeech/abx_items/triphone-whole-dev-clean.item
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UTTERANCES = REPO_ROOT / "data/Information/LibriSpeech/utterances.list"


def load_segments(utterances_fn: Path) -> dict[str, tuple[str, float]]:
    """seg_id -> (recording id, segment start time in seconds)."""

    segments: dict[str, tuple[str, float]] = {}
    with utterances_fn.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seg_id, start, _end = line.split()
            # "1272_1272-128104-0000_000" -> "1272-128104-0000"
            recording = seg_id.split("_", 1)[1].rsplit("_", 1)[0]
            segments[seg_id] = (recording, float(start))
    return segments


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Rebase a segment-keyed item file onto whole-recording features."
    )
    parser.add_argument("item_file", help="segment-keyed item file to convert")
    parser.add_argument("out_file", help="where to write the recording-keyed item file")
    parser.add_argument(
        "--utterances",
        default=str(DEFAULT_UTTERANCES),
        help=f"utterances.list giving each segment's start time (default: {DEFAULT_UTTERANCES})",
    )
    return parser.parse_args()


def main() -> None:
    """Convert one item file from segment-relative to recording-absolute."""

    args = parse_args()
    item_fn, out_fn = Path(args.item_file), Path(args.out_file)
    if not item_fn.is_file():
        print(f"Error: item file not found at {item_fn}", file=sys.stderr)
        raise SystemExit(1)

    segments = load_segments(Path(args.utterances))

    lines = item_fn.read_text(encoding="utf-8").splitlines()
    header, body = lines[0], [l for l in lines[1:] if l.strip()]
    if not header.startswith("#file"):
        print(f"Error: {item_fn} has no '#file ...' header line", file=sys.stderr)
        raise SystemExit(1)

    out = [header]
    missing: set[str] = set()
    for line in body:
        seg_id, onset, offset, phone, prev_phone, next_phone, speaker = line.split()
        if seg_id not in segments:
            missing.add(seg_id)
            continue
        recording, seg_start = segments[seg_id]
        out.append(
            f"{recording} {float(onset) + seg_start:.4f} {float(offset) + seg_start:.4f} "
            f"{phone} {prev_phone} {next_phone} {speaker}"
        )

    out_fn.parent.mkdir(parents=True, exist_ok=True)
    out_fn.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"Wrote {len(out) - 1} rows to {out_fn}")
    if missing:
        print(
            f"Warning: {len(missing)} segment ids had no utterances.list entry and were "
            f"dropped (e.g. {sorted(missing)[:3]})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

"""Build a triphone ABX item file from phone-level forced alignments.

Columns: #file onset offset #phone prev-phone next-phone speaker, as in the triphone ABX
task of Renshaw et al. (Interspeech 2015). Every phone with a phone on both sides within
the same utterance (utterances.list) gives one row spanning the previous phone's start
to the next phone's end. #file is the utterance ID (e.g. s07_s0701a_000) and times are
relative to the utterance start, matching extract_abx_phone_features.py.

Phones that don't make it into a row are listed in <subset>.dropped_phones.txt, as
utterance_edge (expected) or outside_any_segment (the phone and utterance lists disagree).

    python evaluation/build_abx_item_file.py --corpus buckeye --subset zs
    python evaluation/build_abx_item_file.py --corpus xitsonga --subset test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

# Silence and spoken-noise labels, which can't be triphone neighbours. Compared
# case-insensitively: LibriSpeech writes "SIL" but "spn".
NON_PHONES = {"SIL", "SPN"}


def is_non_phone(label: str) -> bool:
    """Whether a forced-alignment label is silence/noise rather than a phone."""

    return label.upper() in NON_PHONES

# Tolerance (s) when matching phone spans to utterance boundaries.
EPS = 1e-3


CORPORA = {
    "buckeye": {
        "phn_fn": REPO_ROOT / "data/Information/Buckeye/english.phn",
        "utterances_fn": REPO_ROOT / "data/Information/Buckeye/utterances.list",
        "splits_dir": REPO_ROOT / "data/Information/Buckeye/Splits",
        "speakers_pattern": "buckeye_{subset}_speakers.list",
        "out_dir": REPO_ROOT / "data/Information/Buckeye/abx_items",
        # utterances.list seg_id: "<speaker>_<file_label>_<seg_idx>", and
        # file_label (e.g. "s0701a") is the key used in english.phn.
        "file_label_of_seg": lambda seg_id: seg_id.split("_")[1],
        "normalise_speaker": lambda s: s,
    },
    "xitsonga": {
        "phn_fn": REPO_ROOT / "data/Information/Buckeye_tsonga/xitsonga.phn",
        "utterances_fn": REPO_ROOT / "data/Information/Buckeye_tsonga/utterances.list",
        "splits_dir": REPO_ROOT / "data/Information/Buckeye_tsonga/Splits",
        "speakers_pattern": "tsong{subset}_speakers.list",
        "out_dir": REPO_ROOT / "data/Information/Buckeye_tsonga/abx_items",
        # seg_id is <speaker>_<file_label>_<seg_idx>; file_label (e.g. nchlt_tso_001m_0007)
        # is the key in xitsonga.phn.
        "file_label_of_seg": lambda seg_id: seg_id.split("_", 1)[1].rsplit("_", 1)[0],
        # Splits files list speakers as "nchlt-tso-001m"; seg_id speaker
        # field (and xitsonga.phn's embedded speaker) uses "001m".
        "normalise_speaker": lambda s: s.split("-")[-1],
    },
    "librispeech": {
        "phn_fn": REPO_ROOT / "data/Information/LibriSpeech/librispeech_dev-clean.phn",
        "utterances_fn": REPO_ROOT / "data/Information/LibriSpeech/utterances.list",
        "splits_dir": REPO_ROOT / "data/Information/LibriSpeech/Splits",
        "speakers_pattern": "librispeech_{subset}_speakers.list",
        "out_dir": REPO_ROOT / "data/Information/LibriSpeech/abx_items",
        # seg_id is <speaker>_<file_label>_<seg_idx>; file_label (e.g. 1272-128104-0000)
        # is the key in librispeech_dev-clean.phn.
        "file_label_of_seg": lambda seg_id: seg_id.split("_")[1],
        "normalise_speaker": lambda s: s,
    },
    "zs17french": {
        "phn_fn": REPO_ROOT / "data/Information/ZS17French/french.phn",
        "utterances_fn": REPO_ROOT / "data/Information/ZS17French/utterances.list",
        "splits_dir": REPO_ROOT / "data/Information/ZS17French/Splits",
        "speakers_pattern": "zs17french{subset}_speakers.list",
        "out_dir": REPO_ROOT / "data/Information/ZS17French/abx_items",
        # One recording per speaker: seg_id is <speaker>_<seg_idx> ("F01R_000"), and
        # french.phn names the recording with the underscore ("F01_R").
        "file_label_of_seg": lambda seg_id: (
            lambda spk: spk[:-1] + "_" + spk[-1]
        )(seg_id.rsplit("_", 1)[0]),
        "normalise_speaker": lambda s: s,
    },
}


def available_subsets(splits_dir: Path, pattern: str) -> list[str]:
    """Subsets that have a speakers list (for the error message)."""

    prefix, suffix = pattern.split("{subset}")
    names = []
    for fn in sorted(splits_dir.glob(f"{prefix}*{suffix}")):
        if fn.name.startswith(prefix) and fn.name.endswith(suffix):
            names.append(fn.name[len(prefix): len(fn.name) - len(suffix)])
    return names


def speaker_of_seg(seg_id: str) -> str:
    """Speaker: the first field of the segment ID."""

    return seg_id.split("_", 1)[0]


def load_phones(phn_fn: Path) -> dict[str, list[tuple[float, float, str]]]:
    """A .phn file as file_label -> [(start, end, phone), ...] in time order (SIL/SPN included)."""

    phones_by_file: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with phn_fn.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            file_label, start, end, label = line.split()
            phones_by_file[file_label].append((float(start), float(end), label))
    return phones_by_file


def load_segments(utterances_fn: Path) -> list[tuple[str, float, float]]:
    """Parse utterances.list into a list of (seg_id, start, end)."""

    segments = []
    with utterances_fn.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_id, start, end = line.split()
            segments.append((seg_id, float(start), float(end)))
    return segments


def load_subset_speakers(splits_dir: Path, pattern: str, subset: str, normalise) -> set[str]:
    """Load the speaker IDs for a subset, normalised to match seg_id's speaker field."""

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
        return {normalise(line.strip()) for line in f if line.strip()}


def triphone_rows_for_segment(
    seg_id: str,
    seg_start: float,
    seg_end: float,
    seg_phones: list[tuple[float, float, str]],
) -> tuple[list[tuple[str, float, float, str, str, str, str]], list[tuple[float, float, str]]]:
    """Item rows for one utterance's phones (sorted by start time).

    Every phone except the first and last gets a row, with times relative to seg_start;
    the first and last are returned as dropped utterance-edge phones.
    """

    rows = []
    speaker = speaker_of_seg(seg_id)
    for i in range(1, len(seg_phones) - 1):
        prev_start, _prev_end, prev_label = seg_phones[i - 1]
        _cur_start, _cur_end, cur_label = seg_phones[i]
        _next_start, next_end, next_label = seg_phones[i + 1]

        onset = prev_start - seg_start
        offset = next_end - seg_start
        if offset <= onset:
            continue

        rows.append((seg_id, onset, offset, cur_label, prev_label, next_label, speaker))

    edge_dropped = []
    if seg_phones:
        edge_dropped.append(seg_phones[0])
        if len(seg_phones) > 1:
            edge_dropped.append(seg_phones[-1])

    return rows, edge_dropped


def build_item_file(corpus: str, subset: str) -> Path:
    """Build and write the triphone item file for one corpus/subset."""

    config = CORPORA[corpus]

    print(f"Loading phone alignments from {config['phn_fn']}...")
    phones_by_file = load_phones(config["phn_fn"])

    print(f"Loading segments from {config['utterances_fn']}...")
    segments = load_segments(config["utterances_fn"])

    speakers = load_subset_speakers(
        config["splits_dir"], config["speakers_pattern"], subset, config["normalise_speaker"]
    )
    print(f"Subset '{subset}' has {len(speakers)} speakers")

    file_label_of_seg = config["file_label_of_seg"]

    # Group utterances by recording, so phones outside every utterance can be reported.
    segments_by_file: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for seg_id, seg_start, seg_end in segments:
        if speaker_of_seg(seg_id) not in speakers:
            continue
        segments_by_file[file_label_of_seg(seg_id)].append((seg_id, seg_start, seg_end))

    rows: list[tuple[str, float, float, str, str, str, str]] = []
    # (file_label, start, end, phone, reason)
    dropped: list[tuple[str, float, float, str, str]] = []
    n_segments_used = 0

    for file_label, segs in segments_by_file.items():
        real_phones = [
            (start, end, label)
            for start, end, label in phones_by_file.get(file_label, [])
            if not is_non_phone(label)
        ]
        claimed = [False] * len(real_phones)

        for seg_id, seg_start, seg_end in segs:
            seg_indices = sorted(
                i
                for i, (start, end, _label) in enumerate(real_phones)
                if start >= seg_start - EPS and end <= seg_end + EPS
            )
            for i in seg_indices:
                claimed[i] = True
            seg_phones = [real_phones[i] for i in seg_indices]

            seg_rows, edge_dropped = triphone_rows_for_segment(seg_id, seg_start, seg_end, seg_phones)
            if seg_rows:
                n_segments_used += 1
            rows.extend(seg_rows)
            dropped.extend((file_label, start, end, label, "utterance_edge") for start, end, label in edge_dropped)

        # Phones that fell inside none of the recording's utterances.
        dropped.extend(
            (file_label, start, end, label, "outside_any_segment")
            for is_claimed, (start, end, label) in zip(claimed, real_phones)
            if not is_claimed
        )

    out_dir = config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = out_dir / f"{subset}.item"

    with out_fn.open("w", encoding="utf-8") as f:
        f.write("#file onset offset #phone prev-phone next-phone speaker\n")
        for seg_id, onset, offset, phone, prev_phone, next_phone, speaker in rows:
            f.write(
                f"{seg_id} {onset:.4f} {offset:.4f} {phone} {prev_phone} {next_phone} {speaker}\n"
            )

    dropped_fn = out_dir / f"{subset}.dropped_phones.txt"
    with dropped_fn.open("w", encoding="utf-8") as f:
        for file_label, start, end, label, reason in dropped:
            f.write(f"{label} {file_label} {start:.4f}-{end:.4f} {reason}\n")

    n_edge = sum(1 for row in dropped if row[4] == "utterance_edge")
    n_outside = len(dropped) - n_edge

    phone_types = {row[3] for row in rows}
    print(
        f"Wrote {len(rows)} triphone rows ({len(phone_types)} phone types) "
        f"from {n_segments_used} segments to {out_fn}"
    )
    print(
        f"Wrote {len(dropped)} dropped phones to {dropped_fn} "
        f"({n_edge} at utterance edges [expected], {n_outside} outside any segment "
        f"[possible english.phn/utterances.list boundary mismatch - worth checking if this is large])"
    )
    return out_fn


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Build a triphone-context ABX item file from forced alignments."
    )
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument(
        "--subset",
        required=True,
        help="any subset with a Splits/*_speakers.list file, e.g. devpart1/devpart2/zs "
        "for buckeye, dev/test for xitsonga",
    )
    return parser.parse_args()


def main() -> None:
    """Build the triphone item file for the requested corpus/subset."""

    args = parse_args()
    build_item_file(args.corpus, args.subset)


if __name__ == "__main__":
    main()

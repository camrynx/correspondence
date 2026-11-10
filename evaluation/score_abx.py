"""Score a directory of ABX features against an item file with fastabx.

The directory holds one .npy per utterance, as written by extract_abx_phone_features.py;
the item file comes from build_abx_item_file.py. Writes abx_result.txt under results/abx/.

    python evaluation/score_abx.py data/Features/Buckeye_MFCCs/zs.abx_cAE \
        data/Information/Buckeye/abx_items/zs.item

    python evaluation/score_abx.py data/Features/Xitsonga_HuBERT/Normalised/test.abx_layer24 \
        data/Information/Buckeye_tsonga/abx_items/test.item --speaker within
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch

# Shares the scoring helpers in run_abx_eval.py (same directory).
import run_abx_eval as abx_common

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Score a features directory against an ABX item file "
        "(triphone-context, ZeroSpeech-style)."
    )
    parser.add_argument(
        "features_dir",
        help="Directory of per-segment .npy files, e.g. as written by "
        "extract_abx_phone_features.py (raw, cAE, or AE; any feature type, "
        "corpus, or UTD/DUSTED/PoorManUTD variant).",
    )
    parser.add_argument(
        "item_file",
        help="Item file naming the #file keys to score, e.g. "
        "data/Information/Buckeye/abx_items/zs.item (see build_abx_item_file.py).",
    )
    parser.add_argument(
        # Repeatable flag rather than nargs="+", which would swallow the positionals.
        "--speaker",
        action="append",
        choices=["within", "across"],
        help="Speaker condition, repeatable: 'within' pairs A/B/X from the same "
        "speaker, 'across' takes X from a different speaker than A/B (default: "
        "both - pass e.g. '--speaker within' to select just one).",
    )
    parser.add_argument(
        "--context",
        action="append",
        choices=["within", "any"],
        help="Context condition, repeatable: 'within' requires A/B/X to share "
        "prev-phone and next-phone, 'any' matches on the centre phone alone. "
        "Defaults to 'within' only - every item file this script scores is "
        "triphone-keyed (prev+centre+next), and fastabx recommends 'within' for "
        "those (its neighbours are part of the compared audio, so 'any' would "
        "pair spans whose neighbours differ freely).",
    )
    parser.add_argument(
        "--distance",
        default="angular",
        help="Frame-level distance fastabx uses inside its DTW (default: angular, "
        "the ZeroSpeech default).",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=None,
        help="Feature frame rate in Hz; inferred from the features path if not "
        "given (100 for MFCC, 50 for WavLM/HuBERT).",
    )
    parser.add_argument(
        "--max-size-group",
        type=int,
        default=abx_common.MAX_SIZE_GROUP,
        help=f"Max instances of A/B/X per cell (default {abx_common.MAX_SIZE_GROUP}, "
        f"the standard ZeroSpeech default).",
    )
    parser.add_argument(
        "--max-x-across",
        type=int,
        default=abx_common.MAX_X_ACROSS,
        help=f"Max distinct X speakers per cell, for --speaker across only "
        f"(default {abx_common.MAX_X_ACROSS}, the standard ZeroSpeech default).",
    )
    return parser.parse_args()


def main() -> None:
    """Score every requested speaker/context condition and write abx_result.txt."""

    args = parse_args()
    features_dir = Path(args.features_dir).resolve()
    item_fn = Path(args.item_file).resolve()

    if not features_dir.is_dir():
        print(f"Error: features directory not found at {features_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not item_fn.is_file():
        print(f"Error: item file not found at {item_fn}", file=sys.stderr)
        raise SystemExit(1)

    # Every file the item file names must exist (catches wrongly keyed features early).
    needed = abx_common.item_file_keys(item_fn)
    have = {p.stem for p in features_dir.glob("*.npy")}
    missing = needed - have
    if missing:
        print(
            f"Error: {features_dir} is missing {len(missing)} of {len(needed)} "
            f"features the item file needs (e.g. {sorted(missing)[:3]}) - wrong "
            f"directory, or extraction didn't finish",
            file=sys.stderr,
        )
        raise SystemExit(1)

    frequency = args.frequency or abx_common.infer_frequency(features_dir)
    # Defaults applied here, since action="append" would add to a default list.
    speakers = list(dict.fromkeys(args.speaker or ["within", "across"]))
    contexts = list(dict.fromkeys(args.context or ["within"]))

    print(f"Features:  {features_dir}  ({len(have)} files)")
    print(f"Item file: {item_fn}  ({len(needed)} keys)")
    print(f"Frequency: {frequency} Hz")
    conditions = [f"{s}-speaker/{c}-context" for s in speakers for c in contexts]
    print(f"Scoring:   {', '.join(conditions)}")

    scored: dict[tuple[str, str], float] = {}
    for speaker in speakers:
        for context in contexts:
            print(f"\n{speaker}-speaker, {context}-context...")
            kwargs = dict(
                item=item_fn,
                root=features_dir,
                max_size_group=args.max_size_group,
                speaker=speaker,
                context=context,
                distance=args.distance,
                frequency=frequency,
                feature_maker=abx_common.load_npy,
                extension=".npy",
            )
            # max_x_across only means anything when X comes from another
            # speaker; fastabx rejects it in the within-speaker task.
            if speaker == "across":
                kwargs["max_x_across"] = args.max_x_across

            error = abx_common.zerospeech_abx_with_cpu_fallback(**kwargs)

            # Free fastabx's GPU tensors between conditions.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  -> {error * 100:.2f}% error")
            scored[(speaker, context)] = error * 100

    result_dir = abx_common.distances_dir_for(features_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_fn = result_dir / "abx_result.txt"
    abx_common.merge_result_file(result_fn, "triphone", scored)
    print(f"\nWrote {result_fn}")
    print(result_fn.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()

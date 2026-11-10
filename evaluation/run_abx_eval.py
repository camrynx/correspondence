"""ABX over a grid of conditions: speaker (within/across) x context (within/any) x unit
(triphone/phoneme) x layer. Used for the LibriSpeech HuBERT experiments.

    --unit      triphone | phoneme | triphone_whole
    --speaker   within | across
    --context   within | any
    --layers    N [N ...]

The two LibriSpeech item files are keyed differently: triphone-dev-clean.item by
utterance segment (segment-relative times), phoneme-dev-clean.item by recording
(recording-relative times). Triphone units get segment-keyed features (one forward pass
per segment); phoneme units get one forward pass per recording. Features are extracted
with extract_abx_phone_features.extract_wavlm() and cached in the same place as its
--raw output, so each layer is extracted once (--force-extract to redo).

    python evaluation/run_abx_eval.py --unit phoneme --layers 12 --speaker within across --context any
    python evaluation/run_abx_eval.py --unit triphone --layers 6 12 18 24
    python evaluation/run_abx_eval.py --unit phoneme --layers 6 12 18 24 --extract-only
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import fastabx.dataset
from fastabx import zerospeech_abx


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
FEATURES_ROOT = REPO_ROOT / "data" / "Features"
DISTANCES_ROOT = REPO_ROOT / "results" / "abx"
ITEMS_DIR = REPO_ROOT / "data/Information/LibriSpeech/abx_items"

MAX_SIZE_GROUP = 10  # standard ZeroSpeech ABX default
MAX_X_ACROSS = 5  # standard ZeroSpeech ABX default, only used for speaker="across"

# The --unit choices: whole_recording is passed to extract_wavlm(); dir_suffix keeps the
# feature caches apart.
UNITS = {
    "triphone": dict(
        item_file=ITEMS_DIR / "triphone-dev-clean.item",
        whole_recording=False,
        dir_suffix="",
        keying="utterances.list segment id, segment-relative onsets",
        # A triphone item spans the previous, centre and next phones, so only
        # context="within" makes sense (as fastabx recommends).
        default_contexts=("within",),
        sound_contexts={"within"},
    ),
    # The triphone task on whole-recording features (items rebased with
    # rebase_item_file.py). Segment-keyed features cut the model's context at pauses;
    # this shares the "phoneme" feature cache.
    "triphone-whole": dict(
        item_file=ITEMS_DIR / "triphone-whole-dev-clean.item",
        whole_recording=True,
        dir_suffix="_whole",
        keying="whole-recording id, recording-relative onsets (rebased triphone)",
        default_contexts=("within",),
        sound_contexts={"within"},
    ),
    "phoneme": dict(
        item_file=ITEMS_DIR / "phoneme-dev-clean.item",
        whole_recording=True,
        dir_suffix="_whole",
        keying="whole-recording id, recording-relative onsets",
        # A phoneme item spans the centre phone alone, so both context modes
        # are meaningful: the full ZeroSpeech 2021 2x2 speaker x context grid.
        default_contexts=("within", "any"),
        sound_contexts={"within", "any"},
    ),
}


# --------------------------------- FASTABX ------------------------------------

# fastabx always builds its tensors on default_device() (the GPU if there is one). On a
# CUDA out-of-memory error we switch it to the CPU for the rest of the run.
_forced_cpu = False


def zerospeech_abx_with_cpu_fallback(**kwargs: object) -> float:
    """Run zerospeech_abx, falling back to CPU on a CUDA OOM."""

    global _forced_cpu
    if _forced_cpu:
        return zerospeech_abx(**kwargs)
    try:
        return zerospeech_abx(**kwargs)
    except torch.OutOfMemoryError:
        print("CUDA out of memory, retrying this condition on CPU...", file=sys.stderr)
        _forced_cpu = True
        gc.collect()
        torch.cuda.empty_cache()
        fastabx.dataset.default_device = lambda: torch.device("cpu")
        return zerospeech_abx(**kwargs)


def load_npy(path: str | Path) -> torch.Tensor:
    """Load a per-segment .npy feature file as a torch.Tensor (fastabx feature_maker)."""

    return torch.from_numpy(np.load(path))


# ------------------------------ PATHS / ITEM FILES -----------------------------

def infer_frequency(features_dir: Path) -> int:
    """Frame rate from the path: 100 Hz for MFCCs, 50 Hz otherwise."""

    name = str(features_dir).lower()
    if "mfcc" in name:
        return 100
    if "wavlm" in name or "hubert" in name or "spidr" in name:
        return 50
    raise ValueError(
        f"Could not infer feature frequency from {features_dir}; pass --frequency explicitly"
    )


def distances_dir_for(features_dir: Path) -> Path:
    """results/abx/<features path relative to data/Features>."""

    resolved = features_dir.resolve()
    try:
        subpath = resolved.relative_to(FEATURES_ROOT)
    except ValueError:
        subpath = Path(features_dir.name)
    return DISTANCES_ROOT / subpath


def item_file_keys(item_fn: Path) -> set[str]:
    """The distinct #file values in an item file, i.e. the feature files fastabx will load."""

    keys: set[str] = set()
    with item_fn.open(encoding="utf-8") as f:
        next(f, None)  # "#file onset offset #phone prev-phone next-phone speaker"
        for line in f:
            if line.strip():
                keys.add(line.split(maxsplit=1)[0])
    return keys


# ------------------------------ FEATURE EXTRACTION -----------------------------

_extractor = None


def load_extractor():
    """Import extract_abx_phone_features (cheap: models load only when extracting)."""

    global _extractor
    if _extractor is None:
        sys.path.insert(0, str(EVAL_DIR))
        import extract_abx_phone_features  # noqa: PLC0415

        _extractor = extract_abx_phone_features
    return _extractor


def model_tag_for(model_name: str | None) -> str:
    """Directory tag for a --model override (e.g. "_hubert-base-ls960"); empty by default.

    Different checkpoints can have different hidden sizes, so they need separate caches.
    """

    if not model_name:
        return ""
    slug = model_name.rsplit("/", 1)[-1]
    return "_" + re.sub(r"[^0-9A-Za-z._-]+", "-", slug)


def features_dir_for(config: dict, subset: str, unit: str, layer: int, model_name: str | None = None) -> Path:
    """Cache directory for one (unit, layer, model); the default model shares
    extract_abx_phone_features.py --raw's paths."""

    return (
        FEATURES_ROOT
        / config["base_feature_set"]
        / config["normalised_subdir"]
        / f"{subset}.abx{UNITS[unit]['dir_suffix']}{model_tag_for(model_name)}_layer{layer:02d}"
    )


def ensure_features(
    corpus: str,
    feature_type: str,
    subset: str,
    unit: str,
    layer: int,
    config: dict,
    item_fn: Path,
    force: bool,
    model_name: str | None = None,
) -> Path:
    """Features directory for (unit, layer), extracting it first if the cache is incomplete."""

    out_dir = features_dir_for(config, subset, unit, layer, model_name)
    needed = item_file_keys(item_fn)
    have = {p.stem for p in out_dir.glob("*.npy")} if out_dir.is_dir() else set()
    missing = needed - have

    if not missing and not force:
        print(f"  Reusing cached features ({len(have)} files) at {out_dir}")
        return out_dir
    if out_dir.is_dir() and not force:
        print(
            f"  Cached features at {out_dir} are incomplete "
            f"({len(missing)} of {len(needed)} item-file keys have no .npy) - re-extracting"
        )

    ext = load_extractor()

    # extract_wavlm reads the model name from the module's {prefix}_MODEL_NAME, so
    # override it there. The framing constants are the same for base and large HuBERT.
    if model_name:
        module = ext._neural_module(feature_type, corpus)
        setattr(module, f"{config['const_prefix']}_MODEL_NAME", model_name)
        print(f"  Model override: {model_name}")

    print(
        f"  Extracting layer {layer} "
        f"({'whole recordings' if UNITS[unit]['whole_recording'] else 'utterances.list segments'})..."
    )
    features = ext.extract_wavlm(
        corpus,
        feature_type,
        subset,
        config,
        layer=layer,
        whole_recording=UNITS[unit]["whole_recording"],
    )

    # As in extract_abx_phone_features.py --raw: whole-recording features and sets with
    # normalise=False are not normalised.
    if config.get("normalise", True) and not UNITS[unit]["whole_recording"]:
        stats_fn = ext.norm_stats_path(config, layer)
        if not stats_fn.is_file():
            raise SystemExit(
                f"Error: no normalisation stats at {stats_fn} - run the matching "
                f"features/normalise*.py with --layer {layer} first"
            )
        norm_stats = torch.load(stats_fn, map_location="cpu")
        mean = norm_stats["mean"].numpy().astype(np.float32)
        std = norm_stats["std"].numpy().astype(np.float32)
        features = {k: (arr.astype(np.float32) - mean) / std for k, arr in features.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, frames in tqdm(sorted(features.items()), desc="  Saving features"):
        np.save(out_dir / f"{key}.npy", frames.astype(np.float32))
    print(f"  Wrote {len(features)} feature files to {out_dir}")

    still_missing = needed - {p.stem for p in out_dir.glob("*.npy")}
    if still_missing:
        print(
            f"  Warning: {len(still_missing)} of {len(needed)} item-file keys still have no "
            f"feature file (e.g. {sorted(still_missing)[:3]}); fastabx will fail on these",
            file=sys.stderr,
        )

    # extract_wavlm holds every segment's frames in one dict (several GB for a
    # full subset at 1024 dims); drop it before the next layer's pass.
    del features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


# ----------------------------------- RESULTS -----------------------------------

SUMMARY_HEADER = "unit\tlayer\tspeaker\tcontext\tdistance\terror_pct\tfeatures_dir\titem_file\ttimestamp"

RESULT_LINE_RE = re.compile(
    r"^(?P<speaker>\w+)-speaker, (?P<context>\w+)-context, (?P<unit>\w+) ABX error: "
    r"(?P<error>[\d.]+)%$"
)


def merge_result_file(result_fn: Path, unit: str, scored: dict[tuple[str, str], float]) -> None:
    """Write the result file for one (unit, layer), keeping conditions from earlier runs."""

    merged: dict[tuple[str, str], float] = {}
    if result_fn.is_file():
        for line in result_fn.read_text(encoding="utf-8").splitlines():
            match = RESULT_LINE_RE.match(line.strip())
            if match and match["unit"] == unit:
                merged[(match["speaker"].lower(), match["context"])] = float(match["error"])
    merged.update(scored)

    lines = [
        f"{speaker.capitalize()}-speaker, {context}-context, {unit} ABX error: {error:.2f}%"
        for (speaker, context), error in sorted(merged.items())
    ]
    result_fn.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_summary(summary_fn: Path, rows: list[str]) -> None:
    """Append result rows to the run-log TSV, writing the header if it's new."""

    summary_fn.parent.mkdir(parents=True, exist_ok=True)
    new = not summary_fn.exists()
    with summary_fn.open("a", encoding="utf-8") as f:
        if new:
            f.write(SUMMARY_HEADER + "\n")
        for row in rows:
            f.write(row + "\n")


# ------------------------------------ MAIN -------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Score ABX over a chosen grid of speaker (within/across), context "
        "(any/within), unit (triphone/isolated phoneme) and HuBERT layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--unit",
        nargs="+",
        required=True,
        choices=sorted(UNITS),
        help="Which item file to score, and therefore how features are keyed. "
        "'triphone' = triphone-dev-clean.item, keyed by utterances.list segment id "
        "with segment-relative onsets (segment-keyed features). 'phoneme' = "
        "phoneme-dev-clean.item, keyed by whole-recording id with recording-relative "
        "onsets, keeping utterance-edge phones (recording-keyed features). The two "
        "are not interchangeable - each needs its own extraction.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
        help="HuBERT/WavLM layer(s) to score, as indices into hidden_states "
        "(0 = feature-extractor output, i = output of transformer layer i; "
        "hubert-large-ll60k has 24 transformer layers).",
    )
    parser.add_argument(
        "--speaker",
        nargs="+",
        default=["within", "across"],
        choices=["within", "across"],
        help="Speaker condition(s): 'within' pairs A/B/X from the same speaker, "
        "'across' takes X from a different speaker than A/B (default: both).",
    )
    parser.add_argument(
        "--context",
        nargs="+",
        default=None,
        choices=["within", "any"],
        help="Context condition(s): 'within' requires A/B/X to share prev-phone and "
        "next-phone, 'any' matches on the centre phone alone. Defaults to the modes "
        "that are meaningful for each --unit: 'within' only for triphone (its items "
        "span prev+centre+next, so the neighbours are part of the compared audio - "
        "fastabx: \"Always use 'within' with representations of triphones\"), both "
        "for phoneme (whose items span the centre phone alone). Setting this "
        "explicitly overrides that, with a warning for unsound combinations.",
    )
    parser.add_argument(
        "--feature-set",
        default="LibriSpeech_HuBERT",
        help="Feature set name or directory, resolved by "
        "extract_abx_phone_features.resolve_feature_set (default: LibriSpeech_HuBERT, "
        "i.e. facebook/hubert-large-ll60k on LibriSpeech).",
    )
    parser.add_argument(
        "--subset",
        default="dev-clean",
        help="Subset with a Splits/*_speakers.list file (default: dev-clean).",
    )
    parser.add_argument(
        "--item-file",
        default=None,
        help="Override the item file for this run. Only valid with a single --unit, "
        "since the unit is what normally selects it; the chosen unit still decides "
        "how features are keyed.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuBERT/WavLM checkpoint to extract with, overriding the extraction "
        "module's hardcoded '{WAVLM,HUBERT}_MODEL_NAME' (e.g. "
        "'facebook/hubert-base-ls960' instead of the default "
        "'facebook/hubert-large-ll60k'). Features go to a directory tagged with "
        "the checkpoint name, so different models never share a cache - "
        "necessary since their hidden sizes differ (768 vs 1024). Remember that "
        "--layer indexes THIS model's hidden_states: base has 12 transformer "
        "layers, large has 24.",
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
        help="Feature frame rate in Hz; inferred from the features path if not given "
        "(100 for MFCC, 50 for WavLM/HuBERT).",
    )
    parser.add_argument(
        "--max-size-group",
        type=int,
        default=MAX_SIZE_GROUP,
        help=f"Max instances of A/B/X per cell, passed to fastabx's Subsampler "
        f"(default {MAX_SIZE_GROUP}, the standard ZeroSpeech default).",
    )
    parser.add_argument(
        "--max-x-across",
        type=int,
        default=MAX_X_ACROSS,
        help=f"Max distinct X speakers per cell, for --speaker across only "
        f"(default {MAX_X_ACROSS}, the standard ZeroSpeech default).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract features even when a complete cache already exists.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract (and cache) the features for every requested unit/layer, then "
        "stop without scoring.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="TSV run log to append every result row to (default: "
        "results/abx/<Set>/<normalised_subdir>/abx_summary.tsv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the extraction/scoring plan and exit, without extracting or scoring.",
    )
    return parser.parse_args()


def main() -> None:
    """Extract (or reuse) features and score every requested ABX condition."""

    args = parse_args()

    # Resolve paths against the current directory before changing to the repo root
    # (the extractors use paths relative to it).
    item_override = Path(args.item_file).resolve() if args.item_file else None
    summary_override = Path(args.summary).resolve() if args.summary else None
    os.chdir(REPO_ROOT)

    if item_override is not None and len(set(args.unit)) > 1:
        print(
            "Error: --item-file applies to one item file, so it can't be combined with "
            "several --unit values (each unit has its own item file)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ext = load_extractor()
    try:
        corpus, feature_type, variant = ext.resolve_feature_set(args.feature_set)
    except ext.UnknownFeatureSetError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    if variant:
        print(
            f"Error: --feature-set takes a base set (raw features don't depend on which "
            f"cAE variant would encode them); pass the base set without the "
            f"'{variant}' suffix",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if feature_type == "mfcc":
        print(
            "Error: this script scores WavLM/HuBERT layers; MFCC has no --layers to "
            "choose from",
            file=sys.stderr,
        )
        raise SystemExit(1)
    config = ext.CONFIGS[(corpus, feature_type)]

    # dict.fromkeys de-duplicates while keeping the order given on the command
    # line, so "--layers 12 6 12" scores each layer once, in the order asked for.
    units = list(dict.fromkeys(args.unit))
    layers = list(dict.fromkeys(args.layers))
    speakers = list(dict.fromkeys(args.speaker))
    # contexts_for() is called once for the score count, once per unit for the
    # plan printout, and again inside the scoring loop; warn only on the first.
    warned: set[tuple[str, str]] = set()

    def contexts_for(unit: str) -> list[str]:
        """Context modes for a unit: by default only the meaningful ones. An explicit
        --context is honoured, with a warning if it doesn't suit the unit."""

        if args.context is None:
            return list(UNITS[unit]["default_contexts"])
        chosen = list(dict.fromkeys(args.context))
        for context in chosen:
            if context not in UNITS[unit]["sound_contexts"] and (unit, context) not in warned:
                warned.add((unit, context))
                print(
                    f"Warning: context='{context}' is not meaningful for --unit {unit}: "
                    f"a {unit} item spans "
                    f"{'prev+centre+next, so its neighbours are part of the compared audio' if unit == 'triphone' else 'the centre phone alone'}"
                    f" (fastabx: \"Always use 'within' with representations of triphones\"). "
                    f"Scoring it anyway, since it was requested explicitly.",
                    file=sys.stderr,
                )
        return chosen

    jobs = [
        (unit, layer, item_override or UNITS[unit]["item_file"])
        for unit in units
        for layer in layers
    ]
    n_scores = sum(len(speakers) * len(contexts_for(unit)) for unit, _layer, _item in jobs)

    print(f"Feature set: {config['base_feature_set']}  (subset: {args.subset})")
    if args.model:
        print(f"Model:       {args.model}  (overrides the extraction module default)")
    print(f"Units:       {', '.join(units)}")
    print(f"Layers:      {', '.join(str(layer) for layer in layers)}")
    for unit in units:
        conds = [f"{s}-speaker/{c}-context" for s in speakers for c in contexts_for(unit)]
        print(f"Conditions:  [{unit}] {', '.join(conds)}")
    print(f"Distance:    {args.distance}")
    print(f"Total:       {len(jobs)} extraction(s), {n_scores} score(s)")

    for unit, layer, item_fn in jobs:
        out_dir = features_dir_for(config, args.subset, unit, layer, args.model)
        print(f"\n[{unit} layer {layer:02d}]")
        print(f"  Item file: {item_fn}  ({UNITS[unit]['keying']})")
        print(f"  Features:  {out_dir}")
        if args.dry_run:
            continue
        if not item_fn.is_file():
            print(f"Error: item file not found at {item_fn}", file=sys.stderr)
            raise SystemExit(1)

    if args.dry_run:
        print("\n(dry run - nothing extracted or scored)")
        return

    summary_fn = summary_override or (
        DISTANCES_ROOT / config["base_feature_set"] / config["normalised_subdir"] / "abx_summary.tsv"
    )
    table: list[tuple[str, int, str, str, float]] = []

    for unit, layer, item_fn in jobs:
        print(f"\n=== {unit}, layer {layer:02d} ===")
        features_dir = ensure_features(
            corpus, feature_type, args.subset, unit, layer, config, item_fn, args.force_extract,
            args.model,
        )
        if args.extract_only:
            continue

        frequency = args.frequency or infer_frequency(features_dir)
        rows: list[str] = []
        scored: dict[tuple[str, str], float] = {}

        for speaker, context in [(s, c) for s in speakers for c in contexts_for(unit)]:
            print(f"  Scoring {speaker}-speaker, {context}-context ({unit}, layer {layer})...")
            kwargs = dict(
                item=item_fn,
                root=features_dir,
                max_size_group=args.max_size_group,
                speaker=speaker,
                context=context,
                distance=args.distance,
                frequency=frequency,
                feature_maker=load_npy,
                extension=".npy",
            )
            # max_x_across only means anything when X comes from another
            # speaker; fastabx rejects it in the within-speaker task.
            if speaker == "across":
                kwargs["max_x_across"] = args.max_x_across

            error = zerospeech_abx_with_cpu_fallback(**kwargs)

            # Free fastabx's GPU tensors between conditions.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"    -> {error * 100:.2f}% error")
            table.append((unit, layer, speaker, context, error * 100))
            scored[(speaker, context)] = error * 100
            rows.append(
                "\t".join(
                    [
                        unit,
                        str(layer),
                        speaker,
                        context,
                        args.distance,
                        f"{error * 100:.4f}",
                        str(features_dir),
                        str(item_fn),
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                    ]
                )
            )

        result_dir = distances_dir_for(features_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_fn = result_dir / f"abx_{unit}_result.txt"
        merge_result_file(result_fn, unit, scored)
        print(f"  Wrote {result_fn}")
        append_summary(summary_fn, rows)

    if args.extract_only:
        print("\nExtraction complete (--extract-only; nothing scored).")
        return

    print(f"\n{'unit':<10}{'layer':>6}{'speaker':>10}{'context':>10}{'ABX error':>12}")
    for unit, layer, speaker, context, error in table:
        print(f"{unit:<10}{layer:>6}{speaker:>10}{context:>10}{error:>11.2f}%")
    print(f"\nRun log: {summary_fn}")


if __name__ == "__main__":
    main()

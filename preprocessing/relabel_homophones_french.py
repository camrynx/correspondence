"""
Apply the ZeroSpeech 2017 French homophone map to the keys of extracted features.

Keys start with the word label ("suite_F01R_000_000055-000083"), so after merging
homophones the keys have to change too. The frames don't, so this rewrites the keys
instead of re-extracting. Alignments are not rewritten: merged labels create new pairs,
so regenerate them afterwards (training/align.py).

    python preprocessing/relabel_homophones_french.py           # dry run
    python preprocessing/relabel_homophones_french.py --apply
"""

import argparse
from collections import Counter
from os import path, replace
from pathlib import Path

import numpy as np


DEFAULT_MAP = "data/Information/ZS17French/homophones.map"
DEFAULT_FEATURES_ROOT = "data/Features"

# Only word-level sets: the ABX archives are keyed by utterance, not label.
DEFAULT_DIR_GLOB = "ZS17French*/**/*.samediff*"


def read_homophone_map(map_fn):
    """Read "<original> <canonical>" lines into {original: canonical}."""
    homophone_map = {}
    with open(map_fn, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            homophone_map[parts[0]] = parts[1]
    return homophone_map


def relabel_key(key, homophone_map):
    """ "<label>_<rest>" -> "<canonical>_<rest>" (labels contain no underscores)."""
    label, sep, rest = key.partition("_")
    if not sep:
        return key
    canonical = homophone_map.get(label)
    return key if canonical is None else canonical + "_" + rest


def relabel_npz(npz_fn, homophone_map, apply_changes):
    """Rewrite one archive's keys via a temp file. Returns (n_keys, n_changed)."""
    with np.load(npz_fn, allow_pickle=False) as archive:
        keys = list(archive.files)
        new_keys = [relabel_key(key, homophone_map) for key in keys]
        n_changed = sum(1 for old, new in zip(keys, new_keys) if old != new)

        if not n_changed or not apply_changes:
            return len(keys), n_changed

        if len(set(new_keys)) != len(new_keys):
            duplicates = [k for k, n in Counter(new_keys).items() if n > 1]
            raise ValueError(
                "relabelling {} would collide on {} key(s), e.g. {}".format(
                    npz_fn, len(duplicates), duplicates[:3]
                )
            )

        relabelled = {
            new: archive[old] for old, new in zip(keys, new_keys)
        }

    # Write through a file handle: savez_compressed would append .npz to the temp name.
    tmp_fn = npz_fn.with_name(npz_fn.name + ".tmp")
    with open(tmp_fn, "wb") as f_tmp:
        np.savez_compressed(f_tmp, **relabelled)
    replace(tmp_fn, npz_fn)
    return len(keys), n_changed


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite word-level .npz keys to the canonical homophone "
                    "labels produced by "
                    "ZS17Frenchthe utterance/word-list preprocessing."
    )
    parser.add_argument(
        "--map-fn", default=DEFAULT_MAP,
        help="Homophone merge map to apply (default: %(default)s)",
    )
    parser.add_argument(
        "--features-root", default=DEFAULT_FEATURES_ROOT,
        help="Root of the feature tree to walk (default: %(default)s)",
    )
    parser.add_argument(
        "--dir-glob", default=DEFAULT_DIR_GLOB,
        help="Glob, relative to --features-root, selecting the directories to "
             "relabel (default: %(default)s)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually rewrite the archives. Without this the run only "
             "reports what would change.",
    )
    args = parser.parse_args()

    homophone_map = read_homophone_map(args.map_fn)
    print("Read {} merged spellings from {}".format(
        len(homophone_map), args.map_fn
    ))

    root = Path(args.features_root)
    npz_fns = sorted(
        npz_fn
        for directory in root.glob(args.dir_glob)
        if directory.is_dir()
        for npz_fn in directory.rglob("*.npz")
    )
    if not npz_fns:
        print("No .npz files matched {}/{}".format(
            args.features_root, args.dir_glob
        ))
        return

    total_keys = total_changed = 0
    changed_files = 0
    for npz_fn in npz_fns:
        n_keys, n_changed = relabel_npz(npz_fn, homophone_map, args.apply)
        total_keys += n_keys
        total_changed += n_changed
        if n_changed:
            changed_files += 1

    verb = "Relabelled" if args.apply else "Would relabel"
    print("{} {} of {} keys across {} of {} archives".format(
        verb, total_changed, total_keys, changed_files, len(npz_fns)
    ))
    if not args.apply:
        print("Nothing was written. Re-run with --apply to rewrite the keys.")
    else:
        print(
            "Now regenerate the alignments so the new same-word pairs get DTW "
            "paths, e.g.\n"
            "  python training/align.py "
            "--n-cpus 8"
        )


if __name__ == "__main__":
    main()

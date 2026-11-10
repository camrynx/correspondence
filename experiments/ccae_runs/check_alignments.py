"""Fail loudly if pre-generated alignments don't match the (re-extracted) features they index.

train_ccae.py silently skips alignment pairs whose token key is missing from the
features, so stale alignments would quietly shrink the training set. This checks every
pair: both token keys must exist, and every frame index must be in range.

    python experiments/ccae_runs/check_alignments.py <alignment_dir> <features_dir>
"""
import pickle
import sys
from pathlib import Path

import numpy as np

align_dir, feat_dir = Path(sys.argv[1]), Path(sys.argv[2])
lengths = {}
for fn in sorted(feat_dir.glob("*.npz")):
    with np.load(fn, allow_pickle=False) as f:
        lengths.update({k: f[k].shape[0] for k in f.files})

n_pairs = n_missing = n_oob = 0
missing_examples = []
# Read the same chunks the trainer reads: metadata.pkl's total_chunks when present
# (a regenerated directory can still hold stale higher-numbered chunks from a larger
# earlier run, which the trainer never touches), else every chunk file.
meta_fn = align_dir / "metadata.pkl"
if meta_fn.is_file():
    with open(meta_fn, "rb") as f:
        n_chunks = pickle.load(f)["total_chunks"]
    chunks = [align_dir / f"alignments.chunk_{i:04d}.pkl" for i in range(n_chunks)]
    n_stale = len(list(align_dir.glob("alignments.chunk_*.pkl"))) - n_chunks
    if n_stale > 0:
        print(f"  note: {n_stale} stale chunk files beyond metadata's {n_chunks} are ignored (as the trainer does)")
else:
    chunks = sorted(align_dir.glob("alignments.chunk_*.pkl"))

for chunk in chunks:
    with open(chunk, "rb") as f:
        data = pickle.load(f)
    for alignments in data.values():
        for a in alignments:
            n_pairs += 1
            u1, u2 = a["pair"]
            if u1 not in lengths or u2 not in lengths:
                n_missing += 1
                if len(missing_examples) < 3:
                    missing_examples.append((u1, u2))
                continue
            fp = np.asarray(a["frame_pairs"])
            if len(fp) and (fp[:, 0].max() >= lengths[u1] or fp[:, 1].max() >= lengths[u2]):
                n_oob += 1

print(f"{align_dir}: {n_pairs} pairs vs {len(lengths)} feature tokens in {feat_dir}")
print(f"  pairs with a missing token: {n_missing}  {missing_examples}")
print(f"  pairs with out-of-range frame indices: {n_oob}")
if n_pairs == 0 or n_missing or n_oob:
    sys.exit("MISMATCH: alignments do not match these features - regenerate the alignments")
print("OK")

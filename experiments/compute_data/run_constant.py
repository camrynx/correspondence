#!/usr/bin/env python
"""Same pairs as run.py, but each small subset trains for as many weight updates as the full-data run of that model
(30 epochs fixed gives small subsets far fewer updates). Separates "too little data" from "too little training".
Uses each size's existing features/alignments read-only, in a new sandbox: sizes/<size>/box_const.
    python experiments/compute_data/run_constant.py
"""
import math
import re
import shutil

from run import (CAE, CODE_DIRS, DUST_REL, EXP, GOLD_REL, ITEMS, LOGS, PY, RES, STATE, log, stage)

BATCH = 2048
ALIGN_REL = "data/Alignments/Buckeye_WavLMDUSTED"


def items(name):
    m = re.search(r"Total frame pairs indexed: (\d+)", (LOGS / f"{name}_train_cAE.log").read_text(errors="replace"))
    return int(m.group(1))


def steps_per_epoch(n):
    return math.ceil(n / BATCH)


def build(name):
    src, box = EXP / "sizes" / name / "box", EXP / "sizes" / name / "box_const"
    if (box / ".built").exists():
        return src, box
    if box.exists():
        shutil.rmtree(box)
    for d in CODE_DIRS:
        shutil.copytree(CAE / d, box / d, ignore=__import__("shutil").ignore_patterns("__pycache__", "APResults", "logs"))
    (box / "third_party").symlink_to(CAE / "third_party")
    (box / "data").mkdir()
    (box / "data/Datasets").symlink_to(CAE / "data/Datasets")
    (box / "data/Information").symlink_to(src / "data/Information")
    (box / GOLD_REL).parent.mkdir(parents=True)
    (box / GOLD_REL).symlink_to(src / GOLD_REL)
    (box / ALIGN_REL).parent.mkdir(parents=True)
    (box / ALIGN_REL).symlink_to(src / ALIGN_REL)
    norm = box / DUST_REL / "Normalised"
    norm.mkdir(parents=True)
    for s in ("devpart1", "devpart2", "zs"):
        p = src / DUST_REL / "Normalised" / f"{s}.samediff"
        if p.exists():
            (norm / f"{s}.samediff").symlink_to(p)
    for d in ("models/Buckeye_WavLM", "results/abx", "results/ap"):
        (box / d).mkdir(parents=True, exist_ok=True)
    (box / ".built").touch()
    return src, box


def main():
    full = items("full")
    target_cae = 30 * steps_per_epoch(full)
    target_nce = 30 * steps_per_epoch(full // 2)
    log(f"NOTE  constant updates: cAE {target_cae}, NCE {target_nce}")
    for name in ("2h", "1h", "30min", "10min"):
        src, box = build(name)
        n = items(name)
        ep_cae = math.ceil(target_cae / steps_per_epoch(n))
        ep_nce = math.ceil(target_nce / steps_per_epoch(n // 2))
        log(f"NOTE  {name}: cAE {ep_cae} epochs, NCE {ep_nce} epochs")
        wb = {"WANDB_DIR": str(box), "WANDB_MODE": "disabled"}
        stage(f"const_{name}_train_cAE", [PY, "training/train_cae.py", "--features", "wavlm", "--corpus", "buckeye", "--pairs", "dusted", "--epochs", ep_cae], box, env=wb, gpu=True)
        stage(f"const_{name}_train_NCE", [PY, "training/train_ccae.py", "--corpus", "buckeye", "--pairs", "dusted",
                                          "--init", "random", "--layer", 21, "--epochs", ep_nce], box, env=wb, gpu=True)
        for model, prefix, flags in (("cAE", "cAE", []), ("NCE", "ccAErandinit", ["--ccae", "--randinit"])):
            stage(f"const_{name}_ap_{model}", [PY, "evaluation/evaluate_ap.py", f"{DUST_REL}/Normalised/zs.samediff_{prefix}/layer_21"], box, gpu=True)
            stage(f"const_{name}_abxfeat_{model}", [PY, "evaluation/extract_abx_phone_features.py", "Buckeye_WavLMDUSTED", "--subset", "zs", *flags, "--layer", 21], box, gpu=True)
            feats = f"{DUST_REL}/Normalised/zs.abx_{prefix}_layer21"
            if stage(f"const_{name}_abx_{model}", [PY, "evaluation/score_abx.py", feats, ITEMS], box, gpu=True):
                shutil.rmtree(box / feats, ignore_errors=True)
    (STATE / "CURRENT").unlink(missing_ok=True)
    log("CONSTANT UPDATES DONE")


if __name__ == "__main__":
    main()

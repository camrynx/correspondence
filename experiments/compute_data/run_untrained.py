#!/usr/bin/env python
"""What does a randomly initialised, untrained encoder score on the 10-minute subset's features? (--epochs 0)"""
import shutil
from run import CAE, CODE_DIRS, DUST_REL, EXP, GOLD_REL, ITEMS, PY, log, stage

ALIGN_REL = "data/Alignments/Buckeye_WavLMDUSTED"
name = "10min"
src, box = EXP / "sizes" / name / "box", EXP / "sizes" / name / "box_untrained"
if not (box / ".built").exists():
    if box.exists():
        shutil.rmtree(box)
    for d in CODE_DIRS:
        shutil.copytree(CAE / d, box / d, ignore=shutil.ignore_patterns("__pycache__", "APResults", "logs"))
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
wb = {"WANDB_DIR": str(box), "WANDB_MODE": "disabled"}
stage("untrained_10min_train", [PY, "training/train_ccae.py", "--corpus", "buckeye", "--pairs", "dusted", "--init", "random", "--layer", 21, "--epochs", 0], box, env=wb, gpu=True)
stage("untrained_10min_ap", [PY, "evaluation/evaluate_ap.py", f"{DUST_REL}/Normalised/zs.samediff_ccAErandinit/layer_21"], box, gpu=True)
stage("untrained_10min_abxfeat", [PY, "evaluation/extract_abx_phone_features.py", "Buckeye_WavLMDUSTED", "--subset", "zs", "--ccae", "--randinit", "--layer", 21], box, gpu=True)
feats = f"{DUST_REL}/Normalised/zs.abx_ccAErandinit_layer21"
if stage("untrained_10min_abx", [PY, "evaluation/score_abx.py", feats, ITEMS], box, gpu=True):
    shutil.rmtree(box / feats, ignore_errors=True)
log("UNTRAINED CHECK DONE")

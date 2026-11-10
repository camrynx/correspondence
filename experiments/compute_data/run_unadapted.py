#!/usr/bin/env python
"""Unadapted SpidR-Adapt checkpoints on each language's test set, no fine-tuning, features unnormalised.

    python experiments/compute_data/run_unadapted.py

Each row has its own box under sizes/unadapted/. GPU stages wait until nobody else is on the GPU, and the whole
run waits for retime_extract.sh first. Results: python experiments/compute_data/unadapted_results.py
"""
import shutil
import subprocess
import time

import run
from run import CAE, CODE_DIRS, EXP, LOGS, PY, RES, SPIDR_INIT, SPIDR_SRC, STATE, log, set_subsets, stage

SL_URL = "https://dl.fbaipublicfiles.com/shared/devai/assets/spidr-adapt/checkpoints/spidr_reptile_sl_vp19lang/step_200000.pt"
SL = EXP / "checkpoints/spidr_reptile_sl_vp19lang_step_200000.pt"
CORPORA = {
    "english": ("Buckeye_SpidR_adapted", "zs", "features/extras/extractFeaturesSpidR.py", "data/Information/Buckeye/abx_items/zs.item"),
    "xitsonga": ("Xitsonga_SpidR_adapted", "test", "features/extras/extractFeaturesSpidRTSO.py", "data/Information/Buckeye_tsonga/abx_items/test.item"),
    "french": ("ZS17French_SpidR_unadapted", "test", None, "data/Information/ZS17French/abx_items/test.item"),
}
run.GATE_MB = 16000  # SpidR ABX scoring peaks near 14 GB, and a smaller gate collided with another user's job
RUNS = [("reptile", SPIDR_INIT, "xitsonga", 5), ("reptile", SPIDR_INIT, "french", 6),
        ("reptile_sl", SL, "english", 7), ("reptile_sl", SL, "xitsonga", 7), ("reptile_sl", SL, "french", 8)]


def build(box):
    if (box / ".built").exists():
        return
    if box.exists():
        shutil.rmtree(box)
    for d in CODE_DIRS:
        shutil.copytree(CAE / d, box / d, ignore=shutil.ignore_patterns("__pycache__", "APResults", "logs"))
    (box / "third_party").symlink_to(CAE / "third_party")
    (box / "data").mkdir()
    (box / "data/Datasets").symlink_to(CAE / "data/Datasets")
    (box / "data/Information").symlink_to(CAE / "data/Information")
    (box / "results" / "abx").mkdir(parents=True)
    (box / ".built").touch()


def evaluate(ckpt_name, ckpt, corpus, layer):
    name = f"unadapted_{ckpt_name}_{corpus}_L{layer}"
    box = EXP / "sizes/unadapted" / f"{ckpt_name}_{corpus}_L{layer}" / "box"
    build(box)
    feature_set, subset, script, items = CORPORA[corpus]
    rel = f"data/Features/{feature_set}/NotNormalised"
    env = {"SD_MODEL": str(ckpt), "SD_LAYERS": str(layer), "SD_FEATURE_SET": feature_set,
           "PYTHONPATH": f"{EXP/'stubs'}:{SPIDR_SRC}"}
    french = [PY, EXP / "spidr_french.py"]
    if script:
        set_subsets(box / script, [subset])
        stage(f"{name}_extract", [PY, script], box, env=env, gpu=True, strict=True)
    else:
        stage(f"{name}_extract", [*french, "extract", ckpt, layer], box, env=env, gpu=True, strict=True)
    stage(f"{name}_ap", [PY, "evaluation/evaluate_ap.py", f"{rel}/{subset}.samediff/layer_{layer:02d}", "--out-dir", box / "spidr_ap"],
          box, gpu=True, strict=True)
    if script:
        stage(f"{name}_abxfeat", [PY, "evaluation/extract_abx_phone_features.py", feature_set, "--subset", subset, "--raw", "--layer", layer],
              box, env=env, gpu=True, strict=True)
    else:
        stage(f"{name}_abxfeat", [*french, "abx", ckpt, layer], box, env=env, gpu=True, strict=True)
    feats = f"{rel}/{subset}.abx_layer{layer:02d}"
    if stage(f"{name}_abx", [PY, "evaluation/score_abx.py", feats, items, "--frequency", 50], box, gpu=True, strict=True):
        shutil.rmtree(box / feats, ignore_errors=True)


def main():
    while subprocess.run(["pgrep", "-f", "retime_extract.sh"], capture_output=True).returncode == 0:
        time.sleep(60)
    stage("unadapted_download_sl", ["curl", "-fL", "-o", SL, SL_URL], EXP)
    assert SL.stat().st_size == 739415090, SL.stat().st_size
    for row in RUNS:
        for attempt in range(3):  # finished stages are skipped, so a retry resumes where the last one failed
            try:
                evaluate(*row)
                break
            except SystemExit:
                if attempt == 2:
                    raise
                log(f"NOTE  retrying {row[0]} {row[2]} L{row[3]} in 10 min")
                time.sleep(600)
    (STATE / "CURRENT").unlink(missing_ok=True)
    log("UNADAPTED DONE")


if __name__ == "__main__":
    for d in (STATE, LOGS, RES):
        d.mkdir(parents=True, exist_ok=True)
    main()

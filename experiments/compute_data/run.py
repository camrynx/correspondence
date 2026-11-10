#!/usr/bin/env python
"""English (Buckeye) compute + amount-of-data experiments for the paper.

For each amount of training speech (devpart1, equal minutes per speaker, each speaker's first minutes), this
reruns the paper's pipeline end to end on that audio only:

  DUSTED:  prepare -> encode -> segment -> match -> format   (../UTD_DUSTED_Clean, English HuBERT + released codebook)
  Pairs:   WavLM term features -> normalise -> DTW alignments -> alignment check
  Models:  cAE and NCE (random init), then AP and ABX on zs
  SpidR:   SpidR-Adapt [SSL] + MAdaPT-Reptile fine-tuned on the same clips (paper's settings), layer 6, AP and ABX on zs

Each size has its own sandbox (a copy of the repo's code, with read-only data symlinked), so the real data/, models/
and results/abx/ are never written to. Normalisation stats come from that size's own gold words, as in the paper's
pipeline with less data. Every stage records wall time, CPU time, peak RSS, peak GPU memory and whether another user
was on the GPU. Resumable via state/*.done.

    python experiments/compute_data/run.py            # everything
    python experiments/compute_data/results.py        # tables so far
"""
import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent
CAE = EXP.parents[1]
UTD = Path(os.environ.get("UTD_DUSTED_ROOT", CAE.parent / "UTD_DUSTED_Clean"))  # DUSTED pipeline checkout
SPIDR_SRC = Path(os.environ.get("SPIDR_SRC", CAE.parent / "spidr-adapt" / "src"))  # SpidR-Adapt source
PY = os.environ.get("PY", sys.executable)
SPIDR_BIN = EXP / "spidr_env/bin"
SPIDR_INIT = EXP / "checkpoints/spidr_reptile_vp19lang_step_200000.pt"
SPIDR_LAYER = 6  # the layer the paper's table reports for English
STATE, LOGS, RES = EXP / "state", EXP / "logs", EXP / "results"
TIMINGS = RES / "timings.jsonl"
GATE_MB = 5500
ME = os.environ.get("USER", "")
ITEMS = "data/Information/Buckeye/abx_items/zs.item"
CODE_DIRS = ("features", "training", "evaluation")
SIZES = {"full": None, "4h": 240, "2h": 120, "1h": 60, "30min": 30, "10min": 10}  # total speech minutes (12 speakers)
GOLD_REL = "data/Features/Buckeye_WavLM"
DUST_REL = "data/Features/Buckeye_WavLMDUSTED"
BASE_ENV = {"PYTHONUNBUFFERED": "1", "WANDB_MODE": "offline", "CUDA_VISIBLE_DEVICES": "0"}


def log(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a") as f:
        f.write(line + "\n")


def foreign_gpu_users():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    users = set()
    for pid in out:
        u = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if u and u != ME:
            users.add(u)
    return users


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def wait_for_gpu(strict):
    """strict (timed compute runs): nobody else on the GPU. Otherwise only enough free memory for our job."""
    while True:
        used, free = gpu_mem()
        others = foreign_gpu_users()
        if free >= GATE_MB and not (strict and others):
            return
        with open(LOGS / "gpu_gate.log", "a") as f:
            f.write(f"[{datetime.now():%F %T}] waiting: free={free} MiB others={sorted(others)}\n")
        time.sleep(60)


def stage(name, cmd, cwd, env=None, gpu=False, strict=False, marks=()):
    """Run one stage once. Returns False if already done."""
    done = STATE / f"{name}.done"
    if done.exists():
        return False
    if gpu:
        wait_for_gpu(strict)
    (STATE / "CURRENT").write_text(name)
    log(f"START {name}")
    cmd = [str(c) for c in cmd]
    tfile = RES / f".time_{name}"
    logfile = LOGS / f"{name}.log"
    base_used, _ = gpu_mem()
    peak = [base_used]
    others = set(foreign_gpu_users())
    stop = threading.Event()

    def sampler():
        while not stop.wait(2):
            used, _ = gpu_mem()
            peak[0] = max(peak[0], used)
            others.update(foreign_gpu_users())

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    seen = {}
    start = time.time()
    tail = b""
    with open(logfile, "wb") as out:
        out.write(f"$ (cd {cwd} && {' '.join(cmd)})\n".encode())
        out.flush()
        proc = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(tfile), *cmd], cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env={**os.environ, **BASE_ENV, **(env or {})})
        while chunk := os.read(proc.stdout.fileno(), 65536):
            out.write(chunk)
            out.flush()
            tail = (tail + chunk)[-4096:]
            for m in marks:
                if m not in seen and m.encode() in tail:
                    seen[m] = time.time() - start
        rc = proc.wait()
    wall = time.time() - start
    stop.set()
    th.join()
    tv = tfile.read_text() if tfile.exists() else ""
    grab = lambda pat: (re.search(pat, tv) or [None, "0"])[1]
    rec = {
        "stage": name, "wall_s": round(wall, 2), "user_s": float(grab(r"User time \(seconds\): ([\d.]+)")),
        "sys_s": float(grab(r"System time \(seconds\): ([\d.]+)")),
        "max_rss_mb": round(int(grab(r"Maximum resident set size \(kbytes\): (\d+)")) / 1024),
        "gpu_peak_mb": peak[0] - base_used if gpu else 0, "others_on_gpu": sorted(others),
        "load_start": round(os.getloadavg()[0], 1), "marks": {k: round(v, 2) for k, v in seen.items()},
        "exit": rc, "finished": f"{datetime.now():%F %T}",
    }
    with open(TIMINGS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tfile.unlink(missing_ok=True)
    if rc != 0:
        (STATE / "CURRENT").unlink(missing_ok=True)
        log(f"FAIL  {name} (see {logfile})")
        sys.exit(1)
    done.touch()
    log(f"DONE  {name}  {wall:.0f}s  gpu+{rec['gpu_peak_mb']}MiB" + (f"  OTHERS={sorted(others)}" if others else ""))
    return True


# ------------------------------------------------------------------ data subsets

def subset_lists(name, minutes, out):
    """Each devpart1 speaker's first minutes/12 of speech (run_trial.buckeye_subset's rule), plus gold words inside it."""
    out.mkdir(parents=True, exist_ok=True)
    utt_fn, words_fn = out / "utterances.list", out / "words.list"
    if utt_fn.exists() and words_fn.exists():
        return utt_fn, words_fn
    info = CAE / "data/Information/Buckeye"
    speakers = set((info / "Splits/buckeye_devpart1_speakers.list").read_text().split())
    by_spk = {}
    for line in (info / "utterances.list").read_text().splitlines():
        if line.strip():
            n, a, b = line.split()
            if n.split("_")[0] in speakers:
                by_spk.setdefault(n.split("_")[0], []).append((n.split("_")[1], float(a), float(b), n))
    quota = None if minutes is None else minutes * 60 / len(speakers)
    kept, segs, secs = [], {}, 0.0
    for spk in sorted(by_spk):
        total = 0.0
        for sess, a, b, n in sorted(by_spk[spk]):
            if quota is not None and total >= quota:
                break
            kept.append(f"{n} {a:.4f} {b:.4f}")
            segs.setdefault(sess, []).append((a, b))
            total += b - a
        secs += total
    utt_fn.write_text("\n".join(kept) + "\n")
    for v in segs.values():
        v.sort()
    n_w, lines = 0, []
    for line in (info / "words.list").read_text().splitlines():
        p = line.split()
        if p[1][:3] in speakers:
            s = segs.get(p[1], [])
            a, b = float(p[2]), float(p[3])
            i = bisect.bisect_right(s, (a, float("inf"))) - 1
            if not (i >= 0 and s[i][0] - 1e-6 <= a and b <= s[i][1] + 1e-6):
                continue
            n_w += 1
        lines.append(line)
    words_fn.write_text("\n".join(lines) + "\n")
    (out / "subset.json").write_text(json.dumps({"size": name, "minutes": minutes, "speech_h": secs / 3600,
                                                 "clips": len(kept), "gold_tokens": n_w, "speakers": len(by_spk)}))
    log(f"subset {name}: {secs/3600:.3f} h, {len(kept)} clips, {n_w} gold word tokens")
    return utt_fn, words_fn


def build_box(box, words_fn=None):
    """Sandbox: copied code + symlinked read-only data; everything it writes stays inside."""
    if (box / ".built").exists():
        return
    if box.exists():
        shutil.rmtree(box)
    for d in CODE_DIRS:
        shutil.copytree(CAE / d, box / d, ignore=shutil.ignore_patterns("__pycache__", "APResults", "logs"))
    (box / "experiments/ccae_runs").mkdir(parents=True)
    shutil.copy2(CAE / "experiments/ccae_runs/check_alignments.py", box / "experiments/ccae_runs")
    (box / "third_party").symlink_to(CAE / "third_party")
    (box / "data").mkdir()
    (box / "data/Datasets").symlink_to(CAE / "data/Datasets")
    info = box / "data/Information/Buckeye"
    info.mkdir(parents=True)
    for p in (CAE / "data/Information/Buckeye").iterdir():
        if p.name not in {"UTDWordsDUSTED.txt", "words.list"}:
            (info / p.name).symlink_to(p)
    shutil.copy2(words_fn or CAE / "data/Information/Buckeye/words.list", info / "words.list")
    for d in ("models/Buckeye_WavLM", "results/abx", "results/ap", GOLD_REL, DUST_REL):
        (box / d).mkdir(parents=True, exist_ok=True)
    (box / ".built").touch()


def set_subsets(script, subsets):
    t = script.read_text()
    t2 = re.sub(r'^SUBSETS = \[.*\]$', f"SUBSETS = {subsets!r}", t, flags=re.M)
    assert t2 != t or repr(subsets) in t, script
    script.write_text(t2)


# ------------------------------------------------------------------ stages

def shared_gold():
    box = EXP / "shared_box"
    build_box(box)
    stage("gold_extract_full", [PY, "features/extract_ssl.py", "--model", "wavlm", "--corpus", "buckeye"], box, gpu=True, strict=True)
    return box


def dusted(name, utt_fn, strict):
    d = EXP / "sizes" / name / "dusted"
    d.mkdir(parents=True, exist_ok=True)
    stage(f"{name}_dusted_prepare", [PY, "prepare.py", "buckeye", d / "wavs", "--utterances", utt_fn], UTD)
    stage(f"{name}_dusted_encode", [PY, "encode.py", d / "wavs", d / "features", "english", "--windows", "buckeye"],
          UTD, gpu=True, strict=strict)
    stage(f"{name}_dusted_segment", [PY, "segment.py", d / "features", d / "segments", "--codebook", "english"], UTD)
    stage(f"{name}_dusted_match", [PY, "tools/match_inmemory.py", d / "segments", d / "pairs.csv"], UTD)
    lst = d / "pairs.txt"
    stage(f"{name}_dusted_format", [PY, "format_pairs.py", d / "pairs.csv", lst, "--min_duration", "0.5"], UTD)
    return d, lst


def models(name, box, lst, shared, strict):
    info = box / "data/Information/Buckeye"
    if not (info / "UTDWordsDUSTED.txt").exists():
        shutil.copy2(lst, info / "UTDWordsDUSTED.txt")
    raw = box / GOLD_REL / "NotNormalised"
    raw.mkdir(parents=True, exist_ok=True)
    shared_raw = shared / GOLD_REL / "NotNormalised"
    for s in ("devpart2", "zs") if name != "full" else ("devpart1", "devpart2", "zs"):
        link = raw / f"{s}.samediff"
        if not link.exists():
            link.symlink_to(shared_raw / f"{s}.samediff")
    if name != "full":
        stage(f"{name}_gold_extract", [PY, "features/extract_ssl.py", "--model", "wavlm", "--corpus", "buckeye",
                                       "--splits", "devpart1"], box, gpu=True)
    stage(f"{name}_gold_norm", [PY, "features/normalise.py", "--features", "wavlm", "--corpus", "buckeye", "--layer", 21], box)
    stage(f"{name}_terms_extract", [PY, "features/extract_ssl.py", "--model", "wavlm", "--corpus", "buckeye", "--pairs", "dusted"], box,
          gpu=True, strict=strict)
    stage(f"{name}_terms_norm", [PY, "features/normalise.py", "--features", "wavlm", "--corpus", "buckeye", "--pairs", "dusted", "--layer", 21], box)
    stage(f"{name}_align", [PY, "training/align.py", "--features", "wavlm", "--corpus", "buckeye", "--pairs", "dusted", "--n-cpus", 8],
          box)
    align = "data/Alignments/Buckeye_WavLMDUSTED/devpart1.samediff.alignments/layer_21"
    stage(f"{name}_align_check", [PY, "experiments/ccae_runs/check_alignments.py", align, f"{DUST_REL}/Normalised/devpart1.samediff/layer_21"], box)
    wb = {"WANDB_DIR": str(box)}
    marks = ("Epoch 0/30", "Saved trained")
    stage(f"{name}_train_cAE", [PY, "training/train_cae.py", "--features", "wavlm", "--corpus", "buckeye", "--pairs", "dusted"], box, env=wb, gpu=True, strict=strict, marks=marks)
    stage(f"{name}_train_NCE", [PY, "training/train_ccae.py", "--corpus", "buckeye", "--pairs", "dusted", "--init", "random", "--layer", 21],
          box, env=wb, gpu=True, strict=strict, marks=marks)
    for model, prefix, flags in (("cAE", "cAE", []), ("NCE", "ccAErandinit", ["--ccae", "--randinit"])):
        stage(f"{name}_ap_{model}", [PY, "evaluation/evaluate_ap.py", f"{DUST_REL}/Normalised/zs.samediff_{prefix}/layer_21"], box, gpu=True)
        stage(f"{name}_abxfeat_{model}", [PY, "evaluation/extract_abx_phone_features.py", "Buckeye_WavLMDUSTED", "--subset", "zs", *flags, "--layer", 21], box, gpu=True)
        feats = f"{DUST_REL}/Normalised/zs.abx_{prefix}_layer21"
        if stage(f"{name}_abx_{model}", [PY, "evaluation/score_abx.py", feats, ITEMS], box, gpu=True):
            shutil.rmtree(box / feats, ignore_errors=True)  # 3-4 GB each, only needed for scoring


def spidr_eval(name, box, ckpt):
    if not (box / "features/extras/extractFeaturesSpidR.py").exists():
        raise SystemExit("box not built")
    set_subsets(box / "features/extras/extractFeaturesSpidR.py", ["zs"])
    env = {"SD_MODEL": str(ckpt), "SD_LAYERS": str(SPIDR_LAYER), "SD_FEATURE_SET": "Buckeye_SpidR_adapted",
           "PYTHONPATH": f"{EXP/'stubs'}:{SPIDR_SRC}"}
    rel = "data/Features/Buckeye_SpidR_adapted/NotNormalised"
    stage(f"{name}_spidr_extract", [PY, "features/extras/extractFeaturesSpidR.py"], box, env=env, gpu=True)
    stage(f"{name}_spidr_ap", [PY, "evaluation/evaluate_ap.py", f"{rel}/zs.samediff/layer_{SPIDR_LAYER:02d}", "--out-dir", box / "spidr_ap"], box, gpu=True)
    stage(f"{name}_spidr_abxfeat", [PY, "evaluation/extract_abx_phone_features.py", "Buckeye_SpidR_adapted", "--subset", "zs", "--raw", "--layer", SPIDR_LAYER], box, env=env, gpu=True)
    feats = f"{rel}/zs.abx_layer{SPIDR_LAYER:02d}"
    if stage(f"{name}_spidr_abx", [PY, "evaluation/score_abx.py", feats, ITEMS, "--frequency", 50], box, gpu=True):
        shutil.rmtree(box / feats, ignore_errors=True)


def spidr(name, box, wavs, strict):
    run = EXP / "sizes" / name / "spidr"
    stage(f"{name}_spidr_cfg", [PY, EXP / "make_spidr_cfg.py", wavs, run, name, SPIDR_INIT], EXP)
    stage(f"{name}_spidr_adapt", [SPIDR_BIN / "torchrun", "--standalone", "--nproc-per-node", 1, EXP / "spidr_local.py", run / "config.toml"],
          EXP, gpu=True, strict=strict, marks=("Starting training loop",))
    spidr_eval(name, box, run / "spidr_ft" / name / "final.pt")


def main():
    for d in (STATE, LOGS, RES):
        d.mkdir(parents=True, exist_ok=True)
    shared = shared_gold()
    # cAE/NCE first (the main result), largest first so a full-data sanity check against the paper comes early.
    built = {}
    for name, minutes in SIZES.items():
        strict = name == "full"
        utt_fn, words_fn = subset_lists(name, minutes, EXP / "sizes" / name)
        box = EXP / "sizes" / name / "box"
        build_box(box, words_fn)
        d, lst = dusted(name, utt_fn, strict)
        n_pairs = sum(1 for _ in open(lst)) // 2
        built[name] = (box, d)
        if n_pairs == 0:
            log(f"NOTE  {name}: DUSTED found no pairs >= 0.5 s; skipping cAE/NCE")
            continue
        log(f"NOTE  {name}: {n_pairs} DUSTED pairs")
        models(name, box, lst, shared, strict)
    # SpidR-Adapt: before adaptation, then adapted on each size.
    pre = EXP / "sizes" / "preadapt" / "box"
    build_box(pre)
    spidr_eval("preadapt", pre, SPIDR_INIT)
    for name in SIZES:
        box, d = built[name]
        spidr(name, box, d / "wavs", strict=name == "full")
    (STATE / "CURRENT").unlink(missing_ok=True)
    log("ALL DONE")


if __name__ == "__main__":
    main()

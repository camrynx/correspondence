#!/usr/bin/env python
"""Constant-updates check vs the fixed 30 epochs. python experiments/compute_data/const_results.py"""
import json, re
from pathlib import Path
EXP = Path(__file__).resolve().parent
def read(box, prefix):
    a = box / f"results/abx/Buckeye_WavLMDUSTED/Normalised/zs.abx_{prefix}_layer21/abx_result.txt"
    p = box / f"results/ap/Buckeye_WavLMDUSTED/Normalised/zs.samediff_{prefix}/layer_21/results.json"
    r = ["-"] * 5
    if a.is_file():
        t = a.read_text(); r[0], r[1] = [re.search(rf"{k}-speaker.*?([\d.]+)%", t).group(1) for k in ("Within", "Across")]
    if p.is_file():
        j = json.loads(p.read_text()); r[2:] = [f"{j[k]:.3f}" for k in ("AP", "AP SWSP", "AP SWDP")]
    return r
print(f"{'':<24}{'WS':>7}{'AS':>7}{'AP':>7}{'SWSP':>7}{'SWDP':>7}")
print(f"{'WavLM baseline':<24}{'5.35':>7}{'6.91':>7}{'0.860':>7}{'0.939':>7}{'0.832':>7}")
for model, pre in (("cAE", "cAE"), ("NCE", "ccAErandinit")):
    print(f"{model+' full (30 ep)':<24}" + "".join(f"{v:>7}" for v in read(EXP / "sizes/full/box", pre)))
    for s in ("2h", "1h", "30min", "10min"):
        print(f"{model+' '+s+' 30 epochs':<24}" + "".join(f"{v:>7}" for v in read(EXP / f"sizes/{s}/box", pre)))
        print(f"{model+' '+s+' same updates':<24}" + "".join(f"{v:>7}" for v in read(EXP / f"sizes/{s}/box_const", pre)))

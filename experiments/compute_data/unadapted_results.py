#!/usr/bin/env python
"""Table of unadapted SpidR-Adapt results.   python experiments/compute_data/unadapted_results.py"""
import json
import re
from pathlib import Path

EXP = Path(__file__).resolve().parent
PATHS = {"english": "Buckeye_SpidR_adapted/NotNormalised/zs", "xitsonga": "Xitsonga_SpidR_adapted/NotNormalised/test",
         "french": "ZS17French_SpidR_unadapted/NotNormalised/test"}
ROWS = [("MAdaPT-Reptile", "preadapt", "english", 6), ("MAdaPT-Reptile", "reptile", "xitsonga", 5),
        ("MAdaPT-Reptile", "reptile", "french", 6), ("MAdaPT-Reptile (SSL/SL)", "reptile_sl", "english", 7),
        ("MAdaPT-Reptile (SSL/SL)", "reptile_sl", "xitsonga", 7), ("MAdaPT-Reptile (SSL/SL)", "reptile_sl", "french", 8)]

print("Checkpoint\tLanguage\tLayer\tWS\tAS\tAP\tSWSP\tSWDP")
for label, tag, corpus, layer in ROWS:
    box = EXP / "sizes" / ("preadapt" if tag == "preadapt" else f"unadapted/{tag}_{corpus}_L{layer}") / "box"
    abx_fn = box / "results" / "abx" / f"{PATHS[corpus]}.abx_layer{layer:02d}/abx_result.txt"
    ap_fn = box / "spidr_ap/results.json"
    ws = as_ = ap = swsp = swdp = "?"
    if abx_fn.is_file():
        t = abx_fn.read_text()
        ws, as_ = (re.search(rf"{k}-speaker.*?([\d.]+)%", t).group(1) for k in ("Within", "Across"))
    if ap_fn.is_file():
        r = json.loads(ap_fn.read_text())
        ap, swsp, swdp = (f"{100 * r[k]:.1f}" for k in ("AP", "AP SWSP", "AP SWDP"))
    print(f"{label}\t{corpus.capitalize()}\t{layer}\t{ws}\t{as_}\t{ap}\t{swsp}\t{swdp}")

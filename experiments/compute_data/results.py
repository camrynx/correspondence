#!/usr/bin/env python
"""Tables for experiments/compute_data so far.   python experiments/compute_data/results.py"""
import json
import re
from pathlib import Path

EXP = Path(__file__).resolve().parent
SIZES = ["preadapt", "10min", "30min", "1h", "2h", "4h", "full"]
PAPER = {"cAE": (4.32, 5.87, .931, .953, .923), "NCE": (4.98, 6.56, .926, .956, .916), "SpidR": (5.70, 7.58, .848, .934, .818)}


def abx(fn):
    if not fn.is_file():
        return None, None
    t = fn.read_text()
    return tuple(float(re.search(rf"{k}-speaker.*?([\d.]+)%", t).group(1)) for k in ("Within", "Across"))


def ap(fn):
    if not fn.is_file():
        return None, None, None
    r = json.loads(fn.read_text())
    return r["AP"], r["AP SWSP"], r["AP SWDP"]


def fmt(v, n):
    return "-" if v is None else f"{v:.{n}f}"


def row(label, vals):
    print(f"{label:<24}" + "".join(f"{fmt(v, 2 if i < 2 else 3):>8}" for i, v in enumerate(vals)))


def info(s):
    p = EXP / "sizes" / s / "subset.json"
    i = json.loads(p.read_text()) if p.exists() else {}
    lst = EXP / "sizes" / s / "dusted/pairs.txt"
    i["pairs"] = sum(1 for _ in open(lst)) // 2 if lst.exists() else None
    log = EXP / "logs" / f"{s}_train_cAE.log"
    if log.exists():
        m = re.search(r"Total frame pairs indexed: (\d+)", log.read_text(errors="replace"))
        i["cae_items"] = int(m.group(1)) if m else None
    return i


print("Amount of speech")
print(f"{'':<8}{'hours':>8}{'clips':>8}{'gold tok':>9}{'pairs':>8}{'cAE items':>11}{'updates':>9}")
for s in SIZES[1:]:
    i = info(s)
    items = i.get("cae_items")
    print(f"{s:<8}{fmt(i.get('speech_h'),2):>8}{i.get('clips','-'):>8}{i.get('gold_tokens','-'):>9}{str(i.get('pairs','-')):>8}"
          f"{str(items if items else '-'):>11}{str(30 * ((items + 2047) // 2048) if items else '-'):>9}")

print(f"\n{'English zs, by amount':<24}" + "".join(f"{h:>8}" for h in ("WS", "AS", "AP", "SWSP", "SWDP")))
print("WavLM baseline (paper)   " + "    5.35    6.91   0.860   0.939   0.832"[1:])
for model, prefix in (("cAE", "cAE"), ("NCE", "ccAErandinit")):
    for s in SIZES[1:]:
        b = EXP / "sizes" / s / "box"
        vals = (*abx(b / f"results/abx/Buckeye_WavLMDUSTED/Normalised/zs.abx_{prefix}_layer21/abx_result.txt"),
                *ap(b / f"results/ap/Buckeye_WavLMDUSTED/Normalised/zs.samediff_{prefix}/layer_21/results.json"))
        row(f"{model} {s}", vals)
    row(f"{model} paper table", PAPER[model])
for s in SIZES:
    b = EXP / "sizes" / s / "box"
    vals = (*abx(b / "results/abx/Buckeye_SpidR_adapted/NotNormalised/zs.abx_layer06/abx_result.txt"), *ap(b / "spidr_ap/results.json"))
    row(f"SpidR L6 {s}", vals)
row("SpidR paper table", PAPER["SpidR"])

recs = {}
fn = EXP / "results/timings.jsonl"
if fn.exists():
    for line in fn.read_text().splitlines():
        r = json.loads(line)
        recs[r["stage"]] = r
print(f"\n{'Compute (timed runs)':<30}{'wall s':>9}{'CPU s':>9}{'RAM GB':>8}{'GPU GB':>8}  others")
for name, r in recs.items():
    if name.startswith(("full_", "gold_", "1h_")):
        print(f"{name:<30}{r['wall_s']:>9.0f}{r['user_s'] + r['sys_s']:>9.0f}{r['max_rss_mb'] / 1024:>8.1f}{r['gpu_peak_mb'] / 1024:>8.1f}  {','.join(r['others_on_gpu'])}"
              + (f"  marks={r['marks']}" if r["marks"] else ""))

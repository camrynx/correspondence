#!/usr/bin/env python
"""Amount-of-data figure, after SpidR-Adapt's Fig. 2: ABX (mean of within- and across-speaker) and AP vs adaptation
speech, English test set. Reads whatever has finished, so re-run it as sizes complete.
    python experiments/compute_data/plot.py    -> figures/data_amount.{pdf,png}, figures/data_amount.csv
"""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent
SIZES = ["10min", "30min", "1h", "2h", "4h", "full"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE_GREY = "#9A9891"  # \definecolor{baseGrey} in main.tex
WAVLM = {"WS": 5.35, "AS": 6.91, "AP": 0.860}  # baseline row of Table II, which reproduces on disk
SERIES = [  # colour = main.tex's caeBlue, then slots 2 and 3 of the validated default palette; markers carry identity too
    ("cAE", "#2A78D6", "o", lambda b: (b / "results/abx/Buckeye_WavLMDUSTED/Normalised/zs.abx_cAE_layer21/abx_result.txt",
                                       b / "results/ap/Buckeye_WavLMDUSTED/Normalised/zs.samediff_cAE/layer_21/results.json")),
    ("NCE", "#EB6834", "s", lambda b: (b / "results/abx/Buckeye_WavLMDUSTED/Normalised/zs.abx_ccAErandinit_layer21/abx_result.txt",
                                       b / "results/ap/Buckeye_WavLMDUSTED/Normalised/zs.samediff_ccAErandinit/layer_21/results.json")),
    ("SpidR-Adapt", "#1BAF7A", "^", lambda b: (b / "results/abx/Buckeye_SpidR_adapted/NotNormalised/zs.abx_layer06/abx_result.txt",
                                               b / "spidr_ap/results.json")),
]


def read(paths):
    abx_fn, ap_fn = paths
    if not (abx_fn.is_file() and ap_fn.is_file()):
        return None
    t = abx_fn.read_text()
    ws, as_ = (float(re.search(rf"{k}-speaker.*?([\d.]+)%", t).group(1)) for k in ("Within", "Across"))
    return ws, as_, json.loads(ap_fn.read_text())["AP"]


hours = {s: json.loads((EXP / "sizes" / s / "subset.json").read_text())["speech_h"]
         for s in SIZES if (EXP / "sizes" / s / "subset.json").exists()}
data = {name: {s: r for s in hours if (r := read(paths(EXP / "sizes" / s / "box")))} for name, _, _, paths in SERIES}
pre = read(SERIES[2][3](EXP / "sizes/preadapt/box"))

with open(EXP / "figures/data_amount.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "size", "speech_h", "abx_within", "abx_across", "abx_mean", "ap"])
    for name, rows in data.items():
        for s, (ws, as_, ap) in rows.items():
            w.writerow([name, s, f"{hours[s]:.3f}", ws, as_, f"{(ws + as_) / 2:.3f}", f"{ap:.4f}"])

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharex=True, constrained_layout=True)  # IEEE column width
panels = [(axes[0], "ABX error (%), lower is better", lambda r: (r[0] + r[1]) / 2, (WAVLM["WS"] + WAVLM["AS"]) / 2),
          (axes[1], "AP, higher is better", lambda r: r[2], WAVLM["AP"])]
xs = sorted(hours.values()) or [1 / 6, 6]
for ax, ylabel, val, base in panels:
    ax.axhline(base, color=BASE_GREY, lw=1.2, ls=(0, (4, 2)), zorder=1, label="WavLM baseline")
    if pre:
        ax.axhline(val(pre), color=SERIES[2][1], lw=1.0, ls=(0, (1, 1.5)), zorder=1, label="SpidR-Adapt, unadapted")
    for name, color, marker, _ in SERIES:
        rows = data[name]
        if rows:
            order = sorted(rows, key=lambda s: hours[s])
            ax.plot([hours[s] for s in order], [val(rows[s]) for s in order], color=color, lw=1.4, marker=marker, ms=4.2,
                    markeredgecolor="white", markeredgewidth=0.5, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.14, 7.2)
    ticks = [(1 / 6, "10m"), (0.5, "30m"), (1, "1h"), (2, "2h"), (3.91, "4h"), (5.98, "6h")]
    ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Target-domain speech")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5, labelsize=7)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=len(l) if len(l) <= 4 else 3, frameon=False, fontsize=7, handlelength=2.2, columnspacing=1.0)
fig.savefig(EXP / "figures/data_amount.pdf")
fig.savefig(EXP / "figures/data_amount.png", dpi=300)
print({k: sorted(v, key=lambda s: hours[s]) for k, v in data.items()}, "preadapt" if pre else "no preadapt")

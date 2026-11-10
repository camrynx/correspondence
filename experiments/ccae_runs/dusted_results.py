"""Results of run_dusted_cae.sh: cAE + random-init ccAE on Xitsonga / French DUSTED (test, layer 21).

    python experiments/ccae_runs/dusted_results.py

"new" = written by today's run (after the 2026-10-07 backup); otherwise the value shown is still
the old file. "before" = the backed-up value from before this run (experiments/ccae_runs/backup_before_2026-10-07/).
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / "experiments/ccae_runs/backup_before_2026-10-07"
RUN_START = datetime(2026, 10, 7, 10, 36)


def read(base, set_, prefix):
    ap_fn = base / f"results/ap/{set_}/Normalised/test.samediff_{prefix}/layer_21/results.json"
    abx_fn = base / f"results/abx/{set_}/Normalised/test.abx_{prefix}_layer21/abx_result.txt"
    vals = ["-"] * 5
    new = [False, False]
    if abx_fn.is_file():
        t = abx_fn.read_text()
        for i, k in enumerate(("Within", "Across")):
            m = re.search(rf"{k}-speaker.*?([\d.]+)%", t)
            vals[i] = m.group(1) if m else "-"
        new[0] = datetime.fromtimestamp(abx_fn.stat().st_mtime) >= RUN_START
    if ap_fn.is_file():
        r = json.loads(ap_fn.read_text())
        vals[2:] = [f"{r[k]:.4f}" for k in ("AP", "AP SWSP", "AP SWDP")]
        new[1] = datetime.fromtimestamp(ap_fn.stat().st_mtime) >= RUN_START
    return vals, new


print(f"{'':<34}{'Within':>8}{'Across':>8}{'AP':>9}{'SWSP':>9}{'SWDP':>9}   status (ABX / AP)")
for lang, set_ in (("Xitsonga", "Xitsonga_WavLMDUSTED"), ("French", "ZS17French_WavLMDUSTED")):
    for model, prefix in (("cAE", "cAE"), ("ccAE", "ccAErandinit")):
        vals, new = read(ROOT, set_, prefix)
        status = " / ".join("new" if n else "old (pending)" for n in new)
        print(f"{lang} {model} (UTD)".ljust(34) + "".join(f"{v:>8}" if i < 2 else f"{v:>9}" for i, v in enumerate(vals)) + f"   {status}")
        before, _ = read(BACKUP, set_, prefix)
        print(f"  before this run".ljust(34) + "".join(f"{v:>8}" if i < 2 else f"{v:>9}" for i, v in enumerate(before)))
    print()

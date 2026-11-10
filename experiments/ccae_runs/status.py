"""Progress + results table for experiments/ccae_runs/run_all.sh.

    python experiments/ccae_runs/status.py              # random-init ccAE (ccAErandinit)
    watch -n 30 python experiments/ccae_runs/status.py
"""
import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE, LOGS = ROOT / "experiments/ccae_runs/state", ROOT / "experiments/ccae_runs/logs"

CORPORA = [("English", "buckeye", "Buckeye_WavLM", "devpart2"),
           ("Xitsonga", "xitsonga", "Xitsonga_WavLM", "test"),
           ("French", "zs17french", "ZS17French_WavLM", "test")]
ROWS = [("ccAE", "gold", ""), ("ccAE (UTD)", "dusted", "DUSTED"), ("ccAE (Poor Man UTD)", "poorman", "PoorManUTD")]


def abx(set_, split, prefix):
    fn = ROOT / f"results/abx/{set_}/Normalised/{split}.abx_{prefix}_layer21/abx_result.txt"
    if not fn.is_file():
        return "-", "-"
    txt = fn.read_text()
    w = re.search(r"Within-speaker.*?([\d.]+)%", txt)
    a = re.search(r"Across-speaker.*?([\d.]+)%", txt)
    return (w.group(1) if w else "-"), (a.group(1) if a else "-")


def ap(set_, split, prefix):
    fn = ROOT / f"results/ap/{set_}/Normalised/{split}.samediff_{prefix}/layer_21/results.json"
    if not fn.is_file():
        return "-", "-", "-"
    r = json.loads(fn.read_text())
    return tuple(f"{r[k]:.4f}" for k in ("AP", "AP SWSP", "AP SWDP"))


def last_progress_line(log):
    """Last non-empty line of a log, with tqdm carriage-return updates collapsed."""
    try:
        data = log.read_bytes()[-20000:].decode(errors="replace")
    except OSError:
        return ""
    lines = [l for l in re.split(r"[\r\n]+", data) if l.strip()]
    return lines[-1].strip()[:160] if lines else ""


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--prefix", default=None)
    prefix = ap_.parse_args().prefix or "ccAErandinit"

    print(f"=== ccAE pipeline status  ({time.strftime('%F %T')}, model prefix: {prefix}) ===\n")
    done = sorted(p.stem for p in STATE.glob("*.done")) if STATE.is_dir() else []
    failed = sorted(p.stem for p in STATE.glob("*.failed")) if STATE.is_dir() else []
    cur = (STATE / "CURRENT").read_text().strip() if (STATE / "CURRENT").is_file() else None
    total = 58  # wait_gpu + 12 prep/alignment stages + 9 alignment checks + 9 runs x (train, AP, ABX features, ABX score)
    print(f"Stages done: {len(done)}/{total}" + (f"   FAILED: {', '.join(failed)}" if failed else ""))
    if cur:
        print(f"Running now: {cur}")
        print(f"  {last_progress_line(LOGS / f'{cur}.log')}")
    else:
        print("Running now: (nothing - finished, failed, or not started)")
    print()

    hdr = f"{'':<36}{'Within-Spk':>11}{'Across-Spk':>11}{'AP':>9}{'AP SWSP':>9}{'AP SWDP':>9}"
    for lang, _corpus, base, split in CORPORA:
        print(f"{lang}, WavLM Large Layer 21".ljust(36) + hdr[36:])
        for label, _pairs, suffix in ROWS:
            set_ = base + suffix
            w, a = abx(set_, split, prefix)
            print(f"  {label:<34}{w:>11}{a:>11}" + "".join(f"{v:>9}" for v in ap(set_, split, prefix)))
        print()

    log = LOGS / "pipeline.log"
    if log.is_file():
        print("Recent pipeline events:")
        for line in log.read_text().splitlines()[-6:]:
            print("  " + line)


if __name__ == "__main__":
    main()

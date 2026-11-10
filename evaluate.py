#!/usr/bin/env python
"""Same-different AP and triphone ABX for trained models.

    python evaluate.py --buckeye --mfcc --gold --cae
    python evaluate.py --french --wavlm --poorman --cae --ccae     # every combination
    python evaluate.py --xitsonga --wavlm --dusted --ccae -n       # dry run

AP runs evaluation/evaluate_ap.py on the word features written by training. ABX encodes
the item file's segments with the checkpoint (extract_abx_phone_features.py) and scores
them (score_abx.py). Logs go to results/logs/ (--verbose to stream them).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "results" / "logs"

# Steps run under the same interpreter as this script, so run it from the buckeye env
# (conda activate buckeye): the step scripts need its torch/samediff/fastabx.
PYTHON = sys.executable


@dataclass(frozen=True)
class Corpus:
    label: str
    prefix: str  # feature sets are <prefix>_<feature tag>, e.g. Buckeye_WavLM
    subset: str  # default evaluation subset (the one the results tables report)
    items_dir: str  # ABX item files, one <subset>.item per subset
    ccae_name: str  # train_ccae.py --corpus value


CORPORA = {
    "buckeye": Corpus("Buckeye", "Buckeye", "zs", "data/Information/Buckeye/abx_items", "buckeye"),
    "xitsonga": Corpus("Xitsonga", "Xitsonga", "test", "data/Information/Buckeye_tsonga/abx_items", "xitsonga"),
    "french": Corpus("French", "ZS17French", "test", "data/Information/ZS17French/abx_items", "zs17french"),
}


@dataclass(frozen=True)
class FeatureType:
    label: str
    tag: str  # feature-set and models/ directory name component
    default_layer: int | None  # None = no layers (MFCC): no Normalised/ or layer_NN/ dirs either


FEATURE_TYPES = {
    "mfcc": FeatureType("MFCC", "MFCCs", None),
    "wavlm": FeatureType("WavLM", "WavLM", 21),
    "hubert": FeatureType("HuBERT", "HuBERT", 24),
}


@dataclass(frozen=True)
class Pairing:
    label: str
    variant: str  # suffix on the feature set and checkpoint name ("" for gold)
    ccae_pairs: str | None  # train_ccae.py --pairs value; None = no ccAE for this pairing


PAIRINGS = {
    "gold": Pairing("Gold", "", "gold"),
    "dusted": Pairing("DUSTED", "DUSTED", "dusted"),
    "poorman": Pairing("Poor Man's UTD", "PoorManUTD", "poorman"),
    # Same-speaker neighbours allowed (other word tokens only); WavLM cAE only.
    "poorman_anyspk": Pairing("Poor Man's UTD, any speaker", "PoorManUTDAnySpk", None),
    "utd": Pairing("UTD", "UTD", "utd"),
}


@dataclass(frozen=True)
class Model:
    label: str
    prefix: str  # checkpoint / output name prefix
    abx_flags: tuple[str, ...]  # extract_abx_phone_features.py flags selecting the checkpoint


MODELS = {
    "cae": Model("cAE", "cAE", ()),
    "ccae": Model("ccAE", "ccAErandinit", ("--ccae", "--randinit")),
}

ABX_LINE_RE = re.compile(r"(Within|Across)-speaker, within-context, triphone ABX error: ([\d.]+)%")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def has_files(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() for p in path.iterdir())


def fmt_time(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


class Job:
    """One (corpus, feature type, pairing, model) combination and every path its evaluation touches."""

    def __init__(self, corpus_key: str, feature_key: str, pairing_key: str, model_key: str,
                 subset: str | None, layer: int | None):
        self.corpus = CORPORA[corpus_key]
        self.feature = FEATURE_TYPES[feature_key]
        self.pairing = PAIRINGS[pairing_key]
        self.model = MODELS[model_key]
        self.subset = subset or self.corpus.subset
        self.layer = None if self.feature.default_layer is None else (layer or self.feature.default_layer)

        base = f"{self.corpus.prefix}_{self.feature.tag}"
        variant = self.pairing.variant
        self.feature_set = base + variant
        prefix = self.model.prefix
        features = REPO_ROOT / "data/Features"

        # MFCC sets have no layers and no global normalisation (speaker MVN at extraction), so
        # their directories sit straight under the set; WavLM/HuBERT nest Normalised/.../layer_NN.
        if self.layer is None:
            layer_dir, layer_tag, abx_layer, normalised = (), "", "", ()
        else:
            layer_dir = (f"layer_{self.layer:02d}",)
            layer_tag, abx_layer, normalised = f"_layer_{self.layer:02d}", f"_layer{self.layer:02d}", ("Normalised",)

        ckpt_variant = f"_{variant}" if variant else ""
        self.ckpt = REPO_ROOT / "models" / base / f"{prefix}{ckpt_variant}{layer_tag}.pt"
        encoded_sub = Path(self.feature_set, *normalised, f"{self.subset}.samediff_{prefix}", *layer_dir)
        self.encoded = features / encoded_sub
        self.ap_result = REPO_ROOT / "results/ap" / encoded_sub / "results.json"
        abx_sub = Path(self.feature_set, *normalised, f"{self.subset}.abx_{prefix}{abx_layer}")
        self.abx_feats = features / abx_sub
        self.abx_result = REPO_ROOT / "results" / "abx" / abx_sub / "abx_result.txt"
        self.item_file = REPO_ROOT / self.corpus.items_dir / f"{self.subset}.item"

        abx_layer_args = ("--layer", str(self.layer)) if self.layer is not None else ()
        self.ap_cmd = [PYTHON, "evaluation/evaluate_ap.py", rel(self.encoded)]
        self.abxfeat_cmd = [PYTHON, "evaluation/extract_abx_phone_features.py", self.feature_set,
                            "--subset", self.subset, *self.model.abx_flags, *abx_layer_args]
        self.abx_cmd = [PYTHON, "evaluation/score_abx.py", rel(self.abx_feats), rel(self.item_file)]

    @property
    def title(self) -> str:
        layer = f", layer {self.layer}" if self.layer is not None else ""
        return (f"{self.feature.label} {self.model.label} {self.pairing.label} "
                f"({self.corpus.label}, {self.subset}{layer})")

    @property
    def log_stem(self) -> str:
        layer = f"_layer{self.layer}" if self.layer is not None else ""
        return f"{self.feature_set}_{self.subset}_{self.model.prefix}{layer}"

    def train_hint(self) -> str:
        if self.model.prefix.startswith("ccAE"):
            if self.pairing.ccae_pairs is None:
                return f"train_ccae.py has no '{self.pairing.label}' pairing"
            feature_args = {"WavLM": f"--layer {self.layer}", "HuBERT": f"--features hubert --layer {self.layer}",
                            "MFCCs": "--features mfcc"}[self.feature.tag]
            return (f"written by: python training/train_ccae.py --corpus {self.corpus.ccae_name} "
                    f"--pairs {self.pairing.ccae_pairs} --init random {feature_args}")
        if self.pairing.ccae_pairs is None and self.pairing.variant.endswith("AnySpk"):
            pairs = ["--pairs", "poorman", "--allow-same-speaker"]
        else:
            pairs = ["--pairs", self.pairing.ccae_pairs]
        feature = {"MFCCs": "mfcc", "WavLM": "wavlm", "HuBERT": "hubert"}[self.feature.tag]
        corpus = "french" if self.corpus.ccae_name == "zs17french" else self.corpus.ccae_name
        return "written by: " + shlex.join(["python", "training/train_cae.py", "--features", feature,
                                            "--corpus", corpus, *pairs])

    def ap_problem(self) -> str | None:
        if not has_files(self.encoded):
            return f"no features at {rel(self.encoded)}\n      {self.train_hint()}"
        return None

    def abx_problem(self) -> str | None:
        if not self.ckpt.is_file():
            return f"no checkpoint at {rel(self.ckpt)}\n      {self.train_hint()}"
        if not self.item_file.is_file():
            return f"no item file at {rel(self.item_file)} (see evaluation/build_abx_item_file.py)"
        return None

    def warnings(self) -> list[str]:
        """The AP features were encoded by an older checkpoint than the one ABX will use."""

        if self.ckpt.is_file() and has_files(self.encoded):
            ckpt_t = self.ckpt.stat().st_mtime
            feats_t = min(p.stat().st_mtime for p in self.encoded.iterdir() if p.is_file())
            if feats_t < ckpt_t:
                return [f"{rel(self.encoded)} ({fmt_time(feats_t)}) is older than {rel(self.ckpt)} "
                        f"({fmt_time(ckpt_t)}): AP and ABX would come from different models"]
        return []

    def results(self) -> dict[str, float]:
        res = {}
        if self.ap_result.is_file():
            data = json.loads(self.ap_result.read_text())
            for key in ("AP", "AP SWSP", "AP SWDP"):
                if key in data:
                    res[key] = data[key]
        if self.abx_result.is_file():
            for cond, val in ABX_LINE_RE.findall(self.abx_result.read_text()):
                res[f"ABX {cond}"] = float(val)
        return res

    def print_results(self, do_ap: bool, do_abx: bool, ran: set[str] | None = None) -> None:
        """Print the metrics; with `ran`, metrics whose step failed show 'not run'."""

        res = self.results()
        names = (["AP", "AP SWSP", "AP SWDP"] if do_ap else []) + (["ABX Within", "ABX Across"] if do_abx else [])
        for name in names:
            v = res.get(name)
            if ran is not None and name.split()[0] not in ran:
                shown = "not run"
            elif v is None:
                shown = "-"
            else:
                shown = f"{v:.2f}%" if name.startswith("ABX") else f"{v:.4f}"
            print(f"{name}: {shown}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Several flags in one group run every combination, e.g. "
               "'--buckeye --xitsonga --wavlm --gold --cae --ccae' evaluates four models.",
    )
    corpus = parser.add_argument_group("corpus (at least one)")
    corpus.add_argument("--buckeye", action="store_true", help="English, Buckeye (default subset zs, the held-out test speakers)")
    corpus.add_argument("--xitsonga", action="store_true", help="Xitsonga, NCHLT (default subset test)")
    corpus.add_argument("--french", "--zs17french", dest="french", action="store_true",
                        help="French, ZeroSpeech 2017 (default subset test)")
    feats = parser.add_argument_group("feature type (at least one)")
    feats.add_argument("--mfcc", action="store_true", help="MFCCs")
    feats.add_argument("--wavlm", action="store_true", help="WavLM Large (default layer 21)")
    feats.add_argument("--hubert", action="store_true", help="HuBERT Large (default layer 24)")
    pairs = parser.add_argument_group("training pairs (at least one)")
    pairs.add_argument("--gold", action="store_true", help="forced-alignment word pairs")
    pairs.add_argument("--dusted", action="store_true", help="DUSTED discovered-term pairs")
    pairs.add_argument("--poorman", action="store_true", help="Poor Man's UTD frame pairs")
    pairs.add_argument("--poorman-anyspk", action="store_true",
                       help="Poor Man's UTD frame pairs with same-speaker neighbours allowed - WavLM cAE only")
    pairs.add_argument("--utd", action="store_true", help="classic UTD (ZRTools) pairs - Buckeye/Xitsonga only")
    model = parser.add_argument_group("model (at least one)")
    model.add_argument("--cae", action="store_true", help="correspondence autoencoder (MSE)")
    model.add_argument("--ccae", action="store_true", help="contrastive cAE (InfoNCE), random init")

    opts = parser.add_argument_group("options")
    opts.add_argument("--subset", "--split", dest="subset",
                      help="subset to evaluate, e.g. devpart1/devpart2/zs (Buckeye), dev/test (Xitsonga), "
                           "train/test (French); default: each corpus's reported subset")
    only = opts.add_mutually_exclusive_group()
    only.add_argument("--ap-only", action="store_true", help="skip ABX")
    only.add_argument("--abx-only", action="store_true", help="skip same-different AP")
    opts.add_argument("--layer", type=int, help="WavLM/HuBERT layer (default: 21 for WavLM, 24 for HuBERT)")
    opts.add_argument("--dry-run", "-n", action="store_true",
                      help="print the exact commands and the results already on disk, run nothing")
    opts.add_argument("--verbose", "-v", action="store_true",
                      help="stream each step's output instead of writing it to results/logs/")
    args = parser.parse_args()

    groups = {
        "a corpus (--buckeye/--xitsonga/--french)": CORPORA,
        "a feature type (--mfcc/--wavlm/--hubert)": FEATURE_TYPES,
        "training pairs (--gold/--dusted/--poorman/--poorman-anyspk/--utd)": PAIRINGS,
        "a model (--cae/--ccae)": MODELS,
    }
    chosen = {name: [k for k in options if getattr(args, k)] for name, options in groups.items()}
    missing = [name for name, keys in chosen.items() if not keys]
    if missing:
        parser.error("choose " + ", ".join(missing))
    args.corpora, args.features, args.pairings, args.models = chosen.values()
    return args


def run_step(job: Job, name: str, cmd: list[str], verbose: bool) -> bool:
    start = time.perf_counter()
    if verbose:
        print(f"  $ {shlex.join(cmd)}", flush=True)
        ok = subprocess.run(cmd, cwd=REPO_ROOT).returncode == 0
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_fn = LOG_DIR / f"{job.log_stem}_{name.replace(' ', '').lower()}.log"
        print(f"  {name:<9} running (log: {rel(log_fn)})", flush=True)
        with log_fn.open("w") as log:
            log.write(f"$ {shlex.join(cmd)}\n")
            log.flush()
            ok = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode == 0
        if not ok:
            tail = log_fn.read_text(errors="replace").splitlines()[-15:]
            print("\n".join("    | " + line for line in tail))
    print(f"  {name:<9} {'done' if ok else 'FAILED'} in {time.perf_counter() - start:.0f} s")
    return ok


def main() -> int:
    args = parse_args()
    do_ap, do_abx = not args.abx_only, not args.ap_only

    jobs = [Job(c, f, p, m, args.subset, args.layer)
            for c in args.corpora for f in args.features for p in args.pairings for m in args.models]


    failed = False
    ran: dict[int, set[str]] = {}  # job index -> metrics ("AP", "ABX") computed successfully this run
    for i, job in enumerate(jobs):
        print(f"\nEvaluating: {job.title}")
        for w in job.warnings():
            print(f"  WARNING: {w}")

        groups: list[tuple[str, list[tuple[str, list[str]]]]] = []
        for metric, wanted, problem, steps in (
            ("AP", do_ap, job.ap_problem(), [("AP", job.ap_cmd)]),
            ("ABX", do_abx, job.abx_problem(), [("ABX feats", job.abxfeat_cmd), ("ABX", job.abx_cmd)]),
        ):
            if not wanted:
                continue
            if problem:
                print(f"  cannot run {metric}: {problem}")
                failed = True
            else:
                groups.append((metric, steps))

        ran[i] = set()
        for metric, steps in groups:
            for name, cmd in steps:
                if args.dry_run:
                    print(f"  would run {name}:\n    $ {shlex.join(cmd)}")
                elif not run_step(job, name, cmd, args.verbose):
                    failed = True
                    break  # the rest of this group (ABX scoring) needs this step's output
            else:
                ran[i].add(metric)

        job.print_results(do_ap, do_abx, None if args.dry_run else ran[i])

    if len(jobs) > 1 and not args.dry_run:
        print("\n==================== Summary ====================")
        for i, job in enumerate(jobs):
            print(f"\nEvaluating: {job.title}")
            job.print_results(do_ap, do_abx, ran[i])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

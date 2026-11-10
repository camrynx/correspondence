"""Unadapted SpidR-Adapt on ZS17 French (the repo has no SpidR extractor for French). Runs inside a box.

Runs features/extract_ssl.py with a SpidR model dressed up as a HF model
(hidden_states[L] == SpidR layer L). Features are left unnormalised, as for extractFeaturesSpidR(TSO).py.

    python spidr_french.py <extract|abx|check> <checkpoint> <layer>
"""
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

BOX = Path.cwd()
FEATURES = "data/Features/ZS17French_SpidR_unadapted/NotNormalised"
sys.path.insert(0, str(BOX / "evaluation"))
step, ckpt, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])


class SpidRAsHF(nn.Module):
    def __init__(self):
        super().__init__()
        from spidr_adapt.models import build_model

        self.m = build_model(model_type="spidr_reset", checkpoint=ckpt)
        self.config = SimpleNamespace(feat_extract_norm="n/a (SpidR-Adapt)")
        print(f"Loaded {ckpt}: {len(self.m.student.layers)} layers")

    def forward(self, x, output_hidden_states=True):
        return SimpleNamespace(hidden_states=[None] + self.m.get_intermediate_outputs(x))


fake = types.ModuleType("transformers")
fake.AutoModel = SimpleNamespace(from_pretrained=lambda *a, **k: SpidRAsHF())
sys.modules["transformers"] = fake


def load_extractor():
    """features/extract_ssl.py, with the fake transformers above and SpidR's 768-d layers."""
    spec = importlib.util.spec_from_file_location("extract_ssl", BOX / "features/extract_ssl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODEL_DIMENSIONS = 768
    return mod


if step == "extract":
    mod = load_extractor()
    os.environ["SD_FEATURE_SET"] = FEATURES.split("/")[2]  # -> data/Features/<set>/NotNormalised
    sys.argv = ["extract_ssl.py", "--model", "wavlm", "--corpus", "french", "--splits", "test", "--layers", str(layer)]
    mod.main()
elif step == "abx":
    import extract_abx_phone_features as ex

    feats = ex.extract_wavlm("zs17french", "wavlm", "test", ex.CONFIGS[("zs17french", "wavlm")], layer=layer)
    out = BOX / FEATURES / f"test.abx_layer{layer:02d}"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in feats.items():
        np.save(out / f"{k}.npy", v.astype(np.float32))
    print(f"Wrote {len(feats)} segment files to {out}")
elif step == "check":
    mod = load_extractor()
    import extract_abx_phone_features as ex

    x = torch.zeros(1, 16000)
    hs = mod.AutoModel.from_pretrained(ckpt)(x, output_hidden_states=True).hidden_states
    print("layers", len(hs) - 1, "layer shape", hs[layer].shape, "config", ex.CONFIGS[("zs17french", "wavlm")]["const_prefix"])

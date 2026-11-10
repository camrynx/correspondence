#!/usr/bin/env python
"""Does one reset at the start reproduce the paper's SpidR row better than no reset? Full English data only."""
from run import EXP, PY, SPIDR_BIN, SPIDR_INIT, build_box, log, spidr_eval, stage

name = "fullreset"
run = EXP / "sizes" / name / "spidr"
box = EXP / "sizes" / name / "box"
build_box(box)
wavs = EXP / "sizes/full/dusted/wavs"
stage(f"{name}_spidr_cfg", [PY, EXP / "make_spidr_cfg.py", wavs, run, name, SPIDR_INIT], EXP)
stage(f"{name}_spidr_adapt", [SPIDR_BIN / "torchrun", "--standalone", "--nproc-per-node", 1, EXP / "spidr_local_reset.py", run / "config.toml"], EXP, gpu=True)
spidr_eval(name, box, run / "spidr_ft" / name / "final.pt")
log("RESET CHECK DONE")

"""Single-GPU, no-Slurm launcher for the released SpidR-Adapt fine-tuning recipe (finetuning.py does the same via submitit).
    torchrun --standalone --nproc-per-node 1 spidr_local.py config.toml

The released train() never calls model.set_task_interval(), so SpidRWithReset.should_reset() raises
AttributeError on the first step. We supply the value finetuning.toml already sets (task_interval = 20001):
no reset fires: the checkpoint's step counter is 200000 on the first step (200000 % 20001 != 0), then 1..8000.
"""
import sys
from spidr_adapt.config import read_config
from spidr_adapt.models.spidr_reset import SpidRWithReset
from spidr_adapt.train import train
cfg = read_config(sys.argv[1])
SpidRWithReset.task_interval = cfg.meta_update.task_interval
train(cfg)

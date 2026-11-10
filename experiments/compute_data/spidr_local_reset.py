"""Variant of spidr_local.py: Active Forgetting (codebook + head reset) fires exactly once, on the first training step.
With the checkpoint's current_step = 200000, the task_interval = 20001 route in spidr_local.py never fires at all.
"""
import sys
from spidr_adapt.config import read_config
from spidr_adapt.models.spidr_reset import SpidRWithReset
from spidr_adapt.train import train
cfg = read_config(sys.argv[1])


def once(self, step):
    if not self.training or getattr(self, "_reset_done", False):
        return False
    self._reset_done = True
    return True


SpidRWithReset.should_reset = once
train(cfg)

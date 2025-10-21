# avg_tb.py
import os, random
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from torch.utils.tensorboard import SummaryWriter as _TBWriter

class AvgLogger:
    def __init__(self):
        self._sum = defaultdict(lambda: defaultdict(float))
        self._cnt = defaultdict(lambda: defaultdict(int))

    def add(self, tag, step, val):
        if step is None:
            return
        s = int(step)
        self._sum[tag][s] += float(val)
        self._cnt[tag][s] += 1

    def write_avg(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        w = _TBWriter(logdir)
        for tag, steps in self._sum.items():
            for step in sorted(steps):
                c = self._cnt[tag][step]
                if c:
                    w.add_scalar(tag, steps[step] / c, step)
        w.flush(); w.close()

class SummaryWriter(_TBWriter):
    """Drop-in: mirrors add_scalar to an aggregator when provided."""
    def __init__(self, log_dir, aggregator=None, **kwargs):
        super().__init__(log_dir, **kwargs)
        self._aggregator = aggregator

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        if self._aggregator is not None:
            self._aggregator.add(tag, global_step, scalar_value)
        return super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)

def set_seed(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

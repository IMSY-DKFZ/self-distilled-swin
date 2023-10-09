# helper functions
import os
import random

import numpy as np
import torch


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Table for printing results
header = r"""

Epoch |  loss | vloss | mAP | imAP | Time, m
"""
#          Epoch         metrics            time
raw_line = "{:6d}" + "\u2502{:7.3f}" * 4 + "\u2502{:6.2f}"

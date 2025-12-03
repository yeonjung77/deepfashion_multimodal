import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Align random, numpy, torch seeds. Optionally enforce determinism on CUDA kernels.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

"""
Set random seeds for reproducibility across different libraries.
"""

import random
import numpy as np


def set_seed(seed: int):
    """
    Set random seeds for Python, NumPy, and PyTorch (if available).
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


"""
Differential privacy parameter tracking and noise mechanisms.
"""

import numpy as np
from typing import Dict


def compute_privacy_parameters(n: int, epsilon: float, delta: float = None) -> Dict[str, float]:
    """
    Compute differential privacy parameters.
    
    Args:
        n: Dataset size.
        epsilon: Privacy budget (lower is more private).
        delta: Failure probability (defaults to 1/n^2).
    
    Returns:
        Dictionary with computed parameters.
    """
    if delta is None:
        delta = 1.0 / (n ** 2)
    
    return {
        'epsilon': epsilon,
        'delta': delta,
        'n': n
    }


def laplace_noise_scale(sensitivity: float, epsilon: float) -> float:
    """
    Compute Laplace noise scale for (epsilon, 0)-differential privacy.
    
    The Laplace mechanism with scale b = sensitivity/epsilon provides
    epsilon-differential privacy.
    
    Args:
        sensitivity: Maximum change in query output when one record changes.
        epsilon: Privacy budget.
    
    Returns:
        Scale parameter b for Laplace distribution.
    """
    return sensitivity / epsilon


def laplace_noise(shape, scale: float, seed: int = None) -> np.ndarray:
    """
    Generate Laplace noise for differential privacy.
    
    Args:
        shape: Shape of output array.
        scale: Scale parameter (b = sensitivity / epsilon).
        seed: Random seed.
    
    Returns:
        Array of i.i.d. Laplace random variables.
    """
    if seed is not None:
        np.random.seed(seed)
    
    u = np.random.uniform(low=-0.5, high=0.5, size=shape)
    noise = -scale * np.sign(u) * np.log(1 - 2 * np.abs(u))
    return noise


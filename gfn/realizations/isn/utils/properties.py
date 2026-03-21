"""
Centralized utilities for extracting and validating entity properties.
Used by WorldPhysics, EntityFactory, and Coherence verification.
"""

import torch
import numpy as np


def is_prime(n: int) -> bool:
    """Standard primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True


def extract_numeric_properties(value: float, d_properties: int) -> torch.Tensor:
    """
    Standard property vector extraction for numbers.
    
    p = [magnitude, sign, parity, is_prime, magnitude_order, ...]
    """
    properties = torch.zeros(d_properties)
    
    properties[0] = value  # magnitude
    properties[1] = np.sign(value)  # sign
    properties[2] = int(abs(value)) % 2  # parity
    properties[3] = float(is_prime(int(abs(value))))  # primality
    
    if abs(value) > 1e-10:
        properties[4] = np.floor(np.log10(abs(value)))  # order of magnitude
    else:
        properties[4] = -10.0 # Smallest log10 scale
        
    return properties


def validate_number_properties(properties: torch.Tensor) -> int:
    """
    Validate if a property vector satisfies numeric constraints.
    Returns the number of violations.
    """
    violations = 0
    
    # Check magnitude is finite
    if not torch.isfinite(properties[0]):
        violations += 1
    
    # Check sign is -1, 0, or 1
    sign = properties[1]
    if sign not in [-1, 0, 1]:
        violations += 1
    
    # Check parity is 0 or 1
    parity = properties[2]
    if parity not in [0, 1]:
        violations += 1
        
    return violations

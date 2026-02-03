"""
Unit Test Configuration
=======================

Pytest configuration and fixtures for GFN tests.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_batch():
    """Small batch size for quick tests."""
    return 2


@pytest.fixture
def medium_batch():
    """Medium batch size for standard tests."""
    return 8


@pytest.fixture
def small_dim():
    """Small dimension for quick tests."""
    return 32


@pytest.fixture
def medium_dim():
    """Medium dimension for standard tests."""
    return 64


@pytest.fixture
def large_dim():
    """Large dimension for stress tests."""
    return 256


@pytest.fixture
def short_seq():
    """Short sequence length."""
    return 10


@pytest.fixture
def medium_seq():
    """Medium sequence length."""
    return 50


@pytest.fixture
def long_seq():
    """Long sequence length."""
    return 200


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

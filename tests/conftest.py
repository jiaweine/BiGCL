"""Shared fixtures for BiGCL tests."""

import sys
import os

import pytest
import torch
import numpy as np

# Add BiGCL package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BiGCL"))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def rng():
    return np.random.default_rng(42)

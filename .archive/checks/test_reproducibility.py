"""Check evaluator RNG repeatability without downloading judge weights."""
import random

import numpy as np
import torch

from driftbench_runner.reproducibility import seed_evaluator


def test_evaluator_repeats_and_records_controls(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '42')
    prior = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
             torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)
    try:
        policy = seed_evaluator()
        first = (random.random(), np.random.rand(), torch.rand(4))
        assert seed_evaluator() == policy
        second = (random.random(), np.random.rand(), torch.rand(4))
        assert first[:2] == second[:2] and torch.equal(first[2], second[2])
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert policy['seed'] == 42 and policy['python_hash_seed'] == '42'
    finally:
        torch.use_deterministic_algorithms(prior[0])
        torch.backends.cudnn.benchmark = prior[1]
        torch.backends.cudnn.deterministic = prior[2]
        torch.backends.cuda.matmul.allow_tf32 = prior[3]
        torch.backends.cudnn.allow_tf32 = prior[4]

"""Controls for the common PyTorch evaluators (not vendor serving kernels)."""
import os
import random

SEED = 42


def seed_evaluator():
    # Set before CUDA is initialized. Unsupported deterministic operations fail
    # instead of silently producing scores with a weaker evaluator policy.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"seed": SEED, "deterministic_algorithms": True,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "allow_tf32": False, "cublas_workspace_config": ":4096:8",
            "python_hash_seed": os.environ.get("PYTHONHASHSEED")}

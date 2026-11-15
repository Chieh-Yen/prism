"""PRISM: geometric risk-discrepancy bounds for proxy models."""

# Intra-op CPU thread cap for the CPU-side linear algebra (eigh / SVD / matmul
# and the reductions over them). The experiments were run with 8 threads;
# PRISM_CPU_THREADS overrides it and PRISM_CPU_THREADS=0 leaves the PyTorch
# default untouched.
import os as _os

import torch as _torch

_n = int(_os.environ.get("PRISM_CPU_THREADS", "8"))
if _n > 0:
    _torch.set_num_threads(_n)

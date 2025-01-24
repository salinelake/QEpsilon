"""
This module contains some utility functions.
"""

import torch as th
import numpy as np


def compose(op_sequence: list[th.Tensor]):
    """
    This function composes a sequence of operators.
    Args:
        op_sequence: a list of operators.
    """
    op = op_sequence[0]
    for i in range(1, len(op_sequence)):
        op = th.kron(op, op_sequence[i])
    return op

def ABAd(A: th.Tensor, B: th.Tensor):
    """
    This function computes A @ B @ A.conj().T.
    """
    if A.dim() != B.dim():
        raise ValueError("The dimensions of A and B do not match.")
    if A.dim() == 2:
        return th.matmul(A, th.matmul(B, A.conj().T))
    elif A.dim() == 3:
        if A.is_sparse:
            nb = A.shape[0]
            result = []
            for i in range(nb):
                result.append(th.matmul(A[i], th.matmul(B[i], A[i].conj().T)))
            return th.stack(result)
        else:
            return th.matmul(A, th.matmul(B, A.conj().permute(0, 2, 1)))
    else:
        raise ValueError("A and B must be 2D or 3D tensors.")

def trace(rho: th.Tensor):
    """
    This function traces out the density matrix.
    Args:
        rho: the density matrix to be traced out. Shape: (nb, ns, ns) or (ns, ns).
    Returns:
        trace: a (nb) tensor or a scalar, the trace of the density matrix.
    """
    if rho.dim() == 3:
        return th.einsum('ijj', rho).real
    elif rho.dim() == 2:
        return th.trace(rho).real
    else:
        raise ValueError("Density matrix must have shape (dim, dim) or (batchsize, dim, dim).")

def bin2idx(config: th.Tensor):
    """
    This function converts a binary numb to an index.
    """
    if config.dtype != th.int64:
        raise ValueError("Config must be an integer tensor (th.int64).")
    if config.dim() != 1:
        raise ValueError("Config must be a 1D tensor.")
    nd = config.shape[0]
    multiplier = th.tensor([2**(nd-i-1) for i in range(nd)], dtype=th.int64, device=config.device)
    return th.matmul(config, multiplier)


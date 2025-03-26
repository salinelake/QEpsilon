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

def apply_to_pse(pse: th.Tensor, ops: th.Tensor):
    ## TODO: fix error when pse is sparse.
    """
    This function applies an operator to pure state ensemble.
    Args:
        pse: the pure state ensemble to be acted on. Shape: (nb, ns).
        ops: the operator to be applied. Shape: (nb, ns, ns) or (ns, ns).
    Returns:
        pse_new: the pure state ensemble after applying the operator. Shape: (nb, ns).
    """
    if pse.dim() == 2:
        pass
    else:
        raise ValueError("Pure state ensemble must have shape (nb, ns).")

    if ops.dim() == 2:
        return th.matmul(pse, ops.T)
    elif ops.dim() == 3:
        return (ops * pse[:,None,:]).sum(-1)
    else:
        raise ValueError("Operator must have shape (nb, ns, ns) or (ns, ns).")

def expectation_pse(pse: th.Tensor, ops: th.Tensor):
    """
    This function computes the expectation value of an operator on a pure state ensemble: <psi|ops|psi>/<psi|psi>.
    Args:
        pse: the pure state ensemble. Shape: (nb, ns).
        ops: the operator. Shape: (nb, ns, ns) or (ns, ns).
    Returns:
        exp: the expectation value. Shape: (nb).
    """
    # sanity check
    if pse.dim() == 2:
        pass
    else:
        raise ValueError("Pure state ensemble must have shape (nb, ns).")
    if ops.dim() == 2 or ops.dim() == 3:
        pass
    else:
        raise ValueError("Operator must have shape (nb, ns, ns) or (ns, ns).")
    # compute the expectation value
    norm2 = th.sum(pse * pse.conj(), dim=-1)
    ops_pse = apply_to_pse(pse, ops)
    return (ops_pse * pse.conj()).sum(dim=-1) / norm2
    
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

def qubitconf2idx(config: th.Tensor):
    """
    This function converts a binary qubit/spin configuration to an index.
    Args:
        config: a 0 or 1 tensor that specifies the qubit/spin configuration. Shape: (#qubits). Example for 2-qubit system: [0, 1] means |01>. 
        By our convention of Pauli-operator composition, the all-qubit-up configuration (|11...1>) is the first configuration. So the index is 0.
    Returns:
        idx: an integer, the index of the configuration.
    """
    ## sanity check
    if config.dtype != th.int64:
        raise ValueError("Config must be an integer tensor (th.int64).")
    if config.dim() != 1:
        raise ValueError("Config must be a 1D tensor.")
    if max(config) > 1 or min(config) < 0:
        raise ValueError("Config must be made of 0 or 1.")
    ## convert
    ndigits = config.shape[0]
    binary = 1 - config  # 1 for spin-down, 0 for spin-up. Do this to make the all-spin-up configuration the first configuration.
    multiplier = th.tensor([2**(ndigits-i-1) for i in range(ndigits)], dtype=th.int64, device=config.device)
    return th.dot(binary, multiplier)

def idx2qubitconf(idx: int, n_qubits: int):
    """
    This function converts an index to a binary number.
    """
    binary = th.tensor([int(i) for i in bin(idx)[2:].zfill(n_qubits)], dtype=th.int64)
    ## make sure the all-qubit-up configuration (|11...1>) is the first configuration
    return 1 - binary  

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

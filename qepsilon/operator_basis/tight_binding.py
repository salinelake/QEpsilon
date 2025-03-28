"""
This module deals with tight binding operators and related operations.
"""

import torch as th
from qepsilon.utilities import compose

class TightBinding(th.nn.Module):
    """
    This class deals with tight binding operators and related operations.
    """
    def __init__(self, n_sites: int, pbc: bool = True):
        super().__init__()
        """
        Initialize the tight binding operator class.
        Args:
            n_sites: the number of sites.
        """
        self.ns = n_sites
        self.pbc = pbc
        ## A list of possible operators. No matrix needed to be stored since there won't be any kronecker product. Values are meaningless. 
        self.register_buffer("L", None)
        self.register_buffer("R", None)
        self.register_buffer("X", th.zeros(2,2, dtype=th.cfloat))
        self.register_buffer("N", None)

    def get_composite_ops(self, name_sequence: str):
        """
        This function returns a composite tight binding operator.
        Args:
            name_sequence: a string of tight binding operator names. Examples: 
                (1) "XXLXX", meaning |1><2|, the particle on third site hops to the left. 
                (2) "XXRXX", meaning |3><2|, the particle on third site hops to the right.
                (3) "XXNXX", meaning |2><2|, identity for the third site.
        """
        ## assert there are only "X", "L", "R", "N" in the name_sequence, otherwise raise ValueError
        if not all(c in ["X", "L", "R", "N"] for c in name_sequence):
            raise ValueError("Only X, L, R, N are allowed in the name_sequence")
        ## assert the length of the name_sequence is n_sites
        if len(name_sequence) != self.ns:
            raise ValueError("The length of the name_sequence must be n_sites")
        ## find the position of the non-X character
        non_X_pos = [i for i, c in enumerate(name_sequence) if c != "X"]
        ## assert the non-X character is only one
        if len(non_X_pos) != 1:
            raise ValueError("There must be exactly one non-X character in the name_sequence")
        ## get the position of the site to be acted on
        idx = non_X_pos[0]
        op = name_sequence[idx]
        op_matrix = th.zeros(self.ns, self.ns, dtype=th.cfloat, device=self.X.device)

        if self.pbc is False:
            if op == "L":
                op_matrix[idx, idx-1] = 1
            elif op == "R":
                op_matrix[idx, idx+1] = 1
            elif op == "N":
                op_matrix[idx, idx] = 1
        else:
            if op == "L":
                op_matrix[idx, (idx-1)%self.ns] = 1
            elif op == "R":
                op_matrix[idx, (idx+1)%self.ns] = 1
            elif op == "N":
                op_matrix[idx, idx] = 1
        return op_matrix
        
        
        
        
        
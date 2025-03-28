"""
This module deals with Pauli operators and related operations.
"""

import torch as th
from qepsilon.utilities import compose

class Boson(th.nn.Module):
    """
    This class deals with boson operators.
    """

    def _I(self, ns: int):
        """
        This function returns the identity operator for a mode with maximum occupation number ns.
        """
        return th.eye(ns, dtype=th.cfloat)

    def _U(self, ns: int):
        """
        This function returns the creation operator for a mode with maximum occupation number ns.
        """
        x = th.zeros((ns, ns), dtype=th.cfloat)
        for i in range(1, ns):
            x[i, i-1] = (i*1.0)**0.5
        return x
    
    def _D(self, ns: int):
        """
        This function returns the annihilation operator for a mode with maximum occupation number ns.
        """
        x = th.zeros((ns, ns), dtype=th.cfloat)
        for i in range(ns-1):
            x[i, i+1] = (i+1.0)**0.5
        return x
    
    def _N(self, ns: int):
        """
        This function returns the number operator for a mode with maximum occupation number ns.
        """
        return th.diag(th.arange(ns)).to(dtype=th.cfloat)

    def __init__(self, nmax: int):
        super().__init__()
        """
        Initialize the boson operator class.
        Args:
            nmax: the maximum occupation number of bosons.
        """
        self.nmax = nmax
        self.ns = nmax+1
        self.register_buffer("U", self._U(self.ns))
        self.register_buffer("D", self._D(self.ns))
        self.register_buffer("I", self._I(self.ns))
        self.register_buffer("N", self._N(self.ns))
    
    def get_sequence_ops(self, name_sequence: str):
        """
        This function returns a sequence of boson operators for a bosonic system with multiple modes.
        Args:
            name_sequence: a string of boson operator names. Example: "UDI" for creation of mode 0, annihilation of mode 1, identity of mode 2.
        """
        op = []
        for s in name_sequence:
            op.append(getattr(self, s))
        return op
    
    def get_composite_ops(self, name_sequence: str):
        """
        This function returns a composite boson operator.
        """
        return compose(self.get_sequence_ops(name_sequence))

 
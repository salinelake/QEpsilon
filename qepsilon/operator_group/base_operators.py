import torch as th
import numpy as np


class OperatorGroup(th.nn.Module):
    """
    This is a base class for operator groups.
    """
    def __init__(self, id: str, num_states: int, batchsize: int = 1):
        super().__init__()
        self.ns = num_states
        self.id = id
        self.nb = batchsize
        self._ops = []
        self._prefactor = []

    def reset(self):
        """
        Reset the coefficients of the operator group.
        """
        pass

    def add_operator(self, prefactor: float=1.0):
        """
        Add an operator (with a prefactor) to the group. Description of the operator is stored in self._ops. To be implemented in subclasses.
        Args:
            prefactor: float, the prefactor of the operator.
        """
        pass

    def sum_operators(self):
        """
        Sum up the operators in the group. Coefficients not multiplied! To be implemented in subclasses.
        Returns a matrix of shape (self.ns, self.ns).
        """
        pass

    def sample(self):
        """
        Sample the total operator with static or stochastic coefficients. To be implemented in subclasses.
        Returns the operator matrix of shape (self.ns, self.ns) and a list of coefficients. The length of the list is the batchsize.
        """
        pass

class ComposedOperatorGroups(OperatorGroup):
    def __init__(self, id: str, OP1: OperatorGroup, OP2: OperatorGroup):
        """
        Compose two operator groups.
        Args:
            OP1: the first operator group. 
            OP2: the second operator group.
        Returns:
            The composed operator group.
        """
        if OP1.nb != OP2.nb:
            raise ValueError("The batchsize of OP1 and OP2 do not match.")
        super().__init__(id, num_states=OP1.ns * OP2.ns, batchsize=OP1.nb)
        self.OP1 = OP1
        self.OP2 = OP2
        
    def reset(self):
        self.OP1.reset()
        self.OP2.reset()

    def add_operator(self, prefactor: float=1.0):
        raise ValueError("Cannot add operator to a composed operator group. The operators in the subsystems are specified at initialization.")

    def sum_operators(self):
        raise ValueError("Cannot sum operators in a composed operator group. There won't be multiple terms in the composed operator group.")

    def sample(self, dt: float):
        OP1_ops, OP1_coef = self.OP1.sample(dt)
        OP2_ops, OP2_coef = self.OP2.sample(dt)
        ops = th.kron(OP1_ops, OP2_ops)
        coef = OP1_coef * OP2_coef
        return ops, coef
        

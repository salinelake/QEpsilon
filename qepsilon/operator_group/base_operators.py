import torch as th
import numpy as np


class OperatorGroup(th.nn.Module):
    """
    This is a base class for operator groups.
    """
    def __init__(self, id: str, num_states: int, batchsize: int = 1, static: bool = False):
        super().__init__()
        self.ns = num_states
        self.static = static
        self.id = id
        self.nb = batchsize
        self._ops = []
        self._prefactors = []
        self.op_static = None
        self.coef_static = None

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

    def sample(self, dt: float=None):
        if self.static:
            if self.op_static is None:
                op, coef = self._sample(dt)
                self.op_static = op.clone()
                self.coef_static = coef.clone()
            return self.op_static, self.coef_static
        else:
            op, coef = self._sample(dt)
            return op, coef

    def _sample(self, dt: float):
        """
        Sample the total operator with static or stochastic coefficients. To be implemented in subclasses.
        Returns the operator matrix of shape (self.ns, self.ns) and a list of coefficients. The length of the list is the batchsize.
        """
        pass

class ComposedOperatorGroups(OperatorGroup):
    def __init__(self, id: str, OP_list: list[OperatorGroup], static: bool = False):
        """
        Compose multiple operator groups.
        Args:
            OP_list: a list of operator groups.
        Returns:
            The composed operator group.
        """
        nOP = len(OP_list)
        nb = OP_list[0].nb
        for i in range(nOP):
            if OP_list[i].nb != nb:
                raise ValueError("The batchsize of OP_list[{}] and OP_list[0] do not match.".format(i))
        ns = 1
        for i in range(nOP):
            ns *= OP_list[i].ns
        super().__init__(id, num_states=ns, batchsize=nb, static=static)
        self.OP_list = OP_list

    def reset(self):
        for OP in self.OP_list:
            OP.reset()

    def to(self, device='cuda'):
        for OP in self.OP_list:
            OP.to(device=device)

    def add_operator(self, prefactor: float=1.0):
        raise ValueError("Cannot add operator to a composed operator group. The operators in the subsystems are specified at initialization.")

    def sum_operators(self):
        raise ValueError("Cannot sum operators in a composed operator group. There won't be multiple terms in the composed operator group.")

    def _sample(self, dt: float):
        ops_composed, coef_composed = self.OP_list[0].sample(dt)
        for OP in self.OP_list[1:]:
            ops, coef = OP.sample(dt)
            ops_composed = th.kron(ops_composed, ops)
            coef_composed *= coef
        return ops_composed, coef_composed
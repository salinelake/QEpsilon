"""
This module deals with Pauli operators and related operations.
"""

import torch as th
from qepsilon.utilities import compose

class Pauli(th.nn.Module):
    """
    This class deals with Pauli operators.
    """
    def _X(self):
        """
        This function returns the Pauli X operator as a 2x2 complex tensor.
        """
        return th.tensor([[0, 1], [1, 0]], dtype=th.cfloat)

    def _Y(self):
        """
        This function returns the Pauli Y operator as a 2x2 complex tensor.
        """
        return th.tensor([[0, -1j], [1j, 0]], dtype=th.cfloat)

    def _Z(self):
        """
        This function returns the Pauli Z operator as a 2x2 complex tensor.
        """
        return th.tensor([[1, 0], [0, -1]], dtype=th.cfloat)
    
    def _I(self):
        """
        This function returns the identity operator as a 2x2 complex tensor.
        """
        return th.eye(2, dtype=th.cfloat)

    def _U(self):
        """
        This function returns the raising operator as a 2x2 complex tensor.
        """
        return th.tensor([[0, 1], [0, 0]], dtype=th.cfloat)
    
    def _D(self):
        """
        This function returns the lowering operator as a 2x2 complex tensor.
        """
        return th.tensor([[0, 0], [1, 0]], dtype=th.cfloat)


    def _N(self):
        """
        This function returns the number operator as a 2x2 complex tensor.
        """
        return th.tensor([[1, 0], [0, 0]], dtype=th.cfloat)

    def __init__(self, n_qubits: int):
        super().__init__()
        """
        Initialize the Pauli operator class. One-body Pauli operators are registered as buffers.
        Args:
            n_qubits: the number of qubits.
        """
        self.nq = n_qubits
        self.register_buffer("X", self._X())
        self.register_buffer("Y", self._Y())
        self.register_buffer("Z", self._Z())
        self.register_buffer("I", self._I())
        self.register_buffer("U", self._U())
        self.register_buffer("D", self._D())
        self.register_buffer("N", self._N())
    
    def along_u(self, u: th.FloatTensor):
        """
        This function returns (Pauli vector dot u). u is a direction vector.
        Args:
            u: the direction vector. Should be a real-valued tensor. Shape: (3) or (nbatch, 3)
        Returns:
            M: the (batched) Pauli operator. Shape: (2,2) or (nbatch, 2, 2)
        """
        if u.dim() == 1:
            if u.shape != (3,):
                raise ValueError("u must be a 3-dimensional vector.")
            u = u / th.norm(u)
            u = u.to(self.X.device)
            return self.X * u[0] + self.Y * u[1] + self.Z * u[2]
        elif u.dim() == 2:
            if u.shape[1] != 3:
                raise ValueError("u must be a 2D tensor with the last dimension being 3. ")
            u = u / th.norm(u, dim=1, keepdim=True)
            u = u.to(self.X.device)
            M = self.X[None, :, :] * u[:, 0, None, None] + self.Y[None, :, :] * u[:, 1, None, None] + self.Z[None, :, :] * u[:, 2, None, None]
            return M
        else:
            raise ValueError("u must be a 1D or 2D tensor.")

    def SU2_rotation(self, u: th.Tensor, theta: th.Tensor):
        """
        This function returns the SU(2) rotation operator about the direction u by angle theta.
        Args:
            u: the (batched) direction of the rotation. Shape: (3) or (nbatch, 3)
            theta: the (batched) angle of the rotation. Shape: (1) or (nbatch)
        Returns:
            M: the (batched) rotation operator. Shape: (2,2) or (nbatch, 2, 2)
        """
        if u.dim() == 1:
            return th.cos(theta / 2) * self.I - 1j * th.sin(theta / 2) * self.along_u(u)
        elif u.dim() == 2:
            if theta.shape != u[:,0].shape:
                raise ValueError("The first dimension of u and theta must be the same.")
            theta = theta.reshape(-1, 1, 1)
            return th.cos(theta / 2) * self.I.unsqueeze(0) - 1j * th.sin(theta / 2) * self.along_u(u)
        else:
            raise ValueError("u must be a 1D or 2D tensor.")

    def get_sequence_ops(self, name_sequence: str):
        """
        This function returns a sequence of Pauli operators.
        Args:
            name_sequence: a string of Pauli operator names. Example: "XYZI"
        """
        op = []
        for s in name_sequence:
            op.append(getattr(self, s))
        return op
    
    def get_composite_ops(self, name_sequence: str):
        """
        This function returns a composite Pauli operator.
        """
        return compose(self.get_sequence_ops(name_sequence))


class Kraus(Pauli):
    """
    This class deals with Kraus operators.
    """
    def __init__(self):
        super().__init__()

    def depolarizing_operators(self, p: float):
        """
        This function returns the Kraus operators of the depolarizing channel.
        This channel represents a situation where the system is subject to a random unitary error with equal probability between X, Y, and Z.
        """
        ops = []
        ops.append(th.sqrt(1 - p) * self.I)
        ops.append(th.sqrt(p / 3) * self.X)
        ops.append(th.sqrt(p / 3) * self.Y)
        ops.append(th.sqrt(p / 3) * self.Z)
        return ops
    
    def dephasing_operators(self, gamma: float):
        """
        This function returns the Kraus operators of the dephasing channel.
        This channel represents a situation where the system is subject to a random phase error.
        """
        ops = []
        ops.append(th.sqrt(1 - gamma) * self.I)
        ops.append(th.sqrt(gamma) * self.Z)
        return ops
    
    def amplitude_damping_operators(self, gamma: float):
        """
        This function returns the Kraus operators of the amplitude damping channel.
        This channel represents a situation where the system experience energy dissipation, such as spontaneous emission.
        """
        raise NotImplementedError("Amplitude damping channel is not implemented yet.")

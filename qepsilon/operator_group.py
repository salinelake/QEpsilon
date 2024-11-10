import torch as th
import numpy as np
from qepsilon.tls import Pauli
from qepsilon.utility import compose

class OperatorGroup(th.nn.Module):
    """
    This is a base class for operator groups.
    """
    def __init__(self, n_qubits: int, id: str):
        super().__init__()
        self.nq = n_qubits
        self.ns = 2**n_qubits
        self.id = id
        self._ops = []
    
    def add_operator(self):
        """
        Add an operator to the group. Description of the operator is stored in self._ops. To be implemented in subclasses.
        """
        pass
    
    def sum_operators(self):
        """
        Sum up the operators in the group. Coefficients not multiplied! To be implemented in subclasses.
        """
        pass

    def sample(self):
        """
        Sample the total operator with static or stochastic coefficients. To be implemented in subclasses.
        """
        pass

class PauliOperatorGroup(OperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems). 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str):
        super().__init__(n_qubits, id)
        self.pauli = Pauli(n_qubits)
    
    def add_operator(self, PauliSequence: str):
        """
        Add an operator to the group.
        """
        self._ops.append(PauliSequence)
        return
    
    def sum_operators(self):
        """
        Sum up the operators in the group.
        """
        total_ops = 0
        for op in self._ops:
            total_ops += self.pauli.get_composite_ops(op)
        return total_ops


class StaticPauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a static coefficient. 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, coef: float, requires_grad: bool = False):
        super().__init__(n_qubits, id)
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(th.tensor(coef)))
        else:
            self.register_buffer("coef", th.tensor(coef))
    
    def sample(self, dt: float):
        """
        This function sum up the operators in the group.
        """
        return self.sum_operators() * self.coef

class WhiteNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a white noise coefficient. 
    """
    def __init__(self, n_qubits: int, id: str,  amp: float, requires_grad: bool = False):
        super().__init__(n_qubits, id)
        if requires_grad:
            self.register_parameter("amp", th.nn.Parameter(th.tensor(amp)))
        else:
            self.register_buffer("amp", th.tensor(amp))
    
    def sample(self, dt: float):
        """
        This function sample the average of the total operator for a time step dt. 
        Note that the accumulation of the white noise is a Wiener process. dW ~ N(0, sqrt(dt)). 
        """
        noise = np.random.normal(0, 1) * self.amp
        return noise * self.sum_operators() / np.sqrt(dt)

class ColorNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a fluctuating coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, damping: float, amp: float, requires_grad: bool = False):
        super().__init__(n_qubits, id)
        if requires_grad:
            self.register_parameter("damping", th.nn.Parameter(th.tensor(damping)))
            self.register_parameter("amp", th.nn.Parameter(th.tensor(amp)))
        else:
            self.register_buffer("damping", th.tensor(damping))
            self.register_buffer("amp", th.tensor(amp))

        self.register_buffer("noise", th.tensor(0))

    def z1(self, dt: float):
        return th.exp(-self.damping * dt)
    
    def z2(self, dt: float):
        return th.sqrt(1 - th.exp(-2 * self.damping * dt))

    def sample(self, dt: float):
        """
        This function steps the color noise for a time step dt, then return the total operator.
        dx = -damping * x + amp * dW
        """
        drive = np.random.normal(0, 1) * self.amp
        noise_new = self.noise * self.z1(dt) + drive * self.z2(dt)   
        self.noise = noise_new
        return self.noise * self.sum_operators()
 
class CustomNoiseOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a custom noise. 
    """
    def __init__(self, n_qubits: int, id: str, noise: callable, shift: float=0, scale: float=1, requires_grad: bool = False):
        super().__init__(n_qubits, id)
        if requires_grad:
            self.register_parameter("shift", th.nn.Parameter(th.tensor(shift)))
            self.register_parameter("scale", th.nn.Parameter(th.tensor(scale)))
        else:
            self.register_buffer("shift", th.tensor(shift))
            self.register_buffer("scale", th.tensor(scale))
    
    def sample(self, dt: float):  # TODO: implement this function
        """
        This function steps the custom noise for a time step dt, then return the total operator.
        """
        pass
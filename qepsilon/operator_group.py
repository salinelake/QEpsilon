import torch as th
import numpy as np
from qepsilon.tls import Pauli
from qepsilon.utility import compose

class OperatorGroup(th.nn.Module):
    """
    This is a base class for operator groups.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1):
        super().__init__()
        self.nq = n_qubits
        self.ns = 2**n_qubits
        self.id = id
        self.nb = batchsize
        self._ops = []
    
    def reset(self):
        """
        Reset the coefficients of the operator group.
        """
        pass

    def add_operator(self):
        """
        Add an operator to the group. Description of the operator is stored in self._ops. To be implemented in subclasses.
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

class PauliOperatorGroup(OperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems). 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1):
        super().__init__(n_qubits, id, batchsize)
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
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, coef: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(th.tensor(coef, dtype=th.float)))
        else:
            self.register_buffer("coef", th.tensor(coef, dtype=th.float))
    
    def sample(self, dt: float):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ops = self.sum_operators() 
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device) * self.coef

class WhiteNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a white noise coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        if requires_grad:
            self.register_parameter("amp", th.nn.Parameter(th.tensor(amp, dtype=th.float)))
        else:
            self.register_buffer("amp", th.tensor(amp, dtype=th.float))
    
    def sample(self, dt: float):
        """
        This function sample the average of the total operator for a time step dt. 
        Note that the accumulation of the white noise is a Wiener process. dW ~ N(0, sqrt(dt)). 
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        noise = np.random.normal(0, 1, self.nb) 
        noise = th.tensor(noise, dtype=self.amp.dtype, device=self.amp.device) * self.amp / np.sqrt(dt)
        return self.sum_operators(), noise

class ColorNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a fluctuating coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        if requires_grad:
            self.register_parameter("tau", th.nn.Parameter(th.tensor(tau, dtype=th.float)))
            self.register_parameter("amp", th.nn.Parameter(th.tensor(amp, dtype=th.float)))
        else:
            self.register_buffer("tau", th.tensor(tau, dtype=th.float))
            self.register_buffer("amp", th.tensor(amp, dtype=th.float))

        self.register_buffer("noise", th.randn(self.nb) * amp)
    
    @property
    def damping(self):
        return 1 / th.abs(self.tau)
    
    def reset(self):
        self.noise = th.randn(self.nb, device=self.amp.device) * self.amp

    def z1(self, dt: float):
        return th.exp(- self.damping * dt)
    
    def z2(self, dt: float):
        return th.sqrt(1 - th.exp(-2 * self.damping * dt))

    def sample(self, dt: float):
        """
        This function steps the color noise for a time step dt, then return the total operator.
        dx = -damping * x + amp * dW
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        drive = np.random.normal(0, 1, self.nb)
        drive = th.tensor(drive, dtype=self.amp.dtype, device=self.amp.device) * self.amp
        noise_new = self.noise * self.z1(dt) + drive * self.z2(dt)   
        self.noise = noise_new
        return self.sum_operators(), noise_new

class ColorNoisePauliOperatorGroup_Conv(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a fluctuating coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        self.conv_cutoff = 5
        self.l0 = 1000
        if requires_grad:
            self.register_parameter("tau", th.nn.Parameter(th.tensor(tau, dtype=th.float)))
            self.register_parameter("amp", th.nn.Parameter(th.tensor(amp, dtype=th.float)))
        else:
            self.register_buffer("tau", th.tensor(tau, dtype=th.float))
            self.register_buffer("amp", th.tensor(amp, dtype=th.float))

        ##  initialize the noise history, head is the newest
        self.register_buffer("noise", th.randn((self.nb, self.l0)))

    @property
    def damping(self):
        return 1 / th.abs(self.tau)
    
    def reset(self):
        self.noise = th.randn((self.nb, self.l0), device=self.noise.device)

    def sample(self, dt: float):
        """
        This function steps the color noise for a time step dt, then return the total operator.
        c(t) = \sqrt(2*damping) * amp * \int_{-\infty}^t exp(-damping * (t-s)) w(s) ds
        where w(s) is a white noise with zero mean and unit variance.
        Discretize time with t=n*dt, then
        c(n) = \sqrt(2*damping) * amp * sum_{i=0}^{N} exp(-damping * (i+0.5) * dt) * dW(n-i)
        where dW(n) is the increment of a Wiener process at time n. standard deviation is sqrt(dt).
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        l_kernel = int(self.conv_cutoff / (self.damping * dt)) + 1
        l_noise = self.noise.shape[1]
        if l_noise < l_kernel:
            ## pad the noise history with white noise
            self.noise = th.cat([self.noise, th.randn((self.nb, l_kernel - l_noise), device=self.noise.device)], dim=1)
        else:
            ## new white noise, move the stack forward
            self.noise = th.cat([th.randn((self.nb, 1), device=self.noise.device), self.noise[:, :-1]], dim=1)
        print(self.noise.shape)
        conv_kernel = th.exp(-self.damping * (th.arange(l_kernel, dtype=self.noise.dtype, device=self.noise.device) + 0.5) * dt)
        coef = th.sum(self.noise[:, :l_kernel] * conv_kernel, dim=1)
        coef = th.sqrt(2 * self.damping) * self.amp * coef * np.sqrt(dt)
        return self.sum_operators(), coef



class CustomNoiseOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a custom noise. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int, noise: callable, shift: float=0, scale: float=1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
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
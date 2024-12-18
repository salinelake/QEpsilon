import numpy as np
import torch as th
from ..boson import Boson
from .base_operators import OperatorGroup

class BosonOperatorGroup(OperatorGroup):
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1):
        self.nm = num_modes  # number of modes
        self.ns = (nmax+1)**num_modes  # number of states
        super().__init__(id, self.ns, batchsize)
        self.boson = Boson(nmax)

    def add_operator(self, boson_sequence: str):
        """
        Add an operator to the group. Stored as a string of boson operator names. 
        Args:
            boson_sequence: str, the boson sequence. Example: "+-1" for creation of mode 0, annihilation of mode 1, identity of mode 2.
        """
        if len(boson_sequence) != self.nm:
            raise ValueError("length of boson_sequence must be the number of modes")
        self._ops.append(boson_sequence)
        return
    
    def sum_operators(self):
        total_ops = 0
        for op in self._ops:
            total_ops += self.boson.get_composite_ops(op)
        return total_ops

class IdentityBosonOperatorGroup(BosonOperatorGroup):
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1):
        super().__init__(num_modes, id, nmax, batchsize)
        self.add_operator("I"*num_modes)

    def sample(self, dt: float):
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device)

class StaticBosonOperatorGroup(BosonOperatorGroup):
    """
    This class deals with a group of operators (composite boson operators on n-mode systems) and a static coefficient. 
    Each operator is a direct product of boson operators. It is specified by a string of boson operator names.  
    For example, "UDI" is the 2-body operator $$a^\dagger_0 \otimes a_1 \otimes I_2$$.
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1, coef: float = 1.0, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize)
        ## require coef is a scalar
        if not isinstance(coef, float):
            print(type(coef))
            raise ValueError("coef must be a float scalar")
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(th.tensor(coef, dtype=th.float)))
        else:
            self.register_buffer("coef", th.tensor(coef, dtype=th.float))

    def sample(self, dt: float):
        """
        This function returns the sum of the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device) * self.coef
    
class HarmonicOscillatorBosonOperatorGroup(BosonOperatorGroup):
    """
    static operator H = \sum_i \omega_i (a_i^\dagger a_i + 1/2)
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int, omega: th.Tensor, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize)
        if omega.shape != (self.nm,):
            raise ValueError("omega specifies the frequency of each mode. It must have shape (num_modes,)")
        if omega.dtype != th.float:
            raise ValueError("omega specifies the frequency of each mode. It must be a real float tensor.")
        if omega.min()<=0:
            raise ValueError("omega specifies the frequency of each mode. It must be all positive.")
        log_omega = th.log(omega)
        if requires_grad:
            self.register_parameter("log_omega", th.nn.Parameter(log_omega))
        else:
            self.register_buffer("log_omega", log_omega)
        for idx in range(self.nm):
            _ops = ['I'] * self.nm
            _ops[idx] = 'N'
            self.add_operator(''.join(_ops))
        self.add_operator('I'*self.nm)
    @property
    def omega(self):
        return th.exp(self.log_omega)
    
    def sample(self, dt: float):
        """
        This function returns H = \sum_i \omega_i (a_i^\dagger a_i + 1/2) and a all-one coefficient tensor.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        total_ops = 0
        for idx, op in enumerate(self._ops[:-1]):
            total_ops += self.boson.get_composite_ops(op) * self.omega[idx]
        ## add zero-point energy
        total_ops += self.boson.get_composite_ops(self._ops[-1]) * self.omega.sum() * 0.5
        return total_ops, th.ones(self.nb, dtype=total_ops.dtype, device=total_ops.device)
    
class WhiteNoiseBosonOperatorGroup(BosonOperatorGroup):
    """
    This class deals with a group of operators (composite boson operators on n-mode systems) and a white noise coefficient. 
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize)
        if amp<0:
            raise ValueError("amp must be non-negative")
        logamp = th.log(th.tensor(amp, dtype=th.float))
        if requires_grad:
            self.register_parameter("logamp", th.nn.Parameter(logamp))
        else:
            self.register_buffer("logamp", logamp)

    @property
    def amp(self):
        return th.exp(self.logamp)

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
    
class LangevinNoiseBosonOperatorGroup(BosonOperatorGroup):
    """
    This class deals with a group of operators (composite boson operators on n-mode systems) and a Langevin noise coefficient. 
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1, amp: float = 1, tau: float = 1, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize)
        if amp<0:
            raise ValueError("amp must be non-negative")
        if tau<=0:
            raise ValueError("tau must be positive")
        logamp = th.log(th.tensor(amp, dtype=th.float))
        logtau = th.log(th.tensor(tau, dtype=th.float))
        if requires_grad:
            self.register_parameter("logtau", th.nn.Parameter(logtau))
            self.register_parameter("logamp", th.nn.Parameter(logamp))
        else:
            self.register_buffer("logtau", logtau)
            self.register_buffer("logamp", logamp)

        self.register_buffer("noise", th.randn(self.nb) * amp)

    @property
    def amp(self):
        return th.exp(self.logamp)

    @property
    def tau(self):
        return th.exp(self.logtau)
    
    @property
    def damping(self):
        return 1 / self.tau
    
    def reset(self):
        self.noise = th.randn(self.nb, device=self.amp.device) * self.amp

    def z1(self, dt: float):
        return th.exp(- self.damping * dt)
    
    def z2(self, dt: float):
        return th.sqrt(1 - th.exp(-2 * self.damping * dt))

    def sample(self, dt: float):
        """
        This function steps the noise for a time step dt, then return the total operator.
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
    
class ColorNoiseBosonOperatorGroup(LangevinNoiseBosonOperatorGroup):  ## TODO: test autocorrelation
    """
    This class deals with a group of operators (composite boson operators on n-mode systems) and a color noise coefficient. 
    The autocorrelation function of the color noise is ~exp(-t/tau)cos(omega*t). 
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1, amp: float = 1, tau: float = 1, omega: float = 1, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize, amp, tau, requires_grad)
        if requires_grad:
            self.register_parameter("omega", th.nn.Parameter(th.tensor(omega, dtype=th.float)))
        else:
            self.register_buffer("omega", th.tensor(omega, dtype=th.float))
        self.register_buffer("phase", th.rand(self.nb) * 2 * np.pi)
        self.register_buffer("time", th.zeros(1, dtype=th.float))

    def reset(self):
        self.time = th.zeros(1, dtype=th.float)
        self.phase = th.rand(self.nb) * 2 * np.pi
        self.noise = th.randn(self.nb, device=self.amp.device) * self.amp

    def sample(self, dt: float):
        """
        This function steps the noise for a time step dt, then return the total operator.
        The coefficient is a stochastic process: xi(t) = sqrt(2)*cos(omega*t+phi) * x(t). phi is a random phase.
        x(t) is a Langevin process: dx = -damping * x + amp * dW
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ## update the Langevin process
        drive = np.random.normal(0, 1, self.nb)
        drive = th.tensor(drive, dtype=self.amp.dtype, device=self.amp.device) * self.amp
        noise_new = self.noise * self.z1(dt) + drive * self.z2(dt)   
        self.noise = noise_new
        ## get the coefficient
        self.time += dt
        phase = self.omega * self.time + self.phase
        coef = np.sqrt(2) * th.cos(phase) * noise_new
        return self.sum_operators(), coef

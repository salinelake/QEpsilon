import torch as th
import numpy as np
from qepsilon.operator_basis.tls import Pauli
from qepsilon.utilities import compose
from qepsilon.system.particles import Particles
from qepsilon.operator_group.base_operators import OperatorGroup
import warnings
from typing import Callable
###########################################################################
# Base class for Pauli operator groups.
###########################################################################
class PauliOperatorGroup(OperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems). 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, static: bool = False):
        self.nq = n_qubits
        ns = 2**n_qubits
        super().__init__(id, ns, batchsize, static)
        self.pauli = Pauli(n_qubits)

    def set_batch_rescaling(self, batch_rescaling: th.Tensor):
        """
        Set the batch rescaling of the operator group. The final coefficiant will be multiplied by the batch rescaling.
        Args:
            batch_rescaling: th.Tensor, the batch rescaling of shape (self.nb,).
        """
        if batch_rescaling.shape[0] != self.nb:
            raise ValueError("The shape of batch_rescaling must be (self.nb,).")
        ## check if the batch_rescaling is already a registered buffer
        if hasattr(self, "batch_rescaling"):
            raise ValueError("batch_rescaling is already set. If you want to change the batch rescaling, you need to create a new operator group.")
        else:
            self.register_buffer("batch_rescaling", batch_rescaling)

    def add_operator(self, PauliSequence: str, prefactor: float = 1):
        """
        Add an operator to the group. Stored as a string of Pauli operator names. 
        Args:
            PauliSequence: str, the Pauli sequence. Example: "XI"
        """
        if len(PauliSequence) != self.nq:
            raise ValueError("length of PauliSequence must be the number of qubits")
        self._ops.append(PauliSequence)
        self._prefactors.append(prefactor)
        return
    
    def sum_operators(self):
        """
        Sum up the operators in the group.
        Returns:
            total_ops: th.Tensor, the total operator matrix of shape (self.ns, self.ns).
        """
        total_ops = 0
        for op, prefactor in zip(self._ops, self._prefactors):
            total_ops += self.pauli.get_composite_ops(op) * prefactor
        return total_ops


class IdentityPauliOperatorGroup(PauliOperatorGroup):
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1):
        super().__init__(n_qubits, id, batchsize)
        self.add_operator("I"*n_qubits)

    def _sample(self, dt: float):
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device)

class StaticPauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a static coefficient. 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, coef: float = 1, static: bool = True, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, static)
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(th.tensor(coef, dtype=th.float)))
        else:
            self.register_buffer("coef", th.tensor(coef, dtype=th.float))
    
    def _sample(self, dt: float = 1.0):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ops = self.sum_operators() 
        coef_scalar = self.coef
        if hasattr(self, "batch_rescaling"):
            coef = coef_scalar * self.batch_rescaling
        else:
            coef = coef_scalar * th.ones(self.nb, dtype=ops.dtype, device=ops.device)
        return ops, coef

class ScheduledPauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a scheduled coefficient as a function of time. 
    The coefficient will be specified by a python function taking time as input.
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, coef_fn: Callable = None, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        self.time = 0
        self.coef_fn = coef_fn
        self.required_grad = requires_grad
        if requires_grad:
            raise ValueError("requires_grad must be False for ScheduledPauliOperatorGroup because the schedule function is not required to be differentiable")
        if coef_fn is None:
            raise ValueError("coef_fn must be specified for ScheduledPauliOperatorGroup. It takes time as input and returns the coefficient.")
    

    def reset(self):
        self.time = 0

    def _sample(self, dt: float = 1.0):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        self.time += dt
        coef_scalar = self.coef_fn(self.time)
        ops = self.sum_operators() 
        if hasattr(self, "batch_rescaling"):
            coef = coef_scalar * self.batch_rescaling
        else:
            coef = coef_scalar * th.ones(self.nb, dtype=ops.dtype, device=ops.device)
        return ops, coef
 
###########################################################################
# Operators groups for Rydberg atoms. 
###########################################################################
class RydbergCouplingOperatorGroup(PauliOperatorGroup):
    """
    This class deals with Rydberg interaction between all pairs of atoms. Static positional disorder is considered. 
    The interaction is given by:
    H = \sum_{i,j} C6 / |r_i - r_j|^6
    where C6 is the Rydberg coupling constant, and r_i is the position of the i-th atom.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int, atom_centers: th.Tensor, C6_normalized: float = 1, position_std: float = 1, static: bool = False, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, static)
        ## set variables
        if requires_grad:
            self.register_parameter("log_std", th.nn.Parameter(th.log(th.tensor(position_std, dtype=th.float))))
        else:
            self.register_buffer("log_std", th.log(th.tensor(position_std, dtype=th.float)))
        self.register_buffer("C6_normalized", th.tensor(C6_normalized, dtype=th.float))
        self.register_buffer("rand_normal", th.randn((n_qubits, batchsize, 3), dtype=th.float))
        self.register_buffer("atom_centers", atom_centers.reshape(n_qubits, 1, 3).to(dtype=th.float))
        
        ## add interaction operators
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                op_name = ['I'] * n_qubits
                op_name[i] = 'N'
                op_name[j] = 'N'
                self._ops.append(''.join(op_name))

    def add_operator(self, PauliSequence):
        raise ValueError("RydbergCouplingOperatorGroup does not support custom operators. The operators are sum_ij (N_iN_j) for each pair of atoms.")
    
    def reset(self):
        self.rand_normal = th.randn((self.nq, self.nb, 3), dtype=self.log_std.dtype, device=self.log_std.device)
        self.clear_buffer()

    @property
    def position_std(self):
        return th.exp(self.log_std)
    
    @property
    def atomic_positions(self):
        return self.atom_centers + self.position_std * self.rand_normal  # shape: (nq, nb, 3)

    def _sample(self, dt: float):
        """
        This function sum up pair-wise Rydberg interaction operators.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        positions = self.atomic_positions  # shape: (nq, nb, 3)
        total_ops = 0
        idx = 0
        for i in range(self.nq):
            for j in range(i+1, self.nq):
                prefactor = self.C6_normalized / th.norm(positions[i] - positions[j], dim=-1)**6  ## shape: (nb,)
                total_ops += self.pauli.get_composite_ops(self._ops[idx]).reshape(1, self.ns, self.ns) * prefactor.reshape(self.nb, 1, 1)
                idx += 1
        return total_ops, th.ones(self.nb, dtype=total_ops.dtype, device=total_ops.device)

class RydbergCouplingThermalOperatorGroup(PauliOperatorGroup):
    """
    This class deals with Rydberg interaction between all pairs of atoms. Atoms are modeled as classical particles with constant-velocity motion.
    The interaction is given by:
    H = \sum_{i,j} C6 / |r_i - r_j|^6
    where C6 is the Rydberg coupling constant, and r_i is the position of the i-th atom.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int, 
                atom_centers: th.Tensor, 
                C6_normalized: float = 1, 
                position_std: float = 1, 
                velocity_std: float = 1,
                static: bool = False, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, static)
        ## set variables
        if requires_grad:
            raise NotImplementedError("RydbergCoupling with flexible initial position/velocity standard deviation is not implemented yet.")
        else:
            self.register_buffer("x_std", th.tensor(position_std, dtype=th.float))
            self.register_buffer("v_std", th.tensor(velocity_std, dtype=th.float))
        self.register_buffer("atom_centers", atom_centers.reshape(n_qubits, 1, 3).to(dtype=th.float))
        self.register_buffer("C6_normalized", th.tensor(C6_normalized, dtype=th.float))
        self.register_buffer("seed_x", th.randn((n_qubits, batchsize, 3), dtype=th.float))
        self.register_buffer("seed_v", th.randn((n_qubits, batchsize, 3), dtype=th.float))
        self.time = 0
        
        ## add interaction operators
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                op_name = ['I'] * n_qubits
                op_name[i] = 'N'
                op_name[j] = 'N'
                self._ops.append(''.join(op_name))

    def add_operator(self, PauliSequence):
        raise ValueError("RydbergCouplingOperatorGroup does not support custom operators. The operators are sum_ij (N_iN_j) for each pair of atoms.")
    
    def reset(self):
        self.time = 0
        self.seed_x = th.randn((self.nq, self.nb, 3), dtype=self.x_std.dtype, device=self.x_std.device)
        self.seed_v = th.randn((self.nq, self.nb, 3), dtype=self.v_std.dtype, device=self.v_std.device)
        self.clear_buffer()

    @property
    def initial_position(self):
        return self.atom_centers + self.x_std * self.seed_x  # shape: (nq, nb, 3)

    @property
    def initial_velocity(self):
        return self.v_std * self.seed_v  # shape: (nq, nb, 3)

    @property
    def atomic_positions(self):
        return self.initial_position + self.initial_velocity * self.time  # shape: (nq, nb, 3)

    def _sample(self, dt: float):
        """
        This function sum up pair-wise Rydberg interaction operators.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        self.time += dt
        positions = self.atomic_positions  # shape: (nq, nb, 3)
        total_ops = 0
        idx = 0
        for i in range(self.nq):
            for j in range(i+1, self.nq):
                distance = th.norm(positions[i] - positions[j], dim=-1)  ## shape: (nb,)
                prefactor = self.C6_normalized / distance**6  ## shape: (nb,)
                total_ops += self.pauli.get_composite_ops(self._ops[idx]).reshape(1, self.ns, self.ns) * prefactor.reshape(self.nb, 1, 1)
                idx += 1
        return total_ops, th.ones(self.nb, dtype=total_ops.dtype, device=total_ops.device)

class RydbergInhomoOperatorGroup(PauliOperatorGroup):
    """
    This class deals with spatially inhomogeneous detuning/Rabi-Frequency noise for each qubit. The noise is shot-by-shot, hence fixed in time.
    H = \sum_{i} n_i N_i
    or H = \sum_{i} n_i X_i
    where n_i is a random variable drawn from a normal distribution with mean=shift and standard deviation=amp.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, amp: float = 1, shift: float = None, type: str = "detune", static: bool = False, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, static)
        if amp<0:
            raise ValueError("amp must be non-negative")
        logamp = th.log(th.tensor(amp, dtype=th.float))
        if requires_grad:
            self.register_parameter("logamp", th.nn.Parameter(logamp))
        else:
            self.register_buffer("logamp", logamp)
        if shift is None:
            self.register_buffer("shift", th.tensor(0.0, dtype=th.float))
        else:
            if requires_grad:
                self.register_parameter("shift", th.nn.Parameter(th.tensor(shift, dtype=th.float)))
            else:
                self.register_buffer("shift", th.tensor(shift, dtype=th.float))
        self.register_buffer("seed", th.randn((n_qubits, batchsize), dtype=th.float))
        ## add operators
        for i in range(n_qubits):
            op_name = ['I'] * n_qubits
            if type == "detune":
                op_name[i] = 'N'
            elif type == "rabi":
                op_name[i] = 'X'
            else:
                raise ValueError("type must be 'detune' or 'rabi'")
            self._ops.append(''.join(op_name))

    def add_operator(self, PauliSequence):
        raise ValueError("RydbergInhomoDetuneOperatorGroup does not support custom operators.")
    
    @property
    def amp(self):
        return th.exp(self.logamp)
    
    def reset(self):
        self.seed = th.randn((self.nq, self.nb), dtype=self.logamp.dtype, device=self.logamp.device)
        self.clear_buffer()

    def _sample(self, dt: float):
        total_ops = 0
        for i in range(self.nq):
            noise = self.amp * self.seed[i] + self.shift
            total_ops += self.pauli.get_composite_ops(self._ops[i]).reshape(1, self.ns, self.ns) * noise.reshape(self.nb, 1, 1)
        return total_ops, th.ones(self.nb, dtype=total_ops.dtype, device=total_ops.device)

###########################################################################
# Operators groups involving simple stochastic processes. 
###########################################################################

class ShotbyShotNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a shot-by-shot noise coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, amp: float = 1, shift: float = None, static: bool = False, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, static)
        if amp<0:
            raise ValueError("amp must be non-negative")
        logamp = th.log(th.tensor(amp, dtype=th.float))
        ## set the amplitude of shot-by-shot noise
        if requires_grad:
            self.register_parameter("logamp", th.nn.Parameter(logamp))
        else:
            self.register_buffer("logamp", logamp)
        ## set the shift of shot-by-shot noise
        if shift is None:
            self.register_buffer("shift", th.tensor(0.0, dtype=th.float))
        else:
            if requires_grad:
                self.register_parameter("shift", th.nn.Parameter(th.tensor(shift, dtype=th.float)))
            else:
                self.register_buffer("shift", th.tensor(shift, dtype=th.float))
        ## set the seed of shot-by-shot noise
        self.register_buffer("seed", th.randn(self.nb, dtype=logamp.dtype, device=logamp.device))

    @property
    def amp(self):
        return th.exp(self.logamp)
    
    def reset(self):
        self.seed = th.randn(self.nb, dtype=self.logamp.dtype, device=self.logamp.device)
        self.clear_buffer()

    def _sample(self, dt: float):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ops = self.sum_operators() 
        noise = self.amp * self.seed + self.shift
        return ops, noise

class WhiteNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a white noise coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
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

    def _sample(self, dt: float):
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
        noise = th.tensor(noise, dtype=self.logamp.dtype, device=self.logamp.device) * self.amp / np.sqrt(dt)
        return self.sum_operators(), noise

class PeriodicNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a periodic noise. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False, requires_grad_tau_only: bool = False, requires_grad_amp_only: bool = False):
        """
        Args:
            tau: float, the period of the noise.
            amp: float, the amplitude of the noise.
        """
        super().__init__(n_qubits, id, batchsize)
        if requires_grad_tau_only and requires_grad_amp_only:
            raise ValueError("requires_grad_tau_only and requires_grad_amp_only cannot be both True")
        if requires_grad_tau_only or requires_grad_amp_only:
            if requires_grad is False:
                raise ValueError("requires_grad must be True if requires_grad_tau_only or requires_grad_amp_only is True")
        # self.register_buffer("tau", th.tensor(tau, dtype=th.float))
        self.register_buffer("phase", th.rand(self.nb) * 2 * np.pi )
        # self.phase = th.rand(self.nb) * 2 * np.pi
        self.time = 0
        if amp<0:
            raise ValueError("amp must be non-negative")
        logamp = th.log(th.tensor(amp, dtype=th.float))
        logtau = th.log(th.tensor(tau, dtype=th.float))
        if requires_grad:
            if (requires_grad_tau_only is False) and (requires_grad_amp_only is False):
                self.register_parameter("logamp", th.nn.Parameter(logamp))
                self.register_parameter("logtau", th.nn.Parameter(logtau))
            elif requires_grad_tau_only:
                self.register_parameter("logtau", th.nn.Parameter(logtau))
                self.register_buffer("logamp", logamp)
            elif requires_grad_amp_only:
                self.register_parameter("logamp", th.nn.Parameter(logamp))
                self.register_buffer("logtau", logtau)
        else:
            self.register_buffer("logamp", logamp)
            self.register_buffer("logtau", logtau)
    @property
    def amp(self):
        return th.exp(self.logamp)
    @property
    def tau(self):
        return th.exp(self.logtau)
    
    def reset(self):
        self.phase = th.rand(self.nb, device=self.phase.device) * 2 * np.pi 
        self.time = 0
    
    def _sample(self, dt: float):
        """
        This function steps the periodic noise for a time step dt, then return the total operator.
        """
        self.time += dt
        # self.phase += 2 * np.pi * dt / self.tau
        phase = self.phase + 2 * np.pi * self.time / self.tau
        noise = self.amp * th.sin(phase)
        return self.sum_operators(), noise

class LangevinNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a fluctuating coefficient. 
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
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
        self.noise = th.randn(self.nb, device=self.logamp.device) * self.amp

    def z1(self, dt: float):
        return th.exp(- self.damping * dt)
    
    def z2(self, dt: float):
        return th.sqrt(1 - th.exp(-2 * self.damping * dt))

    def _sample(self, dt: float):
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
        drive = th.tensor(drive, dtype=self.logamp.dtype, device=self.logamp.device) * self.amp
        noise_new = self.noise * self.z1(dt) + drive * self.z2(dt)   
        self.noise = noise_new
        return self.sum_operators(), noise_new
 
class ColorNoisePauliOperatorGroup(LangevinNoisePauliOperatorGroup):  ## TODO: test autocorrelation
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a color noise coefficient. 
    The autocorrelation function of the color noise is ~exp(-t/tau)cos(omega*t). 
    """
    def __init__(self, n_qubits, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, omega: float = 1, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, tau, amp, requires_grad)
        if requires_grad:
            self.register_parameter("omega", th.nn.Parameter(th.tensor(omega, dtype=th.float)))
        else:
            self.register_buffer("omega", th.tensor(omega, dtype=th.float))
        self.register_buffer("phase", th.rand(self.nb) * 2 * np.pi)
        self.time = 0

    def reset(self):
        self.time = 0
        self.phase = th.rand(self.nb, device=self.logamp.device) * 2 * np.pi
        self.noise = th.randn(self.nb, device=self.logamp.device) * self.amp.detach()

    def _sample(self, dt: float):
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
        drive = th.tensor(drive, dtype=self.logamp.dtype, device=self.logamp.device) * self.amp
        noise_new = self.noise * self.z1(dt) + drive * self.z2(dt)   
        self.noise = noise_new
        ## get the coefficient
        self.time += dt
        phase = self.omega * self.time + self.phase
        coef = np.sqrt(2) * th.cos(phase) * noise_new
        return self.sum_operators(), coef

###########################################################################
# Operators groups involving motion of classical particles.
###########################################################################

class DipolarInteraction(PauliOperatorGroup):
    """
    This class deals with dipolar-dipole interactions.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int, particles: Particles, connectivity: th.Tensor, prefactor: float=1, average_nsteps: int = 100, qaxis: th.Tensor=th.tensor([0.0, 1.0, 0.0]), requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        """
        dipolar interaction between pairs of particles.
        Args:
            n_qubits: int, the number of qubits.
            id: str, the id of the operator group.
            batchsize: int, the batch size.
            particles: Particles, the particles object.
            connectivity: th.Tensor, the connectivity matrix of shape (nq, nq) and dtype bool.
            prefactor: float, the prefactor in dipole-dipole interaction: d^2/(4*pi*epsilon0). Unit: hbar * hz * (um)^3.
            average_nsteps: int, the number of steps to average the dipole-dipole interaction.
        """
        if requires_grad:
            self.register_parameter("prefactor", th.nn.Parameter(th.tensor(prefactor, dtype=th.float)))
        else:
            self.register_buffer("prefactor", th.tensor(prefactor, dtype=th.float))
        self.particles = particles
        self.average_nsteps = average_nsteps
        self.qaxis = qaxis
        # self.dcut = 0.1
        ## make connectivity a upper triangular matrix
        self.connectivity = connectivity & (th.triu(th.ones_like(connectivity))==1)
        self.pair_list = th.nonzero(self.connectivity)
        self.npair = self.pair_list.shape[0]
        ## add XX+YY operator for each connected pair
        pair_interaction_list = []
        for pair in self.pair_list:
            pauli_seq = ["I"]*self.nq
            pauli_seq[pair[0]] = "X"
            pauli_seq[pair[1]] = "X"
            pauli_seq = "".join(pauli_seq)
            op = self.pauli.get_composite_ops(pauli_seq)
            pauli_seq = ["I"]*self.nq
            pauli_seq[pair[0]] = "Y"
            pauli_seq[pair[1]] = "Y"
            pauli_seq = "".join(pauli_seq)
            op += self.pauli.get_composite_ops(pauli_seq)
            pair_interaction_list.append(op)
        self.register_buffer("pair_interactions", th.stack(pair_interaction_list))

    def add_operator(self, prefactor: float = 1):
        raise ValueError("DipolarInteraction does not support custom operators. The operators are sum_ij (X_iX_j+Y_iY_j) for each connected pair.")

    def sum_operators(self):
        raise ValueError("DipolarInteraction does not support summing operators")

    def get_dipole_dipole_coef(self):
        """
        Get the dipole-dipole interaction coefficients.
        Returns:
            coef: th.Tensor, the dipole-dipole interaction coefficients of shape (nb, npair).
        """
        particles = self.particles
        nb = particles.nb
        nq = particles.nq
        axis = self.qaxis / th.norm(self.qaxis)
        if len(particles.traj) == 0:
            ## if the particles have no trajectory, use the current positions
            nsteps = 1
            pos = particles.get_positions()
        else:
            ## if the particles have a trajectory, use the last nsteps positions
            nsteps = self.average_nsteps
            pos = th.stack(particles.traj[-nsteps:])  # shape: (nsteps, nb, nq, 3)
        distance = pos.reshape(nsteps, nb, nq, 1, 3) - pos.reshape(nsteps, nb, 1, nq, 3)
        separation = th.norm(distance, dim=-1)
        axis = axis.to(device=separation.device)
        # ## clip separation
        # separation = th.clip(separation, min=self.dcut)
        # ## renormalize distance
        # distance = distance / th.norm(distance+1e-8, dim=-1)[...,None] * separation[...,None]

        cos_theta = (distance * axis[None,None,None,None,:]).sum(dim=-1) / separation
        coef = (1 - 3 * cos_theta**2) / separation**3  # shape: (nsteps, nb, nq, nq)
        coef_select = coef[self.connectivity.repeat(nsteps,nb,1,1)].reshape(nsteps, nb, self.npair)
        coef_avg = th.mean(coef_select, dim=0)
        coef_avg = coef_avg.to(device=self.prefactor.device)
        return self.prefactor * coef_avg

    def _sample(self, dt = None):   
        """
        This function steps the custom noise for a time step dt, then return the total operator.
        Args:
            dt: float, the time step. Not used here.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.nb, self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        total_ops = 0
        coef = self.get_dipole_dipole_coef()
        # for idx, op in enumerate(self._ops):
        #     total_ops += op[None,:,:] * coef[:,idx, None,None]
        total_ops = (self.pair_interactions[None,:,:,:] * coef[:,:,None,None]).sum(dim=1)   # pair interaction (npair, ns, ns) , coef (nb, npair)
        return total_ops, th.ones_like(coef[:,0], device=total_ops.device)

class spin_oscillators_interaction(PauliOperatorGroup):
    """
    Spin-(classical) Oscillator interaction as an approximation to spin-boson interaction. 
    For each oscillator, the interaction is coef * x * N, where x is the position of the oscillator, and N is the number operator of the qubit (1 for spin-up, 0 for spin-down).
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int, particles: Particles, coef: th.Tensor, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        """
        Args:
            n_qubits: int, the number of qubits.
            id: str, the id of the operator group.
            batchsize: int, the batch size.
            particles: Particles, the particles object.
        """
        if particles.ndim != 1:
            raise ValueError(f"The number of dimensions of a particle must be 1 for it to approximate a bosonic mode")
        self.particles = particles
        self.nmodes = particles.nq

        if coef.shape != (self.nmodes,):
            raise ValueError(f"The number of coefficients must be equal to the number of modes.")
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(coef))
        else:
            self.register_buffer("coef", coef)

    def _sample(self, dt = None):   
        """
        Sample the spin-oscillators interaction for a time step dt.
        Args:
            dt: float, the time step. Not used here.
        Returns:
            ops: th.Tensor, the operator matrix of shape ( self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ## sanity check
        if dt is not None:
            warnings.warn("spin_oscillators_interaction does not integrate oscillator motion. dt is ignored.")
        ## get the positions of the oscillators
        oscillator_positions = self.particles.get_positions()
        if oscillator_positions.shape != (self.nb, self.nmodes, 1):
            raise ValueError(f"The shape of the oscillator positions must be (self.nb, self.nmodes, 1).")
        oscillator_positions = oscillator_positions.to(device=self.coef.device)
        coef = 0 
        for i in range(self.nmodes):
            coef += self.coef[i] * oscillator_positions[:,i,0]
        # get the operator
        if self.op_static is None:
            self.op_static = self.sum_operators()
        return self.op_static, coef
        # ops = self.sum_operators()
        # return ops, coef


###########################################################################
# Operators groups serving as quantum channels.
###########################################################################

class DepolarizationChannel(PauliOperatorGroup):
    """
    This class deals with a group of Kraus operators.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int, p: float, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize)
        if self.static:
            raise ValueError("DepolarizationChannel can not be set to static. ")
        if p<0 or p>1:
            raise ValueError("p must be between 0 and 1")
        _p = th.atanh(th.tensor(p, dtype=th.float) * 2 - 1)
        if requires_grad:
            self.register_parameter("_p", th.nn.Parameter(_p))
        else:
            self.register_buffer("_p", _p)
    @property
    def p(self):
        return (th.tanh(self._p)+1)/2.0

    def add_operator(self, prefactor: float = 1):
        raise ValueError("DepolarizationChannel does not support adding operators")

    def sum_operators(self):
        raise ValueError("DepolarizationChannel does not support summing operators")

    def _sample(self, dt=None):
        """
        This function sample the Kraus operators of the depolarizing channel.
        """
        ops = [th.sqrt(1 - self.p) * self.pauli.I, 
               th.sqrt(self.p / 3) * self.pauli.X, 
               th.sqrt(self.p / 3) * self.pauli.Y, 
               th.sqrt(self.p / 3) * self.pauli.Z]
        return ops, th.ones(4, device=self.pauli.I.device)



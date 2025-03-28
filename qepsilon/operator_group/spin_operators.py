import torch as th
import numpy as np
from qepsilon.operator_basis.tls import Pauli
from qepsilon.utilities import compose
from qepsilon.system.particles import Particles
from qepsilon.operator_group.base_operators import OperatorGroup
import warnings

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
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device) * self.coef

###########################################################################
# Operators groups involving simple stochastic processes. 
###########################################################################

class ShotbyShotNoisePauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a shot-by-shot noise coefficient. 
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
        self.register_buffer("seed", th.randn(self.nb, dtype=logamp.dtype, device=logamp.device))
        self.tau = None

    @property
    def amp(self):
        return th.exp(self.logamp)
    
    def reset(self):
        self.seed = th.randn(self.nb, dtype=self.logamp.dtype, device=self.logamp.device)

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
        noise = self.amp * self.seed
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
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False):
        """
        Args:
            tau: float, the period of the noise.
            amp: float, the amplitude of the noise.
        """
        super().__init__(n_qubits, id, batchsize)
        # self.register_buffer("tau", th.tensor(tau, dtype=th.float))
        self.register_buffer("phase", th.rand(self.nb) * 2 * np.pi )
        # self.phase = th.rand(self.nb) * 2 * np.pi
        self.time = 0
        if amp<0:
            raise ValueError("amp must be non-negative")
        logamp = th.log(th.tensor(amp, dtype=th.float))
        if requires_grad:
            self.register_parameter("logamp", th.nn.Parameter(logamp))
            self.register_parameter("logtau", th.nn.Parameter(th.log(th.tensor(tau, dtype=th.float))))
        else:
            self.register_buffer("logamp", logamp)
            self.register_buffer("logtau", th.log(th.tensor(tau, dtype=th.float)))
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

# class LangevinNoisePauliOperatorGroup_Conv(PauliOperatorGroup):
#     """
#     This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a fluctuating coefficient. 
#     """
#     def __init__(self, n_qubits: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False):
#         super().__init__(n_qubits, id, batchsize)
#         self.conv_cutoff = 5
#         self.l0 = 1000
#         if requires_grad:
#             self.register_parameter("tau", th.nn.Parameter(th.tensor(tau, dtype=th.float)))
#             self.register_parameter("amp", th.nn.Parameter(th.tensor(amp, dtype=th.float)))
#         else:
#             self.register_buffer("tau", th.tensor(tau, dtype=th.float))
#             self.register_buffer("amp", th.tensor(amp, dtype=th.float))

#         ##  initialize the noise history, head is the newest
#         self.register_buffer("noise", th.randn((self.nb, self.l0)))

#     @property
#     def damping(self):
#         return 1 / th.abs(self.tau)
    
#     def reset(self):
#         self.noise = th.randn((self.nb, self.l0), device=self.noise.device)

#     def sample(self, dt: float):
#         """
#         This function steps the noise for a time step dt, then return the total operator.
#         c(t) = \sqrt(2*damping) * amp * \int_{-\infty}^t exp(-damping * (t-s)) w(s) ds
#         where w(s) is a white noise with zero mean and unit variance.
#         Discretize time with t=n*dt, then
#         c(n) = \sqrt(2*damping) * amp * sum_{i=0}^{N} exp(-damping * (i+0.5) * dt) * dW(n-i)
#         where dW(n) is the increment of a Wiener process at time n. standard deviation is sqrt(dt).
#         Args:
#             dt: float, the time step.
#         Returns:
#             ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
#             coef: th.Tensor, the coefficient of shape (self.nb,).
#         """
#         l_kernel = int(self.conv_cutoff / (self.damping * dt)) + 1
#         l_noise = self.noise.shape[1]
#         if l_noise < l_kernel:
#             ## pad the noise history with white noise
#             self.noise = th.cat([self.noise, th.randn((self.nb, l_kernel - l_noise), device=self.noise.device)], dim=1)
#         else:
#             ## new white noise, move the stack forward
#             self.noise = th.cat([th.randn((self.nb, 1), device=self.noise.device), self.noise[:, :-1]], dim=1)
#         conv_kernel = th.exp(-self.damping * (th.arange(l_kernel, dtype=self.noise.dtype, device=self.noise.device) + 0.5) * dt)
#         coef = th.sum(self.noise[:, :l_kernel] * conv_kernel, dim=1)
#         coef = th.sqrt(2 * self.damping) * self.amp * coef * np.sqrt(dt)
#         return self.sum_operators(), coef


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
            self._ops.append(op)

    def add_operator(self, prefactor: float = 1):
        raise ValueError("DipolarInteraction does not support adding operators")

    def sum_operators(self):
        raise ValueError("DipolarInteraction does not support summing operators")

    def get_dipole_dipole_coef(self):
        """
        Get the dipole-dipole interaction coefficients.
        Returns:
            coef: th.Tensor, the dipole-dipole interaction coefficients of shape (nb, npair).
        """
        particles = self.particles
        nsteps = self.average_nsteps
        nb = particles.nb
        nq = particles.nq
        axis = self.qaxis / th.norm(self.qaxis)
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
        for idx, op in enumerate(self._ops):
            total_ops += op[None,:,:] * coef[:,idx, None,None]
        return total_ops, th.ones_like(coef[:,0])

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

    def _sample(self):
        """
        This function sample the Kraus operators of the depolarizing channel.
        """
        ops = [th.sqrt(1 - self.p) * self.pauli.I, 
               th.sqrt(self.p / 3) * self.pauli.X, 
               th.sqrt(self.p / 3) * self.pauli.Y, 
               th.sqrt(self.p / 3) * self.pauli.Z]
        return ops



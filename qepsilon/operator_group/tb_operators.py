import torch as th
import numpy as np
from qepsilon.operator_basis.tight_binding import TightBinding
from qepsilon.system.particles import Particles
from qepsilon.operator_group.base_operators import OperatorGroup
import warnings


###########################################################################
# Base class for Tight Binding operator groups.
###########################################################################

class TightBindingOperatorGroup(OperatorGroup):
    r"""
    This class represents a group of composite Tight Binding operators on n-site systems.

    Each operator in this group is specified by a string of Tight Binding operator names.
    For example, "XXLXX" is the hopping operator :math:`| 1\rangle\langle 2 |`.
    """
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, static: bool = False):
        self.ns = n_sites
        super().__init__(id, n_sites, batchsize, static)
        self.tb = TightBinding(n_sites)
    
    def add_operator(self, TBSequence: str, prefactor: float = 1):
        """
        Add an operator to the group. Stored as a string of Tight Binding operator names. 
        Args:
            TBSequence: str, the Tight Binding sequence. Example: "XXLXX. Contains one and only one non-`X` character, choosing from `L`, `R`, `N`.
        """
        if len(TBSequence) != self.ns:
            raise ValueError("length of TBSequence must be the number of sites")
        self._ops.append(TBSequence)
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
            total_ops += self.tb.get_composite_ops(op) * prefactor
        return total_ops


class IdentityTightBindingOperatorGroup(TightBindingOperatorGroup):
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, static: bool = True):
        super().__init__(n_sites, id, batchsize, static)
        self.add_operator("X"*n_sites)
    def _sample(self, dt: float = 1.0):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
        """
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device)

class StaticTightBindingOperatorGroup(TightBindingOperatorGroup):
    """
    This class deals with a group of operators (composite Tight Binding operators on n-site systems) and a static coefficient. 
    Each operator in this group is specified by a string of Tight Binding operator names.  For example, "XXLXX" is the hopping operator |1><2|.
    """
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, coef: float = 1, static: bool = True, requires_grad: bool = False):
        super().__init__(n_sites, id, batchsize, static)
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

class PeriodicNoiseTightBindingOperatorGroup(TightBindingOperatorGroup):
    """
    This class deals with a group of operators (composite Tight Binding operators on n-site systems) and a periodic noise. 
    """
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, tau: float = 1, amp: float = 1, requires_grad: bool = False):
        """
        Args:
            tau: float, the period of the noise.
            amp: float, the amplitude of the noise.
        """
        super().__init__(n_sites, id, batchsize)
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

class WhiteNoiseTightBindingOperatorGroup(TightBindingOperatorGroup):
    """
    This class deals with a group of operators (composite Tight Binding operators on n-site systems) and a white noise coefficient. 
    """
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, amp: float = 1, requires_grad: bool = False):
        super().__init__(n_sites, id, batchsize)
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

###########################################################################
# Operators groups involving motion of classical particles.
###########################################################################

class tb_oscillators_interaction(TightBindingOperatorGroup):
    """
    Tight Binding particle-(classical) Oscillator interaction as an approximation to Tight binding particle-boson interaction. 
    For each oscillator, the interaction is coef * x * N, where x is the position of the oscillator, and N is the number operator of the tight binding site.
    """
    def __init__(self, n_sites: int, id: str, batchsize: int, particles: Particles, coef: th.Tensor, requires_grad: bool = False):
        super().__init__(n_sites, id, batchsize)
        """
        Args:
            n_sites: int, the number of sites.
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
        Sample the TB-oscillators interaction for a time step dt.
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

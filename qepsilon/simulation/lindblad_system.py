"""
This file contains the Lindblad simulation class for the QEpsilon project.
"""

import numpy as np
import torch as th
from qepsilon.operator_group import *
from qepsilon.system.density_matrix import DensityMatrix, QubitDensityMatrix
from qepsilon.system.particles import Particles
from qepsilon.utilities import ABAd, trace
import logging
import warnings
from time import time as timer

class LindbladSystem(th.nn.Module):
    """
    This class represents a generic open quantum system.
    The states are represented by a density matrix. 
    The evolution of the system is governed by the Lindblad equation, which is a generalization of the Schrodinger equation for open quantum systems.
    Both the Hamiltonian and the jump operators here allows fluctuating coefficients. So technically, we are dealing with a stochastic master equation. 
    
    The Lindblad equation is given as
    $d\rho(t) / dt = -i [H(t), \rho(t)] + \sum_k L_k(t) \rho(t) L_k(t)^\dagger - 1/2 \{L_k(t)^\dagger L_k(t), \rho(t)\}$
    where H(t) is the Hamiltonian, L_k(t) is the jump operator. Note that the coefficients of the jump operators are absorbed into L_k(t).

    In this class, each component of the Hamiltonian $H(t)$, and each jump operator $L_k(t)$, is represented by a OperatorGroup object. 
    Each OperatorGroup object contains a group of operators and the corresponding coefficients. 
    The operators themselves are time-independent, like a Pauli operator. But the coefficients can be time-dependent stochastic processes. 
    This corresponds to the case where the qubits are coupled to noisy environments such as inhomogeneous, fluctuating magnetic fields. 
    """

    def __init__(self, num_states: int, batchsize: int = 1):
        super().__init__()
        self.ns = num_states
        self.nb = batchsize 
        self.density_matrix = DensityMatrix(num_states, batchsize)
        self._hamiltonian_operator_group_dict = th.nn.ModuleDict()
        self._jumping_group_dict = th.nn.ModuleDict()
        self._channel_group_dict = th.nn.ModuleDict()
        self._ham = None
        self._jump = None
        self._ham_eff = None
    
    def to(self, device='cuda'):
        self.density_matrix.to(device=device)
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.to(device=device)
        for operator_group in self._jumping_group_dict.values():
            operator_group.to(device=device)
        for operator_group in self._channel_group_dict.values():
            operator_group.to(device=device)
        if self._ham is not None:
            self._ham = self._ham.to(device=device)
        if self._jump is not None:
            self._jump = [jump.to(device=device) for jump in self._jump]
    ############################################################
    # Setter and getter  
    ############################################################
    @property
    def hamiltonian_operator_groups(self):
        return list(self._hamiltonian_operator_group_dict.values()) 
    
    @property
    def jumping_operator_groups(self):
        return list(self._jumping_group_dict.values()) 

    @property
    def channel_operator_groups(self):
        return list(self._channel_group_dict.values()) 

    def get_hamiltonian_operator_group_by_ID(self, id: str):
        if id in self._hamiltonian_operator_group_dict:
            return self._hamiltonian_operator_group_dict[id]
        else:
            raise ValueError(f"The ID {id} does not exist in the Hamiltonian operator group dictionary.")
        
    def get_jumping_operator_group_by_ID(self, id: str):
        if id in self._jumping_group_dict:
            return self._jumping_group_dict[id]
        else:
            raise ValueError(f"The ID {id} does not exist in the jumping operator group dictionary.")
        
    def get_channel_operator_group_by_ID(self, id: str):
        if id in self._channel_group_dict:
            return self._channel_group_dict[id]
        else:
            raise ValueError(f"The ID {id} does not exist in the channel operator group dictionary.")
        
    @property
    def rho(self):
        """
        This property returns the density matrix of the system.
        """
        rho = self.density_matrix.get_rho()
        if rho is None:
            raise ValueError("The density matrix has not been initialized.")
        return rho

    @rho.setter
    def rho(self, rho: th.Tensor):
        if rho.dtype != th.cfloat:
            _rho = rho.to(dtype=th.cfloat)
        else:
            _rho = rho
        self.density_matrix.set_rho(_rho)
    

    @property
    def hamiltonian(self):
        return self._ham
    
    # do not set the Hamiltonian buffer directly.
    # @hamiltonian.setter
    # def hamiltonian(self, hamiltonian: th.Tensor):
    #     self._ham = hamiltonian

    @property
    def jump(self):
        return self._jump

    # do not set the jump buffer directly.
    # @jump.setter
    # def jump(self, jump: list[th.Tensor]):
    #     self._jump = jump

    def reset(self):
        """
        Reset the system.
        """
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.reset()
        for operator_group in self._jumping_group_dict.values():
            operator_group.reset()

    def HamiltonianParameters(self):
        parameters_list = []
        for operator_group in self._hamiltonian_operator_group_dict.values():
            parameters_list.extend(list(operator_group.parameters()))
        return parameters_list
        
    def JumpingParameters(self):
        parameters_list = []
        for operator_group in self._jumping_group_dict.values():
            parameters_list.extend(list(operator_group.parameters()))
        return parameters_list
    
    def ChannelParameters(self):
        parameters_list = []
        for operator_group in self._channel_group_dict.values():
            parameters_list.extend(list(operator_group.parameters()))
        return parameters_list

    ############################################################
    # Methods for adding operator groups
    ############################################################
    def add_operator_group_to_hamiltonian(self, operator_group: OperatorGroup):
        """
        This function adds an operator group to the Hamiltonian part of the system.
        Args:
            operator_group: an OperatorGroup object, containing a group of operators and the corresponding coefficients.
        """
        if operator_group.id in self._hamiltonian_operator_group_dict:
            raise ValueError(f"The ID {operator_group.id} already exists.")
        if operator_group.nb != self.nb:
            raise ValueError(f"The batchsize of the operator group {operator_group.id} does not match the batchsize of the system.")
        if operator_group.ns != self.ns:
            raise ValueError(f"The dimension of the operator group {operator_group.id} does not match the number of states of the system.")
        self._hamiltonian_operator_group_dict[operator_group.id] = operator_group

    def add_operator_group_to_jumping(self, operator_group: OperatorGroup):
        """
        This function adds an operator group to the jumping part of the system.
        Args:
            operator_group: an OperatorGroup object, containing a group of operators and the corresponding coefficients.
        """
        if operator_group.id in self._jumping_group_dict:
            raise ValueError(f"The ID {operator_group.id} already exists.")
        if operator_group.nb != self.nb:
            raise ValueError(f"The batchsize of the operator group {operator_group.id} does not match the batchsize of the system.")
        if operator_group.ns != self.ns:
            raise ValueError(f"The dimension of the operator group {operator_group.id} does not match the number of states of the system.")
        self._jumping_group_dict[operator_group.id] = operator_group

    def add_operator_group_to_channel(self, operator_group: OperatorGroup):
        """
        This function adds an operator group to the channel part of the system.
        """
        if operator_group.id in self._channel_group_dict:
            raise ValueError(f"The ID {operator_group.id} already exists.")
        ## TODO: check if self.ns=operator_group.ns; check also if self.nb=operator_group.nb
        self._channel_group_dict[operator_group.id] = operator_group

    ############################################################
    # Methods for evolving the system
    ############################################################
    def step_hamiltonian(self, dt: float, set_buffer: bool = False):
        """
        This function steps the Hamiltonian for a time step dt. 
        Args:
            dt: a float, the time step.
        Returns:
            hamiltonian: a (self.nb, self.ns, self.ns) tensor, the Hamiltonian operator at time t.
        """
        hamiltonian = 0
        for operator_group in self._hamiltonian_operator_group_dict.values():
            ops, coefs = operator_group.sample(dt)
            ## sanitary check
            if coefs.shape != (self.nb,):
                raise ValueError("The coefficients sampled from an operator group should be a 1D tensor of length equal to the batchsize.")
            ## no broadcasting if the operators is already batched
            if ops.shape == (self.nb, self.ns, self.ns):
                hamiltonian += ops * coefs[:, None, None]
            ## broadcast if the operators is not already batched
            elif ops.shape == (self.ns, self.ns):
                hamiltonian += ops[None, :, :] * coefs[:, None, None]
            else:
                raise ValueError(f"The shape of the operator sampled from operator group {operator_group.id} should either be (batchsize, n_states, n_states) or (n_states, n_states).")
        if set_buffer:
            self._ham = hamiltonian
        return hamiltonian

    def step_jumping(self, dt: float, set_buffer: bool = False):
        """
        This function steps the density matrix for a time step dt.
        Args:
            dt: a float, the time step.
        Returns:
            jump_operator_list: a list of (self.nb, self.ns, self.ns) tensors, the jump operators at time t.
        """
        jump_operator_list = []
        for operator_group in self._jumping_group_dict.values():
            ops, coefs = operator_group.sample(dt)
            ## sanitary check
            if coefs.shape != (self.nb,):
                raise ValueError("The coefficients sampled from an operator group should be a 1D tensor of length equal to the batchsize.")
            ## no broadcasting if the operators is already batched
            if ops.shape == (self.nb, self.ns, self.ns):
                jump_operator_list.append(ops * coefs[:, None, None])
            ## broadcast if the operators is not already batched
            elif ops.shape == (self.ns, self.ns):
                jump_operator_list.append(ops[None, :, :] * coefs[:, None, None])
            else:
                raise ValueError(f"The shape of the operator sampled from operator group {operator_group.id} should be (batchsize, n_states, n_states) or (n_states, n_states).")
        if set_buffer:
            self._jump = jump_operator_list
        return jump_operator_list

    def step_rho(self, dt: float, hamiltonian: th.Tensor, jump_operators: list[th.Tensor], time_independent: bool = False, profile: bool = False):
        """
        This function steps the density matrix for a time step dt.
        Args:
            dt: a float, the time step.
            hamiltonian: a (self.nb, self.ns, self.ns) tensor, the Hamiltonian operator at time t.
            jump_operators: a list of (self.nb, self.ns, self.ns) tensors, the jump operators at time t.
        Returns:
            rho_new: a (self.nb, self.ns, self.ns) tensor, the density matrix at time t+dt.
        """
        rho = self.rho
        identity = th.eye(self.ns, dtype=th.cfloat).to(rho.device).unsqueeze(0).repeat(self.nb, 1, 1)
        ## get the effective Hamiltonian=H + 1/2j \sum_k L_k^\dagger L_k
        if profile:
            t0 = timer()
            th.cuda.synchronize()
        if time_independent and self._ham_eff is not None:
            ham_eff = self._ham_eff
        else:
            ham_eff = hamiltonian.clone()
            for jump_operator in jump_operators:
                ham_eff -= 1j * 0.5 * th.matmul(jump_operator.conj().permute(0, 2, 1), jump_operator)
            if time_independent:
                self._ham_eff = ham_eff
        if profile:
            th.cuda.synchronize()
            t1 = timer()
            logging.info(f"The time taken for getting the effective Hamiltonian is {t1 - t0}s.")
        ## sparsify tensors if the dimension of the matrix is larger than 2000
        if self.ns > 2000:
            identity = identity.to_sparse()
            ham_eff = ham_eff.to_sparse()
            # rho = rho.to_sparse()
            jump_operators = [jump_operator.to_sparse() for jump_operator in jump_operators]
        ## first part: (1-i*dt*H_eff)rho(1+i*dt*H_eff)
        rho_new = ABAd(identity - 1j * dt * ham_eff, rho)
        if profile:
            th.cuda.synchronize()
            t2 = timer()
            logging.info(f"The time taken for the unitary contribution to the density matrix evolution is {t2 - t1}s.")
        ## second part: dt * \sum_k L_k \rho L_k^\dagger
        for jump_operator in jump_operators:
            rho_new += ABAd(jump_operator, rho) * dt
        if rho_new.is_sparse:
            rho_new = rho_new.to_dense()
        if profile:
            th.cuda.synchronize()
            t3 = timer()
            logging.info(f"The time taken for the Lindblad contribution to the density matrix evolution is {t3 - t2}s.")
        ###  Do not normalize here. Sometimes we may be evolving a general operator instead of density matrix ###
        # ## normalize if the trace of rho_new is too large.
        # rho_trace = trace(rho_new)
        # if th.abs(rho_trace.mean()-1) > 0.1:
        #     print('normalize')
        #     rho_new = rho_new / rho_trace[:, None, None]
        return rho_new

    def normalize(self):
        rho_trace = trace(self.rho)
        self.rho = self.rho / rho_trace[:, None, None]
        return

    def step(self, dt: float, set_buffer: bool = False, time_independent: bool = False, profile: bool = False):
        """
        This function steps the system for a time step dt.
        """
        if profile:
            t0 = timer()
            th.cuda.synchronize()
        hamiltonian = self.step_hamiltonian(dt, set_buffer)
        if profile:
            th.cuda.synchronize()
            t1 = timer()
            logging.info(f"The time taken for stepping the Hamiltonian is {t1 - t0}s.")
        if self._jumping_group_dict:
            jump_operator_list = self.step_jumping(dt, set_buffer)
        else:
            jump_operator_list = []
        if profile:
            th.cuda.synchronize()
            t2 = timer()
            logging.info(f"The time taken for stepping the jump operators is {t2 - t1}s.")
        self.rho = self.step_rho(dt, hamiltonian, jump_operator_list, time_independent, profile=profile)
        if profile:
            th.cuda.synchronize()
            t3 = timer()
            logging.info(f"The time taken for stepping the density matrix is {t3 - t2}s.")
        return self.rho

    def step_unitary(self, dt: float, set_buffer: bool = False):
        """
        This function steps the system for a time step dt without considering the jump operators.
        """
        hamiltonian = self.step_hamiltonian(dt, set_buffer)
        rho = self.rho
        evo_op = th.linalg.matrix_exp(-1j * dt * hamiltonian)
        rho_new = ABAd(evo_op, rho)
        self.rho = rho_new
        return self.rho

    def observe(self, operator):
        """
        This function observes the system with an operator.
        """
        if isinstance(operator, OperatorGroup) is False:
            raise ValueError("The operator must be an OperatorGroup object. It should not be a plain array or tensor.")
        ops, coefs = operator.sample(dt=0)
        ## sanitary check
        if coefs.shape != (self.nb,):
            raise ValueError("The coefficients sampled from an operator group should be a 1D tensor of length equal to the batchsize.")
        ## no broadcasting if the operators is already batched
        if ops.shape == (self.nb, self.ns, self.ns):
            ops_batched = ops * coefs[:, None, None]
        ## broadcast if the operators is not already batched
        elif ops.shape == (self.ns, self.ns):
            ops_batched = ops[None, :, :] * coefs[:, None, None]
        return trace(th.matmul(ops_batched, self.rho)) / trace(self.rho)
        
    
class QubitLindbladSystem(LindbladSystem):
    """
    This class represents the states of n physical qubits as a open quantum system.
    The states are represented by a density matrix. 
    The evolution of the system is governed by the Lindblad equation, which is a generalization of the Schrodinger equation for open quantum systems.
    Both the Hamiltonian and the jump operators in the Lindblad equation allows fluctuating coefficients. So technically, we are dealing with a stochastic master equation. 
    
    The Lindblad equation is given as
    $d\rho(t) / dt = -i [H(t), \rho(t)] + \sum_k L_k(t) \rho(t) L_k(t)^\dagger - 1/2 \{L_k(t)^\dagger L_k(t), \rho(t)\}$
    where H(t) is the Hamiltonian, L_k(t) is the jump operator. Note that the coefficients of the jump operators are absorbed into L_k(t).

    In this class, each component of the Hamiltonian $H(t)$, and each jump operator $L_k(t)$, is represented by a OperatorGroup object. 
    Each OperatorGroup object contains a group of operators and the corresponding coefficients. 
    The operators themselves are time-independent, like a Pauli operator. But the coefficients can be time-dependent stochastic processes. 
    This corresponds to the case where the qubits are coupled to noisy environments such as inhomogeneous, fluctuating magnetic fields. 
    """
    def __init__(self, n_qubits: int, batchsize: int):
        self.nq = n_qubits
        super().__init__(2**n_qubits, batchsize)
        self.density_matrix = QubitDensityMatrix(n_qubits=self.nq, batchsize=batchsize)


    def set_rho_by_config(self, config):
        if isinstance(config, th.Tensor):
            _c = config
        elif isinstance(config, np.ndarray):
            if config.shape != (self.nq,):
                raise ValueError("Config must have shape (#qubits).")
            if config.dtype != np.int64:
                raise ValueError("Config must be an integer array.")
            _c = th.tensor(config, dtype=th.int64)
        elif isinstance(config, list):
            if len(config) != self.nq:
                raise ValueError("Config must have length (#qubits).")
            if not all(isinstance(i, int) for i in config):
                raise ValueError("Config must be a list of integers.")
            _c = th.tensor(config, dtype=th.int64)
        else:
            raise ValueError("Config must be a 0/1 tensor, a 0/1 numpy array, or a list of 0/1 integers.")
        if not all(i in [0, 1] for i in _c):
            raise ValueError("Config must be a sequence of 0/1 integers. Example for 2-qubit system: [0, 1] means |01>.")
        self.density_matrix.set_rho_by_config(_c)

    def rotate(self, direction: th.Tensor, angle: float, config=None):
        """
        Apply a rotation operator about the Cartesian axes in the Bloch Basis.
        """
        rho_new = self.density_matrix.apply_unitary_rotation(self.rho, direction, angle, config)
        self.rho = rho_new
        return self.rho

    def kraus_operate(self, kraus_operators: list[th.Tensor], config=None):
        """
        Apply a Kraus operation.
        """
        self.rho = self.density_matrix.apply_kraus_operation(self.rho, kraus_operators, config)
        return self.rho
    
class ParticleLindbladSystem(QubitLindbladSystem):
    """
    This class represents the states of n physical qubits as a open quantum system.
    The quantum states include the two-level states of each qubit and the center-of-mass position and momentum of each qubit. 
    We take a ring-polymer (number of beads = p) representation of the center-of-mass state for sampling the qubit distribution. 
    The two-level states will be represented by a (p x 2^n x 2^n) density matrix.
    The center-of-mass position will be represented by a (p x 3n) matrix.

    For harmonic trapping with frequency omega, the number of beads needed for convergence can be estimated by  (hbar omega / kT).
    """
    def __init__(self, n_qubits: int, batchsize: int, particles: Particles):
        super().__init__(n_qubits, batchsize)
        self.particles = particles

    def reset(self):
        """
        Reset the system. 
        In addition to the reset of the quantum system, the center-of-mass positions and momenta of the particles are also reset to arbitrary thermal states.
        """
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.reset()
        for operator_group in self._jumping_group_dict.values():
            operator_group.reset()
        self.particles.reset()
        # self.step_particles(100) # TODO: remove this line

    def step_particles(self, dt: float):
        """
        This function steps the thermal motino of the particles for a time step dt.
        """
        dt_thermal = self.particles.dt
        substeps = int(np.round(dt / dt_thermal, 0))
        if (dt / dt_thermal) != substeps:
            raise ValueError("The time step dt of integrating Lindblad equation must be an integer multiple of the time step dt_thermal for integrating the thermal motion of particles.")
        for _ in range(substeps):
            self.particles.zero_forces()
            self.particles.modify_forces(self.particles.get_trapping_forces())
            self.particles.step_langevin(record_traj=False)
        return

    def step(self, dt: float, set_buffer: bool = False, profile: bool = False):
        """
        This function steps the system for a time step dt.
        """
        if profile:
            t0 = timer()
        self.step_particles(dt)  
        if profile:
            t1 = timer()
            logging.info(f"The time taken for stepping the thermal motion of qubits is {t1 - t0}s.")
        hamiltonian = self.step_hamiltonian(dt, set_buffer=True)
        if self._jumping_group_dict:
            jump_operator_list = self.step_jumping(dt, set_buffer)
        else:
            jump_operator_list = []
        self.rho = self.step_rho(dt, hamiltonian, jump_operator_list)
        if profile:
            t2 = timer()
            logging.info(f"The time taken for stepping the quantum states of qubits is {t2 - t1}s.")
        return self.rho

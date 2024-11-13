"""
This file contains the system class for the QEpsilon project.
"""

import torch as th
from qepsilon.density_matrix import DensityMatrix
from qepsilon.operator_group import *
from qepsilon.utility import ABAd

class LindbladSystem(th.nn.Module):
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

    def __init__(self, n_qubits: int, batchsize: int = 1):
        super().__init__()
        self.nq = n_qubits
        self.ns = 2**n_qubits
        self.nb = batchsize 
        self.density_matrix = DensityMatrix(n_qubits, batchsize)
        self._hamiltonian_operator_group_dict = {}
        self._jumping_group_dict = {}
        self._ham = None
        self._jump = None

    ############################################################
    # Setter and getter  
    ############################################################
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
        self.density_matrix.set_rho(rho)
    
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
        self._jumping_group_dict[operator_group.id] = operator_group

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
            hamiltonian += ops[None, :, :] * coefs[:, None, None]
            # hamiltonian += operator_group.sample(dt)
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
            jump_operator_list.append(ops[None, :, :] * coefs[:, None, None])
            # jump_operator_list.append(operator_group.sample(dt))
        if set_buffer:
            self._jump = jump_operator_list
        return jump_operator_list

    def step_rho(self, dt: float, hamiltonian: th.Tensor, jump_operators: list[th.Tensor]):
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
        identity = th.eye(self.ns, dtype=th.cfloat).to(rho.device).unsqueeze(0)
        rho_new = ABAd(identity - 1j * dt * hamiltonian, rho)
        for jump_operator in jump_operators:
            rho_new += ABAd(jump_operator, rho) * dt
        return rho_new

    def step(self, dt: float, set_buffer: bool = False):
        """
        This function steps the system for a time step dt.
        """
        hamiltonian = self.step_hamiltonian(dt, set_buffer)
        if self._jumping_group_dict:
            jump_operator_list = self.step_jumping(dt, set_buffer)
        else:
            jump_operator_list = []
        self.rho = self.step_rho(dt, hamiltonian, jump_operator_list)
        return self.rho

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
 
class MolecularLindbladSystem(LindbladSystem):
    """
    This class represents the states of n physical qubits as a open quantum system.
    The quantum states include the two-level states of each qubit and the center-of-mass position and momentum of each qubit. 
    We take a ring-polymer (number of beads = p) representation of the center-of-mass state for sampling the qubit distribution. 
    The two-level states will be represented by a (p x 2^n x 2^n) density matrix.
    The center-of-mass position will be represented by a (p x 3n) matrix.

    For harmonic trapping with frequency omega, the number of beads needed for convergence can be estimated by  (hbar omega / kT).
    """
    pass
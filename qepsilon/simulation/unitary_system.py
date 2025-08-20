"""
This file contains the unitary simulation class for the QEpsilon project.
"""

import numpy as np
import torch as th
from qepsilon.operator_group import *
from qepsilon.system.pure_ensemble import *
from qepsilon.utilities import expectation_pse, apply_to_pse
import logging
from time import time as timer

class UnitarySystem(th.nn.Module):
    """
    This class represents an ensemble of unitary pure quantum systems.
    Each pure state is represented by a vector. 
    The evolution of the system is governed by Schrodinger equation for pure states.
    
    In this class, each component of the Hamiltonian $H(t)$ is represented by a OperatorGroup object. 
    Each OperatorGroup object contains a group of operators and the corresponding coefficients. 
    The operators themselves are time-independent, like a Pauli operator. But the coefficients can be time-dependent stochastic processes. 
    This corresponds to the case where the qubits are coupled to noisy environments such as inhomogeneous, fluctuating magnetic fields. 
    """

    def __init__(self, num_states: int, batchsize: int = 1):
        super().__init__()
        self.ns = num_states
        self.nb = batchsize 
        self.pure_ensemble = PureStatesEnsemble(num_states, batchsize)
        self._hamiltonian_operator_group_dict = th.nn.ModuleDict()
        self._ham = None
        self._evo = None
    
    ############################################################
    ## methods for storing the evolution operator. Needed when, e.g., evaluating correlation functions.
    ############################################################
    def reset_evo(self):
        self._evo = th.eye(self.ns, dtype=th.cfloat, device=self.pse.device).unsqueeze(0).repeat(self.nb, 1, 1)

    def accumulate_evo(self, evo: th.Tensor):
        if self._evo is None:
            self._evo = th.eye(self.ns, dtype=th.cfloat, device=evo.device).unsqueeze(0).repeat(self.nb, 1, 1)
        self._evo =  th.matmul(evo, self._evo)

    @property
    def evo(self):
        return self._evo.clone()

    ############################################################
    ## methods for moving the system to GPU, overiding the .to() method of th.nn.Module
    ############################################################
    def to(self, device='cuda'):
        """
        This overrides the ``to`` method of PyTorch Module. It is used to move all relevant components of the system to a specific device.
        """
        self.pure_ensemble.to(device=device)
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.to(device=device)
        if self._ham is not None:
            self._ham = self._ham.to(device=device)
        if self._evo is not None:
            self._evo = self._evo.to(device=device)
        
    ############################################################
    # Setter and getter for the Hamiltonian operator groups and the pure state ensemble
    ############################################################
    @property
    def hamiltonian_operator_groups(self):
        return list(self._hamiltonian_operator_group_dict.values()) 
    
    def get_hamiltonian_operator_group_by_ID(self, id: str):
        if id in self._hamiltonian_operator_group_dict:
            return self._hamiltonian_operator_group_dict[id]
        else:
            raise ValueError(f"The ID {id} does not exist in the Hamiltonian operator group dictionary.")
        
    @property
    def pse(self):
        """
        This property returns the pure state ensemble of the system.
        """
        pse = self.pure_ensemble.get_pse()
        if pse is None:
            raise ValueError("The pure state ensemble has not been initialized.")
        return pse

    @pse.setter
    def pse(self, pse: th.Tensor):
        if pse.dtype != th.cfloat:
            _pse = pse.to(dtype=th.cfloat)
        else:
            _pse = pse
        self.pure_ensemble.set_pse(_pse)
    

    @property
    def hamiltonian(self):
        return self._ham
    
    def reset(self):
        """
        Reset the system.
        """
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.reset()

    def HamiltonianParameters(self):
        parameters_list = []
        for operator_group in self._hamiltonian_operator_group_dict.values():
            parameters_list.extend(list(operator_group.parameters()))
        return parameters_list

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

    ############################################################
    # Methods for evolving the system
    ############################################################
    def normalize(self):
        pse_norm = self.pure_ensemble.norm
        self.pse = self.pse / pse_norm[:, None]
        return

    def step_hamiltonian(self, dt: float, set_buffer: bool = False, verbose: bool = False):
        """
        This function steps the Hamiltonian for a time step dt. 
        Args:
            dt: a float, the time step.
        Returns:
            hamiltonian: a (self.nb, self.ns, self.ns) tensor, the Hamiltonian operator at time t.
        """
        hamiltonian = 0
        for operator_group in self._hamiltonian_operator_group_dict.values():
            #### sample the operator group and get the coefficients
            ops, coefs = operator_group.sample(dt)
            ## sanitary check
            if coefs.shape != (self.nb,):
                raise ValueError("The coefficients sampled from an operator group should be a 1D tensor of length equal to the batchsize.")
            #### add the interaction terms to the Hamiltonian, this is much more computationally expensive than sampling the operator group
            ## no broadcasting if the operators is already batched
            if ops.shape == (self.nb, self.ns, self.ns):
                hamiltonian += ops * coefs[:, None, None]
            ## broadcast if the operators is not already batched
            elif ops.shape == (self.ns, self.ns):
                hamiltonian += ops[None, :, :] * coefs[:, None, None]
            else:
                raise ValueError(f"The shape of the operator sampled from operator group {operator_group.id} should either be (batchsize, n_states, n_states) or (n_states, n_states).")
            if verbose:
                print(f"get operator matrix from operator group {operator_group.id}:")
                print(f"coefficient={coefs}")
        if set_buffer:
            self._ham = hamiltonian
        return hamiltonian

    def step_pse(self, dt: float, hamiltonian: th.Tensor, set_buffer=False, set_buffer_evo=False):
        """
        This function steps the pure state ensemble for a time step dt.
        """
        identity = th.eye(self.ns, dtype=th.cfloat).to(hamiltonian.device).unsqueeze(0).repeat(self.nb, 1, 1)
        evolution_matrix = identity - 1j * dt * hamiltonian
        ## sparsify tensors if the dimension of the matrix is larger than 2000
        if self.ns > 10000:
            evolution_matrix = evolution_matrix.to_sparse()
        if set_buffer_evo:
            self.accumulate_evo(evolution_matrix)
        pse_new = apply_to_pse(self.pse, evolution_matrix)
        if pse_new.is_sparse:
            pse_new = pse_new.to_dense()
        return pse_new

    def step(self, dt: float, set_buffer: bool = False, time_independent: bool = False, set_buffer_evo: bool = False, profile: bool = False):
        """
        This function steps the system for a time step dt.
        """
        ## sanity check
        if time_independent:
            if set_buffer is False:
                raise ValueError("If time_independent is True, set_buffer must be True.")
        ##
        if profile:
            t0 = timer()
            th.cuda.synchronize()
        if time_independent:
            if self._ham is not None:
                hamiltonian = self._ham
            else:
                hamiltonian = self.step_hamiltonian(dt, set_buffer)
                self._ham = hamiltonian
        else:
            hamiltonian = self.step_hamiltonian(dt, set_buffer)
        if profile:
            th.cuda.synchronize()
            t1 = timer()
            logging.info(f"The time taken for stepping the Hamiltonian is {t1 - t0}s.")
        self.pse = self.step_pse(dt, hamiltonian, set_buffer, set_buffer_evo)
        if profile:
            th.cuda.synchronize()
            t2 = timer()
            logging.info(f"The time taken for stepping the pure state ensemble is {t2 - t1}s.")
        return self.pse
 
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
        return expectation_pse(self.pse, ops_batched)
        
    
class TightBindingUnitarySystem(UnitarySystem):
    """
    This class represents the states of an ensemble of tight binding systems.
    """
    def __init__(self, n_sites: int, batchsize: int):
        super().__init__(num_states=n_sites, batchsize=batchsize)
        self.pure_ensemble = TightBindingPureStatesEnsemble(n_sites=n_sites, batchsize=batchsize)
        
    def set_pse_by_config(self, config: th.Tensor):
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
        self.pure_ensemble.set_pse_by_config(_c)

class QubitUnitarySystem(UnitarySystem):
    """
    This class represents the states of an ensemble of pure n-qubit systems.
    Each pure state is represented by a vector. 
    The evolution of the system is governed by the Schrodinger equation.
    
    In this class, each component of the Hamiltonian $H(t)$ is represented by a OperatorGroup object. 
    Each OperatorGroup object contains a group of operators and the corresponding coefficients. 
    The operators themselves are time-independent, like a Pauli operator. But the coefficients can be time-dependent stochastic processes. 
    This corresponds to the case where the qubits are coupled to noisy environments such as inhomogeneous, fluctuating magnetic fields. 
    """
    def __init__(self, n_qubits: int, batchsize: int):
        self.nq = n_qubits
        super().__init__(2**n_qubits, batchsize)
        self.pure_ensemble = QubitPureStatesEnsemble(n_qubits=self.nq, batchsize=batchsize)


    def set_pse_by_config(self, config):
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
        self.pure_ensemble.set_pse_by_config(_c)

    def rotate(self, direction: th.Tensor, angle: float, config=None):
        """
        Apply a rotation operator about the Cartesian axes in the Bloch Basis.
        """
        pse_new = self.pure_ensemble.apply_unitary_rotation(self.pse, direction, angle, config)
        self.pse = pse_new
        return self.pse

    
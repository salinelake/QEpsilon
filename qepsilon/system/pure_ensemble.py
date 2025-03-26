"""
This file contains the pure states ensemble class for the QEpsilon project.
"""

import torch as th
from qepsilon.tls import Pauli
from qepsilon.utilities import compose, qubitconf2idx, apply_to_pse, expectation_pse

class PureStatesEnsemble(th.nn.Module):
    """
    Base class for ensembles of pure states.
    """
    def __init__(self, num_states: int, batchsize: int = 1):
        super().__init__()
        self.ns = num_states
        self.nb = batchsize
        self.register_buffer("_pse", None) ## initialize later. Shape will be (self.nb, self.ns)

    ############################################################
    # Getters and setters for the pure states
    ############################################################
    def get_pse(self):
        return self._pse
    
    def set_pse(self, pse: th.Tensor):
        """
        This function sets the pure states.
        Args:
            pse: a complex tensor. Shape: (self.nb, self.ns).
        """
        if pse.dtype != th.cfloat:
            raise ValueError("Pure states must be a complex tensor (th.cfloat).")
        if pse.shape == (self.ns,):
            pse = pse.unsqueeze(0)
            if self._pse is None:
                self._pse = pse.repeat(self.nb, 1)
            else:
                self._pse = pse.repeat(self.nb, 1).to(self._pse.device)
        elif pse.shape == (self.nb, self.ns):
            if self._pse is None:
                self._pse = pse
            else:
                self._pse = pse.to(self._pse.device)
        else:
            raise ValueError("Pure states must have shape (ns) or (batchsize, ns).")
            
    @property
    def norm(self):
        return th.norm(self._pse, dim=1)

    ############################################################
    # Basic operations on pure state ensemble. Methods below do not update the stored pure states. Use setter if you want to update.
    ############################################################
    def normalize(self, pse: th.Tensor):
        """
        This function normalizes the pure states.
        Args:
            pse: the pure states to be normalized. Shape: (batchsize, ns).
        """
        return pse / self.norm[:, None]
    
    def get_expectation(self, operator: th.Tensor):
        """
        This function computes the expectation of an operator on the pure state ensemble.
        Args:
            operator: the operator to get the expectation. Shape: (ns, ns).
        Returns:
            expectation: the expectation of the operator. Shape: (batchsize).
        """
        if operator.shape == (self.ns, self.ns) or operator.shape == (self.nb, self.ns, self.ns):
            pass
        else:
            raise ValueError("Operator must have shape (ns, ns) or (batchsize, ns, ns).")
        if operator.dtype != th.cfloat:
            raise ValueError("Operator must be a complex tensor (th.cfloat).")
        return expectation_pse(self._pse, operator)



class QubitPureStatesEnsemble(PureStatesEnsemble):
    """
    This class deals with pure states of an ensemble of n-qubit systems. Basic quantum operations on the ensemble of pure states are implemented.
    """
    def __init__(self, n_qubits: int = 1, batchsize: int = 1):
        self.nq = n_qubits
        self.ns = 2**n_qubits
        super().__init__(num_states=self.ns, batchsize=batchsize)
        self.pauli = Pauli(n_qubits)

    ############################################################
    # Getters and setters for the density matrix
    ############################################################
    def set_pse_by_config(self, config: th.Tensor):
        """
        This function sets all pure states as |config>.
        Args:
            config: a 0 or 1 tensor that specifies the spin configuration. Shape: (#qubits). Example for 2-qubit system: [0, 1] means |01>.
        """
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.int64:
            raise ValueError("Config must be an integer tensor (th.int64).")
        idx = qubitconf2idx(config)
        pse = th.zeros(self.nb, self.ns, dtype=th.cfloat)
        pse[:, idx] += 1
        self.set_pse(pse)
        return
    
    ############################################################
    # Basic operations on pure states
    ############################################################
    def partial_trace(self, pse: th.Tensor, config: th.Tensor):
        """
        This function traces out the qubits specified in config.
        Args:
            pse: the (ns) pure states to be traced out.
            config: a boolean tensor that specifies the qubits to be kept. config[i]==False means the i-th qubit will be traced out. Shape: (#qubits).
        """ 
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.bool:
            raise ValueError("Config must be a boolean tensor (th.bool).")
        
        # Determine the indices of qubits to keep
        keep_indices = [i for i, keep in enumerate(config) if keep]
        trace_indices = [i for i, keep in enumerate(config) if not keep]
        raise NotImplementedError("Partial trace for n-qubit system has not been implemented.")
    
    def apply_unitary_rotation(self, pse: th.Tensor, u: th.Tensor, theta: float, config=None):
        """
        This function applies the unitary rotation operator about the direction u by angle theta to the pure states. The rotation is simultaneous performed on selected qubits.
        Args:
            pse: the pure states to be rotated.
            direction: the direction of the rotation. Shape: (3)  
            angle: the angle of the rotation. 
            config: a boolean tensor that specifies the qubits to be rotated. config[i]==True means the i-th qubit is included in the rotation. Shape: (#qubits). If not specified, all qubits are included in the rotation.  
        """
        ## sanity check
        if config is None:
            config = th.ones(self.nq, dtype=th.bool)
        else:
            if config.shape != (self.nq,):
                raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.bool:
            raise ValueError("Config must be a boolean tensor (th.bool).")
        if u.dtype != th.float:
            raise ValueError("Direction must be a real-valued tensor (th.float).")
        ## apply the rotation
        theta = th.tensor(theta).to(u.device)
        M = self.pauli.SU2_rotation(u, theta)
        one_body_ops = [M if config[i] else self.pauli.I for i in range(self.nq)]
        ops = compose(one_body_ops).unsqueeze(0)
        pse_new = apply_to_pse(pse, ops)
        return pse_new

    ############################################################
    # Observing the pure state ensemble
    ############################################################

    def observe_one_qubit(self, pse: th.Tensor, observable: th.Tensor, idx: int):
        """
        This function observes the one-qubit observable on the idx-th qubit.
        Args:
            pse: the pure states to be observed.
            observable: the one-qubit observable.
            idx: the index of the qubit to be observed.
        """
        if observable.shape != (2, 2):
            raise ValueError("One-qubit observable must have shape (2, 2).")
        if observable.dtype != th.cfloat:
            raise ValueError("One-qubit observable must be a complex tensor (th.cfloat).")
        one_body_ops = [self.pauli.I] * self.nq
        one_body_ops[idx] = observable
        ops = compose(one_body_ops).unsqueeze(0)
        O_pse = apply_to_pse(pse, ops)
        pse_O_pse = (pse * O_pse.conj()).sum(dim=-1)
        return pse_O_pse

    def observe_paulix_one_qubit(self, pse: th.Tensor, idx: int):
        return self.observe_one_qubit(pse, self.pauli.X, idx)
    
    def observe_pauliy_one_qubit(self, pse: th.Tensor, idx: int):
        return self.observe_one_qubit(pse, self.pauli.Y, idx)
    
    def observe_pauliz_one_qubit(self, pse: th.Tensor, idx: int):
        return self.observe_one_qubit(pse, self.pauli.Z, idx)
    
    def observe_prob_by_config(self, pse: th.Tensor, config: th.Tensor):
        """
        This function observes the probability of the spin configuration specified by config.
        Args:
            pse: the pure states. Shape: (batchsize, ns). 
            config: a 0 or 1 tensor that specifies the spin configuration. Shape: (#qubits). Example for 2-qubit system: [0, 1] means |01>.
        Returns:
            prob: the probability of the spin configuration. Shape: (batchsize).
        """
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.int64:
            raise ValueError("Config must be an integer tensor (th.int64).")
        idx = qubitconf2idx(config)
        prob = th.abs(pse[:, idx]/ self.norm)**2 
        return prob
 
    
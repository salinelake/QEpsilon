"""
This module deals with density matrices.
"""

import torch as th
from qepsilon.tls import Pauli
from qepsilon.utility import compose

class DensityMatrix(th.nn.Module):
    """
    This class deals with instantaneous quantum operations on n-qubit systems. The operation is not necessarily unitary. A quantum operation is also called a quantum channel. 
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.nq = n_qubits
        self.ns = 2**n_qubits
        self.pauli = Pauli(n_qubits)
        self.register_buffer("_rho", None) ## initialize later
    

    ############################################################
    # Getters and setters for the density matrix
    ############################################################
    def get_rho(self):
        return self._rho
    
    def set_rho(self, rho: th.Tensor):
        """
        This function sets the density matrix.
        Args:
            rho: a 2^n x 2^n complex tensor.
        """
        if rho.shape != (self.ns, self.ns):
            raise ValueError("Density matrix must have shape (2^n, 2^n).")
        if rho.dtype != th.cfloat:
            raise ValueError("Density matrix must be a complex tensor (th.cfloat).")
        if self._rho is None:
            self._rho = rho
        else:
            self._rho = rho.to(self._rho.device)
    
    def set_rho_by_config(self, config: th.Tensor):
        """
        This function sets the density matrix as |config><config|.
        Args:
            config: a 0 or 1 tensor that specifies the spin configuration. Shape: (#qubits). Example for 2-qubit system: [0, 1] means |01>.
        """
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.int64:
            raise ValueError("Config must be an integer tensor (th.int64).")
        one_body_rho = []
        for s in config:
            if s not in [0, 1]:
                raise ValueError("Config must be a 0 or 1 tensor.")
            if s == 0:
                one_body_rho.append(th.tensor([[1, 0], [0, 0]], dtype=th.cfloat))
            else:
                one_body_rho.append(th.tensor([[0, 0], [0, 1]], dtype=th.cfloat))
        self.set_rho(compose(one_body_rho))

    ############################################################
    # Basic operations on density matrices. Methods below do not update the stored density matrix. Use setter if you want to update.
    ############################################################
    def trace(self, rho: th.Tensor):
        """
        This function traces out the density matrix.
        Args:
            rho: the density matrix to be traced out.
        """
        return th.trace(rho)
    
    def normalize(self, rho: th.Tensor):
        """
        This function normalizes the density matrix.
        Args:
            rho: the density matrix to be normalized.
        """
        return rho / th.trace(rho)
    
    def partial_trace(self, rho: th.Tensor, config: th.Tensor):
        """
        This function traces out the qubits specified in config.
        Args:
            rho: the (2^n x 2^n) density matrix to be traced out.
            config: a boolean tensor that specifies the qubits to be kept. config[i]==False means the i-th qubit will be traced out. Shape: (#qubits).
        """
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.bool:
            raise ValueError("Config must be a boolean tensor (th.bool).")
        
        # Determine the indices of qubits to keep
        keep_indices = [i for i, keep in enumerate(config) if keep]
        trace_indices = [i for i, keep in enumerate(config) if not keep]
        
        # Reshape the density matrix to separate the qubits
        reshaped_rho = rho.reshape([2] * (2 * self.nq))
        
        # Perform the partial trace by summing over the specified axes
        if self.nq > 26:
            ## it is unlikely to have more than 26 qubits because we store the density matrix plainly.
            raise NotImplementedError("Partial trace for a system with more than 26 qubits is not implemented.")
        else:
            ## einstein summation
            ein_equation = [chr(ord('a') + i) for i in range(self.nq)] + [chr(ord('A') + i) for i in range(self.nq)]
            for idx in trace_indices:
                ein_equation[idx] = ein_equation[idx + self.nq]
            traced_rho = th.einsum(ein_equation, reshaped_rho)
        
        # Reshape the result back to a matrix
        new_shape = (2**len(keep_indices), 2**len(keep_indices))
        return traced_rho.reshape(new_shape)

    def apply_unitary_rotation(self, rho: th.Tensor, u: th.Tensor, theta: float, config=None):
        """
        This function applies the unitary rotation operator about the direction u by angle theta to the density matrix. The rotation is simultaneous performed on selected qubits.
        Args:
            rho: the density matrix to be rotated.
            direction: the direction of the rotation. Shape: (3)  
            angle: the angle of the rotation. 
            config: a boolean tensor that specifies the qubits to be rotated. config[i]==True means the i-th qubit is included in the rotation. Shape: (#qubits). If not specified, all qubits are included in the rotation.  
            set_state: a boolean flag to set the density matrix to the new state.
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
        ops = compose(one_body_ops)
        rho_new = ops @ rho @ ops.conj().T  
        return rho_new
    
    def apply_kraus_operation(self, rho: th.Tensor, kraus_operators: list[th.Tensor], config=None):
        """
        This function applies the Kraus operation to the density matrix. The operation is simultaneous performed on selected qubits.
        Args:
            rho: the density matrix to be acted on.
            kraus_operators: a list of Kraus operators.
            config: a boolean tensor that specifies the qubits to be acted on. config[i]==True means the i-th qubit is included in the operation. Shape: (#qubits). If not specified, all qubits are included in the operation.
            set_state: a boolean flag to set the density matrix to the new state.
        """
        if config is None:
            config = th.ones(self.nq, dtype=th.bool)
        else:
            if config.shape != (self.nq,):
                raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.bool:
            raise ValueError("Config must be a boolean tensor (th.bool).")
        for idx, s in enumerate(config):
            if s:
                composite_kraus = []
                for K in kraus_operators:
                    one_body_ops = [self.pauli.I] * self.nq
                    one_body_ops[idx] = K
                    ops = compose(one_body_ops)
                    composite_kraus.append(ops)
                new_rho = 0
                for K in composite_kraus:
                    new_rho += K @ rho @ K.conj().T
        return new_rho

    ############################################################
    # Observing the density matrix
    ############################################################
    def observe_one_qubit(self, rho: th.Tensor, observable: th.Tensor, idx: int):
        """
        This function observes the one-qubit observable on the idx-th qubit.
        Args:
            rho: the density matrix to be observed.
            observable: the one-qubit observable.
            idx: the index of the qubit to be observed.
        """
        if observable.shape != (2, 2):
            raise ValueError("One-qubit observable must have shape (2, 2).")
        if observable.dtype != th.cfloat:
            raise ValueError("One-qubit observable must be a complex tensor (th.cfloat).")
        one_body_ops = [self.pauli.I] * self.nq
        one_body_ops[idx] = observable
        ops = compose(one_body_ops)
        th.trace(ops @ rho)
        return
    
    def observe_paulix_one_qubit(self, rho: th.Tensor, idx: int):
        return self.observe_one_qubit(rho, self.pauli.X, idx)
    
    def observe_pauliy_one_qubit(self, rho: th.Tensor, idx: int):
        return self.observe_one_qubit(rho, self.pauli.Y, idx)
    
    def observe_pauliz_one_qubit(self, rho: th.Tensor, idx: int):
        return self.observe_one_qubit(rho, self.pauli.Z, idx)

    def get_diagonal_by_config(self, rho: th.Tensor, config: th.Tensor):
        """
        This function gets the diagonal elements of the density matrix specified by config.
        Args:
            rho: the density matrix.
            config: a 0 or 1 tensor that specifies the spin configuration. Shape: (#qubits). Example for 2-qubit system: [0, 1] means |01>.
        """
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.int64:
            raise ValueError("Config must be an integer tensor (th.int64).")
        rho_diag = rho.diag().reshape([2] * self.nq)
        indices = tuple(config.numpy())
        return rho_diag[indices]
    
    def observe_prob_by_config(self, rho: th.Tensor, config: th.Tensor):
        """
        This function observes the probability of the spin configuration specified by config.
        Args:
            rho: the density matrix.
            config: a 0 or 1 tensor that specifies the spin configuration. Shape: (#qubits). Example for 2-qubit system: [0, 1] means |01>.
        """
        if config.shape != (self.nq,):
            raise ValueError("Config must have shape (#qubits).")
        if config.dtype != th.int64:
            raise ValueError("Config must be an integer tensor (th.int64).")
        rho_diag = rho.diag()
        rho_trace = rho_diag.sum()
        rho_diag = rho_diag.reshape([2] * self.nq)
        indices = tuple(config.numpy())
        prob = rho_diag[indices] / rho_trace
        return prob.real



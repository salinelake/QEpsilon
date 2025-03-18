import qepsilon as qe

class holstein_1D:
    def __init__(self, nsite, batchsize):
        self.nsite = nsite
        self.batchsize = batchsize
        self.system = qe.QubitLindbladSystem(n_qubits=nsite, batchsize=batchsize)

    def get_hopping_operator_group(self, id, coef, requires_grad=False, pbc=True):
        if pbc is False:
            raise NotImplementedError("Currently only PBC is supported")
        nsite = self.nsite
        op_hop = qe.StaticPauliOperatorGroup(n_qubits=nsite, id=id, batchsize=self.batchsize, coef=coef, requires_grad=requires_grad)
        for idx in range(nsite):
            hop_seq_1 = ['I'] * nsite 
            hop_seq_1[idx] = 'U'
            hop_seq_1[(idx+1)%nsite] = 'D'
            hop_seq_1 = "".join(hop_seq_1)
            op_hop.add_operator(hop_seq_1)

            hop_seq_2 = ['I'] * nsite 
            hop_seq_2[idx] = 'D'
            hop_seq_2[(idx+1)%nsite] = 'U'
            hop_seq_2 = "".join(hop_seq_2)
            op_hop.add_operator(hop_seq_2)
        return op_hop
    
    def get_current_operator_group(self, id, coef=1.0, requires_grad=False, pbc=True):
        if pbc is False:
            raise NotImplementedError("Currently only PBC is supported")
        nsite = self.nsite
        op_current = qe.StaticPauliOperatorGroup(n_qubits=nsite, id=id, batchsize=self.batchsize, coef=coef, requires_grad=requires_grad)
        for idx in range(nsite):
            hop_seq_1 = ['I'] * nsite 
            hop_seq_1[idx] = 'U'
            hop_seq_1[(idx+1)%nsite] = 'D'
            hop_seq_1 = "".join(hop_seq_1)

            hop_seq_2 = ['I'] * nsite 
            hop_seq_2[idx] = 'D'
            hop_seq_2[(idx+1)%nsite] = 'U'
            hop_seq_2 = "".join(hop_seq_2)
            op_current.add_operator(hop_seq_1, prefactor=-1.0)
            op_current.add_operator(hop_seq_2, prefactor=1.0) 
        return op_current
     
    def get_jump_operator_group(self, id, coef, requires_grad=False, pbc=True):
        if pbc is False:
            raise NotImplementedError("Currently only PBC is supported")
        nsite = self.nsite
        op_jump = qe.StaticPauliOperatorGroup(n_qubits=nsite, id=id, batchsize=self.batchsize, coef=coef, requires_grad=requires_grad)
        

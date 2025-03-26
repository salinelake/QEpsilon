import qepsilon as qe

def create_hopping_operator_group(id, nsite, batchsize, coef, static=True, requires_grad=False ):
    op_hop = qe.StaticPauliOperatorGroup(n_qubits=nsite, id=id, batchsize=batchsize, coef=coef, static=static, requires_grad=requires_grad)
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

def create_current_operator_group(id, nsite, batchsize, coef=1.0, requires_grad=False):
    op_current = qe.StaticPauliOperatorGroup(n_qubits=nsite, id=id, batchsize=batchsize, coef=coef, requires_grad=requires_grad)
    for idx in range(nsite):
        hop_seq_1 = ['I'] * nsite 
        hop_seq_1[idx] = 'U'
        hop_seq_1[(idx+1)%nsite] = 'D'
        hop_seq_1 = "".join(hop_seq_1)
        op_current.add_operator(hop_seq_1, prefactor=-1.0)

        hop_seq_2 = ['I'] * nsite 
        hop_seq_2[idx] = 'D'
        hop_seq_2[(idx+1)%nsite] = 'U'
        hop_seq_2 = "".join(hop_seq_2)
        op_current.add_operator(hop_seq_2, prefactor=1.0) 
    return op_current
    
def create_local_number_operator_group(nsite, batchsize):
    op_list=[]
    for i in range(nsite):
        op = qe.StaticPauliOperatorGroup(n_qubits=nsite, id='n_{}'.format(i), batchsize=batchsize, coef=1.0, requires_grad=False)
        op.add_operator('I' * i + 'N' + 'I' * (nsite - i - 1))
        op_list.append(op)
    return op_list

def create_local_noise_operator_group(nsite, batchsize, amp):
    op_list=[]
    for i in range(nsite):
        op = qe.WhiteNoisePauliOperatorGroup(n_qubits=nsite, id='w_{}'.format(i), batchsize=batchsize, amp=amp, requires_grad=False)
        op.add_operator('I' * i + 'N' + 'I' * (nsite - i - 1))
        op_list.append(op)
    return op_list
 
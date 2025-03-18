import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import qepsilon as qe
from qepsilon.utilities import Constants_Metal as Constants
from qepsilon.utilities import trace, bin2idx
# log_suffix = time.strftime("%Y%m%d-%H%M%S")
# logging.basicConfig(filename=f'log_{log_suffix}.log', level=logging.INFO)

## do not print the tensor in scientific notation
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)
dev = 'cuda'
################################################
#  define the system
################################################
batchsize = 8
nq = 10
hopping_value = 83
hopping_coef = hopping_value * Constants.meV  # 83meV
# sz_shift_coef = 0 # - 42.06773376464844 
qubit = qe.QubitLindbladSystem(n_qubits=nq, batchsize=batchsize)
static_hamiltonian = 0
################################################
# define the terms in the Hamiltonian
################################################
## add hopping terms. Note that here we will define -H because we want to evolve the current operator instead of the density operator
op_hop = qe.StaticPauliOperatorGroup(n_qubits=nq, id="hop", batchsize=batchsize, coef=-hopping_coef, requires_grad=False)
for idx in range(nq):
    hop_seq_1 = ['I'] * nq 
    hop_seq_1[idx] = 'U'
    hop_seq_1[(idx+1)%nq] = 'D'
    hop_seq_1 = "".join(hop_seq_1)
    op_hop.add_operator(hop_seq_1)

    hop_seq_2 = ['I'] * nq 
    hop_seq_2[idx] = 'D'
    hop_seq_2[(idx+1)%nq] = 'U'
    hop_seq_2 = "".join(hop_seq_2)
    op_hop.add_operator(hop_seq_2)
qubit.add_operator_group_to_hamiltonian(op_hop)
mat_hop, coef = op_hop.sample(0)
static_hamiltonian += - mat_hop[None,:,:] * coef[:,None,None]

## add one-body terms
for idx in range(nq):
    seq = ['I'] * nq
    seq[idx] = 'Z'
    seq = "".join(seq)
    # sz_shift = qe.StaticPauliOperatorGroup(n_qubits=nq, id="sz_shift_{}".format(idx), batchsize=batchsize, coef=-sz_shift_coef, requires_grad=False)
    # sz_shift.add_operator(seq)
    # qubit.add_operator_group_to_hamiltonian(sz_shift)
    # mat_shift, coef = sz_shift.sample(0)
    # static_hamiltonian += - mat_shift * coef[0]

    sz1 = qe.LangevinNoisePauliOperatorGroup(n_qubits=nq, id="sz_noise_langevin_{}".format(idx), batchsize=batchsize, tau=0.20435911417007446, amp=11.90210247039795, requires_grad=False)
    sz1.add_operator(seq)
    qubit.add_operator_group_to_hamiltonian(sz1)

    sz2 = qe.PeriodicNoisePauliOperatorGroup(n_qubits=nq, id="sz_noise_periodic_{}".format(idx), batchsize=batchsize, tau=0.037097856402397156, amp=57.55181884765625, requires_grad=False)
    sz2.add_operator(seq)
    qubit.add_operator_group_to_hamiltonian(sz2)

    sz3 = qe.PeriodicNoisePauliOperatorGroup(n_qubits=nq, id="sz_noise_periodic_2_{}".format(idx), batchsize=batchsize, tau=0.18760807812213898, amp=7.012815475463867, requires_grad=False)
    sz3.add_operator(seq)
    qubit.add_operator_group_to_hamiltonian(sz3)

    # ## try adding the noise operators to the static Hamiltonian and sample the initial density matrix differently.
    # op, coef = sz1.sample(0)
    # static_hamiltonian += - op[None,:,:] * coef[:,None,None]
    # op, coef = sz2.sample(0)
    # static_hamiltonian += - op[None,:,:] * coef[:,None,None]
    # op, coef = sz3.sample(0)
    # static_hamiltonian += - op[None,:,:] * coef[:,None,None]

################################################
# define the jump operators
################################################
for idx in range(nq):
    seq = ['I'] * nq
    seq[idx] = 'Z'
    seq = "".join(seq)
    sz_jump = qe.StaticPauliOperatorGroup(n_qubits=nq, id="sz_jump_{}".format(idx), batchsize=batchsize, coef=3.3476669788360596, requires_grad=False)
    sz_jump.add_operator(seq)
    qubit.add_operator_group_to_jumping(sz_jump)

qubit.to(device=dev)
################################################
# Define the current operator
################################################
obs_current = qe.StaticPauliOperatorGroup(n_qubits=nq, id="current", batchsize=batchsize, coef=1.0)
for idx in range(nq):
    hop_seq_1 = ['I'] * nq 
    hop_seq_1[idx] = 'U'
    hop_seq_1[(idx+1)%nq] = 'D'
    hop_seq_1 = "".join(hop_seq_1)

    hop_seq_2 = ['I'] * nq 
    hop_seq_2[idx] = 'D'
    hop_seq_2[(idx+1)%nq] = 'U'
    hop_seq_2 = "".join(hop_seq_2)
    obs_current.add_operator(hop_seq_1, prefactor=-1.0)
    obs_current.add_operator(hop_seq_2, prefactor=1.0)
obs_current.to(device=dev)

################################################
# Define the thermal state
################################################
temperature = 300 
beta = 1 / (Constants.kb * temperature)
# print('computing thermal state for grand canonical ensemble...')
# thermal_dm_grand = th.matrix_exp(-beta * static_hamiltonian)
# print('computing thermal state for one-electron section ...')
# sec_index = []
# for i in range(nq):
#     config_1bd = th.ones(nq, dtype=int)
#     config_1bd[i] = 0
#     sec_index.append(int(bin2idx(config_1bd).numpy().sum()))
# thermal_dm = th.zeros_like(thermal_dm_grand)
# for i in sec_index:
#     for j in sec_index:
#         thermal_dm[i,j] = thermal_dm_grand[i,j]
# thermal_dm = thermal_dm / trace(thermal_dm)
# thermal_dm = thermal_dm.to(device=dev)
# thermal_dm = thermal_dm[None,:,:].repeat(batchsize,1,1)

########
print('computing thermal state for one-electron section ...')
sec_index = []
for i in range(nq):
    config_1bd = th.ones(nq, dtype=int)
    config_1bd[i] = 0
    sec_index.append(int(bin2idx(config_1bd).numpy().sum()))
dim_sec = len(sec_index)
ham_sec = th.zeros_like(static_hamiltonian[:,:dim_sec,:dim_sec])
for i, idx_i in enumerate(sec_index):
    for j, idx_j in enumerate(sec_index):
        ham_sec[:,i,j] = static_hamiltonian[:,idx_i,idx_j]

thermal_dm_sec = th.matrix_exp(-beta * ham_sec)
thermal_dm = th.zeros_like(static_hamiltonian)
for i, idx_i in enumerate(sec_index):
    for j, idx_j in enumerate(sec_index):
        thermal_dm[:,idx_i,idx_j] = thermal_dm_sec[:,i,j]
thermal_dm = thermal_dm / trace(thermal_dm)[:,None,None]
thermal_dm = thermal_dm.to(device=dev)

################################################
#  get current operator (not including the prefactor) and set it as the "density operator"
################################################
j0, coef = obs_current.sample(0)
j0 = j0.to(device=dev)
qubit.density_matrix.set_rho(j0)
j0 = j0[None,:,:].repeat(batchsize,1,1)

################################################
# Evolve the current operator
################################################
total_t = 50 * Constants.fs
dt = 0.025 * Constants.fs
nsteps = int(total_t / dt)
corr_t = []
corr_current = []
for i in range(nsteps):
    if i % int(Constants.fs/dt) == 0:
        jt = qubit.rho
        corr_jj = th.matmul(th.matmul(jt, j0), thermal_dm).mean(0)
        # print(corr_jj.diagonal())
        corr_jj = th.trace(corr_jj)
        corr_jj *= - (hopping_coef/Constants.Hartree)**2 * Constants.elementary_charge**2 # in unit of R^2
        print('t={:.1f}au, Re C(t)={:.5f}[1e-5], Im C(t)={:.5f}[1e-5]'.format(i*dt/Constants.time_au, corr_jj.real*1e5, corr_jj.imag*1e5))
        corr_t.append(i*dt/Constants.time_au)
        corr_current.append(corr_jj.real.cpu().numpy())
        print()
    qubit.step(dt=dt, set_buffer=False)

##
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(corr_t, corr_current, markersize=1, linewidth=2, linestyle='dashed', label='Simulated')
ax.set_xlabel('Time [a.u.]')
ax.set_ylabel('C(t)')
ax.set_ylim(-0.2e-5, 1.2e-5)
ax.legend()
plt.show()
plt.tight_layout()
plt.savefig('Corr_t_V{}.png'.format(hopping_value), dpi=200)

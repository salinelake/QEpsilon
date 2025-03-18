import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import qepsilon as qe
from qepsilon.utilities import Constants_Metal as Constants
from qepsilon.utilities import trace, bin2idx
from model import holstein_1D

th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)
dev = 'cuda'
################################################
#  define the system
################################################
batchsize = 4
nq = 8
hopping_value = 83 
hopping_coef = hopping_value * Constants.meV  # 83meV
model = holstein_1D(nsite=nq, batchsize=batchsize)
system = model.system
################################################
# define the system (Hamiltonian, noise, jump operators)
################################################
## add hopping terms
op_hop = model.get_hopping_operator_group(id="hop", coef=hopping_coef, requires_grad=False)
system.add_operator_group_to_hamiltonian(op_hop)

## add noise and jump operators
for idx in range(nq):
    seq = ['I'] * nq
    seq[idx] = 'Z'
    seq = "".join(seq)

    ## Langevin Sz noise
    sz1 = qe.LangevinNoisePauliOperatorGroup(n_qubits=nq, id="sz_noise_langevin_{}".format(idx), batchsize=batchsize, tau=0.20435911417007446, amp=11.90210247039795, requires_grad=False)
    sz1.add_operator(seq)
    system.add_operator_group_to_hamiltonian(sz1)
 
## move system to device
system.to(device=dev)

################################################
# thermalize the system
################################################
temperature = 300 
beta = 1 / (Constants.kb * temperature)

## initialization of a fully localized state
mat_hop, coef = op_hop.sample(0)
static_hamiltonian = mat_hop[None,:,:] * coef[:,None,None]
sec_index = []
for i in range(nq):
    config_1bd = th.ones(nq, dtype=int)
    config_1bd[i] = 0
    sec_index.append(int(bin2idx(config_1bd).numpy().sum()))
thermal_dm = th.zeros_like(static_hamiltonian)
for i, idx_i in enumerate(sec_index):
        thermal_dm[:,idx_i,idx_i] = 1 / nq
thermal_dm = thermal_dm.to(device=dev)

## equilibration
system.density_matrix.set_rho(thermal_dm)
total_t = 1000 * Constants.fs
dt = 0.025 * Constants.fs
nsteps = int(total_t / dt)

traj = []
for i in range(nsteps):
    if i % int(Constants.fs/dt) == 0:
        print('Equilibration: t={:.1f}au'.format(i*dt/Constants.time_au))
        system.normalize()
        rho = system.rho
        rho_sec = rho[:, sec_index, :][:, :, sec_index]
        traj.append(rho_sec[0,0])
        print(traj[-1] )
    system.step(dt=dt, set_buffer=False)    
traj = th.stack(traj, dim=0).cpu().numpy()
traj_time = np.arange(traj.shape[0]) * Constants.fs / Constants.time_au
## plot the trajectory
fig, ax = plt.subplots(1,2, figsize=(8, 3))
for i in range(nq):
    ax[0].plot(traj_time, traj[:,i].real, markersize=1, linewidth=2, linestyle='solid', label='{}'.format(i))
    ax[1].plot(traj_time, traj[:,i].imag, markersize=1, linewidth=2, linestyle='solid', label='{}'.format(i))
ax[0].set_xlabel('Time [a.u.]')
ax[1].set_xlabel('Time [a.u.]')
ax[0].set_ylabel('Re rho[0, i](t)')
ax[1].set_ylabel('Im rho[0, i](t)')
ax[0].legend(fontsize=8, loc='upper right')
ax[1].legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('traj_nq{}_V{}.png'.format(nq, hopping_value), dpi=200)

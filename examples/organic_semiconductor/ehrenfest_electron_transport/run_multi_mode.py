import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import logging
import qepsilon as qe
from holstein import *
from qepsilon.simulation.mixed_unitary_system import OscillatorQubitUnitarySystem
from qepsilon.utilities import Constants_Metal as Constants
from qepsilon.utilities import trace, qubitconf2idx, apply_to_pse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='log.txt')
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)
dev = 'cuda'

################################################
#  simulation parameters 
################################################
batchsize = 16
nq = 6
ns = 2**nq
hopping_value = 8.3 
hopping_coef = hopping_value * Constants.meV  # 83meV
nmodes = 9
temperature = 600.0
tau  = 0.1 * Constants.ps       ## very important.
mass = th.ones(nmodes) * 100 * Constants.amu
mass[0] *= 64
# mass[1] *= 4
## equilibration
total_t = 1000 * Constants.fs
dt = 0.1 * Constants.fs
nsteps =  int(total_t / dt)
## sampling.. current correlation is about 20fs.
sample_t = 1000 * Constants.fs
sample_steps = int(sample_t / dt)
sample_interval = int(Constants.fs / dt)
################################################
#  load DFT data 
################################################
data = np.loadtxt('data.csv', delimiter=',')
omega_in_cm = data[:nmodes,0]
lambda_in_cm = data[:nmodes,1] * 2   * 7
g_factor = np.sqrt(lambda_in_cm / omega_in_cm)
g_factor = th.tensor(g_factor, dtype=th.float32)
print('omega[cm^-1]:', omega_in_cm)
print('g_factor:', g_factor)
exit()
omega = Constants.speed_of_light * (omega_in_cm/ Constants.cm) * 2 * np.pi   # ps^-1
omega = th.tensor(omega, dtype=th.float32)
eq_position = - g_factor * np.sqrt(2) / (mass * omega)**0.5
x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5

## initial condition
initial_config = th.zeros(nq, dtype=int)
initial_config[0] = 1
print('#bosonic modes:', nmodes)
print('omega [ps^-1]:', omega)
print('g_factor:', g_factor)
print('reorg energy [meV]:', (g_factor**2 * omega)  / Constants.meV)

################################################
# define the simulation (Hamiltonian, classical oscillators, coupling)
################################################
simulation = OscillatorQubitUnitarySystem(n_qubits=nq, batchsize=batchsize, cls_dt=dt)
simulation.set_pse_by_config(initial_config)
# pse = simulation.pure_ensemble
# new_state = th.zeros_like(pse.get_pse())
# for i in range(nq):
#     qubit_conf = th.zeros(nq, dtype=int)
#     qubit_conf[i] = 1
#     sec_idx = qubitconf2idx(qubit_conf)
#     new_state[:, sec_idx] = 1
# pse.set_pse(new_state)

## add hopping terms
op_hop = create_hopping_operator_group(id="hop", nsite=nq, batchsize=batchsize, coef=hopping_coef, static=True, requires_grad=False)
simulation.add_operator_group_to_hamiltonian(op_hop)

# ## add white noise
# op_noise_list = create_local_noise_operator_group(nsite=nq, batchsize=batchsize, amp=10.0)
# for op in op_noise_list:
#     simulation.add_operator_group_to_hamiltonian(op)

## add classical harmonic oscillators to approximate bosonic environment
for i in range(nq):
    oscilators_id = 'osc_{}'.format(i)
    x0 = eq_position.clone().unsqueeze(-1) if i == 0 else eq_position.clone().unsqueeze(-1) * 0.0
    x0 += x0_std.unsqueeze(-1) * th.randn(nmodes, 1)
    simulation.add_classical_oscillators(id=oscilators_id, nmodes=nmodes, 
        freqs=omega, masses= mass, couplings=g_factor, x0=x0, init_temp=temperature, tau=tau, unit='pm_ps')
    simulation.bind_oscillators_to_qubit(qubit_idx=i, oscillators_id=oscilators_id)

################################################
# define the observables
################################################
## local number operators
local_number_ops = create_local_number_operator_group(nsite=nq, batchsize=batchsize)
## current operator
current_op = create_current_operator_group(id="current", nsite=nq, batchsize=1)

################################################
# move to GPU if dev='cuda'
################################################
if dev=='cuda':
    simulation.to(device=dev)
    for op in local_number_ops:
        op.to(device=dev)
    current_op.to(device=dev)

################################################
# thermalize the system
################################################
current_op_matrix, _ = current_op.sample()
t_traj = []
osc_traj = []
site_occupation_traj = []
corr_jj_0_traj = []
osc_temps_traj = []
for step in range(nsteps):
    if step % int(Constants.fs / dt) == 0:
        print('========Equilibration: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
        t_traj.append(step * dt/Constants.fs)
        ## print the occupation of each site
        site_occupation = [ simulation.observe(local_number_ops[i]).real for i in range(nq)]
        site_occupation_traj.append(th.stack(site_occupation, dim=1))  # (batchsize, nq)
        osc_positions = [ simulation.get_oscilator_by_id('osc_{}'.format(idx))['particles'].get_positions().squeeze(-1) for idx in range(nq)]
        osc_traj.append(th.stack(osc_positions, dim=1))  ## (batchsize, nq, nmodes)
        print('site_occupation_1st_batch:', site_occupation_traj[-1][0])

        ## print oscilators temperature for each site
        osc_temps = [ simulation.get_oscilator_by_id('osc_{}'.format(idx))['particles'].get_temperature().squeeze(-1) for idx in range(nq)]
        osc_temps = th.stack(osc_temps, dim=1).mean() 
        print('osc_temps_batchavg:', osc_temps)
        osc_temps_traj.append(osc_temps)

        ## compute the expectation value of current operator squared
        simulation.normalize()
        pse = simulation.pse
        J_pse = apply_to_pse(pse, current_op_matrix)
        corr_jj_0 = - th.sum(th.abs(J_pse) ** 2, dim=1).mean()
        corr_jj_0_traj.append(corr_jj_0)
        print('corr_jj_0:', corr_jj_0)
    simulation.step(dt, temp=temperature, profile=False)

################################################
# Post-processing
################################################
## process the trajectories
osc_traj = th.stack(osc_traj, dim=0).cpu().numpy()  # (nsteps, batchsize, nq, nmodes)
assert osc_traj.shape[1:] == (batchsize, nq, nmodes)
site_occupation_traj = th.stack(site_occupation_traj, dim=0).cpu().numpy()  # (nsteps, batchsize, nq)
assert site_occupation_traj.shape[1:] == (batchsize, nq)
corr_jj_0_traj = th.stack(corr_jj_0_traj, dim=0).cpu().numpy()  # (nsteps)
corr_jj_0_accumulated = np.cumsum(corr_jj_0_traj) / np.arange(1, corr_jj_0_traj.shape[0]+1)
osc_temps_traj = th.stack(osc_temps_traj, dim=0).cpu().numpy()  # (nsteps, )
osc_temps_accumulated = np.cumsum(osc_temps_traj) / np.arange(1, osc_temps_traj.shape[0]+1)
print('osc_temps_accumulated:', osc_temps_accumulated)
## plot site_occupation_traj
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
for i in range(nq):
    ax[0].plot(t_traj, site_occupation_traj[:,0,i], label='site {}'.format(i))
    ax[1].plot(t_traj, osc_traj[:,0,i,0], label=r'site {}'.format(i))
    # ax[2].plot(t_traj, osc_traj[:,0,i,8], label=r'site {}'.format(i))

# ax[3].plot(t_traj, corr_jj_0_traj )
ax[2].plot(t_traj, osc_temps_accumulated, )
ax[3].plot(t_traj, corr_jj_0_accumulated, '--')
ax[1].axhline(eq_position[0], color='k', linestyle='--', label='polaron eq. pos. ')
# ax[2].axhline(eq_position[8], color='k', linestyle='--', label='polaron eq. pos. ')
ax[1].set_title(r'$\omega_1={}cm^{{-1}}$'.format(omega_in_cm[0]))
# ax[2].set_title(r'$\omega_9={}cm^{{-1}}$'.format(omega_in_cm[8]))
ax[2].set_title(r'oscillator temperature')
ax[3].set_title(r'$C_{JJ}(0)$')
ax[0].legend( frameon=False)
ax[1].legend( frameon=False)
# ax[2].legend( frameon=False)
# ax[3].legend( frameon=False)
ax[0].set_xlabel('time [fs]')
ax[1].set_xlabel('time [fs]')
ax[2].set_xlabel('time [fs]')
ax[3].set_xlabel('time [fs]')
ax[0].set_ylabel('site occupation')
ax[1].set_ylabel('oscillator position [pm]')
# ax[2].set_ylabel('oscillator position [pm]')
ax[2].set_ylabel('oscillator temperature [K]')
ax[3].set_ylabel(r'$\langle \psi(t) | J\cdot J | \psi(t) \rangle$')
plt.tight_layout()
plt.savefig('thermalize_nq{}_T{}_V{}_tau{}_cls.png'.format(nq, temperature, hopping_value, tau/Constants.ps), dpi=200)
plt.close()



################################################
# Sample current correlation
################################################

t_traj = []
psi_traj = []    
evo_op_traj = []
for step in range(sample_steps):
    if step % sample_interval == 0:
        print('========Sampling: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
        t_traj.append(step * dt/Constants.fs)
        if step > 0:
            evo_op_traj.append(simulation.evo)
        simulation.reset_evo()
        simulation.normalize()
        psi_traj.append(simulation.pse)
    simulation.step(dt, temp=temperature, set_buffer=True, profile=False)

nsample = len(psi_traj)
psi_traj = th.stack(psi_traj, dim=0)   # [nsample, batchsize, nstates], the pure state ensemble on t=0, Dt, 2Dt, ...
evo_op_traj = th.stack(evo_op_traj, dim=0)   # [nsample-1, batchsize, nstates, nstates], the evolution operator for the duration [0,Dt], [Dt,2Dt], ...
corr_jj_list = []
corr_T = 48 * Constants.fs  ## the total duration of time we want to calculate the correlation for.
Dt = dt * sample_interval   ## the time resolution of the correlation function
corr_n = int(corr_T / Dt) + 1 ## the number of time slices.
###  get J\psi
J_pse = apply_to_pse(psi_traj.reshape(len(t_traj) * batchsize, ns), current_op_matrix).reshape(len(t_traj), batchsize, ns)   # (nsample, batchsize, ns)
for i in range(corr_n):
    print('calculating the current correlation at t={}fs'.format(i*Dt/Constants.fs))
    if i == 0:
        corr = th.sum(th.abs(J_pse) ** 2, dim=-1).mean()
        corr_jj_list.append(-corr)
    else:
        ## get U(i*Dt): [nsample-i]
        if i == 1:
            evo_accumulated = evo_op_traj.clone()
        else:
            evo_accumulated = th.matmul(evo_op_traj[i-1:], evo_accumulated[:-1] )
        assert evo_accumulated.shape == (nsample-i, batchsize, ns, ns)
        ## get UJ_psi
        UpJUJ_pse = th.matmul(evo_accumulated, J_pse[:-i].unsqueeze(-1))
        assert UpJUJ_pse.shape == (nsample-i, batchsize, ns, 1)
        ## get JUJ_psi, J: (ns, ns)
        UpJUJ_pse = th.matmul(current_op_matrix[None,None,:,:], UpJUJ_pse)  # (1,1,ns,ns) @ (nsample-i, batchsize, ns, 1) -> (nsample-i, batchsize, ns, 1)
        assert UpJUJ_pse.shape == (nsample-i, batchsize, ns, 1)
        ## get U^T J U J \psi
        UpJUJ_pse = th.matmul(evo_accumulated.conj().transpose(2,3), UpJUJ_pse)  # (nsample-i, batchsize, ns, ns) @ (nsample-i, batchsize, ns, 1) -> (nsample-i, batchsize, ns, 1)
        UpJUJ_pse = UpJUJ_pse.squeeze(-1)
        assert UpJUJ_pse.shape == (nsample-i, batchsize, ns)
        corr =  (psi_traj[:-i].conj() * UpJUJ_pse).sum(-1)
        assert corr.shape == (nsample-i, batchsize)
        corr_jj_list.append(corr.mean())
corr_jj_list = th.stack(corr_jj_list, dim=0).cpu().numpy()  # (corr_n)
corr_t = np.arange(corr_n) * Dt 
## get reference correlation
occu_num = 1/(th.exp(omega / Constants.kb / temperature) - 1)  ## thermal average occupation number of the bosonic mode
exponent = [g_factor**2 * (2 * occu_num + 1 - occu_num * (th.exp(-1j * omega * t)) - (occu_num + 1) * (th.exp(1j * omega * t))) for t in corr_t]
exponent = th.stack(exponent, dim=0) # (corr_n, nmodes)
corr_jj_ref = - th.exp( - exponent.sum(dim=-1) ) * 2

## plot corr_jj_list versus time
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(corr_t / Constants.fs / 0.02418, -corr_jj_list.real, label='real')
ax.plot(corr_t / Constants.fs / 0.02418, -corr_jj_list.imag, label='imag')
ax.plot(corr_t / Constants.fs / 0.02418, -corr_jj_ref.real, linestyle='--', linewidth=2, alpha=0.5, label='real, ref')
ax.plot(corr_t / Constants.fs / 0.02418, -corr_jj_ref.imag, linestyle='--', linewidth=2, alpha=0.5, label='imag, ref')
ax.set_xlabel('time [a.u.]')
ax.set_ylabel(r'$-\langle J(t) J(0)  \rangle$')
ax.legend()
plt.tight_layout()
plt.savefig('current_correlation_nq{}_T{}_V{}_tau{}_cls.png'.format(nq, temperature, hopping_value, tau/Constants.ps), dpi=200)
plt.close()





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
dev = 'cpu'

################################################
#  simulation parameters 
################################################
batchsize = 1
nq = 6
hopping_value = 83 
hopping_coef = hopping_value * Constants.meV  # 83meV
nmodes = 9
temperature = 300.0
tau  = 0.1 * Constants.ps       ## very important.
mass = th.ones(nmodes) * 100 * Constants.amu
mass[0] *= 64
mass[1] *= 4
## equilibration
total_t = 10000 * Constants.fs
dt = 0.1 * Constants.fs
nsteps =  int(total_t / dt)

################################################
#  load DFT data 
################################################
data = np.loadtxt('data.csv', delimiter=',')
omega_in_cm = data[:nmodes,0]
lambda_in_cm = data[:nmodes,1]
g_factor = np.sqrt(lambda_in_cm / omega_in_cm)
g_factor = th.tensor(g_factor, dtype=th.float32)
omega = Constants.speed_of_light * (omega_in_cm/ Constants.cm) * 2 * np.pi   # ps^-1
omega = th.tensor(omega, dtype=th.float32)
eq_position = - g_factor * np.sqrt(2) / (mass * omega)**0.5
x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5
print('eq_position:', eq_position)
## initial condition
initial_config = th.zeros(nq, dtype=int)
initial_config[0] = 1
print('#bosonic modes:', nmodes)
print('omega [ps^-1]:', omega)
print('g_factor:', g_factor)

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
    x0 = eq_position.unsqueeze(-1) if i == 0 else eq_position.unsqueeze(-1) * 0.0
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
        print('========step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
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
plt.savefig('thermalize_nq{}_T{}_V{}_tau{}.png'.format(nq, temperature, hopping_value, tau/Constants.ps), dpi=200)
plt.close()

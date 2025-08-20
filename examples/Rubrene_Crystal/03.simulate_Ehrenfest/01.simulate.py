import logging
import time
import numpy as np
import torch as th
import os
from matplotlib import pyplot as plt
import logging
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorTightBindingUnitarySystem
from qepsilon.utilities import Constants_Metal as Constants
import argparse

dev = 'cuda'
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)

##
parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=300.0)
parser.add_argument('--nsites', type=int, default=100)
parser.add_argument('--sample_time', type=float, default=200, help='sampling time in fs')
parser.add_argument('--timestep_in_fs', type=float, default=0.02, help='time step in fs')
args = parser.parse_args()

"""
Preprocessing
"""
## make output folder
out_folder = 'T{:.0f}_ns{}'.format(args.temperature, args.nsites)
os.makedirs(out_folder, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(out_folder, 'log.txt'), filemode='w')

## load coupling parameters
data = np.loadtxt('../coupling_const.csv', delimiter=',')
nmodes = data.shape[0]

##  process system parameters 
ns = args.nsites
hopping_in_meV = 83
hopping_coef = hopping_in_meV * Constants.meV
mass = th.tensor([250 * Constants.amu] * nmodes)
temperature = args.temperature
omega_in_cm = th.tensor(data[:,0], dtype=th.float32)
lambda_in_cm = th.tensor(data[:,1], dtype=th.float32)
g_factor = th.sqrt(lambda_in_cm / omega_in_cm)
omega = omega_in_cm * Constants.cm_inverse_energy
beta = g_factor * omega * th.sqrt(2 * mass * omega)
binding_energy = g_factor **2 * omega

## simulation parameters
batchsize = 128
polaron_eq_position = - g_factor * np.sqrt(2) / (mass * omega)**0.5
x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5
total_t = args.sample_time * Constants.fs
dt = args.timestep_in_fs * Constants.fs
nsteps =  int(total_t / dt)

## report the parameters
logging.info(f'{nmodes} bosonic modes with frequency {omega_in_cm} cm^-1')
logging.info(f'kbT/hw: {(Constants.kb * temperature)/omega}, < 1 means large quantum effect')
logging.info(f'g factor is {g_factor}, binding energy: {binding_energy.sum() / Constants.meV:.5f} meV, hopping energy: {hopping_coef / Constants.meV:.5f} meV')
logging.info(f'mass: {mass}')
logging.info(f'polaron trap depth of classical mode is {polaron_eq_position} pm, std of initial position is {x0_std} pm')

"""
Define the system
"""
simulation = OscillatorTightBindingUnitarySystem(n_sites=ns, batchsize=batchsize, cls_dt=dt)
init_site = ns//2
initial_config = th.zeros(ns, dtype=int)
initial_config[init_site] = 1
simulation.set_pse_by_config(initial_config)

## add hopping terms
op_hop = qe.StaticTightBindingOperatorGroup(n_sites=ns, id="hop", batchsize=batchsize, coef=hopping_coef, static=True, requires_grad=False)
for idx in range(ns):
    hop_seq_1 = ['X'] * ns 
    hop_seq_1[idx] = 'L'
    hop_seq_1 = "".join(hop_seq_1)
    op_hop.add_operator(hop_seq_1)

    hop_seq_2 = ['X'] * ns 
    hop_seq_2[idx] = 'R'
    hop_seq_2 = "".join(hop_seq_2)
    op_hop.add_operator(hop_seq_2)

simulation.add_operator_group_to_hamiltonian(op_hop)

## add classical harmonic oscillators to approximate bosonic environment
for i in range(ns):
    oscilators_id = 'osc_{}'.format(i)
    x0 =  polaron_eq_position.clone() if i == init_site else  th.zeros_like(polaron_eq_position)
    x0 = x0.reshape(1,nmodes,1).repeat(batchsize,1,1)
    x0 += x0_std[None,:,None] * th.randn(batchsize, nmodes, 1)
    simulation.add_classical_oscillators(id=oscilators_id, nmodes=nmodes, 
        freqs=omega, masses= mass, couplings=g_factor, x0=x0, init_temp=temperature, unit='pm_ps')
    simulation.bind_oscillators_to_tb(site_idx=i, oscillators_id=oscilators_id)
    
if dev=='cuda':
    simulation.to(device=dev)

"""
Simulate ehrenfest dynamics with feedback 
"""
t_traj = []
osc_traj = []
site_occupation_traj = []
osc_temps_traj = []
MSD_traj = []
for step in range(nsteps):
    if step % int(Constants.fs / dt) == 0:
        logging.info('========Equilibration: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
        t_traj.append(step * dt/Constants.fs)
        ## record the trajectory of the first oscillator
        osc_positions = simulation.get_oscilator_by_id('osc_{}'.format(init_site))['particles'].get_positions().squeeze(-1)
        osc_traj.append(osc_positions)  ## (batchsize, nmodes)

        ## record oscilators average temperature 
        osc_temps = [ simulation.get_oscilator_by_id('osc_{}'.format(idx))['particles'].get_temperature().squeeze(-1) for idx in range(ns)]
        osc_temps = th.stack(osc_temps, dim=1).mean() 
        logging.info('osc_temps_batchavg: {}K'.format(osc_temps))
        osc_temps_traj.append(osc_temps)

        ## record the occupation of each site
        pure_ensemble = simulation.pure_ensemble
        simulation.normalize()
        site_occupation = pure_ensemble.observe_occupation(simulation.pse)  # (batchsize, ns)
        site_occupation_traj.append(site_occupation)
        # print('site_occupation_1st_batch:', site_occupation_traj[-1][0])
        ## record diffusion-related observables
        expect_r_squared = pure_ensemble.observe_r2(simulation.pse).real
        expect_r = pure_ensemble.observe_r(simulation.pse).real
        MSD = expect_r_squared - expect_r**2
        MSD_traj.append(MSD)
        # print('<r^2>={}, <r>={}, MSD={}'.format(expect_r_squared, expect_r, MSD))
        logging.info('Average MSD={}'.format(MSD.mean()))

    simulation.step(dt, temp=None, profile=False)


"""
Postprocess
"""
## process the oscillator trajectories
nsample = len(t_traj)
osc_traj = th.stack(osc_traj, dim=0).cpu().numpy()  # (nsample, batchsize, nmodes)
assert osc_traj.shape[1:] == (batchsize, nmodes)
## process the oscilator temperature trajectory
osc_temps_traj = th.stack(osc_temps_traj, dim=0).cpu().numpy()  # (nsample, )
osc_temps_accumulated = np.cumsum(osc_temps_traj) / np.arange(1, osc_temps_traj.shape[0]+1)
print('osc_temps_accumulated:', osc_temps_accumulated)
## process the site occupation trajectories
site_occupation_traj = th.stack(site_occupation_traj, dim=0).cpu().numpy()  # (nsample, batchsize, ns)
assert site_occupation_traj.shape[1:] == (batchsize, ns)
## process the MSD trajectory
MSD_traj = th.stack(MSD_traj, dim=0).cpu().numpy()  # (nsample, batchsize)
assert MSD_traj.shape == (nsample, batchsize)
np.save('{}/site_occupation_traj.npy'.format(out_folder), site_occupation_traj)
np.save('{}/MSD_traj.npy'.format(out_folder), MSD_traj)

## plot site_occupation_traj
fig, ax = plt.subplots(1, 4, figsize=(18, 3))
## plot the site occupation as heatmap
ax[0].imshow(-np.log(site_occupation_traj.mean(1)+1e-10), aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, t_traj[-1]], vmin=0, vmax=4)

ax[0].set_title('site occupation')
ax[0].set_xlabel('site')
ax[0].set_ylabel('time [fs]')
## plot others
ax[1].plot(t_traj, osc_traj[:,0,0], label=r'oscillator 1 on site 0 on batch 0')
ax[1].axhline(polaron_eq_position[0], color='k', linestyle='--', label='polaron eq. pos. ')
ax[2].plot(t_traj, osc_temps_accumulated, )
ax[3].plot(t_traj, MSD_traj.mean(-1), '--')
ax[1].set_title(r'$\omega_1={}cm^{{-1}}$'.format(omega_in_cm[0]))
ax[2].set_title(r'oscillator temperature')
ax[3].set_title(r'$Mean Square Displacement$')
ax[1].legend( frameon=False)
ax[1].set_xlabel('time [fs]')
ax[2].set_xlabel('time [fs]')
ax[3].set_xlabel('time [fs]')
ax[1].set_ylabel('oscillator position [pm]')
ax[2].set_ylabel('oscillator temperature [K]')
ax[3].set_ylabel(r'$MSD(t)$')
plt.tight_layout()
plt.savefig(os.path.join(out_folder, 'results.png'), dpi=200)
plt.close()

 

import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import logging
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorTightBindingUnitarySystem
from qepsilon.utilities import Constants_Metal as Constants
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='log.txt', filemode='w')
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)
dev = 'cpu'
 
################################################
#  Parameters 
################################################
## hamiltonian parameters 
hopping_value = - 100   # cm^-1
hopping_coef = hopping_value * Constants.cm_inverse_energy
mass = th.tensor([250 * Constants.amu])
nmodes = 1
temperature = 300.0
# spring_constant = 14500 * Constants.amu / Constants.ps**2 # amu/ps^2
# omega = th.tensor([th.sqrt(spring_constant / mass)])
omega_in_cm = th.tensor([1000])    # cm^-1;   1000cm^-1 --> ~124meV
omega = omega_in_cm * Constants.cm_inverse_energy
# beta = 3500 # cm^-1/A   beta=g * omega * sqrt(2 * mass * omega); default unit should be ps^-1 * sqrt(amu / ps) = 1/pm/ps 
# beta = beta * Constants.cm_inverse_energy / Constants.Angstrom
# g_factor = beta / omega / th.sqrt(2 * mass * omega)
g_factor = th.tensor([0.37])
beta = g_factor * omega * th.sqrt(2 * mass * omega)
binding_energy = g_factor **2 * omega
## simulation parameters
batchsize = 128
ns = 9
tau  = 1.0 * Constants.ps
polaron_eq_position = - g_factor * np.sqrt(2) / (mass * omega)**0.5
x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5

## report the parameters
logging.info(f'{nmodes} bosonic modes with frequency {omega_in_cm} cm^-1')
logging.info(f'kbT/hw: {(Constants.kb * temperature)/omega}, < 1 means large quantum effect')
logging.info(f'g factor is {g_factor}, binding energy: {binding_energy.sum() / Constants.meV:.5f} meV, hopping energy: {hopping_coef / Constants.meV:.5f} meV')
logging.info(f'relative coupling strength= lambda/4/V : {binding_energy / 4 / hopping_coef}, >>1 means strong coupling')
logging.info(f'mass: {mass}')
logging.info(f'polaron trap depth of classical mode is {polaron_eq_position} pm, std of initial position is {x0_std} pm')

## equilibration parameters
total_t = 100 * Constants.fs
dt = 0.025 * Constants.fs
nsteps =  int(total_t / dt)
logging.info('Vdt={}'.format(hopping_coef * dt))

## electron initial condition
init_site = ns//2
initial_config = th.zeros(ns, dtype=int)
initial_config[init_site] = 1

################################################
# define the simulation (Hamiltonian, classical oscillators, coupling)
################################################
simulation = OscillatorTightBindingUnitarySystem(n_sites=ns, batchsize=batchsize, cls_dt=dt)
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
    x0 += x0_std * th.randn(batchsize, nmodes, 1)
    simulation.add_classical_oscillators(id=oscilators_id, nmodes=nmodes, 
        freqs=omega, masses= mass, couplings=g_factor, x0=x0, init_temp=temperature, tau=tau, unit='pm_ps')
    simulation.bind_oscillators_to_tb(site_idx=i, oscillators_id=oscilators_id)
################################################
# move to GPU if dev='cuda'
################################################
if dev=='cuda':
    simulation.to(device=dev)

################################################
# thermalize the system
################################################
t_traj = []
osc_traj = []
site_occupation_traj = []
osc_temps_traj = []
MSD_traj = []
for step in range(nsteps):
    if step % int(Constants.fs / dt) == 0:
        print('========Equilibration: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
        t_traj.append(step * dt/Constants.fs)
        ## record the trajectory of the first oscillator
        osc_positions = simulation.get_oscilator_by_id('osc_{}'.format(init_site))['particles'].get_positions().squeeze(-1)
        osc_traj.append(osc_positions)  ## (batchsize, nmodes)

        ## record oscilators average temperature 
        osc_temps = [ simulation.get_oscilator_by_id('osc_{}'.format(idx))['particles'].get_temperature().squeeze(-1) for idx in range(ns)]
        osc_temps = th.stack(osc_temps, dim=1).mean() 
        print('osc_temps_batchavg:', osc_temps)
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
        print('<r^2>={}, <r>={}, MSD={}'.format(expect_r_squared, expect_r, MSD))

    # simulation.step(dt, temp=temperature, profile=False)
    simulation.step(dt, temp=None, profile=False)


################################################
# Post-processing
################################################
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

## plot site_occupation_traj
fig, ax = plt.subplots(1, 4, figsize=(18, 3))
## plot the site occupation as heatmap
# ax[0].imshow(site_occupation_traj[:,0,:], aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, t_traj[-1]])
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
plt.savefig('ns{}_T{:.0f}_V{}_g{:.2f}_tau{}_adiabatic.png'.format(ns, temperature, hopping_value, g_factor[0], tau/Constants.ps), dpi=200)
plt.close()

 

import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import logging
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorQubitUnitarySystem
from qepsilon.operator_group.spin_operators import StaticPauliOperatorGroup
from qepsilon.utilities import Constants_Metal as Constants
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='log.txt', filemode='w')
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)
dev = 'cpu'
 
## parse a number
parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=300.0)
parser.add_argument('--g_factor', type=float, default=0.5)
args = parser.parse_args()

## load the data
prefix = 'ns{}_nmax{}_T{:.0f}_g{:.1f}_dt{:.3f}'.format(1, 4, 300, 0.5, 0.005)
quantum_sigma_n_list = np.load('data/{}_sigma_n.npy'.format(prefix)).flatten()
quantum_sigma_x_list = np.load('data/{}_sigma_x.npy'.format(prefix)).flatten()
quantum_sigma_y_list = np.load('data/{}_sigma_y.npy'.format(prefix)).flatten()
quantum_t_list = np.arange(quantum_sigma_n_list.shape[0])
sigma_x_ref = th.tensor(quantum_sigma_x_list)

################################################
#  Parameters 
################################################
## hamiltonian parameters 
mass = th.tensor([250 * Constants.amu])
nmodes = 1
temperature = args.temperature
# spring_constant = 14500 * Constants.amu / Constants.ps**2 # amu/ps^2
omega_in_cm = th.tensor([1000])    # cm^-1;   1000cm^-1 --> ~124meV
omega = omega_in_cm * Constants.cm_inverse_energy
g_factor = th.tensor([args.g_factor])
binding_energy = g_factor **2 * omega
## simulation parameters
batchsize = 1024
ns = 1
tau  = 1.0 * Constants.ps
polaron_eq_position = - g_factor * np.sqrt(2) / (mass * omega)**0.5
x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5

## report the parameters
logging.info(f'{nmodes} bosonic modes with frequency {omega_in_cm} cm^-1')
logging.info(f'kbT/hw: {(Constants.kb * temperature)/omega}, < 1 means large quantum effect')
logging.info(f'g factor is {g_factor}, binding energy: {binding_energy.sum() / Constants.meV:.5f} meV')
logging.info(f'mass: {mass}')
logging.info(f'polaron trap depth of classical mode is {polaron_eq_position} pm, std of initial position is {x0_std} pm')

## equilibration parameters
total_t = 200 * Constants.fs
dt = 0.05 * Constants.fs
nsteps =  int(total_t / dt)

################################################
# define the simulation (Hamiltonian, classical oscillators, coupling)
################################################
simulation = OscillatorQubitUnitarySystem(n_qubits=ns, batchsize=batchsize, cls_dt=dt)
# epsilon = 0.1
# simulation.pse = th.tensor([np.sqrt(1-epsilon), np.sqrt(epsilon)], dtype=th.cfloat, device=dev)

## add classical harmonic oscillators to approximate bosonic environment
oscilators_id = 'osc_{}'.format(0)
x0 =  polaron_eq_position.clone()
x0 = x0.reshape(1,nmodes,1).repeat(batchsize,1,1)
x0 += x0_std * th.randn(batchsize, nmodes, 1)
simulation.add_classical_oscillators(id=oscilators_id, nmodes=nmodes, 
    freqs=omega, masses= mass, couplings=g_factor, x0=x0, init_temp=temperature, tau=tau, unit='pm_ps')
simulation.bind_oscillators_to_qubit(qubit_idx=0, oscillators_id=oscilators_id, requires_grad=True)
opg_epc = simulation.get_hamiltonian_operator_group_by_ID(f"site-0_{oscilators_id}_epc")
################################################
# define the operators we want to observe
################################################
oscilator = simulation.get_oscilator_by_id(oscilators_id)
opg_sp_number = StaticPauliOperatorGroup(n_qubits=1, id="sp_number", batchsize=batchsize, coef=1.0)
opg_sp_number.add_operator('N')
## spin sigma_x operator
opg_sp_x = StaticPauliOperatorGroup(n_qubits=1, id="spin_x", batchsize=batchsize, coef=1.0)
opg_sp_x.add_operator('X')
## spin sigma_y operator
opg_sp_y = StaticPauliOperatorGroup(n_qubits=1, id="spin_y", batchsize=batchsize, coef=1.0)
opg_sp_y.add_operator('Y')

################################################
#  train
################################################
nepoch = 200
optimizer = th.optim.Adam([{'params': opg_epc.parameters(), 'lr': 1.0},
                          ])
# ## scheduler
# scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
for epoch in range(nepoch):
    ## reset the system
    epsilon = 0.1
    simulation.pse = th.tensor([np.sqrt(1-epsilon), np.sqrt(epsilon)], dtype=th.cfloat, device=dev)
    oscilator['particles'].reset(temp=temperature)
    x0 =  polaron_eq_position.clone()
    x0 = x0.reshape(1,nmodes,1).repeat(batchsize,1,1)
    x0 += x0_std * th.randn(batchsize, nmodes, 1)
    oscilator['particles'].set_positions(x0)
    effective_g = opg_epc.coef.detach().cpu().numpy().sum() / (omega[0] * th.sqrt(2 * mass[0] * omega[0]))
    reorg_energy = 2 * effective_g**2 * omega[0]
    period = int(2 * np.pi / reorg_energy * 1000)
    ## initialize trajectory holders
    t_list = []
    sigma_x_list = []
    sigma_y_list = []
    sigma_n_list = []
    ## simulate 
    for step in range(nsteps):
        if step % int(Constants.fs / dt) == 0:
            # print('========Equilibration: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
            t_list.append(step * dt/Constants.fs)
            ## record the occupation of each site
            pure_ensemble = simulation.pure_ensemble
            simulation.normalize()
            obs_n = simulation.observe(opg_sp_number).real # (batchsize,)
            obs_x = simulation.observe(opg_sp_x).real # (batchsize,)
            obs_y = simulation.observe(opg_sp_y).real # (batchsize,)
            sigma_n_list.append(obs_n.mean())
            sigma_x_list.append(obs_x.mean())
            sigma_y_list.append(obs_y.mean())
            # print('mean obs_n:{}, mean obs_x:{}, mean obs_y:{}'.format(obs_n.mean(), obs_x.mean(), obs_y.mean()))

        simulation.step(dt, temp=None, profile=False)
    sigma_x_list = th.stack(sigma_x_list)
    sigma_y_list = th.stack(sigma_y_list)
    sigma_n_list = th.stack(sigma_n_list)
    
    diff = (sigma_x_list - sigma_x_ref)[:period]
    loss = th.mean(th.abs(diff)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logging.info(f"Epoch={epoch}, loss={loss.item()}, Effective g={effective_g} -> target: 0.38, period: {period}[fs]")
    logging.info('sigma_x_pred:{}'.format(sigma_x_list[::10]))
    logging.info('sigma_x_ref :{}'.format(sigma_x_ref[::10]))
    ################################################
    # Post-processing
    ################################################
    sigma_n_list = sigma_n_list.detach().cpu().numpy()
    sigma_x_list = sigma_x_list.detach().cpu().numpy()  
    sigma_y_list = sigma_y_list.detach().cpu().numpy() 
    prefix = 'ns{}_cls_T{:.0f}_g{:.2f}_dt{:.3f}'.format(ns, temperature, g_factor.numpy().sum(), dt/Constants.fs)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].plot(t_list, sigma_n_list, label='Ehrenfest')
    ax[0].plot(quantum_t_list, quantum_sigma_n_list, label='Quantum')
    ax[0].set_title(r'$\langle \hat{\sigma}_n \rangle$')
    ax[0].legend()
    ax[1].plot(t_list, sigma_x_list / sigma_x_list[0], label='Ehrenfest')
    ax[1].plot(quantum_t_list, quantum_sigma_x_list / quantum_sigma_x_list[0], label='Quantum')
    ax[1].set_title(r'$Scaled \langle \hat{\sigma}_x \rangle$' + 'iter={}'.format(epoch))
    ax[1].legend()
    ax[2].plot(t_list, sigma_y_list / sigma_x_list[0], label='Ehrenfest')
    ax[2].plot(quantum_t_list, quantum_sigma_y_list / quantum_sigma_x_list[0], label='Quantum')
    ax[2].set_title(r'$Scaled \langle \hat{\sigma}_y \rangle$')
    ax[2].legend()
    for a in ax:
        a.set_xlabel(r'$t$ [fs]')
    plt.tight_layout()
    plt.savefig('train/{}_obs.png'.format(prefix))
    plt.close()



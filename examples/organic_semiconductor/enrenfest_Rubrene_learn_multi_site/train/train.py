import logging
import time
import os
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import logging
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorTightBindingUnitarySystem
from qepsilon.operator_group.tb_operators import StaticTightBindingOperatorGroup
from qepsilon.utilities import Constants_Metal as Constants
import argparse

## parse a number
nmax_list = np.array([8, 3, 1, 1, 1, 1, 1, 1, 1])
parser = argparse.ArgumentParser()
parser.add_argument('--ns', type=int, default=3)
parser.add_argument('--mode_index', type=int, default=5)
parser.add_argument('--temperature', type=float, default=300.0)
parser.add_argument('--hopping_in_meV', type=float, default=8.3)
parser.add_argument('--match_fs', type=int, default=250)
parser.add_argument('--sim_fs', type=float, default=400.0)


args = parser.parse_args()
nmax = nmax_list[args.mode_index]

## load the mode frequency and coupling strength
data = np.loadtxt('../data.csv', delimiter=',')
omega_in_cm = data[args.mode_index,0]
lambda_in_cm = data[args.mode_index,1]
omega = omega_in_cm * Constants.cm_inverse_energy
g_factor =  np.sqrt(lambda_in_cm / omega_in_cm)

## make output folder and log file
out_folder = "T{:.0f}_V{:.1f}meV/mode{:d}".format(args.temperature, args.hopping_in_meV, args.mode_index)
os.makedirs(out_folder, exist_ok=True)
os.makedirs(os.path.join(out_folder, 'figures'), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='{}/train.log'.format(out_folder), filemode='w')
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)
dev = 'cuda'

## load the trajectory data
prefix = "ns{:d}_mode{:d}_nmax{:d}_T{:.0f}_V{:.1f}meV_dt0.005".format(args.ns, args.mode_index, nmax, args.temperature, args.hopping_in_meV)
quantum_occupation_list = np.load('../quantum/data/{}_occupation.npy'.format(prefix))
occ_ref = th.tensor(quantum_occupation_list, device=dev)
################################################
#  Parameters 
################################################
## system parameters 
ns = args.ns
init_site = ns//2
nmodes = 1
temperature = args.temperature
hopping_in_meV = args.hopping_in_meV
hopping_coef = hopping_in_meV * Constants.meV
mass = th.tensor([250 * Constants.amu])
omega_in_cm = th.tensor([omega_in_cm], dtype=th.float32)    # cm^-1; 
omega = omega_in_cm * Constants.cm_inverse_energy
g_factor = th.tensor([g_factor], dtype=th.float32)
binding_energy = g_factor**2 * omega
## simulation parameters
batchsize = 1024
## equilibration parameters
total_sim_time = args.sim_fs * Constants.fs
dt = 0.02 * Constants.fs
nsteps =  int(total_sim_time / dt)
## training parameters
match_fs = args.match_fs # int(2 * np.pi / binding_energy * 1000 * 0.75)

## report the parameters
logging.info(f'{nmodes} bosonic modes with frequency {omega_in_cm} cm^-1')
logging.info(f'kbT/hw: {(Constants.kb * temperature)/omega}, < 1 means large quantum effect')
logging.info(f'g factor is {g_factor}, binding energy: {binding_energy.sum() / Constants.meV:.5f} meV')
logging.info(f'mass: {mass}')
logging.info(f'Train on observation time from 0 to {match_fs} fs')
# logging.info(f'polaron trap depth of classical mode is {polaron_eq_position} pm, std of initial position is {x0_std} pm')


################################################
# define the simulation (Hamiltonian, classical oscillators, coupling)
################################################
simulation = OscillatorTightBindingUnitarySystem(n_sites=ns, batchsize=batchsize, cls_dt=dt)
## add hopping terms
opg_tb_hop = StaticTightBindingOperatorGroup(n_sites=ns, id="hop", batchsize=batchsize, coef=hopping_coef, static=True, requires_grad=False)
for idx in range(ns):
    hop_seq_1 = ['X'] * ns 
    hop_seq_1[idx] = 'L'
    hop_seq_1 = "".join(hop_seq_1)
    opg_tb_hop.add_operator(hop_seq_1)

    hop_seq_2 = ['X'] * ns 
    hop_seq_2[idx] = 'R'
    hop_seq_2 = "".join(hop_seq_2)
    opg_tb_hop.add_operator(hop_seq_2)
simulation.add_operator_group_to_hamiltonian(opg_tb_hop)

## add classical harmonic oscillators to approximate bosonic environment
opg_epc_list = []
oscilator_list = []
for i in range(ns):
    oscilators_id = 'osc_{}'.format(i)
    oscilator = simulation.add_classical_oscillators(id=oscilators_id, nmodes=nmodes, freqs=omega, masses= mass, couplings=g_factor, init_temp=temperature,  unit='pm_ps')
    epc_op = simulation.bind_oscillators_to_tb(site_idx=i, oscillators_id=oscilators_id, requires_grad=True)
    opg_epc_list.append(epc_op)
    oscilator_list.append(oscilator)
## move to device
simulation.to(dev)

################################################
# define the operators we want to observe
################################################
opg_occupation_list = []
for i in range(ns):
    opg_occupation = StaticTightBindingOperatorGroup(n_sites=ns, id="sp_number", batchsize=batchsize, coef=1.0)
    op_seq = ["X"] * ns
    op_seq[i] = "N"
    opg_occupation.add_operator("".join(op_seq))
    opg_occupation.to(dev)
    opg_occupation_list.append(opg_occupation)

################################################
#  train
################################################
nepoch = 100
optimizer = th.optim.Adam([{'params': opg_epc_list[i].parameters(), 'lr': 1.0} for i in range(ns)])
effective_g_list = []
for epoch in range(nepoch):
    # if epoch == nepoch // 2:
    #     match_fs = min(2 * match_fs, 2000)
    ###### reset quantum state
    tb_init = th.zeros(ns, dtype=th.cfloat, device=dev)
    tb_init[init_site] = 1.0
    simulation.pse = tb_init
    ###### reset classical oscillators
    ## update the coupling strength with average gradient
    epc_coef_new = 0
    for i in range(ns):
        epc_coef_new += opg_epc_list[i].coef.detach()
    epc_coef_new /= ns
    g_factor_new = epc_coef_new.cpu().sum() / (omega[0] * th.sqrt(2 * mass[0] * omega[0]))
    effective_g_list.append(g_factor_new.numpy())
    print('epoch = {:d}, effective g = {:.5f}'.format(epoch, effective_g_list[-1]))
    ## reset epc, position and velocity of classical oscillators
    for i in range(ns):
        opg_epc_list[i].coef.data = epc_coef_new * 1.0
        x0 = th.zeros_like(mass) #- g_factor * np.sqrt(2) / (mass * omega)**0.5
        x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5
        x0 = x0.reshape(1,nmodes,1).repeat(batchsize,1,1)
        x0 += x0_std * th.randn(batchsize, nmodes, 1)
        oscilator_list[i].set_positions(x0)
        oscilator_list[i].reset(temp=temperature)
    
    ## initialize trajectory holders
    t_list = []
    occ_list = []
    ## simulate 
    for step in range(nsteps):
        if step % int(Constants.fs / dt) == 0:
            # print('========Equilibration: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
            t_list.append(step * dt/Constants.fs)
            ## record the occupation of each site
            pure_ensemble = simulation.pure_ensemble
            simulation.normalize()
            obs_occupation = [simulation.observe(opg_occupation_list[i]).real for i in range(ns)]
            occ_list.append(th.stack(obs_occupation, dim=-1))  ## list of (batchsize, ns) tensor
        simulation.step(dt, temp=None, profile=False, feedback=False)
    occ_list = th.stack(occ_list, dim=0)   # (#sample, batchsize, ns) tensor
    occ_pred = occ_list.mean(1) ## (#sample, ns) tensor
    diff_occ = occ_pred[:match_fs] - occ_ref[:match_fs]
    loss = th.mean(th.abs(diff_occ)**2)

    optimizer.zero_grad()
    loss.backward()
    logging.info(f"Epoch={epoch}, loss={loss.item()}, Effective g={effective_g_list[-1]}")
    logging.info('occ_pred:{}'.format(occ_pred[::10]))
    logging.info('occ_ref :{}'.format(occ_ref[::10]))
    optimizer.step()
    ################################################
    # Post-processing
    ################################################
    occ_pred_plot = occ_pred.detach().cpu().numpy()
    occ_ref_plot = occ_ref.detach().cpu().numpy()
    occ_ref_plot = occ_ref_plot[:occ_pred_plot.shape[0]]

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].plot(t_list, occ_pred_plot[:,0], label='Ehrenfest')
    ax[0].plot(t_list, occ_ref_plot[:,0], label='Quantum')
    ax[0].set_title('site 0')
    ax[0].legend()

    ax[1].plot(t_list, occ_pred_plot[:,1], label='Ehrenfest')
    ax[1].plot(t_list, occ_ref_plot[:,1], label='Quantum')
    ax[1].set_title('site 1')
    ax[1].legend()

    ax[2].plot(t_list, occ_pred_plot[:,2], label='Ehrenfest')
    ax[2].plot(t_list, occ_ref_plot[:,2], label='Quantum')
    ax[2].set_title('site 2')
    ax[2].legend()
    for a in ax:
        a.set_xlabel(r'$t$ [fs]')
        a.set_ylabel('Occupation')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'figures','train_epoch_{:d}.png'.format(epoch)))
    plt.close()

effective_g_list = np.array(effective_g_list)
np.savetxt(os.path.join(out_folder, 'effective_g_list.txt'), effective_g_list)


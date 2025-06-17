import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorQubitUnitarySystem
from qepsilon.simulation.lindblad_system import QubitLindbladSystem
from qepsilon.operator_group.spin_operators import StaticPauliOperatorGroup, PeriodicNoisePauliOperatorGroup
from qepsilon.utilities import Constants_Metal as Constants
import os
import argparse

## set up device and printing options
dev = 'cuda' if th.cuda.is_available() else 'cpu'
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=300.0, help='temperature in K')
parser.add_argument('--data_dt', type=float, default=0.01, help='time step in fs')
parser.add_argument('--dt', type=float, default=0.05, help='time step in fs')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')
parser.add_argument('--sample_time', type=float, default=100, help='sampling time in fs')
parser.add_argument('--train_frame', type=int, default=70, help='train frame')
parser.add_argument('--batchsize', type=int, default=512, help='batch size')
parser.add_argument('--nepoch', type=int, default=201, help='number of epochs')
args = parser.parse_args()

## load system parameters
data = np.loadtxt('../data.csv', delimiter=',')
nmodes = data.shape[0]
omega_in_cm = th.tensor(data[:,0])
lambda_in_cm = th.tensor(data[:,1])
polaron_binding_energy = - lambda_in_cm.sum() * Constants.cm_inverse_energy
g_factor = th.sqrt(lambda_in_cm / omega_in_cm)
print('polaron_binding_energy:{}'.format(polaron_binding_energy), 'wavenumber:{}/cm'.format(lambda_in_cm.sum()), 'period:{}'.format(1/lambda_in_cm.sum()/3 * 1e5))

## load training data
quantum_data_dir = '../00.data_generation/T{:.0f}_epsilon{:.1f}_dt{:.3f}fs'.format(args.temperature, args.epsilon, args.data_dt)
quantum_sigma_n = np.load('{}/sigma_n.npy'.format(quantum_data_dir))  # (batchsize, nsample)
quantum_sigma_x = np.load('{}/sigma_x.npy'.format(quantum_data_dir))  # (batchsize, nsample)
quantum_sigma_y = np.load('{}/sigma_y.npy'.format(quantum_data_dir))  # (batchsize, nsample)
quantum_t_list = th.arange(quantum_sigma_n.shape[1])
sigma_n_ref = th.tensor(quantum_sigma_n, device=dev)
sigma_x_ref = th.tensor(quantum_sigma_x, device=dev)
sigma_y_ref = th.tensor(quantum_sigma_y, device=dev)

## set up logging
out_dir = 'T{:.0f}_dt{:.3f}fs'.format(args.temperature, args.dt)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='{}/train.log'.format(out_dir), filemode='w')

"""
Process parameters
"""
## system parameters
temperature = args.temperature
kbT = Constants.kb * temperature
mass = th.tensor([250 * Constants.amu] * nmodes)
omega = omega_in_cm * Constants.cm_inverse_energy
epc_cls_amp = th.sqrt(2 * g_factor**2 * omega * kbT)
binding_energy = g_factor **2 * omega
## simulation parameters
ns = 1
batchsize = args.batchsize
x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5

## report the parameters
logging.info(f'{nmodes} bosonic modes with frequency {omega_in_cm} cm^-1')
logging.info(f'kbT/hw: {kbT/omega}, < 1 means large quantum effect')
logging.info(f'g factor is {g_factor}, binding energy: {binding_energy.sum() / Constants.meV:.5f} meV')
logging.info(f'mass: {mass}')

## equilibration parameters
total_sim_time = args.sample_time * Constants.fs
dt = args.dt * Constants.fs   # does not need to be smaller for 200fs simulation. 
nsteps =  int(total_sim_time / dt)

"""
define the simulation
"""
simulation = QubitLindbladSystem(n_qubits=ns, batchsize=batchsize)

# add classical harmonic oscillators
for i in range(nmodes):
    opg_epc = PeriodicNoisePauliOperatorGroup(n_qubits=1, id=f"mode-{i}_epc", batchsize=batchsize, 
        tau=2 * np.pi / omega.numpy()[i], amp=epc_cls_amp.numpy()[i], requires_grad=True, requires_grad_amp_only=True, requires_grad_tau_only=False)
    opg_epc.add_operator('N')
    simulation.add_operator_group_to_hamiltonian(opg_epc)

## add a shift of onsite energy
opg_onsite_energy = StaticPauliOperatorGroup(n_qubits=1, id="onsite_energy", batchsize=batchsize, coef=polaron_binding_energy.numpy(), static=False, requires_grad=True)
opg_onsite_energy.add_operator('N')
simulation.add_operator_group_to_hamiltonian(opg_onsite_energy)

## add a jump operator for dephasing
opg_dephasing = StaticPauliOperatorGroup(n_qubits=1, id="dephasing", batchsize=batchsize, coef=10.0, static=False, requires_grad=True)
opg_dephasing.add_operator('N')
simulation.add_operator_group_to_jumping(opg_dephasing)

## move simulation to device
simulation.to(dev)

# define observables
opg_sp_number = StaticPauliOperatorGroup(n_qubits=1, id="sp_number", batchsize=batchsize, coef=1.0).to(dev)
opg_sp_number.add_operator('N')
## spin sigma_x operator
opg_sp_x = StaticPauliOperatorGroup(n_qubits=1, id="spin_x", batchsize=batchsize, coef=1.0).to(dev)
opg_sp_x.add_operator('X')
## spin sigma_y operator
opg_sp_y = StaticPauliOperatorGroup(n_qubits=1, id="spin_y", batchsize=batchsize, coef=1.0).to(dev)
opg_sp_y.add_operator('Y')

"""
train
"""
nepoch = args.nepoch
optimizer = th.optim.Adam([{'params': opg.parameters(), 'lr': 0.3} for opg in simulation.hamiltonian_operator_groups] +
                           [{'params': opg.parameters(), 'lr': 0.3} for opg in simulation.jumping_operator_groups])
for epoch in range(nepoch):
    ## reset the quantum state
    epsilon = args.epsilon
    sp_init = th.zeros(2,2, dtype=th.cfloat, device=dev)
    sp_init[0,0] =  1-epsilon
    sp_init[1,1] =  epsilon
    sp_init[1,0] = np.sqrt(epsilon * (1-epsilon))
    sp_init[0,1] = np.sqrt(epsilon * (1-epsilon))
    simulation.rho = sp_init.clone()
    ## reset the classical oscillators
    simulation.reset()
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
            simulation.normalize()
            obs_n = simulation.observe(opg_sp_number).real # (batchsize,)
            obs_x = simulation.observe(opg_sp_x).real # (batchsize,)
            obs_y = simulation.observe(opg_sp_y).real # (batchsize,)
            sigma_n_list.append(obs_n)
            sigma_x_list.append(obs_x)
            sigma_y_list.append(obs_y)
            # print('mean obs_n:{}, mean obs_x:{}, mean obs_y:{}'.format(obs_n.mean(), obs_x.mean(), obs_y.mean()))
        simulation.step(dt, profile=False)
    sigma_x_list = th.stack(sigma_x_list, dim=1) ## (batchsize, nsample)
    sigma_y_list = th.stack(sigma_y_list, dim=1) ## (batchsize, nsample)
    sigma_n_list = th.stack(sigma_n_list, dim=1) ## (batchsize, nsample)

    diff_mean_x = (sigma_x_list[:,:args.train_frame].mean(0) - sigma_x_ref[:,:args.train_frame].mean(0))
    diff_mean_y = (sigma_y_list[:,:args.train_frame].mean(0) - sigma_y_ref[:,:args.train_frame].mean(0))
    diff_std_x = (sigma_x_list[:,:args.train_frame].std(0) - sigma_x_ref[:,:args.train_frame].std(0))
    diff_std_y = (sigma_y_list[:,:args.train_frame].std(0) - sigma_y_ref[:,:args.train_frame].std(0))
    loss = th.mean(th.abs(diff_mean_x)**2) + th.mean(th.abs(diff_mean_y)**2)
    loss += 0.1 * th.mean(th.abs(diff_std_x)**2) + 0.1 * th.mean(th.abs(diff_std_y)**2)
    optimizer.zero_grad()
    loss.backward()
    logging.info(f"Epoch={epoch}, loss={loss.item()}")
    for opg in simulation.hamiltonian_operator_groups:
        try:
            logging.info(f"{opg.id} coef={opg.coef.item()}")
        except:
            logging.info("{:s} tau={:.1f}fs amp={:.2f}".format(opg.id, opg.tau.item()*1000, opg.amp.item() ))
    for opg in simulation.jumping_operator_groups:
        logging.info(f"{opg.id}={opg.coef.item()}")
    optimizer.step()
    ################################################
    # Post-processing
    ################################################
    if epoch % 10 == 0:  ## plot 
        sigma_n_ref_mean = sigma_n_ref.detach().cpu().numpy().mean(0)
        sigma_x_ref_mean = sigma_x_ref.detach().cpu().numpy().mean(0)
        sigma_y_ref_mean = sigma_y_ref.detach().cpu().numpy().mean(0)
        sigma_n_ref_std = sigma_n_ref.detach().cpu().numpy().std(0)
        sigma_x_ref_std = sigma_x_ref.detach().cpu().numpy().std(0)
        sigma_y_ref_std = sigma_y_ref.detach().cpu().numpy().std(0)
        sigma_n_pred_mean = sigma_n_list.detach().cpu().numpy().mean(0)
        sigma_x_pred_mean = sigma_x_list.detach().cpu().numpy().mean(0)
        sigma_y_pred_mean = sigma_y_list.detach().cpu().numpy().mean(0)
        sigma_n_pred_std = sigma_n_list.detach().cpu().numpy().std(0)
        sigma_x_pred_std = sigma_x_list.detach().cpu().numpy().std(0)
        sigma_y_pred_std = sigma_y_list.detach().cpu().numpy().std(0)

        fig, ax = plt.subplots(1, 3, figsize=(12, 3))

        ax[0].plot(t_list, sigma_n_pred_mean, label='Ehrenfest')
        ax[0].fill_between(t_list, sigma_n_pred_mean - sigma_n_pred_std, sigma_n_pred_mean + sigma_n_pred_std, alpha=0.5)
        ax[0].plot(quantum_t_list, sigma_n_ref_mean, label='Quantum')
        ax[0].fill_between(quantum_t_list, sigma_n_ref_mean - sigma_n_ref_std, sigma_n_ref_mean + sigma_n_ref_std, alpha=0.5)
        ax[0].set_title(r'$\langle \hat{\sigma}_n \rangle$' + 'iter={}'.format(epoch))
        ax[0].legend()

        ax[1].plot(t_list, sigma_x_pred_mean, label='Ehrenfest')
        ax[1].fill_between(t_list, sigma_x_pred_mean - sigma_x_pred_std, sigma_x_pred_mean + sigma_x_pred_std, alpha=0.5)
        ax[1].plot(quantum_t_list, sigma_x_ref_mean, label='Quantum')
        ax[1].fill_between(quantum_t_list, sigma_x_ref_mean - sigma_x_ref_std, sigma_x_ref_mean + sigma_x_ref_std, alpha=0.5)
        ax[1].set_title(r'$\langle \hat{\sigma}_y \rangle$' + 'iter={}'.format(epoch))
        ax[1].legend()

        ax[2].plot(t_list, sigma_y_pred_mean, label='Ehrenfest')
        ax[2].fill_between(t_list, sigma_y_pred_mean - sigma_y_pred_std, sigma_y_pred_mean + sigma_y_pred_std, alpha=0.5)
        ax[2].plot(quantum_t_list, sigma_y_ref_mean, label='Quantum')
        ax[2].fill_between(quantum_t_list, sigma_y_ref_mean - sigma_y_ref_std, sigma_y_ref_mean + sigma_y_ref_std, alpha=0.5)
        ax[2].set_title(r'$\sigma \langle \hat{\sigma}_x \rangle$' + 'iter={}'.format(epoch))
        ax[2].legend()

        for a in ax:
            a.set_xlabel(r'$t$ [fs]')
        plt.tight_layout()
        plt.savefig('{}/train_iter{}.png'.format(out_dir, epoch))
        plt.close()




import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorQubitUnitarySystem
from qepsilon.simulation.lindblad_system import QubitLindbladSystem
from qepsilon.operator_group.spin_operators import StaticPauliOperatorGroup, PeriodicNoisePauliOperatorGroup
from qepsilon.utilities import Constants_Metal as Constants
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.markersize'] = 5

dev = 'cuda' if th.cuda.is_available() else 'cpu'
temp_list = [200, 250, 300, 350, 400]
epsilon = 0.1
nmodes = 9
timestep_in_fs= 0.05
sample_time = 150
batchsize = 512
temperature = 250
fig, ax = plt.subplots(2, 5, figsize=(16, 5))
fig1, ax1 = plt.subplots(1, 1, figsize=(3, 3))
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for idx, temperature in enumerate(temp_list):
    ## reference data
    ## load training data
    quantum_data_dir = '../00.data_generation/T{:.0f}_epsilon{:.1f}_dt{:.3f}fs'.format(temperature, 0.1, 0.01)
    quantum_sigma_n = np.load('{}/sigma_n.npy'.format(quantum_data_dir))  # (batchsize, nsample)
    quantum_sigma_x = np.load('{}/sigma_x.npy'.format(quantum_data_dir))  # (batchsize, nsample)
    quantum_sigma_y = np.load('{}/sigma_y.npy'.format(quantum_data_dir))  # (batchsize, nsample)
    quantum_t_list = th.arange(quantum_sigma_n.shape[1])
    sigma_n_ref = th.tensor(quantum_sigma_n, device=dev)
    sigma_x_ref = th.tensor(quantum_sigma_x, device=dev)
    sigma_y_ref = th.tensor(quantum_sigma_y, device=dev)

    """
    Process parameters
    """
    ## system parameters

    kbT = Constants.kb * temperature
    mass = th.tensor([250 * Constants.amu] * nmodes)
    data = np.loadtxt('../coupling_const.csv', delimiter=',')
    omega_in_cm = th.tensor(data[:,0])
    omega = omega_in_cm * Constants.cm_inverse_energy
    epc_cls_amp = np.loadtxt('T{:.0f}_dt{:.3f}fs/amp_list.txt'.format(temperature, timestep_in_fs))[-1]
    dephasing = np.loadtxt('T{:.0f}_dt{:.3f}fs/dephasing_list.txt'.format(temperature, timestep_in_fs))[-1]
    onsite_energy = np.loadtxt('T{:.0f}_dt{:.3f}fs/onsite_energy_list.txt'.format(temperature, timestep_in_fs))[-1]
    print('==========Temperature: {:.0f}K, dt: {:.3f}fs, epsilon: {:.1f} =========='.format(temperature, timestep_in_fs, epsilon))
    print('epc_cls_amp: {}'.format(epc_cls_amp))
    print('onsite_energy: {}'.format(onsite_energy))
    print('dephasing: {}'.format(dephasing))

    ## simulation parameters
    ns = 1
    x0_std = (Constants.kb * temperature / (mass * omega**2))**0.5

    ## equilibration parameters
    total_sim_time = sample_time * Constants.fs
    dt = timestep_in_fs * Constants.fs   # does not need to be smaller for 200fs simulation. 
    nsteps =  int(total_sim_time / dt)

    """
    define the simulation
    """
    simulation = QubitLindbladSystem(n_qubits=ns, batchsize=batchsize)

    # add classical harmonic oscillators
    for i in range(nmodes):
        opg_epc = PeriodicNoisePauliOperatorGroup(n_qubits=1, id=f"mode-{i}_epc", batchsize=batchsize, 
            tau=2 * np.pi / omega.numpy()[i], amp=epc_cls_amp[i], requires_grad=False)
        opg_epc.add_operator('N')
        simulation.add_operator_group_to_hamiltonian(opg_epc)

    ## add a shift of onsite energy
    opg_onsite_energy = StaticPauliOperatorGroup(n_qubits=1, id="onsite_energy", batchsize=batchsize, coef=onsite_energy, static=True, requires_grad=False)
    opg_onsite_energy.add_operator('N')
    simulation.add_operator_group_to_hamiltonian(opg_onsite_energy)

    ## add a jump operator for dephasing
    opg_dephasing = StaticPauliOperatorGroup(n_qubits=1, id="dephasing", batchsize=batchsize, coef=dephasing, static=True, requires_grad=False)
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
    Simulate
    """
    ## initialize the quantum state
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
    
    ################################################
    # Post-processing
    ################################################
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

 
    ## plot sigma_x, sigma_y for each temperature seperately
    ax[0,idx].plot(t_list, sigma_x_pred_mean, label='DIQCD')
    ax[0,idx].fill_between(t_list, sigma_x_pred_mean - sigma_x_pred_std, sigma_x_pred_mean + sigma_x_pred_std, alpha=0.5)
    ax[0,idx].plot(quantum_t_list, sigma_x_ref_mean, label='Exact')
    ax[0,idx].fill_between(quantum_t_list, sigma_x_ref_mean - sigma_x_ref_std, sigma_x_ref_mean + sigma_x_ref_std, alpha=0.5)
    ax[0,idx].legend(frameon=False)
    ax[0,idx].set_xlabel(r'$t$ [fs]')
    ax[0,idx].set_ylabel(r'$\langle \hat{\sigma}_x (t)\rangle$')

    ax[1,idx].plot(t_list, sigma_y_pred_mean, label='DIQCD')
    ax[1,idx].fill_between(t_list, sigma_y_pred_mean - sigma_y_pred_std, sigma_y_pred_mean + sigma_y_pred_std, alpha=0.5)
    ax[1,idx].plot(quantum_t_list, sigma_y_ref_mean, label='Exact')
    ax[1,idx].fill_between(quantum_t_list, sigma_y_ref_mean - sigma_y_ref_std, sigma_y_ref_mean + sigma_y_ref_std, alpha=0.5)
    ax[1,idx].legend(frameon=False)
    ax[1,idx].set_xlabel(r'$t$ [fs]')
    ax[1,idx].set_ylabel(r'$\langle \hat{\sigma}_y (t)\rangle$')

    ax[0, idx].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    ax[1, idx].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    ax[0,idx].set_title(r'T={}K'.format(temperature))
    
    if temperature == 300:
        ## plot sigma_x, sigma_y for all temperatures together
        ax1.plot(t_list, sigma_x_pred_mean,  color='tab:blue', linestyle='-', alpha=0.5, linewidth=2, label=r'$\langle \hat{\sigma}_x (t)\rangle$, DIQCD')
        ax1.plot(quantum_t_list, sigma_x_ref_mean, color='tab:blue', linestyle='--', alpha=1.0, linewidth=2, label=r'$\langle \hat{\sigma}_x (t)\rangle$, Exact')
        ax1.plot(t_list, sigma_y_pred_mean,  color='tab:orange', linestyle='-', alpha=0.5, linewidth=2, label=r'$\langle \hat{\sigma}_y (t)\rangle$, DIQCD')
        ax1.plot(quantum_t_list, sigma_y_ref_mean, color='tab:orange', linestyle='--', alpha=1, linewidth=2, label=r'$\langle \hat{\sigma}_y (t)\rangle$, Exact')
        ax1.legend(frameon=False)
        ax1.set_xlabel(r'$t$ [fs]', fontsize=13)
        ax1.set_ylabel('Observation', fontsize=13)

fig.tight_layout()
fig1.tight_layout()
fig.savefig('validation.png', dpi=300)
fig1.savefig('validation_all.png', dpi=300)




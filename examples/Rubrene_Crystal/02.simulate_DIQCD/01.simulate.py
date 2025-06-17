import logging
import time
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import qepsilon as qe
from qepsilon.simulation.mixed_unitary_system import OscillatorTightBindingUnitarySystem
from qepsilon.simulation.lindblad_system import LindbladSystem
from qepsilon.operator_group.tb_operators import PeriodicNoiseTightBindingOperatorGroup, StaticTightBindingOperatorGroup
from qepsilon.utilities import Constants_Metal as Constants
import os
import argparse
dev = 'cuda' if th.cuda.is_available() else 'cpu'
th.set_printoptions(sci_mode=False, precision=5)
np.set_printoptions(suppress=True, precision=5)

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nsites', type=int, default=40)
parser.add_argument('--temperature', type=float, default=300.0)
parser.add_argument('--hopping_in_meV', type=float, default=83.0)
parser.add_argument('--dt', type=float, default=0.01, help='time step in fs')
parser.add_argument('--batchsize', type=int, default=512, help='batch size')
parser.add_argument('--sim_time', type=float, default=100)
args = parser.parse_args()
temperature = args.temperature
hopping_in_meV = args.hopping_in_meV

## make output folder
out_folder = 'T{:.0f}_V{:.1f}meV_ns{}'.format(temperature, hopping_in_meV, args.nsites)
os.makedirs(out_folder, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='{}/log.txt'.format(out_folder), filemode='w')


## load system parameters
data_folder = '../train_lindblad/T{:.0f}_epsilon0.1_dt{:.3f}fs'.format(temperature, 0.05)
with open('{}/train.log'.format(data_folder), 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if "Epoch=199" in line:
            start_idx = idx + 1
            break
    end_idx = start_idx + 11
    data_lines = lines[start_idx:end_idx]
    
    # Extract tau and amp from first 9 lines
    tau_list = []
    amp_list = []
    for i in range(9):
        line = data_lines[i]
        tau = float(line.split('tau=')[1].split('fs')[0])
        amp = float(line.split('amp=')[1].strip())
        tau_list.append(tau)
        amp_list.append(amp)
    
    # Extract coef and dephasing from last 2 lines
    onsite_coef = float(data_lines[-2].split('coef=')[1].strip())
    dephasing = float(data_lines[-1].split('dephasing=')[1].strip())

################################################
#  Parameters 
################################################
## hamiltonian parameters 
ns = args.nsites
nmodes = 9
hopping_coef = hopping_in_meV * Constants.meV
batchsize = args.batchsize
total_t = args.sim_time * Constants.fs
dt = args.dt * Constants.fs
nsteps =  int(total_t / dt)

################################################
# define the simulation (Hamiltonian, classical oscillators, coupling)
################################################
simulation = LindbladSystem(num_states=ns, batchsize=batchsize)

## set initial state
init_site = ns//2
init_state = th.zeros(ns, ns, dtype=th.cfloat)
init_state[init_site, init_site] = 1
simulation.rho = init_state.clone()

## add hopping terms
op_hop = StaticTightBindingOperatorGroup(n_sites=ns, id="hop", batchsize=batchsize, coef=hopping_coef, static=True, requires_grad=False)
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

## add classical harmonic oscillators
for isite in range(ns):
    for imode in range(nmodes):
        opg_epc = PeriodicNoiseTightBindingOperatorGroup(n_sites=ns, id=f"site-{isite}_mode-{imode}_epc", batchsize=batchsize, 
            tau=tau_list[imode], amp=amp_list[imode], requires_grad=False)
        epc_seq = ['X'] * ns 
        epc_seq[isite] = 'N'
        opg_epc.add_operator("".join(epc_seq))
        simulation.add_operator_group_to_hamiltonian(opg_epc)

## add jump operators
for isite in range(ns):
    opg_dephasing = StaticTightBindingOperatorGroup(n_sites=ns, id=f"site-{isite}_dephasing", batchsize=batchsize, coef=dephasing, static=False, requires_grad=False)
    dephasing_seq = ['X'] * ns 
    dephasing_seq[isite] = 'N'
    opg_dephasing.add_operator("".join(dephasing_seq))
    simulation.add_operator_group_to_jumping(opg_dephasing)

## move to GPU if dev='cuda'
simulation.to(dev)

################################################
# Simulation
################################################
t_traj = []
site_occupation_traj = []
MSD_traj = []
for step in range(nsteps):
    if step % int(Constants.fs / dt) == 0:
        print('========Equilibration: step-{}, t={}fs========'.format(step, step * dt/Constants.fs))
        t_traj.append(step * dt/Constants.fs)
        simulation.normalize()
        density_matrix = simulation.rho
        ## get diagonal elements
        site_occupation = density_matrix.diagonal(dim1=1, dim2=2)  # (batchsize, ns)
        site_occupation = site_occupation.real
        assert site_occupation.shape == (batchsize, ns)
        assert site_occupation.sum(dim=-1).allclose(th.ones(batchsize, device=site_occupation.device))
        site_occupation_traj.append(site_occupation)

        pos =  th.arange(ns).to(dtype=site_occupation.dtype, device=site_occupation.device)
        average_r = (site_occupation * pos[None, :]).sum(dim=-1)
        average_r2 = (site_occupation * pos[None, :]**2).sum(dim=-1)
        MSD = average_r2 - average_r**2
        MSD_traj.append(MSD)
        print('Average MSD={}'.format(MSD.mean()))

    simulation.step(dt, profile=False)


################################################
# Post-processing
################################################
nsample = len(t_traj)
## process the site occupation trajectories
site_occupation_traj = th.stack(site_occupation_traj, dim=0).cpu().numpy()  # (nsample, batchsize, ns)
assert site_occupation_traj.shape[1:] == (batchsize, ns)
## process the MSD trajectory
MSD_traj = th.stack(MSD_traj, dim=0).cpu().numpy()  # (nsample, batchsize)
assert MSD_traj.shape == (nsample, batchsize)
np.save('{}/site_occupation_traj.npy'.format(out_folder), site_occupation_traj)
np.save('{}/MSD_traj.npy'.format(out_folder), MSD_traj)

## plot site_occupation_traj
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
## plot the site occupation as heatmap
# ax[0].imshow(site_occupation_traj[:,0,:], aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, t_traj[-1]])
ax[0].imshow(-np.log(site_occupation_traj.mean(1)+1e-10), aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, t_traj[-1]], vmin=0, vmax=4)

ax[0].set_title('site occupation')
ax[0].set_xlabel('site')
ax[0].set_ylabel('time [fs]')
## plot others
ax[1].plot(t_traj, MSD_traj.mean(-1), '--')
ax[1].set_title(r'$Mean Square Displacement$')
ax[1].set_xlabel('time [fs]')
ax[1].set_ylabel(r'$MSD(t)$')
plt.tight_layout()
plt.savefig('{}/results.png'.format(out_folder), dpi=200)
plt.close()

 

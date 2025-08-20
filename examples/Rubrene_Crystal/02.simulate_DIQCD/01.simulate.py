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

##
parser = argparse.ArgumentParser()
parser.add_argument('--nsites', type=int, default=40, help='number of sites')
parser.add_argument('--temperature', type=float, default=300, help='temperature in K')
parser.add_argument('--timestep_in_fs', type=float, default=0.02, help='time step in fs')
parser.add_argument('--batchsize', type=int, default=512, help='batch size')
parser.add_argument('--sample_time', type=float, default=100, help='sampling time in fs')
args = parser.parse_args()

"""
Preprocessing
"""
## parse arguments
nsites= args.nsites
temperature = args.temperature
hopping_in_meV = 83  # in meV
timestep_in_fs = args.timestep_in_fs   # in fs
batchsize = args.batchsize
sim_time = args.sample_time   # in fs

## make output folder
out_folder = 'T{:.0f}_ns{}'.format(temperature, nsites)
os.makedirs(out_folder, exist_ok=True)

## load DIQCD model parameters
data_folder = '../01.train_DIQCD/T{:.0f}_dt{:.3f}fs'.format(temperature, 0.05)
tau_list = np.loadtxt(os.path.join(data_folder, 'tau_list.txt'))[-1] 
amp_list = np.loadtxt(os.path.join(data_folder, 'amp_list.txt'))[-1] 
dephasing = np.loadtxt(os.path.join(data_folder, 'dephasing_list.txt'))[-1] 

print('mode period:', tau_list)
print('mode amplitude:', amp_list)
                       
## process system parameters
ns = nsites
nmodes = tau_list.shape[0]
hopping_coef = hopping_in_meV * Constants.meV
total_t = sim_time * Constants.fs
dt = timestep_in_fs * Constants.fs
nsteps =  int(total_t / dt)

"""
Define the system
"""
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
        opg_epc = PeriodicNoiseTightBindingOperatorGroup(
            n_sites=ns, id=f"site-{isite}_mode-{imode}_epc", batchsize=batchsize,tau=tau_list[imode], amp=amp_list[imode], requires_grad=False)
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

"""
Simulate DIQCD
"""
t_traj = []
site_occupation_traj = []
MSD_traj = []
for step in range(nsteps):
    if step % int(1 / timestep_in_fs) == 0:
        print('======== Step-{}, t={:.0f}fs========'.format(step, step * dt/Constants.fs))
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

"""
Postprocess
"""
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
ax[0].imshow(-np.log(site_occupation_traj.mean(1)+1e-10), aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, t_traj[-1]], vmin=0, vmax=4)
ax[0].set_title('site occupation')
ax[0].set_xlabel('site')
ax[0].set_ylabel('time [fs]')
## plot mean squared displacement
ax[1].plot(t_traj, MSD_traj.mean(-1), '--')
ax[1].fill_between(t_traj, MSD_traj.mean(-1)+MSD_traj.std(-1) ,MSD_traj.mean(-1)-MSD_traj.std(-1) , alpha=0.5)
ax[1].set_title('Mean Square Displacement$')
ax[1].set_xlabel('time [fs]')
ax[1].set_ylabel(r'$MSD(t)$')
plt.tight_layout()
plt.savefig(os.path.join(out_folder,'results.png'), dpi=200)
plt.show()
plt.close()

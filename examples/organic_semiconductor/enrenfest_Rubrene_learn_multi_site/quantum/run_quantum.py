import numpy as np
import torch as th
import matplotlib.pyplot as plt
import qepsilon as qe
from qepsilon.utilities import compose, trace
from qepsilon.utilities import Constants_Metal as Constants
from qepsilon import LindbladSystem
import argparse
from qepsilon.operator_group import *
th.set_printoptions(sci_mode=False, precision=4)
dev = 'cuda'


## parse a number
nmax_list = np.array([8, 3, 1, 1, 1, 1, 1, 1, 1])
parser = argparse.ArgumentParser()
parser.add_argument('--ns', type=int, default=3)
parser.add_argument('--mode_index', type=int, default=0)
# parser.add_argument('--nmax', type=int, default=4)
parser.add_argument('--temperature', type=float, default=300)
parser.add_argument('--hopping_in_meV', type=float, default=83)
args = parser.parse_args()

## load the data
data = np.loadtxt('../data.csv', delimiter=',')
omega_in_cm = data[args.mode_index,0]
lambda_in_cm = data[args.mode_index,1]
omega = omega_in_cm * Constants.cm_inverse_energy
g_factor = np.sqrt(lambda_in_cm / omega_in_cm)

## simulation parameters
ns = args.ns  ## number of sites
batchsize = 1
# dt = 0.025 * Constants.fs    # max 0.1fs
dt = 0.005 * Constants.fs    # max 0.1fs
sample_time = 1000 * Constants.fs
sample_freq = int(Constants.fs/dt)
nsteps = int(sample_time/dt)
temperature = args.temperature
kbT = Constants.kb * temperature  

## hopping parameters
hopping_in_meV = args.hopping_in_meV   # meV
hopping_coef = hopping_in_meV * Constants.meV
## boson bath parameters
num_modes = 1
nmax = nmax_list[args.mode_index]
num_boson_states = ((nmax+1)**num_modes) ** ns
total_num_states = ns * num_boson_states
print('Hopping coefficient [meV]:', hopping_in_meV, 'in internal units [ps^-1]:', hopping_coef)
print('g factor:', g_factor)
print('omega [cm^-1]:', omega_in_cm, 'in internal units:', omega)
print('polaron binding energy [cm^-1]:', - g_factor**2 * omega_in_cm, 'in internal units:', - g_factor**2 * omega)

"""
initiate the system
"""
system = LindbladSystem(num_states=total_num_states, batchsize=batchsize)

"""
Define the Hamiltonian
"""
############################## define identities ##############################
opg_bs_id = IdentityBosonOperatorGroup(num_modes=num_modes, id="boson_id", nmax=nmax, batchsize=batchsize)
opg_tb_id = IdentityTightBindingOperatorGroup(n_sites=ns, id="spin_id", batchsize=batchsize)

############################## add the hopping term to the Hamiltonian ##############################
## tight binding hopping term for electrons
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
## combine and add to the system
opg_hop = ComposedOperatorGroups(id="hop", OP_list=[opg_tb_hop] + [opg_bs_id]*ns, static=True)
system.add_operator_group_to_hamiltonian(opg_hop)

############################## add the bosonic harmonic energy term to the Hamiltonian ##############################
opg_bs_harmonic = HarmonicOscillatorBosonOperatorGroup(num_modes=num_modes, id="boson_harmonic", batchsize=batchsize, nmax=nmax, omega=  omega)   # wb^\dagger b
for i in range(ns):
    OP_list = [opg_tb_id] + [opg_bs_id]*i + [opg_bs_harmonic] + [opg_bs_id]*(ns-i-1)
    opg_harmonic = ComposedOperatorGroups(id="harmonic_{}".format(i), OP_list = OP_list, static=True)
    system.add_operator_group_to_hamiltonian(opg_harmonic)

############################## add the electron-boson coupling term: c^+ c b
opg_bs_x = StaticBosonOperatorGroup(num_modes=num_modes, id="boson_x", nmax=nmax, batchsize=batchsize, coef= g_factor * omega)   # gw(b^\dagger + b)
opg_bs_x.add_operator('U')
opg_bs_x.add_operator('D')
opg_tb_number_list = []
for i in range(ns):
    opg_tb_number = StaticTightBindingOperatorGroup(n_sites=ns, id="tb_number", batchsize=batchsize, coef=1.0)
    OP_seq = ['X'] * ns
    OP_seq[i] = 'N'
    OP_seq = "".join(OP_seq)
    opg_tb_number.add_operator(OP_seq)

    opg_tb_number_list.append(opg_tb_number)
    OP_list = [opg_tb_number] + [opg_bs_id]*i + [opg_bs_x] + [opg_bs_id]*(ns-i-1)
    opg_couple = ComposedOperatorGroups(id="couple_{}".format(i), OP_list = OP_list, static=True)
    system.add_operator_group_to_hamiltonian(opg_couple)

"""
Moving system to the GPU
"""
if dev == 'cuda':
    system.to(dev)

"""
Set the initial state as a localized electron on site n//2 and a thermal boson state
"""
init_site = ns//2
localized_ham = 0
for i in range(ns):
    opg_harmonic_init = ComposedOperatorGroups(id="harmonic_{}".format(i), OP_list = [opg_bs_id]*i + [opg_bs_harmonic] + [opg_bs_id]*(ns-i-1), static=True)
    op, coef = opg_harmonic_init.sample()
    localized_ham += op * coef.mean()
# opg_bs_x_init = ComposedOperatorGroups(id="x_at_init", OP_list = [opg_bs_id]*init_site + [opg_bs_x] + [opg_bs_id]*(ns-init_site-1), static=True)
# op, coef = opg_bs_x_init.sample()
# localized_ham += op * coef.mean()
assert localized_ham.shape == (num_boson_states, num_boson_states)
localized_ham = localized_ham - th.eye(num_boson_states, dtype=localized_ham.dtype, device=dev) * th.diag(localized_ham).real.min()
bs_thermal_state = th.matrix_exp(-localized_ham/kbT)
bs_thermal_state = bs_thermal_state / trace(bs_thermal_state)

tb_init = th.zeros(ns,ns, dtype=th.cfloat, device=dev)
tb_init[init_site, init_site] = 1.0
init_dm = th.kron(tb_init, bs_thermal_state)
system.rho = init_dm

"""
Define the operators we want to observe
"""
obs_list = []
bs_occ_list = []
for i in range(ns):
    OP_list = [opg_tb_number_list[i]] + [opg_bs_id]*ns
    obs_list.append(ComposedOperatorGroups(id="site{}_occupation".format(i), OP_list = OP_list, static=True))
    coef, op = obs_list[-1].sample()
for i in range(ns):
    opg_bs_n = StaticBosonOperatorGroup(num_modes=num_modes, id="boson_n", nmax=nmax, batchsize=batchsize)   # wb^\dagger b
    opg_bs_n.add_operator('N')
    OP_list = [opg_tb_id] + [opg_bs_id]*i + [opg_bs_n] + [opg_bs_id]*(ns-i-1)
    bs_occ_list.append(ComposedOperatorGroups(id="site{}_bs_occupation".format(i), OP_list = OP_list, static=True))
if dev == 'cuda':
    for op in obs_list:
        op.to(dev)
    for op in bs_occ_list:
        op.to(dev)

"""
Run the simulation
"""
t_list = []
occupation_list = []
MSD_list = []
for i in range(nsteps):
    system.step(dt=dt, time_independent = True, set_buffer=True)
    if i%sample_freq==0:    
        system.normalize()
        occupation = np.array([system.observe(obs_list[j]).cpu().numpy().mean() for j in range(ns)])
        bs_occ = np.array([system.observe(bs_occ_list[j]).cpu().numpy().mean() for j in range(ns)])
        MSD = (occupation * np.arange(ns)**2).sum() - (occupation * np.arange(ns)).sum()**2
        occupation_list.append(occupation)
        MSD_list.append(MSD)
        print('t={:.3f}fs, occupation={}, MSD={}'.format(i*dt/Constants.fs, occupation, MSD))
        print('bs_occ={}'.format(bs_occ))
        t_list.append(i*dt/Constants.fs)


### processing
prefix = 'ns{}_mode{:d}_nmax{}_T{:.0f}_V{:.1f}meV_dt{:.3f}'.format(ns, args.mode_index, nmax, temperature, hopping_in_meV, dt/Constants.fs)
occupation_list = np.array(occupation_list).reshape(-1,ns)
MSD_list = np.array(MSD_list).flatten()

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(-np.log(occupation_list+1e-10), aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, t_list[-1]], vmin=0, vmax=4)
ax[0].set_title('site occupation')
ax[0].set_xlabel('site')
ax[0].set_ylabel('time [fs]')
## plot others
ax[1].plot(t_list, MSD_list, '--')
ax[1].set_title(r'$Mean Square Displacement$')
ax[1].set_xlabel('time [fs]')
ax[1].set_ylabel(r'$MSD(t)$')
plt.tight_layout()
plt.savefig('data/{}.png'.format(prefix), dpi=200)
plt.close()
print('figure saved to {}.png'.format(prefix))
np.save('data/{}_occupation.npy'.format(prefix), occupation_list)
np.save('data/{}_MSD.npy'.format(prefix), MSD_list)


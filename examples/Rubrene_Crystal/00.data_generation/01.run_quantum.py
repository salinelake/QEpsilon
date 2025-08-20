import numpy as np
import torch as th
import matplotlib.pyplot as plt
import qepsilon as qe
from qepsilon.utilities import compose, trace
from qepsilon.utilities import Constants_Metal as Constants
from qepsilon import LindbladSystem, UnitarySystem
from qepsilon.operator_group import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=300, help='temperature in K')
parser.add_argument('--dt', type=float, default=0.01, help='time step in fs')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')
parser.add_argument('--sample_time', type=float, default=100, help='sampling time in fs')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--run_id', type=int, default=0, help='run id')
args = parser.parse_args()
epsilon = args.epsilon


th.set_printoptions(sci_mode=False, precision=4)
dev = 'cuda'

def sample_init_state(epsilon=0.1, num_modes=9, nmax=None, omega=None, kbT=None):
    sp_init = th.zeros(2, dtype=th.cfloat)
    sp_init[0] = np.sqrt(1-epsilon)
    sp_init[1] = np.sqrt(epsilon)

    bs_init_total = None
    chosen_levels = []
    for idx in range(num_modes):
        nstate = nmax[idx] + 1
        energy_levels = th.arange(nstate) * omega[idx]
        prob_levels = th.exp(-energy_levels/kbT)
        prob_levels = prob_levels / prob_levels.sum()

        bs_init = th.zeros(nstate, dtype=th.cfloat)
        choose_level = np.random.choice(nstate, p=prob_levels.numpy())
        bs_init[choose_level] = 1.0
        chosen_levels.append(choose_level)
        if idx == 0:
            bs_init_total = bs_init * 1.0
        else:
            bs_init_total = th.kron(bs_init_total, bs_init)
    system_init = th.kron(sp_init, bs_init_total)
    return system_init, chosen_levels

## load the data
data = np.loadtxt('../coupling_const.csv', delimiter=',')
num_modes = 9
omega_in_cm = data[:num_modes,0]
lambda_in_cm = data[:num_modes,1]
g_factor = np.sqrt(lambda_in_cm / omega_in_cm)

## simulation parameters
ns = 1  ## number of sites
batchsize = args.batchsize
dt = args.dt * Constants.fs    # max 0.1fs
sample_time = args.sample_time * Constants.fs
sample_freq = int(Constants.fs/dt)
nsteps = int(sample_time/dt)
temperature = args.temperature
out_folder = 'T{:.0f}_epsilon{:.1f}_dt{:.3f}fs'.format(temperature, epsilon, dt/Constants.fs)
if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)

## boson bath parameters
nmax = np.array([8, 3, 1, 1, 1, 1, 1, 1, 1])[:num_modes]
omega = omega_in_cm * Constants.cm_inverse_energy
spin_boson_coupling = g_factor * omega

## derived parameters
kbT = Constants.kb * temperature  
occupation = 1/(np.exp(omega/kbT)-1)
num_boson_states = np.prod(nmax+1)
total_num_states = 2 * num_boson_states

print('nmodes={}, total_num_states={}'.format(num_modes, total_num_states))
print('g factor:', g_factor)
print('omega [cm^-1]:', omega_in_cm, 'in internal units:', omega)
print('polaron binding energy [cm^-1]:', g_factor**2 * omega_in_cm, 'sum in meV:', (g_factor**2 * omega_in_cm).sum() * Constants.cm_inverse_energy / Constants.meV)
"""
initiate the system
"""
system = UnitarySystem(num_states=total_num_states, batchsize=batchsize)
 
############################## define identities ##############################
opg_bs_id_list = [IdentityBosonOperatorGroup(
    num_modes=1, id="boson_id_{}".format(i), nmax=nmax[i], batchsize=batchsize).to(device=dev)
    for i in range(num_modes)
]
opg_sp_id = IdentityPauliOperatorGroup(n_qubits=1, id="spin_identity", batchsize=batchsize).to(device=dev)

############################## add the bosonic harmonic energy term to the Hamiltonian ##############################
for i in range(num_modes):
    system_harmonic_list = [opg_sp_id]
    for j in range(num_modes):
        if i==j:
            system_harmonic_list.append(
                HarmonicOscillatorBosonOperatorGroup(num_modes=1, id="boson_harmonic_{}{}".format(i,j), batchsize=batchsize, nmax=nmax[i], omega = omega[i]).to(device=dev)
            )
        else:
            system_harmonic_list.append(opg_bs_id_list[j])
    opg_harmonic = ComposedOperatorGroups(id="harmonic_{}".format(i), OP_list = system_harmonic_list , static=True)
    system.add_operator_group_to_hamiltonian(opg_harmonic)

############################## add the electron-boson coupling term: c^+ c b
opg_sp_number = StaticPauliOperatorGroup(n_qubits=1, id="spin_number", batchsize=batchsize, coef=1.0).to(device=dev)
opg_sp_number.add_operator('N')
## add c^\dagger c (b_i + b_i^\dagger) for each mode
for i in range(num_modes):
    system_couple_list = [opg_sp_number]
    for j in range(num_modes):
        if i==j:
            opg_bs_x = StaticBosonOperatorGroup(
                num_modes=1, id="boson_x_{}{}".format(i,j), nmax=nmax[i], batchsize=batchsize, coef= spin_boson_coupling[i]).to(device=dev)   # gw(b^\dagger + b)
            opg_bs_x.add_operator('U')
            opg_bs_x.add_operator('D')
            system_couple_list.append(opg_bs_x)
        else:
            system_couple_list.append(opg_bs_id_list[j])

    opg_couple = ComposedOperatorGroups(id="system_couple_{}".format(i), OP_list=system_couple_list , static=True)
    system.add_operator_group_to_hamiltonian(opg_couple)

""" 
Set the initial state (pure state)
"""
pse_init = []
for idx in range(batchsize):
    system_init, chosen_levels = sample_init_state(epsilon=epsilon, num_modes=num_modes, nmax=nmax, omega=omega, kbT=kbT)
    pse_init.append(system_init)
    print('batch {} chosen levels:'.format(idx), chosen_levels)
pse_init = th.stack(pse_init, dim=0)
system.pse = pse_init.to(device=dev)

"""
Define the operators we want to observe
"""
## spin number operator
obs_1 = ComposedOperatorGroups(id="sp_number", OP_list = [opg_sp_number] + opg_bs_id_list, static=True)
## spin sigma_x operator
opg_sp_x = StaticPauliOperatorGroup(n_qubits=1, id="spin_x", batchsize=batchsize, coef=1.0)
opg_sp_x.add_operator('X')
obs_2 = ComposedOperatorGroups(id="sp_x", OP_list = [opg_sp_x] + opg_bs_id_list, static=True)
## spin sigma_y operator
opg_sp_y = StaticPauliOperatorGroup(n_qubits=1, id="spin_y", batchsize=batchsize, coef=1.0)
opg_sp_y.add_operator('Y')
obs_3 = ComposedOperatorGroups(id="sp_y", OP_list = [opg_sp_y] + opg_bs_id_list, static=True)
obs_list = [obs_1, obs_2, obs_3]
for obs in obs_list:
    obs.to(device=dev)

"""
Run the simulation
"""
t_list = []
sigma_n_list = []
sigma_x_list = []
sigma_y_list = []
for i in range(nsteps):
    system.step(dt=dt, time_independent = True, set_buffer=True, profile=False)
    if i%sample_freq==0:    
        system.normalize()
        observations = [system.observe(obs).cpu().numpy() for obs in obs_list]
        sigma_n_list.append(observations[0])
        sigma_x_list.append(observations[1])
        sigma_y_list.append(observations[2])
        print('t={:.3f}fs, sigma_n={}, sigma_x={}, sigma_y={}'.format(i*dt/Constants.fs, sigma_n_list[-1], sigma_x_list[-1], sigma_y_list[-1]))
        t_list.append(i*dt/Constants.fs)


"""
Post-processing
"""
sigma_n_list = np.array(sigma_n_list)  # (#samples, batchsize)
sigma_x_list = np.array(sigma_x_list)  # (#samples, batchsize)
sigma_y_list = np.array(sigma_y_list)  # (#samples, batchsize)
np.save('{}/sigma_n_run{}.npy'.format(out_folder, args.run_id), sigma_n_list)
np.save('{}/sigma_x_run{}.npy'.format(out_folder, args.run_id), sigma_x_list)
np.save('{}/sigma_y_run{}.npy'.format(out_folder, args.run_id), sigma_y_list)


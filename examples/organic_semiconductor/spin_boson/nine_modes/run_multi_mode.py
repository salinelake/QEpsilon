import numpy as np
import torch as th
import matplotlib.pyplot as plt
import qepsilon as qe
from qepsilon.utilities import compose, trace
from qepsilon.utilities import Constants_Metal as Constants
from qepsilon import LindbladSystem
from qepsilon.operator_group import *
from time import time as timer
import logging

logging.basicConfig(level=logging.INFO, filename="simulation.log")
np.set_printoptions( precision=4)
th.set_printoptions(sci_mode=False, precision=2)
dev = 'cuda'

## load the data
data = np.loadtxt('data.csv', delimiter=',')
omega_in_cm = data[:,0]
lambda_in_cm = data[:,1]

"""
This is a simple spin-boson model with a single spin coupled to a single bosonic mode.
We will initialize the system and define the Hamiltonian and Lindbladian terms.
"""
## simulation parameters
batchsize = 1
dt = 0.01 * Constants.fs  
nsteps = int(0.2 * Constants.ps / dt)

## boson bath parameters
nmax = np.array([8, 3, 1, 1, 1, 1, 1, 1, 1])
# nmax = np.array([12, 4, 1, 1, 1, 1, 1, 1, 1])
keep_modes = 9
omega_in_cm = omega_in_cm[:keep_modes]
lambda_in_cm = lambda_in_cm[:keep_modes]
nmax = nmax[:keep_modes]
num_modes = len(nmax)
assert num_modes == len(omega_in_cm) == len(lambda_in_cm)
num_boson_states = np.prod(nmax+1)
omega = Constants.speed_of_light * (omega_in_cm/ Constants.cm) * 2 * np.pi   # ps^-1

## bath relaxation
tau = 0.1 * Constants.ps  ## relaxation time of the boson bath
tau_in_ps = tau / Constants.ps
gamma = 1 / tau
kbT = Constants.kb * 300  ## room temperature: 25.8 meV, boson occupation decays to 0.01 for level 10,
occupation = 1/(np.exp(omega/kbT)-1)
occupation_finite_size = occupation - nmax / (np.exp(nmax * omega/kbT) - 1)
## spin-boson coupling
g_factor = np.sqrt(lambda_in_cm / omega_in_cm)
spin_boson_coupling = g_factor * omega 
## system parameters
num_states = 2 * num_boson_states  ## we will have one spin coupled to all the boson modes

print('total number of states:', num_states)
## initiate the system with a thermal state
system = LindbladSystem(num_states=num_states, batchsize=batchsize)
rho0_list = [th.tensor([[0.01, 0.1], [0.1, 0.99]])]
for i in range(num_modes):
    rho0_list.append(th.diag(th.exp(-th.arange(nmax[i]+1)*omega[i]/kbT)))
system.rho = compose(rho0_list)  
print('omega:', omega)
print('coupling:', spin_boson_coupling)
print('occupation number:', occupation)
"""
Define the Hamiltonian
"""
## have identity operators for the spin and boson
spin_identity = IdentityPauliOperatorGroup(n_qubits=1, id="spin_identity", batchsize=batchsize)
boson_identity_list = [
    IdentityBosonOperatorGroup(num_modes=1, id="boson_identity_{}".format(i), nmax=nmax[i], batchsize=batchsize)
    for i in range(num_modes)
]

## add the harmonic energy term 
for i in range(num_modes):
    system_harmonic_list = [spin_identity]
    for j in range(num_modes):
        if i==j:
            system_harmonic_list.append(
                HarmonicOscillatorBosonOperatorGroup(num_modes=1, id="boson_harmonic_{}{}".format(i,j), batchsize=batchsize, nmax=nmax[i], omega = omega[i])
            )
        else:
            system_harmonic_list.append(boson_identity_list[j])
    system_harmonic = ComposedOperatorGroups(id="system_harmonic_{}".format(i), OP_list=system_harmonic_list)
    system.add_operator_group_to_hamiltonian(system_harmonic)

## add the spin-boson coupling terms
couple_spin = StaticPauliOperatorGroup(n_qubits=1, id="couple_spin", batchsize=batchsize, coef=1.0)
couple_spin.add_operator('N')
## add c^\dagger c (b_i + b_i^\dagger) for each mode
for i in range(num_modes):
    system_couple_list = [couple_spin]
    for j in range(num_modes):
        if i==j:
            system_couple_list.append(
                StaticBosonOperatorGroup(num_modes=1, id="couple_boson_{}{}".format(i,j), nmax=nmax[i], batchsize=batchsize, coef=spin_boson_coupling[i])
            )
            system_couple_list[-1].add_operator('U')
            system_couple_list[-1].add_operator('D')
        else:
            system_couple_list.append(boson_identity_list[j])
    system_couple = ComposedOperatorGroups(id="system_couple_{}".format(i), OP_list=system_couple_list)
    system.add_operator_group_to_hamiltonian(system_couple)


"""
Define the Lindbladian terms
"""
## add the Lindbladian terms for maintaining a thermal bath
for i in range(num_modes):
    boson_up_jump_list = [spin_identity]
    boson_dn_jump_list = [spin_identity]
    for j in range(num_modes):
        if i==j:
            boson_up_jump_list.append(
                StaticBosonOperatorGroup(num_modes=1, id="boson_up_jump_{}{}".format(i,j), nmax=nmax[i], batchsize=batchsize, coef=np.sqrt(gamma*occupation[i]))
            )
            boson_up_jump_list[-1].add_operator('U')
            boson_dn_jump_list.append(
                StaticBosonOperatorGroup(num_modes=1, id="boson_dn_jump_{}{}".format(i,j), nmax=nmax[i], batchsize=batchsize, coef=np.sqrt(gamma*(1+occupation[i])))
            )
            boson_dn_jump_list[-1].add_operator('D')
        else:
            boson_up_jump_list.append(boson_identity_list[j])
            boson_dn_jump_list.append(boson_identity_list[j])
    boson_up_jump = ComposedOperatorGroups(id="boson_up_jump_{}".format(i), OP_list=boson_up_jump_list)
    boson_dn_jump = ComposedOperatorGroups(id="boson_dn_jump_{}".format(i), OP_list=boson_dn_jump_list)
    system.add_operator_group_to_jumping(boson_up_jump)
    system.add_operator_group_to_jumping(boson_dn_jump)

system.to(device=dev)


"""
Define the operators we want to observe
"""
obs_spin_sz = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sz", batchsize=batchsize, coef=1.0)
obs_spin_sz.add_operator('Z')
obs_system_sz = ComposedOperatorGroups(id="obs_system_sz", OP_list=[obs_spin_sz]+boson_identity_list)
obs_system_sz.to(device=dev)

obs_spin_sx = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sx", batchsize=batchsize, coef=1.0)
obs_spin_sx.add_operator('X')
obs_system_sx = ComposedOperatorGroups(id="obs_system_sx", OP_list=[obs_spin_sx]+boson_identity_list)
obs_system_sx.to(device=dev)

obs_spin_sy = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sy", batchsize=batchsize, coef=1.0)
obs_spin_sy.add_operator('Y')
obs_system_sy = ComposedOperatorGroups(id="obs_system_sy", OP_list=[obs_spin_sy]+boson_identity_list)
obs_system_sy.to(device=dev)


obs_spin_excited = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_excited", batchsize=batchsize, coef=0.5)
obs_spin_excited.add_operator('Z')
obs_spin_excited.add_operator('I')
obs_system_excited = ComposedOperatorGroups(id="obs_system_excited", OP_list=[obs_spin_excited]+boson_identity_list)
obs_system_excited.to(device=dev)

obs_boson_N_list = []
for i in range(num_modes):
    obs_boson_N_op_list = [spin_identity]
    for j in range(num_modes):
        if i==j:
            obs_boson_N_op_list.append(
                StaticBosonOperatorGroup(num_modes=1, id="obs_boson_N_{}{}".format(i,j), nmax=nmax[j], batchsize=batchsize, coef=1.0)
            )
            obs_boson_N_op_list[-1].add_operator('N')
        else:
            obs_boson_N_op_list.append(boson_identity_list[j])
    obs_boson_N_list.append(
        ComposedOperatorGroups(id="obs_boson_N_{}".format(i), OP_list=obs_boson_N_op_list)
        )
    obs_boson_N_list[-1].to(device=dev)

"""
Run the simulation
"""
ne_list = []
t_list = []
sx_list = []
sy_list = []
# sz_list = []
for i in range(nsteps):
    system.step(dt=dt, set_buffer=False, time_independent=True, profile=False)
    if i%int(Constants.fs/dt)==0:    
        # sigma_z = system.observe(obs_system_sz).cpu().numpy().mean()
        sigma_x = system.observe(obs_system_sx).cpu().numpy().mean()
        sigma_y = system.observe(obs_system_sy).cpu().numpy().mean()
        ne = system.observe(obs_system_excited).cpu().numpy().mean()
        boson_N_list = []
        for obs in obs_boson_N_list:
            boson_N_list.append(system.observe(obs).cpu().numpy().mean())
        boson_N_list = np.array(boson_N_list)
        print('t={:.2f}fs, sigma_x={:.3f}, ne={:.3f}'.format(i*dt/Constants.fs,  sigma_x, ne))
        print('boson_N (target(infinite size)={})={}'.format(occupation, boson_N_list))
        ne_list.append(ne)
        t_list.append(i*dt)
        sx_list.append(sigma_x)
        sy_list.append(sigma_y)
        # sz_list.append(sigma_z)

t_list = np.array(t_list)
sx_list = np.array(sx_list)
sy_list = np.array(sy_list)
np.save('sx_tau{:.2f}ps_{:d}modes_dt{:.0f}as.npy'.format(tau_in_ps, num_modes, dt/Constants.As), sx_list)
np.save('sy_tau{:.2f}ps_{:d}modes_dt{:.0f}as.npy'.format(tau_in_ps, num_modes, dt/Constants.As), sy_list)
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(t_list / Constants.fs, ne_list, label='Sim')
ax[0].set_xlabel('t [fs]')
ax[0].set_ylabel(r'$N_e/N$')
ax[0].legend()

ax[1].plot(t_list / Constants.fs, sx_list, label='Sx')
ax[1].plot(t_list / Constants.fs, sy_list, label='Sy')
# ax[1].plot(t_list / Constants.fs, sz_list, label='Sz')
ax[1].set_xlabel('t [fs]')
ax[1].set_ylabel(r'$\langle \sigma_i \rangle$')
ax[1].legend()
plt.savefig('ne_tau{:.2f}ps_{:d}modes_start0.2.png'.format(tau_in_ps, num_modes), dpi=300)

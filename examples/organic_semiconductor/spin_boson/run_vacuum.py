import numpy as np
import torch as th
import matplotlib.pyplot as plt
import qepsilon as qe
from qepsilon.utilities import *
from qepsilon import LindbladSystem
from qepsilon.operator_group import *
th.set_printoptions(sci_mode=False, precision=2)
dev = 'cpu'

"""
This is a simple spin-boson model with a single spin coupled to a single bosonic mode.
We will initialize the system and define the Hamiltonian and Lindbladian terms.
"""
## simulation parameters
batchsize = 1
dt = 0.001
## boson bath parameters
num_modes = 1
omega = 1.1
nmax = 6
num_boson_states = (nmax+1)**num_modes
## spin parameters
detuning = 0.1
spin_omega = omega - detuning
spin_boson_coupling = 1
gamma = 0.0  ## 1/tau for jumping rate
vacuum_rabi_freq = np.sqrt(detuning**2 + spin_boson_coupling**2)
## system parameters
num_states = 2 * num_boson_states  ## we will have one spin coupled to all the boson modes
# kbT = 1.0 / np.log(10)  ## target relative distribution: 1, 0.1, ....
kbT = 0.1
occupation = 1/(np.exp(omega/kbT)-1)

## initiate the system with a pure state
system = LindbladSystem(num_states=num_states, batchsize=batchsize)
# rho0 = th.zeros((num_states, num_states), dtype=th.cfloat)
# rho0[0, 0] = 1
rho0_boson = th.diag(th.exp(-th.arange(num_boson_states)*omega/kbT))
rho0_boson = rho0_boson / trace(rho0_boson)
rho0_spin = th.tensor([[1.0, 0.0], [0.0, 0.0]])
rho0 = th.kron(rho0_spin, rho0_boson)
system.rho=rho0  

"""
Define the Hamiltonian
"""
## have identity operators for the spin and boson
spin_identity = IdentityPauliOperatorGroup(n_qubits=1, id="spin_identity", batchsize=batchsize)
boson_identity = IdentityBosonOperatorGroup(num_modes=num_modes, id="boson_identity", nmax=nmax, batchsize=batchsize)

## add the harmonic energy term
boson_harmonic = HarmonicOscillatorBosonOperatorGroup(num_modes=num_modes, id="boson_harmonic", batchsize=batchsize, nmax=nmax, omega= th.ones(num_modes)*omega)
system_harmonic = ComposedOperatorGroups(id="system_harmonic", OP1=spin_identity, OP2=boson_harmonic)
system.add_operator_group_to_hamiltonian(system_harmonic)

## add the spin TLS term
spin_sz = StaticPauliOperatorGroup(n_qubits=1, id="spin_sz", batchsize=batchsize, coef=spin_omega/2.0)
spin_sz.add_operator('Z')
system_sz = ComposedOperatorGroups(id="system_sz", OP1=spin_sz, OP2=boson_identity)
system.add_operator_group_to_hamiltonian(system_sz)

## add the spin-boson coupling term: c^\dagger b
couple_up_spin = StaticPauliOperatorGroup(n_qubits=1, id="couple_up_spin", batchsize=batchsize, coef=spin_boson_coupling/2.0)
couple_up_spin.add_operator('U')
couple_down_boson = StaticBosonOperatorGroup(num_modes=num_modes, id="couple_down_boson", nmax=nmax, batchsize=batchsize, coef=1.0)
couple_down_boson.add_operator('D')
system_couple_UD = ComposedOperatorGroups(id="system_couple_UD", OP1=couple_up_spin, OP2=couple_down_boson)
system.add_operator_group_to_hamiltonian(system_couple_UD)

## add the spin-boson coupling term: c b^\dagger
couple_down_spin = StaticPauliOperatorGroup(n_qubits=1, id="couple_down_spin", batchsize=batchsize, coef=spin_boson_coupling/2.0)
couple_down_spin.add_operator('D')
couple_up_boson = StaticBosonOperatorGroup(num_modes=num_modes, id="couple_up_boson", nmax=nmax, batchsize=batchsize, coef=1.0)
couple_up_boson.add_operator('U')
system_couple_DU = ComposedOperatorGroups(id="system_couple_DU", OP1=couple_down_spin, OP2=couple_up_boson)
system.add_operator_group_to_hamiltonian(system_couple_DU)

"""
Define the Lindbladian terms
"""
## add the Lindbladian terms for maintaining a thermal bath
boson_up_jump = StaticBosonOperatorGroup(num_modes=num_modes, id="boson_up_jump", nmax=nmax, batchsize=batchsize, coef=np.sqrt(gamma*occupation))
boson_up_jump.add_operator('U')
system_up_jump = ComposedOperatorGroups(id="system_up_jump", OP1=spin_identity, OP2=boson_up_jump)
system.add_operator_group_to_jumping(system_up_jump)

boson_down_jump = StaticBosonOperatorGroup(num_modes=num_modes, id="boson_down_jump", nmax=nmax, batchsize=batchsize, coef=np.sqrt(gamma*(1+occupation)))
boson_down_jump.add_operator('D')
system_down_jump = ComposedOperatorGroups(id="system_down_jump", OP1=spin_identity, OP2=boson_down_jump)
system.add_operator_group_to_jumping(system_down_jump)

"""
Define the operators we want to observe
"""
obs_spin_sz = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sz", batchsize=batchsize, coef=1.0)
obs_spin_sz.add_operator('Z')
obs_system_sz = ComposedOperatorGroups(id="obs_system_sz", OP1=obs_spin_sz, OP2=boson_identity)

obs_spin_excited = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_excited", batchsize=batchsize, coef=0.5)
obs_spin_excited.add_operator('Z')
obs_spin_excited.add_operator('I')
obs_system_excited = ComposedOperatorGroups(id="obs_system_excited", OP1=obs_spin_excited, OP2=boson_identity)

obs_boson_N = StaticBosonOperatorGroup(num_modes=num_modes, id="obs_boson_N", nmax=nmax, batchsize=batchsize, coef=1.0)
obs_boson_N.add_operator('N')
obs_system_N = ComposedOperatorGroups(id="obs_system_N", OP1=spin_identity, OP2=obs_boson_N)

"""
Run the simulation
"""
ne_list = []
t_list = []
for i in range(20000):
    system.step(dt=dt, set_buffer=False)
    system.normalize()
    if i%100==0:    
        sigma_z = system.observe(obs_system_sz).numpy().mean()
        ne = system.observe(obs_system_excited).numpy().mean()
        boson_N = system.observe(obs_system_N).numpy().mean()
        print('t={:.3f}, sigma_z={:.3f}, ne={:.3f}'.format(i*dt, sigma_z, ne))
        print('boson_N (target={:.3f})={:.3f}'.format(occupation, boson_N))
        ne_list.append(ne)
        t_list.append(i*dt)

t_list = np.array(t_list)
fig, ax = plt.subplots()
ax.plot(t_list, ne_list, label='Sim')
ax.set_xlabel('t')
ax.set_ylabel(r'$N_e/N$')
ax.plot(t_list,  0.5 + 0.5 * np.cos(vacuum_rabi_freq*t_list), '--', color='gray', label='Ref')
ax.legend()
plt.savefig('ne_t.png')

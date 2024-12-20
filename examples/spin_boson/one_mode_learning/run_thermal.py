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
dt = 0.005 * Constants.fs  ## in unit of us
nsteps = int(Constants.ps / dt)
## spin parameters
spin_omega = 0

## boson bath parameters
num_modes = 1
nmax = 10
num_boson_states = (nmax+1)**num_modes
omega_in_cm = 84 # cm^-1 => ~10meV  ~ 0.4ps
omega = Constants.speed_of_light * (omega_in_cm/ Constants.cm) * 2 * np.pi   # us^-1

## bath relaxation
tau = 0.1 * Constants.ps  ## relaxation time of the boson bath
tau_in_ps = tau / Constants.ps
gamma = 1 / tau
kbT = Constants.kb * 300  ## room temperature: 25.8 meV, boson occupation decays to 0.01 for level 10,
occupation = 1/(np.exp(omega/kbT)-1)

## spin-boson coupling
g_factor = np.sqrt(75 / omega_in_cm)
spin_boson_coupling = g_factor * omega
## system parameters
num_states = 2 * num_boson_states  ## we will have one spin coupled to all the boson modes

## initiate the system with a thermal state
system = LindbladSystem(num_states=num_states, batchsize=batchsize)
rho0_boson = th.diag(th.exp(-th.arange(num_boson_states)*omega/kbT))
rho0_boson = rho0_boson / trace(rho0_boson)
rho0_spin = th.tensor([[0.5, 0.5], [0.5, 0.5]])
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

## add the spin-boson coupling term: c^\dagger b
couple_spin = StaticPauliOperatorGroup(n_qubits=1, id="couple_spin", batchsize=batchsize, coef=spin_boson_coupling)
couple_spin.add_operator('N')
couple_boson = StaticBosonOperatorGroup(num_modes=num_modes, id="couple_boson", nmax=nmax, batchsize=batchsize, coef=1.0)
couple_boson.add_operator('U')
couple_boson.add_operator('D')
system_couple = ComposedOperatorGroups(id="system_couple", OP1=couple_spin, OP2=couple_boson)
system.add_operator_group_to_hamiltonian(system_couple)
 
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

obs_spin_sx = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sx", batchsize=batchsize, coef=1.0)
obs_spin_sx.add_operator('X')
obs_system_sx = ComposedOperatorGroups(id="obs_system_sx", OP1=obs_spin_sx, OP2=boson_identity)

obs_spin_excited = StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_excited", batchsize=batchsize, coef=1)
obs_spin_excited.add_operator('N')
obs_system_excited = ComposedOperatorGroups(id="obs_system_excited", OP1=obs_spin_excited, OP2=boson_identity)

obs_boson_N = StaticBosonOperatorGroup(num_modes=num_modes, id="obs_boson_N", nmax=nmax, batchsize=batchsize, coef=1.0)
obs_boson_N.add_operator('N')
obs_system_N = ComposedOperatorGroups(id="obs_system_N", OP1=spin_identity, OP2=obs_boson_N)

"""
Run the simulation
"""
t_list = []
ne_list = []
sx_list = []
sz_list = []
for i in range(nsteps):
    system.step(dt=dt, set_buffer=False)
    system.normalize()
    if i%100==0:    
        sigma_z = system.observe(obs_system_sz).numpy().mean()
        sigma_x = system.observe(obs_system_sx).numpy().mean()
        ne = system.observe(obs_system_excited).numpy().mean()
        boson_N = system.observe(obs_system_N).numpy().mean()
        print('t={:.3f}ps, sigma_z={:.3f}, sigma_x={:.3f}, ne={:.3f}'.format(i*dt*1e6, sigma_z, sigma_x, ne))
        print('boson_N (target={:.3f})={:.3f}'.format(occupation, boson_N))
        ne_list.append(ne)
        t_list.append(i*dt)
        sx_list.append(sigma_x)
        sz_list.append(sigma_z)

t_list = np.array(t_list)
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(t_list / Constants.fs, ne_list, label='Sim')
ax[0].set_xlabel('t [fs]')
ax[0].set_ylabel(r'$N_e/N$')
ax[0].legend()

ax[1].plot(t_list / Constants.fs, sx_list, label='Sx')
ax[1].plot(t_list / Constants.fs, sz_list, label='Sz')
ax[1].set_xlabel('t [fs]')
ax[1].set_ylabel(r'$\langle \sigma_i \rangle$')
ax[1].legend()
plt.savefig('ne_tau{:.2f}ps_nmax{:d}.png'.format(tau_in_ps, nmax), dpi=300)

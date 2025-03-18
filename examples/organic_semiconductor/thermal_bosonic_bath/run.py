import numpy as np
import torch as th
import qepsilon as qe
from qepsilon.utilities import *
from qepsilon import LindbladSystem
from qepsilon.operator_group import *
th.set_printoptions(sci_mode=False, precision=5)
dev = 'cpu'


batchsize = 100
num_modes = 1
nmax = 10
num_states = (nmax+1)**num_modes
dt = 0.001
omega = 1.0
kbT = 1.0 / np.log(10)  ## target relative distribution: 1, 0.1, ....
gamma = 0.1
occupation = 1/(np.exp(omega/kbT)-1)

system = LindbladSystem(num_states=num_states, batchsize=batchsize)
rho0 = th.zeros((num_states, num_states), dtype=th.cfloat)
rho0[0, 0] = 1
system.rho=rho0  ## set the initial state to be vacuum

harmonic = HarmonicOscillatorBosonOperatorGroup(num_modes=num_modes, id="harmonic", batchsize=batchsize, nmax=nmax, omega= th.ones(num_modes)*omega)
system.add_operator_group_to_hamiltonian(harmonic)

up_jump = StaticBosonOperatorGroup(num_modes=num_modes, id="up_jump", nmax=nmax, batchsize=batchsize, coef=np.sqrt(gamma*occupation), requires_grad=False)
up_jump.add_operator('U')
system.add_operator_group_to_jumping(up_jump)

down_jump = StaticBosonOperatorGroup(num_modes=num_modes, id="down_jump", nmax=nmax, batchsize=batchsize, coef=np.sqrt(gamma*(1+occupation)), requires_grad=False)
down_jump.add_operator('D')
system.add_operator_group_to_jumping(down_jump)


for i in range(100000):
    system.step(dt=dt, set_buffer=False)
    system.normalize()
    if i%100==0:    
        prob = th.diag(system.rho.mean(0)).real
        print('t={:.1f}, prob={}'.format(i*dt, prob / prob[0] ))




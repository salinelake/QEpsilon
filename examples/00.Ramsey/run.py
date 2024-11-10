import numpy as np
import torch as th
import qepsilon as qe
from qepsilon import StaticPauliOperatorGroup, ColorNoisePauliOperatorGroup
## do not print the tensor in scientific notation
th.set_printoptions(sci_mode=False)

## define the noisy terms in the Hamiltonian
sx = ColorNoisePauliOperatorGroup(n_qubits=1, id="sx_noise", damping=1/20, amp=1, requires_grad=False)
sx.add_operator('X')
sy = ColorNoisePauliOperatorGroup(n_qubits=1, id="sy_noise", damping=1/20, amp=1, requires_grad=False)
sy.add_operator('Y')

sx_jump = StaticPauliOperatorGroup(n_qubits=1, id="sx_jump", coef=1/250, requires_grad=False)
sx_jump.add_operator('X')
sy_jump = StaticPauliOperatorGroup(n_qubits=1, id="sy_jump", coef=1/250, requires_grad=False)
sy_jump.add_operator('Y')

## define the system
qubit = qe.LindbladSystem(n_qubits=1)
qubit.add_operator_group_to_hamiltonian(sx)
qubit.add_operator_group_to_hamiltonian(sy)
qubit.add_operator_group_to_jumping(sx_jump)
qubit.add_operator_group_to_jumping(sy_jump)

## Simulation parameters
dt = 0.01
T = 2.5
nsteps = int(T / dt)
n_epochs = 1000
theta_list = np.array([0, 60, 120, 180]) * np.pi / 180
P0_list = []

for theta in theta_list:
    P0 = []
    for epoch in range(n_epochs):
        ## set the initial state
        qubit.set_rho_by_config([0])
        # first pi/2 rotation along x
        qubit.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)

        # free evolution
        for i in range(nsteps):
            qubit.step(dt=dt)

        # second pi/2 rotation along x
        u = th.tensor([np.cos(theta), np.sin(theta), 0.0], dtype=th.float)
        qubit.rotate(direction=u, angle=np.pi/2)
        prob_0 = qubit.density_matrix.observe_prob_by_config(qubit.rho, th.tensor([0]))
        P0.append(prob_0)
    P0_list.append(th.tensor(P0).mean())

print(f"Ramsey Fringes for T={T}, theta = {theta_list} are {P0_list}")
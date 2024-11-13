import numpy as np
import torch as th
import qepsilon as qe
from qepsilon import WhiteNoisePauliOperatorGroup, ColorNoisePauliOperatorGroup, StaticPauliOperatorGroup
## do not print the tensor in scientific notation
th.set_printoptions(sci_mode=False)

################################################
#  define the system
################################################
batchsize = 400
qubit = qe.LindbladSystem(n_qubits=1, batchsize=batchsize)

################################################
# define the noisy terms in the Hamiltonian
################################################
sx = ColorNoisePauliOperatorGroup(n_qubits=1, id="sx_noise", batchsize=batchsize, damping=1/20, amp=0.1, requires_grad=True)
# sx = WhiteNoisePauliOperatorGroup(n_qubits=1, id="sx_noise", batchsize=batchsize,  amp=0.1, requires_grad=True)
sx.add_operator('X')
qubit.add_operator_group_to_hamiltonian(sx)

sy = ColorNoisePauliOperatorGroup(n_qubits=1, id="sy_noise", batchsize=batchsize, damping=1/20, amp=0.1, requires_grad=True)
# sy = WhiteNoisePauliOperatorGroup(n_qubits=1, id="sy_noise", batchsize=batchsize,  amp=0.1, requires_grad=True)
sy.add_operator('Y')
qubit.add_operator_group_to_hamiltonian(sy)

################################################
# define the jump operators
################################################
sx_jump = StaticPauliOperatorGroup(n_qubits=1, id="sx_jump", batchsize=batchsize, coef=np.sqrt(0.002), requires_grad=False)
sx_jump.add_operator('X')
qubit.add_operator_group_to_jumping(sx_jump)

sz_jump = StaticPauliOperatorGroup(n_qubits=1, id="sz_jump", batchsize=batchsize, coef=np.sqrt(0.002), requires_grad=False)
sz_jump.add_operator('Z')
qubit.add_operator_group_to_jumping(sz_jump)
################################################
# Simulation parameters
################################################
dt = 0.1
T = 2.5
nsteps = int(T / dt)
theta_list = np.array([0, 60, 120, 180]) * np.pi / 180
P0_list_ref = [0.3, 0.4, 0.5, 0.6] 
################################################
#  train
################################################
nepoch = 100
for epoch in range(nepoch):
    P0_list = []
    for theta in theta_list:
        sx.reset_history()
        sy.reset_history()
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
        # observe the probability of being in the state |0>
        prob_0 = qubit.density_matrix.observe_prob_by_config(qubit.rho, th.tensor([0]))
        P0_list.append(prob_0.mean())
    print(f"Epoch={epoch}, Ramsey Fringes for T={T}, theta = {th.tensor(theta_list, dtype=th.float)} are {th.tensor(P0_list, dtype=th.float).detach()}")
    print(f'Sx Damping={sx.damping.mean()}, Amp={sx.amp.mean()}, Sy Damping={sy.damping.mean()}, Amp={sy.amp.mean()}')


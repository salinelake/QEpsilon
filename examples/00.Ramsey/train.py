import numpy as np
import torch as th
import qepsilon as qe
from qepsilon import WhiteNoisePauliOperatorGroup,  StaticPauliOperatorGroup, ColorNoisePauliOperatorGroup
from qepsilon.task import *
from matplotlib import pyplot as plt
from time import time as timer  

## do not print the tensor in scientific notation
th.set_printoptions(sci_mode=False, precision=3)
dev = 'cpu'
################################################
#  define the system
################################################
batchsize = 1000
qubit = qe.LindbladSystem(n_qubits=1, batchsize=batchsize).to(dev)
qubit.set_rho_by_config([0])
qubit.to(dev)

################################################
# define the terms in the Hamiltonian
################################################
sx = ColorNoisePauliOperatorGroup(n_qubits=1, id="sx_noise", batchsize=batchsize, tau=5, amp=0.2, requires_grad=True).to(dev)
sx.add_operator('X')
qubit.add_operator_group_to_hamiltonian(sx)

sy = ColorNoisePauliOperatorGroup(n_qubits=1, id="sy_noise", batchsize=batchsize, tau=5, amp=0.2, requires_grad=True).to(dev)
sy.add_operator('Y')
qubit.add_operator_group_to_hamiltonian(sy)

sz = ColorNoisePauliOperatorGroup(n_qubits=1, id="sz_noise", batchsize=batchsize, tau=5, amp=0.2, requires_grad=True).to(dev)
sz.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz)

################################################
# define the jump operators
################################################
sx_jump = StaticPauliOperatorGroup(n_qubits=1, id="sx_jump", batchsize=batchsize, coef=np.sqrt(0.002), requires_grad=True).to(dev)
sx_jump.add_operator('X')
qubit.add_operator_group_to_jumping(sx_jump)

sz_jump = StaticPauliOperatorGroup(n_qubits=1, id="sz_jump", batchsize=batchsize, coef=np.sqrt(0.002), requires_grad=True).to(dev)
sz_jump.add_operator('Z')
qubit.add_operator_group_to_jumping(sz_jump)
################################################
# Experimental data
################################################
preperation_rate = 0.824
data_plain = th.tensor([[0.260, 0.709],[0.780, 0.571],[1.300, 0.515],[1.820, 0.429],[2.340, 0.275],[2.860, 0.187]]).to(dev)
data_plain[:, 1] /= preperation_rate

data_echo = th.tensor([ [1.820, 0.752],[4.939, 0.712],[9.619, 0.605],[14.818, 0.501],[19.757, 0.384],[24.957, 0.299],[29.636, 0.315],[34.835, 0.219],[40.035, 0.160]]).to(dev)
data_echo[:, 1] /= preperation_rate

data_XY8 = th.tensor([ [3.120, 0.760],[16.378, 0.680],[32.756, 0.597],[49.393, 0.533],[82.409, 0.544],[115.165, 0.424],[148.180, 0.339]]).to(dev)
data_XY8[:, 1] /= preperation_rate


fringe_blue_t3 = th.tensor([[28.076, 1.048],[36.915, 0.925],[43.674, 0.840],[50.433, 0.808],[57.192, 0.880],[64.211, 0.965],[72.790, 1.053]]).to(dev)
fringe_blue_t3[:, 1] /= preperation_rate
fringe_blue_t148 = th.tensor([[441.628, 0.169],[510.698, 0.051],[565.116, -0.076],[621.628, -0.161],[676.047, -0.127],[730.465, 0.144],[799.535, 0.153]]).to(dev)
fringe_blue_t148[:, 1] /= preperation_rate
XY8_cycle_time = 1.6
################################################
#  train
################################################
nepoch = 100
optimizer = th.optim.Adam(qubit.HamiltonianParameters() + qubit.JumpingParameters(), lr=0.01)
for epoch in range(nepoch):
    print(f"Epoch={epoch}")
    print(f'Hamiltonian: Sx Tau={sx.tau.mean()}ms, Amp={sx.amp.mean()}, Sy Tau={sy.tau.mean()}ms, Amp={sy.amp.mean()}')
    print(f'Jump: Sx Amp={sx_jump.coef.mean()}, Sz Amp={sz_jump.coef.mean()}')
    loss = 0

    ## Ramsey experiment without echo
    Ramsey_Plain_P0 = RamseyScan(qubit, dt=0.01, T=3, theta_list=[0, np.pi], observe_at=data_plain[:, 0])
    Ramsey_Plain_Contrast = th.abs(Ramsey_Plain_P0[:, 1] - Ramsey_Plain_P0[:, 0])
    loss += ((Ramsey_Plain_Contrast - data_plain[:, 1]) ** 2).mean()
    print(f"Ramsey Plain Data={data_plain[:, 1]}")
    print(f"Ramsey Plain Simulated={Ramsey_Plain_Contrast.detach()}")

    ## Ramsey experiment with echo
    Ramsey_Echo_Contrast = []
    for T in data_echo[:, 0]:
        Ramsey_Echo_P0 = RamseySpinEcho(qubit, dt=0.1, T=T, theta_list=[0, np.pi])
        Ramsey_Echo_Contrast.append(th.abs(Ramsey_Echo_P0[1] - Ramsey_Echo_P0[0]))
    Ramsey_Echo_Contrast = th.stack(Ramsey_Echo_Contrast)
    loss += ((Ramsey_Echo_Contrast - data_echo[:, 1]) ** 2).mean()
    print(f"Ramsey Echo Data={data_echo[:, 1]}")
    print(f"Ramsey Echo Simulated={Ramsey_Echo_Contrast.detach()}")

    ## Ramsey experiment with XY8 sequence
    # Ramsey_XY8_Contrast = []
    # for T in data_XY8[:, 0]:
    #     Ramsey_XY8_P0 = RamseyXY8(qubit, dt=0.1, T=T, cycle_time=XY8_cycle_time, theta_list=[0, np.pi])
    #     Ramsey_XY8_Contrast.append(th.abs(Ramsey_XY8_P0[1] - Ramsey_XY8_P0[0]))
    # Ramsey_XY8_Contrast = th.stack(Ramsey_XY8_Contrast)
    Ramsey_XY8_P0 = RamseyScan_XY8(qubit, dt=0.1, T=150, cycle_time=XY8_cycle_time, theta_list=[0, np.pi], observe_at=data_XY8[:, 0])
    Ramsey_XY8_Contrast = th.abs(Ramsey_XY8_P0[:, 1] - Ramsey_XY8_P0[:, 0])
    loss += ((Ramsey_XY8_Contrast - data_XY8[:, 1]) ** 2).mean()
    print(f"Ramsey XY8 Data={data_XY8[:, 1]}")
    print(f"Ramsey XY8 Simulated={Ramsey_XY8_Contrast.detach()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(data_plain[:, 0], Ramsey_Plain_Contrast.detach().numpy() * preperation_rate, label='Simulated', color='green')
ax.plot(data_plain[:, 0], data_plain[:, 1] * preperation_rate, marker='^', color='green', label='Data', linestyle='None') # triangle marker
ax.plot(data_echo[:, 0], Ramsey_Echo_Contrast.detach().numpy() * preperation_rate, label='Simulated', color='red')
ax.plot(data_echo[:, 0], data_echo[:, 1] * preperation_rate, marker='s', color='red', label='Data', linestyle='None') # square marker
ax.plot(data_XY8[:, 0], Ramsey_XY8_Contrast.detach().numpy() * preperation_rate, label='Simulated', color='blue')
ax.plot(data_XY8[:, 0], data_XY8[:, 1] * preperation_rate, marker='o', color='blue', label='Data', linestyle='None') # circle marker
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Contrast')
ax.legend()
plt.show()
plt.tight_layout()
plt.savefig('ramsey_contrast.png')

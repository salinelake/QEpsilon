import numpy as np
import torch as th
import qepsilon as qe
from qepsilon import *
from qepsilon.task import *
from matplotlib import pyplot as plt
import logging
import time

log_suffix = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=f'log_{log_suffix}.log', level=logging.INFO)

## do not print the tensor in scientific notation
th.set_printoptions(sci_mode=False, precision=3)
dev = 'cpu'
################################################
#  define the system
################################################
batchsize = 1000
qubit = qe.QubitLindbladSystem(n_qubits=1, batchsize=batchsize).to(dev)
qubit.set_rho_by_config([0])
qubit.to(dev)

################################################
# define the terms in the Hamiltonian
################################################
# sx = LangevinNoisePauliOperatorGroup(n_qubits=1, id="sx_noise", batchsize=batchsize, tau=5, amp=0.2, requires_grad=True).to(dev)
# sx.add_operator('X')
# qubit.add_operator_group_to_hamiltonian(sx)

# sy = LangevinNoisePauliOperatorGroup(n_qubits=1, id="sy_noise", batchsize=batchsize, tau=5, amp=0.2, requires_grad=True).to(dev)
# sy.add_operator('Y')
# qubit.add_operator_group_to_hamiltonian(sy)
sz_shot = ShotbyShotNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_shot", batchsize=batchsize, amp=0.1, requires_grad=True).to(dev)
sz_shot.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz_shot)

sz0 = LangevinNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_color", batchsize=batchsize, tau=20, amp=0.01, requires_grad=True).to(dev)
sz0.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz0)

sz1 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_60hz", batchsize=batchsize, tau=(1000/60 ), amp=0.01, requires_grad=True).to(dev)
sz1.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz1)

sz2 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_120hz", batchsize=batchsize, tau=(1000/120), amp=0.01, requires_grad=True).to(dev)
sz2.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz2)

sz3 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_180hz", batchsize=batchsize, tau=(1000/180), amp=0.01, requires_grad=True).to(dev)
sz3.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz3)

sz4 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_240hz", batchsize=batchsize, tau=(1000/240), amp=0.01, requires_grad=True).to(dev)
sz4.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz4)

################################################
# define the jump operators
################################################
sx_jump = StaticPauliOperatorGroup(n_qubits=1, id="sx_jump", batchsize=batchsize, coef=np.sqrt(0.0001), requires_grad=True).to(dev)
sx_jump.add_operator('X')
qubit.add_operator_group_to_jumping(sx_jump)

# sy_jump = StaticPauliOperatorGroup(n_qubits=1, id="sy_jump", batchsize=batchsize, coef=np.sqrt(0.0001), requires_grad=True).to(dev)
# sy_jump.add_operator('Y')
# qubit.add_operator_group_to_jumping(sy_jump)

sz_jump = StaticPauliOperatorGroup(n_qubits=1, id="sz_jump", batchsize=batchsize, coef=np.sqrt(0.0001), requires_grad=True).to(dev)
sz_jump.add_operator('Z')
qubit.add_operator_group_to_jumping(sz_jump)

################################################
# define the error channel of each pulse
################################################
depol_channel = DepolarizationChannel(n_qubits=1, id="depol_channel", batchsize=batchsize, p=0.002, requires_grad=True).to(dev)
qubit.add_operator_group_to_channel(depol_channel)

################################################
# Experimental data
################################################
preperation_rate = 0.8 #0.824
## load csv data
data_plain = np.loadtxt('./Data/Fig3C_GreenTriangles.csv', delimiter=',', skiprows=1)
data_plain = th.tensor(data_plain, dtype=th.float).to(dev)
data_plain[:, 1] /= preperation_rate
data_plain = data_plain[1:]

data_echo = np.loadtxt('./Data/Fig3C_RedSquares.csv', delimiter=',', skiprows=1)
data_echo = th.tensor(data_echo, dtype=th.float).to(dev)
data_echo[:, 1] /= preperation_rate
data_echo = data_echo[1:]


data_XY8 = np.loadtxt('./Data/Fig3C_BlueCircles.csv', delimiter=',', skiprows=1)
data_XY8 = th.tensor(data_XY8, dtype=th.float).to(dev)
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
# optimizer = th.optim.Adam(qubit.HamiltonianParameters() + qubit.JumpingParameters() + qubit.ChannelParameters(), lr=0.03)
## set learning rate by group
optimizer = th.optim.Adam([{'params': qubit.HamiltonianParameters(), 'lr': 0.1},
                          {'params': qubit.JumpingParameters(), 'lr': 0.001},
                          {'params': qubit.ChannelParameters(), 'lr': 0.1}])
for epoch in range(nepoch):
    logging.info(f"Epoch={epoch}")
    for op in qubit.hamiltonian_operator_groups:
        if op.tau is not None:
            logging.info(f'Hamiltonian: {op.id} Tau={op.tau.mean()}ms, Amp={op.amp.mean()}')
        else:
            logging.info(f'Hamiltonian: {op.id} Amp={op.amp.mean()}')
    for op in qubit.jumping_operator_groups:
        logging.info(f'Jump: {op.id} Amp={op.coef.mean()}')
    logging.info(f"depol_channel: p={depol_channel.p.mean()}")
    loss = 0

    ## Ramsey experiment without echo
    Ramsey_Plain_P0 = RamseyScan(qubit, dt=0.01, T=3, theta_list=[0, np.pi], observe_at=data_plain[:, 0])
    Ramsey_Plain_Contrast = th.abs(Ramsey_Plain_P0[:, 1] - Ramsey_Plain_P0[:, 0])
    loss += ((Ramsey_Plain_Contrast - data_plain[:, 1]) ** 2).mean()
    logging.info(f"Ramsey Plain Data={data_plain[:, 1]}")
    logging.info(f"Ramsey Plain Simulated={Ramsey_Plain_Contrast.detach()}")

    ## Ramsey experiment with echo
    Ramsey_Echo_Contrast = []
    for T in data_echo[:, 0]:
        Ramsey_Echo_P0 = RamseySpinEcho(qubit, dt=0.05, T=T, theta_list=[0, np.pi])
        Ramsey_Echo_Contrast.append(th.abs(Ramsey_Echo_P0[1] - Ramsey_Echo_P0[0]))
    Ramsey_Echo_Contrast = th.stack(Ramsey_Echo_Contrast)
    loss += ((Ramsey_Echo_Contrast - data_echo[:, 1]) ** 2).mean()
    logging.info(f"Ramsey Echo Data={data_echo[:, 1]}")
    logging.info(f"Ramsey Echo Simulated={Ramsey_Echo_Contrast.detach()}")

    ## Ramsey experiment with XY8 sequence
    Ramsey_XY8_P0 = RamseyScan_XY8(qubit, dt=0.1, T=150, cycle_time=XY8_cycle_time, theta_list=[0, np.pi], observe_at=data_XY8[:, 0])
    Ramsey_XY8_Contrast = th.abs(Ramsey_XY8_P0[:, 1] - Ramsey_XY8_P0[:, 0])
    loss += ((Ramsey_XY8_Contrast - data_XY8[:, 1]) ** 2).mean()
    logging.info(f"Ramsey XY8 Data={data_XY8[:, 1]}")
    logging.info(f"Ramsey XY8 Simulated={Ramsey_XY8_Contrast.detach()}")
    logging.info(f'loss = {loss}')
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
plt.savefig(f'ramsey_contrast_{log_suffix}.png')

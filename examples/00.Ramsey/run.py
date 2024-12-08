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
qubit = qe.LindbladSystem(n_qubits=1, batchsize=batchsize).to(dev)
qubit.set_rho_by_config([0])
qubit.to(dev)

################################################
# define the terms in the Hamiltonian
################################################
sz_shot = ShotbyShotNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_shot", batchsize=batchsize, amp=0.28886616230010986e-3, requires_grad=False).to(dev)
sz_shot.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz_shot)

sz0 = ColorNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_color", batchsize=batchsize, tau=3.5981204509735107e3, amp=0.05603799596428871e-3, requires_grad=False).to(dev)
sz0.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz0)

sz1 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_60hz", batchsize=batchsize, tau=(1e6/60), amp=0.0365261472761631e-3, requires_grad=False).to(dev)
sz1.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz1)

sz2 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_120hz", batchsize=batchsize, tau=(1e6/120), amp=0.06091545894742012e-3, requires_grad=False).to(dev)
sz2.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz2)

sz3 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_180hz", batchsize=batchsize, tau=(1e6/180), amp=0.03468571603298187e-3, requires_grad=False).to(dev)
sz3.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz3)

sz4 = PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_240hz", batchsize=batchsize, tau=(1e6/240), amp=0.10632599145174026e-3, requires_grad=False).to(dev)
sz4.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz4)

################################################
# define the jump operators
################################################
sx_jump = StaticPauliOperatorGroup(n_qubits=1, id="sx_jump", batchsize=batchsize, coef=0.03112558089196682*(1e-3)**0.5, requires_grad=False).to(dev)
sx_jump.add_operator('X')
qubit.add_operator_group_to_jumping(sx_jump)

sy_jump = StaticPauliOperatorGroup(n_qubits=1, id="sy_jump", batchsize=batchsize, coef=0.03429257124662399*(1e-3)**0.5, requires_grad=False).to(dev)
sy_jump.add_operator('Y')
qubit.add_operator_group_to_jumping(sy_jump)

sz_jump = StaticPauliOperatorGroup(n_qubits=1, id="sz_jump", batchsize=batchsize, coef=0.03246501088142395*(1e-3)**0.5, requires_grad=False).to(dev)
sz_jump.add_operator('Z')
qubit.add_operator_group_to_jumping(sz_jump)

################################################
# define the error channel of each pulse
################################################
depol_channel = DepolarizationChannel(n_qubits=1, id="depol_channel", batchsize=batchsize, p=0.0003319382667541504, requires_grad=False).to(dev)
qubit.add_operator_group_to_channel(depol_channel)

################################################
# Experimental data
################################################
preperation_rate = 0.824
## load csv data
data_plain = np.loadtxt('./Data/Fig3C_GreenTriangles.csv', delimiter=',', skiprows=1)
data_plain = th.tensor(data_plain, dtype=th.float).to(dev)
data_plain[:, 1] /= preperation_rate
data_plain[:, 0] *= 1e3
data_plain = data_plain[1:]

data_echo = np.loadtxt('./Data/Fig3C_RedSquares.csv', delimiter=',', skiprows=1)
data_echo = th.tensor(data_echo, dtype=th.float).to(dev)
data_echo[:, 1] /= preperation_rate
data_echo[:, 0] *= 1e3
data_echo = data_echo[1:]

data_XY8 = np.loadtxt('./Data/Fig3C_BlueCircles.csv', delimiter=',', skiprows=1)
data_XY8 = th.tensor(data_XY8, dtype=th.float).to(dev)
data_XY8[:, 1] /= preperation_rate
data_XY8[:, 0] *= 1e3

XY8_cycle_time = 1600
################################################
#  train
################################################
for op in qubit.hamiltonian_operator_groups:
    if op.tau is not None:
        logging.info(f'Hamiltonian: {op.id} Tau={op.tau.mean()}ms, Amp={op.amp.mean()}')
    else:
        logging.info(f'Hamiltonian: {op.id} Amp={op.amp.mean()}')
for op in qubit.jumping_operator_groups:
    logging.info(f'Jump: {op.id} Amp={op.coef.mean()}')
logging.info(f"depol_channel: p={depol_channel.p.mean()}")

## Ramsey experiment without echo
Ramsey_Plain_P0 = RamseyScan(qubit, dt=10, T=3000, theta_list=[0, np.pi], observe_at=data_plain[:, 0])
Ramsey_Plain_Contrast = th.abs(Ramsey_Plain_P0[:, 1] - Ramsey_Plain_P0[:, 0])
logging.info(f"Ramsey Plain Data={data_plain[:, 1]}")
logging.info(f"Ramsey Plain Simulated={Ramsey_Plain_Contrast.detach()}")

## Ramsey experiment with echo
Ramsey_Echo_Contrast = []
for T in data_echo[:, 0]:
    Ramsey_Echo_P0 = RamseySpinEcho(qubit, dt=100, T=T, theta_list=[0, np.pi])
    Ramsey_Echo_Contrast.append(th.abs(Ramsey_Echo_P0[1] - Ramsey_Echo_P0[0]))
Ramsey_Echo_Contrast = th.stack(Ramsey_Echo_Contrast)
logging.info(f"Ramsey Echo Data={data_echo[:, 1]}")
logging.info(f"Ramsey Echo Simulated={Ramsey_Echo_Contrast.detach()}")

## Ramsey experiment with XY8 sequence
Ramsey_XY8_P0 = RamseyScan_XY8(qubit, dt=100, T=150000, cycle_time=XY8_cycle_time, theta_list=[0, np.pi], observe_at=data_XY8[:, 0])
Ramsey_XY8_Contrast = th.abs(Ramsey_XY8_P0[:, 1] - Ramsey_XY8_P0[:, 0])
logging.info(f"Ramsey XY8 Data={data_XY8[:, 1]}")
logging.info(f"Ramsey XY8 Simulated={Ramsey_XY8_Contrast.detach()}")

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
plt.savefig(f'ramsey_contrast_{log_suffix}_confirm.png')

from matplotlib import pyplot as plt
import logging
import time
import numpy as np
import os
import torch as th
import qepsilon as qe
from qepsilon import *
from qepsilon.task import *
from qepsilon.utility import Constants

logging.basicConfig(filename=f'simulation.log', level=logging.INFO)
## do not print the tensor in scientific notation
th.set_printoptions(sci_mode=False, precision=6)
dev = 'cpu'
################################################
# Load experimental data
################################################
data_folder = '/home/pinchenx/data.gpfs/QEpsilon/examples/two_qubits/data'
data_XY8_193 = np.loadtxt(os.path.join(data_folder, 'Fig3D_BlueCircles.csv'), delimiter=',', skiprows=1)
data_XY8_193 = th.tensor(data_XY8_193, dtype=th.float).to(dev)

data_XY8_235 = np.loadtxt(os.path.join(data_folder, 'Fig3E_PurpleTriangles.csv'), delimiter=',', skiprows=1)
data_XY8_235 = th.tensor(data_XY8_235, dtype=th.float).to(dev)

data_XY8_168 = np.loadtxt(os.path.join(data_folder, 'Fig3E_GreenSquares.csv'), delimiter=',', skiprows=1)
data_XY8_168 = th.tensor(data_XY8_168, dtype=th.float).to(dev)

data_XY8_160 = np.loadtxt(os.path.join(data_folder, 'Fig3E_YellowDiamonds.csv'), delimiter=',', skiprows=1)
data_XY8_160 = th.tensor(data_XY8_160, dtype=th.float).to(dev)

data_XY8_143 = np.loadtxt(os.path.join(data_folder, 'Fig3E_OrangeHexagons.csv'), delimiter=',', skiprows=1)
data_XY8_143 = th.tensor(data_XY8_143, dtype=th.float).to(dev)

data_XY8_126 = np.loadtxt(os.path.join(data_folder, 'Fig3E_RedPentagons.csv'), delimiter=',', skiprows=1)
data_XY8_126 = th.tensor(data_XY8_126, dtype=th.float).to(dev)
################################################
#  define the system
################################################
## simulation parameters
nparticles = 2
batchsize = 1000
dt_thermal = 0.25  # us
dt_quantum = 25  # us
tweezer_sep = 1.93 # um
data_XY8 = data_XY8_193
axial_temperature = 2e-05 # K
radial_temperature = 2e-05 # K
tau = 1000000 # us
preperation_rate = 0.824 # 0.824
data_XY8[:, 1] /= preperation_rate**2
data_XY8[:, 0] *= 1e3
XY8_cycle_time = 1600 if tweezer_sep < 1.6 else 3200
ddinteraction_prefactor = (22.15 * Constants.hbar_Hz) * (2 * np.pi) * (2.4 ** 3) / 4.0  ## J/4 in unit of hbar * MHz * um^3
# _ddinteraction_prefactor = (0.328 * 1.21 * Constants.bohr_radius)**2 / (4 * np.pi * Constants.epsilon0) / 4.0 ## J/4 in unit of hbar * MHz * um^3
## define the thermal states

max_depth = Constants.kb * 0.215e-3 # hbar * MHz 
particles = Particles(n_particles=nparticles, batchsize=batchsize, mass=59.0 * Constants.amu, 
                      radial_temp=radial_temperature , axial_temp=axial_temperature, 
                      dt=dt_thermal, tau = tau)    
particles.init_tweezers('TZ1', min_waist=0.730, wavelength=0.781, max_depth=max_depth, center=th.tensor([0, 0, 0.0]), axis=th.tensor([0, 0, 1.0]))
particles.init_tweezers('TZ2', min_waist=0.730, wavelength=0.781, max_depth=max_depth, center=th.tensor([tweezer_sep, 0, 0.0]), axis=th.tensor([0, 0, 1.0]))
logging.info(f"dt_thermal={dt_thermal}us, dt_quantum={dt_quantum}us, batchsize={batchsize}")
logging.info(f"tweezer_sep={tweezer_sep}um, cycle_time={XY8_cycle_time}us")
logging.info(f"radial_temp={particles.radial_temp*1e6}uK, axial_temp={particles.axial_temp*1e6}uK")
particles.reset()

## define the qubits
# qubit = qe.LindbladSystem(n_qubits=nparticles, batchsize=batchsize).to(dev)
qubit = qe.ParticleLindbladSystem(n_qubits=nparticles, batchsize=batchsize, particles=particles).to(dev)
qubit.set_rho_by_config([0, 0])
qubit.to(dev)
################################################
# define the terms in the Hamiltonian
################################################
## add the two-body dipole-dipole interaction
dipole_int = DipolarInteraction(n_qubits=nparticles, id='dipole_int', batchsize=batchsize, particles=particles, 
                                connectivity=th.tensor([[False, True], [True, False]]), 
                                prefactor=ddinteraction_prefactor, 
                                average_nsteps=int(dt_quantum / dt_thermal), 
                                qaxis=th.tensor([0.0, 1.0, 0.0]),
                                requires_grad=False)
qubit.add_operator_group_to_hamiltonian(dipole_int)

# add one-body terms
sz_shot = ShotbyShotNoisePauliOperatorGroup(n_qubits=nparticles, id="sz_noise_shot", batchsize=batchsize, amp=0.28362715244293213e-3, requires_grad=False).to(dev)
sz_shot.add_operator('ZI')
sz_shot.add_operator('IZ')
qubit.add_operator_group_to_hamiltonian(sz_shot)

sz0 = ColorNoisePauliOperatorGroup(n_qubits=nparticles, id="sz_noise_color", batchsize=batchsize, tau=3.5519464015960693e3, amp=0.05605236068367958e-3, requires_grad=False).to(dev)
sz0.add_operator('ZI')
sz0.add_operator('IZ')
qubit.add_operator_group_to_hamiltonian(sz0)

sz1 = PeriodicNoisePauliOperatorGroup(n_qubits=nparticles, id="sz_noise_60hz", batchsize=batchsize, tau=(1e6/60), amp=0.03714356571435928e-3, requires_grad=False).to(dev)
sz1.add_operator('ZI')
sz1.add_operator('IZ')
qubit.add_operator_group_to_hamiltonian(sz1)

sz2 = PeriodicNoisePauliOperatorGroup(n_qubits=nparticles, id="sz_noise_120hz", batchsize=batchsize, tau=(1e6/120), amp=0.05967099964618683e-3, requires_grad=False).to(dev)
sz2.add_operator('ZI')
sz2.add_operator('IZ')
qubit.add_operator_group_to_hamiltonian(sz2)

sz3 = PeriodicNoisePauliOperatorGroup(n_qubits=nparticles, id="sz_noise_180hz", batchsize=batchsize, tau=(1e6/180), amp=0.03487752377986908e-3, requires_grad=False).to(dev)
sz3.add_operator('ZI')
sz3.add_operator('IZ')
qubit.add_operator_group_to_hamiltonian(sz3)

sz4 = PeriodicNoisePauliOperatorGroup(n_qubits=nparticles, id="sz_noise_240hz", batchsize=batchsize, tau=(1e6/240), amp=0.1100817546248436e-3, requires_grad=False).to(dev)
sz4.add_operator('ZI')
sz4.add_operator('IZ')
qubit.add_operator_group_to_hamiltonian(sz4)

################################################
# define the jump operators.  These twos are significant.
################################################
sx_jump = StaticPauliOperatorGroup(n_qubits=nparticles, id="sx_jump", batchsize=batchsize, coef=0.03111828863620758*(1e-3)**0.5, requires_grad=False).to(dev)
# sx_jump = StaticPauliOperatorGroup(n_qubits=nparticles, id="sx_jump", batchsize=batchsize, coef=0.06*(1e-3)**0.5, requires_grad=False).to(dev)
sx_jump.add_operator('XI')
sx_jump.add_operator('IX')
qubit.add_operator_group_to_jumping(sx_jump)

sz_jump = StaticPauliOperatorGroup(n_qubits=nparticles, id="sz_jump", batchsize=batchsize, coef=0.03243376687169075*(1e-3)**0.5, requires_grad=False).to(dev)
# sz_jump = StaticPauliOperatorGroup(n_qubits=nparticles, id="sz_jump", batchsize=batchsize, coef=0.06 *(1e-3)**0.5, requires_grad=False).to(dev)
sz_jump.add_operator('ZI')
sz_jump.add_operator('IZ')
qubit.add_operator_group_to_jumping(sz_jump)

################################################
# define the error channel of each pulse
################################################
depol_channel = DepolarizationChannel(n_qubits=nparticles, id="depol_channel", batchsize=batchsize, p=0.00033205747604370117, requires_grad=False).to(dev)
qubit.add_operator_group_to_channel(depol_channel)

################################################
#  train
################################################
# nepoch = 100
# optimizer = th.optim.Adam(qubit.HamiltonianParameters() + qubit.JumpingParameters() + qubit.ChannelParameters(), lr=0.03)
## set learning rate by group
# optimizer = th.optim.Adam([{'params': qubit.HamiltonianParameters(), 'lr': 0.1},
#                           {'params': qubit.JumpingParameters(), 'lr': 0.001},
#                           {'params': qubit.ChannelParameters(), 'lr': 0.1}])
# logging.info(f"Epoch={epoch}")
# loss = 0
## Ramsey experiment with XY8 sequence
tmax = data_XY8[:, 0].max() + 3200 #160000 # us
obs_at = np.arange(np.ceil(tmax/3200)) * 3200
Ramsey_XY8_P00, loss = RamseyScan_XY8_TwoQubits(qubit, dt=dt_quantum, T=tmax, cycle_time=XY8_cycle_time, observe_at=obs_at)
# loss += ((Ramsey_XY8_P00 - data_XY8[:, 1]) ** 2).mean()
logging.info(f"Ramsey XY8 Data={data_XY8[:, 1]}")
logging.info(f"Ramsey XY8 Simulated={Ramsey_XY8_P00.detach()}")
# logging.info(f'loss = {loss}')
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

np.save('Ramsey_XY8_t.npy', obs_at)
np.save('Ramsey_XY8_P00.npy', Ramsey_XY8_P00.detach().numpy())
np.save('Ramsey_XY8_loss.npy', loss.detach().numpy())

## plot the probability of |00>
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(obs_at/1000, Ramsey_XY8_P00.detach().numpy() * preperation_rate**2, label='Simulated', color='blue')
ax.plot(data_XY8[:, 0]/1000, data_XY8[:, 1] * preperation_rate**2, marker='o', color='blue', label='Data', linestyle='None') # circle marker
ax.set_xlabel('Time [ms]')
ax.set_ylabel('P00')
ax.legend()
plt.show()
plt.tight_layout()
plt.savefig(f'P00_sep{tweezer_sep:.3f}.png')
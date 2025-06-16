import numpy as np
import torch as th
import qepsilon as qe
from qepsilon.system.particles import OpticalTweezer
from qepsilon.system.particles import ParticlesInTweezers
from qepsilon.utilities import *
import matplotlib.pyplot as plt

max_depth = Constants.kb * 1.28e-3 # hbar * MHz 
w0 = 0.730 # um
wavelength = 0.781 # um
nq = 2 # number of qubits
sep = 1.26 # um
## initialize the particles 
particles = ParticlesInTweezers(n_particles=nq, batchsize=1, mass=59.0 * Constants.amu, 
                    dt = 0.25, tau=None)
## setup the first tweezer, length unit is um, energy unit is hbar*MHz
particles.init_tweezers('TZ1', min_waist=w0, wavelength=wavelength, 
                        max_depth=max_depth, center=th.tensor([0, 0, 0]), axis=th.tensor([0, 0, 1.0]))
 ## setup the second tweezer
particles.init_tweezers('TZ2', min_waist=w0, wavelength=wavelength, 
                        max_depth=max_depth, center=th.tensor([sep, 0, 0]), axis=th.tensor([0, 0, 1.0]))
######################### plot the potential energy surface when there is one tweezer #########################
fig, ax = plt.subplots(figsize=(4, 10))
x = np.linspace(-1.5, 1.5, 100)
z = np.linspace(-8, 8, 100)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)
positions = np.stack([X, Y, Z], axis=-1) # (nx, nz, 3)
positions = th.tensor(positions, dtype=th.float32).reshape(-1,1,3)
potential = particles.get_tweezer_by_id('TZ1').get_pot(positions).reshape(x.shape[0], z.shape[0])
potential = potential / max_depth + 1
levels = np.linspace(0, 0.8, 9)
plt.contour(X, Z, potential, levels=levels, cmap='Spectral', linewidths=2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-8, 8)
plt.colorbar( orientation='horizontal', shrink=0.5)  # Add a colorbar to show the values
## save the figure
plt.tight_layout()
plt.savefig('PES_one_tweezer.png', dpi=300)
plt.close()


######################### plot the potential energy surface when there are two tweezers #########################
fig, ax = plt.subplots(figsize=(6, 10))
x = np.linspace(-1.5, sep + 1.5, 100)
z = np.linspace(-8, 8, 100)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)
positions = np.stack([X, Y, Z], axis=-1) # (nx, nz, 3)
positions = th.tensor(positions, dtype=th.float32).reshape(-1,1,3)
potential = particles.get_tweezer_by_id('TZ1').get_pot(positions).reshape(x.shape[0], z.shape[0])
potential += particles.get_tweezer_by_id('TZ2').get_pot(positions).reshape(x.shape[0], z.shape[0])
potential = potential / max_depth + 1
levels = np.linspace(0, 0.8, 9)
plt.contour(X, Z, potential, levels=levels, cmap='Spectral', linewidths=2)
ax.set_xlim(-1.5, sep + 1.5)
ax.set_ylim(-8, 8)

plt.colorbar( orientation='horizontal', shrink=0.5)  # Add a colorbar to show the values
## save the figure
plt.tight_layout()
plt.savefig('PES_two_tweezers_sep{:.3f}um.png'.format(sep), dpi=300)
plt.close()

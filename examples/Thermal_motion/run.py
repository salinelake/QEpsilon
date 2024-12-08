import qepsilon as qe
from qepsilon.particles import OpticalTweezer
from qepsilon.particles import Particles
from qepsilon.operator_group import DipolarInteraction
from qepsilon.utility import *
import matplotlib.pyplot as plt

radial_temp = 40e-6 # K
axial_temp = 80e-6 # K
max_depth = Constants.kb * 1.28e-3 # hbar * MHz 
nq = 2
nb = 1
dt = 0.25 # us
sep = 1.680
## initialize the particles, dt is time step, tau is the coherent time of thermal motion. 
particles = Particles(n_particles=nq, batchsize=nb, mass=59.0 * Constants.amu, 
                    radial_temp=radial_temp, axial_temp=axial_temp, 
                    dt = dt, tau=100000)
## setup the first tweezer, length unit is um, energy unit is hbar*MHz
particles.init_tweezers('TZ1', min_waist=0.730, wavelength=0.781, 
                        max_depth=max_depth, center=th.tensor([0, 0, 0]), axis=th.tensor([0, 0, 1.0]))
## setup the second tweezer
particles.init_tweezers('TZ2', min_waist=0.730, wavelength=0.781, 
                        max_depth=max_depth, center=th.tensor([sep, 0, 0]), axis=th.tensor([0, 0, 1.0]))
## do the initial thermalization
particles.reset()
print('After initial thermalization, positions=', particles.get_positions())
## run isothermal simulation for 40000 steps with time step dt
nsteps = 40000
temp_list = []
for i in range(nsteps):
    particles.zero_forces()
    particles.modify_forces(particles.get_trapping_forces())
    particles.step_langevin( record_traj=True)
    temp_list.append(particles.get_temperature())
    if i % (nsteps // 10) == 0:
        print(f"Step {i} / {nsteps}")
        print("Avg Temperature in uK: ", th.stack(temp_list).mean(0) * 1e6 )
        # print("Velocities: ", particles.get_velocities())
        print("Positions: ", particles.get_positions())

## get the trajectory
traj = particles.get_trajectory() # (nsteps, nb, nq, 3)



fig, ax = plt.subplots()
######################### plot the trajectory #########################
ax.scatter(traj[:,:,0,0].flatten(), traj[:,:,0,2].flatten(), s=np.ones(traj[:,:,0,0].flatten().shape[0]) * 4, color='black')
ax.scatter(traj[:,:,1,0].flatten(), traj[:,:,1,2].flatten(), s=np.ones(traj[:,:,1,0].flatten().shape[0]) * 4, color='blue')

ax.set_xlabel('x [um]')
ax.set_ylabel('z [um]') 

######################### plot the potential energy surface #########################
# Create a grid of x and z values
x = np.linspace(-1.5, sep + 1.5, 100)
z = np.linspace(-8, 8, 100)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)
positions = np.stack([X, Y, Z], axis=-1) # (nx, nz, 3)
positions = th.tensor(positions, dtype=th.float32).reshape(-1,1,3)
potential = particles.get_tweezer_by_id('TZ1').get_pot(positions).reshape(x.shape[0], z.shape[0])
potential += particles.get_tweezer_by_id('TZ2').get_pot(positions).reshape(x.shape[0], z.shape[0])
levels = np.linspace(potential.min(), potential.max(), 12)
plt.contour(X, Z, potential, levels=levels, cmap='Spectral')
ax.set_xlim(-1.5, sep + 1.5)
ax.set_ylim(-8, 8)

plt.colorbar()  # Add a colorbar to show the values
## save the figure
plt.tight_layout()
plt.savefig('traj_sep{:.3f}um.png'.format(sep))
plt.close()

import numpy as np
import matplotlib.pyplot as plt
from qepsilon.utilities import Constants_Metal as Constants
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.markersize'] = 5


hopping_in_meV = 83
ns = 60
rubrene_R = 7 * Constants.Angstrom
rubrene_V = hopping_in_meV * Constants.meV
data = np.loadtxt('mobility.csv', delimiter=',')

DMRG_temp_list = data[:,0]
DMRG_mobility  = (data[:,1] + data[:,2])/2


temp_list = np.array([200, 250, 300, 350, 400])
DIQCD_mobility_list = []
for temp in temp_list:
    factor = rubrene_R ** 2/ Constants.cm**2 * Constants.eV / Constants.kb / temp / 2.0 * Constants.s / Constants.fs
    MSD_traj = np.load('/home/pinchenx/data.gpfs/QEpsilon/examples/organic_semiconductor/Rubrene_Crystal/simulate_lindblad/T{:.0f}_V{:.1f}meV_ns{}/MSD_traj.npy'.format(temp, hopping_in_meV, ns))
    MSD_mean = MSD_traj.mean(-1)
    slope = (MSD_mean[99] - MSD_mean[49]) / 50
    mobility = slope * factor
    DIQCD_mobility_list.append(mobility)
Ehrenfest_mobility_list = []
for temp in temp_list:
    factor = rubrene_R ** 2/ Constants.cm**2 * Constants.eV / Constants.kb / temp / 2.0 * Constants.s / Constants.fs
    MSD_traj = np.load('/home/pinchenx/data.gpfs/QEpsilon/examples/organic_semiconductor/ehrenfest_Rubrene/classical/MSD_traj_{:.0f}K.npy'.format(temp))
    slope = (MSD_traj[199] - MSD_traj[99]) / 100
    mobility = slope * factor
    Ehrenfest_mobility_list.append(mobility)

fig, ax = plt.subplots(1, 1, figsize=(3,3))
ax.plot(temp_list, Ehrenfest_mobility_list, 'o-', label='Ehrenfest',   markersize=6,  color='tab:purple')
ax.plot(DMRG_temp_list, DMRG_mobility, 's-', label='TD-DMRG', markersize=6,  color='tab:orange')
ax.plot(temp_list, DIQCD_mobility_list, '*-', label='DIQCD', markersize=10, linewidth=2, color='tab:red')
ax.set_ylim(0,230)
# kbT = Constants.kb * temp_list / Constants.eV
# DMRG_kbT = Constants.kb * DMRG_temp_list / Constants.eV
# ax[1].plot(temp_list, Ehrenfest_mobility_list * kbT, 's--', label='Ehrenfest',   markersize=8, alpha=0.8, color='tab:purple')
# ax[1].plot(DMRG_temp_list, DMRG_mobility * DMRG_kbT, 's--', label='TD-DMRG', markersize=8, alpha=0.8, color='tab:orange')
# ax[1].plot(temp_list, DIQCD_mobility_list * kbT, '*-', label='DIQCD', markersize=12, linewidth=2, color='tab:red')
# ax[1].set_ylim(0,3.8)

ax.set_xlim(180, 420)
ax.set_xlabel('T [K]', fontsize=13)
ax.set_ylabel(r'$\mu$ [$\mathrm{cm^2/(V\cdot s)}$]', fontsize=13)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('mobility_V{:.1f}meV.png'.format(hopping_in_meV), dpi=200)




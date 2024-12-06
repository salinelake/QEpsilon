import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
################################################
# Load experimental data
################################################
data_folder = '/home/pinchenx/data.gpfs/QEpsilon/examples/two_qubits/data'
data_XY8_193 = np.loadtxt(os.path.join(data_folder, 'Fig3D_BlueCircles.csv'), delimiter=',', skiprows=1)
data_XY8_235 = np.loadtxt(os.path.join(data_folder, 'Fig3E_PurpleTriangles.csv'), delimiter=',', skiprows=1)
data_XY8_168 = np.loadtxt(os.path.join(data_folder, 'Fig3E_GreenSquares.csv'), delimiter=',', skiprows=1)
data_XY8_160 = np.loadtxt(os.path.join(data_folder, 'Fig3E_YellowDiamonds.csv'), delimiter=',', skiprows=1)
data_XY8_143 = np.loadtxt(os.path.join(data_folder, 'Fig3E_OrangeHexagons.csv'), delimiter=',', skiprows=1)
data_XY8_126 = np.loadtxt(os.path.join(data_folder, 'Fig3E_RedPentagons.csv'), delimiter=',', skiprows=1)
data_list = [data_XY8_126, data_XY8_143, data_XY8_160, data_XY8_168, data_XY8_193, data_XY8_235]
################################################
# Plot
################################################
sep_list = [1.26,1.43, 1.6,1.68, 1.93, 2.35] # um
atemp = 80
rtemp = 40
spam_scale = 0.824 ** 2

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
## set font size globally
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
for idx, sep in enumerate(sep_list):
    folder_name = 'sep{:.2f}um_atemp{:.0f}uK_rtemp{:.0f}uK'.format(sep, atemp, rtemp)
    data = data_list[idx]
    sim_P00 = np.load(os.path.join(folder_name, 'Ramsey_XY8_P00.npy')) * spam_scale
    sim_t = np.load(os.path.join(folder_name, 'Ramsey_XY8_t.npy')) / 1000
    try:
        loss = np.load(os.path.join(folder_name, 'Ramsey_XY8_loss.npy'))
    except:
        loss = np.zeros_like(sim_t)
    ax[0].plot(sim_t, sim_P00+idx*0.5,  )
    ax[0].scatter(data[:,0], data[:,1]+idx*0.5, np.ones_like(data[:,0])*30, marker='*', label=f'r={sep}um')
    ax[1].plot(sim_t, loss*100, label=f'r={sep}um')

# ax[0].legend(fontsize=10)
ax[0].set_title(r'$T_r=40\mu K, T_a=80\mu K$')
ax[0].set_xlabel('t [ms]', fontsize=14)
ax[0].set_ylabel(r'$P_{00}$', fontsize=14)

ax[1].legend(fontsize=8, loc='upper right', frameon=False)
ax[1].set_xlabel('t [ms]', fontsize=14)
ax[1].set_ylabel('Molecular Loss [%]', fontsize=14)
ax[1].set_xlim(-10, 220)
fig.tight_layout()
fig.savefig('compare.png', dpi=300)



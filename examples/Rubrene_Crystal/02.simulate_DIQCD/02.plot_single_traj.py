import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.markersize'] = 5

temp = 300
V = 8.3
ns = 40
data_folder = 'T{:.0f}_V{:.1f}meV_ns{}'.format(temp, V, ns)
x = np.load(os.path.join(data_folder, 'site_occupation_traj.npy'))
nsample, batchsize, ns = x.shape

fig, ax = plt.subplots(1, 4, figsize=(18, 3))
for batch_idx in range(4):
    ax[batch_idx].imshow(-np.log10(x[:,batch_idx,:]+1e-10), aspect='auto', cmap='viridis', origin='lower', extent=[0, ns, 0, nsample], vmin=0, vmax=4)
    ax[batch_idx].set_title('batch {}'.format(batch_idx))
    ax[batch_idx].set_xlabel('site')
    ax[batch_idx].set_ylabel('time [fs]')

plt.savefig(os.path.join(data_folder, 'site_occupation_single_traj.png'))






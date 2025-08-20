import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
epsilon = 0.1


sigma_x_alltemp = []
sigma_y_alltemp = []
temperature_list = [200, 250, 300, 350, 400]
for temperature in temperature_list:

    out_dir = 'T{:.0f}_epsilon{:.1f}_dt{:.3f}fs'.format(temperature, epsilon, dt)
    sigma_n_list = []
    sigma_x_list = []
    sigma_y_list = []
    for run_id in range(8):
        sigma_n = np.load('{}/sigma_n_run{}.npy'.format(out_dir, run_id))
        sigma_x = np.load('{}/sigma_x_run{}.npy'.format(out_dir, run_id))
        sigma_y = np.load('{}/sigma_y_run{}.npy'.format(out_dir, run_id))
        sigma_n_list.append(sigma_n)
        sigma_x_list.append(sigma_x)
        sigma_y_list.append(sigma_y)
    sigma_n_list = np.concatenate(sigma_n_list, axis=1).real
    sigma_x_list = np.concatenate(sigma_x_list, axis=1).real
    sigma_y_list = np.concatenate(sigma_y_list, axis=1).real
    np.save('{}/sigma_n.npy'.format(out_dir), sigma_n_list.transpose())
    np.save('{}/sigma_x.npy'.format(out_dir), sigma_x_list.transpose())
    np.save('{}/sigma_y.npy'.format(out_dir), sigma_y_list.transpose())

    scale = sigma_x_list.mean(axis=1)[0]
    sigma_n_list = sigma_n_list / sigma_n_list.mean(axis=1)[0]
    sigma_x_list = sigma_x_list / scale
    sigma_y_list = sigma_y_list / scale

    nbatch = sigma_n_list.shape[1]
    nsample = sigma_n_list.shape[0]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(nbatch):
        ax[0,0].plot(sigma_n_list[:,i])
        ax[0,1].plot(sigma_x_list[:,i])
        ax[0,2].plot(sigma_y_list[:,i])
    ax[0,0].set_title(r'$\langle \hat{\sigma}_n \rangle$')
    ax[0,1].set_title(r'$\langle \hat{\sigma}_x \rangle$')
    ax[0,2].set_title(r'$\langle \hat{\sigma}_y \rangle$')


    sigma_n_std = sigma_n_list.std(axis=1)
    sigma_x_std = sigma_x_list.std(axis=1)
    sigma_y_std = sigma_y_list.std(axis=1)
    ax[1,0].plot(sigma_n_list.mean(axis=1))
    ax[1,0].fill_between(np.arange(nsample), sigma_n_list.mean(axis=1) - sigma_n_std, sigma_n_list.mean(axis=1) + sigma_n_std, alpha=0.5)
    ax[1,1].plot(sigma_x_list.mean(axis=1))
    ax[1,1].fill_between(np.arange(nsample), sigma_x_list.mean(axis=1) - sigma_x_std, sigma_x_list.mean(axis=1) + sigma_x_std, alpha=0.5)
    ax[1,2].plot(sigma_y_list.mean(axis=1))
    ax[1,2].fill_between(np.arange(nsample), sigma_y_list.mean(axis=1) - sigma_y_std, sigma_y_list.mean(axis=1) + sigma_y_std, alpha=0.5)

    plt.tight_layout()
    plt.savefig('observation_T{:.0f}_epsilon{:.1f}_dt{:.3f}fs.png'.format(temperature, epsilon, dt))

    sigma_x_alltemp.append(sigma_x_list.mean(axis=1))
    sigma_y_alltemp.append(sigma_y_list.mean(axis=1))

fig, ax = plt.subplots(figsize=(4, 3))
for temp, sigma_x in zip(temperature_list, sigma_x_alltemp):
    ax.plot(sigma_x, label=r'T={} K'.format(temp))
ax.legend()
plt.tight_layout()
plt.savefig('observation_epsilon{:.1f}_dt{:.3f}fs.png'.format(epsilon, dt))





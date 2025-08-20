import numpy as np
import matplotlib.pyplot as plt


temp_list = [200, 250, 300, 350, 400]
dt = 0.05
nepoch = 200
nmodes = 9
for temperature in temp_list:
    ## load training results
    data_folder = 'T{:.0f}_dt{:.3f}fs'.format(temperature, dt)
    with open('{}/train.log'.format(data_folder), 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "Epoch=0" in line:
                start_idx = idx
            if "Epoch={:d}".format(nepoch-1) in line:
                end_idx = idx + 11
    ## extract parameters
    tau_list = []
    amp_list = []
    dephasing_list = []
    onsite_energy_list = []
    data_lines = lines[start_idx:end_idx+1]
    for line in data_lines:
        if "dephasing" in line:
            dephasing_list.append(float(line.split('dephasing=')[1].strip()))
        if "tau" in line:
            tau_list.append(float(line.split('tau=')[1].split('fs')[0]))
        if "amp" in line:
            amp_list.append(float(line.split('amp=')[1].strip()))
        if "onsite_energy" in line:
            onsite_energy_list.append(float(line.split('coef=')[1].strip()))

    ## save parameters
    tau_list = np.array(tau_list).reshape(nepoch, 9)
    amp_list = np.array(amp_list).reshape(nepoch, 9)
    dephasing_list = np.array(dephasing_list).reshape(nepoch)
    onsite_energy_list = np.array(onsite_energy_list).reshape(nepoch)
    np.savetxt('T{:.0f}_dt{:.3f}fs/tau_list.txt'.format(temperature, dt), tau_list)
    np.savetxt('T{:.0f}_dt{:.3f}fs/amp_list.txt'.format(temperature, dt), amp_list)
    np.savetxt('T{:.0f}_dt{:.3f}fs/dephasing_list.txt'.format(temperature, dt), dephasing_list)
    np.savetxt('T{:.0f}_dt{:.3f}fs/onsite_energy_list.txt'.format(temperature, dt), onsite_energy_list)




for temperature in temp_list:
    tau = np.loadtxt('T{:.0f}_dt{:.3f}fs/tau_list.txt'.format(temperature, dt))
    amp = np.loadtxt('T{:.0f}_dt{:.3f}fs/amp_list.txt'.format(temperature, dt))
    dephasing = np.loadtxt('T{:.0f}_dt{:.3f}fs/dephasing_list.txt'.format(temperature, dt))
    onsite_energy = np.loadtxt('T{:.0f}_dt{:.3f}fs/onsite_energy_list.txt'.format(temperature, dt))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    for i in range(nmodes):
        ax[0].plot(amp[:, i], label=r'$\gamma_{}$'.format(i))
    ax[0].legend()
    ax[1].plot(dephasing)
    ax[2].plot(onsite_energy)
    plt.savefig('T{:.0f}_dt{:.3f}fs/train_trajectory.png'.format(temperature, dt))
    plt.close()
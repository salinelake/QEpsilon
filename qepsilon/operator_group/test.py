import torch as th
import numpy as np
import qepsilon as qe
import matplotlib.pyplot as plt
op = qe.ColorNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_color1", batchsize=10, 
    tau=50, amp=0.1, omega=3.1415/5, requires_grad=False)
op.add_operator('Z')
noise_list = []
dt= 0.1
for i in range(10000):
    _, coef = op.sample(dt)
    noise_list.append(coef)
noise_list = th.stack(noise_list) # (10000,10)

corr_all = [(noise_list**2).mean()]
for i in range(1,1000):
    corr = (noise_list[i:] * noise_list[:-i]).mean()
    corr_all.append(corr)
corr_all = th.tensor(corr_all) / corr_all[0]

# plt.plot(np.arange(corr_all.shape[0])*dt, corr_all)
plt.plot(noise_list[:3000,0])
plt.savefig('test.png')
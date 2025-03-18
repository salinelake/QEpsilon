import numpy as np
import torch as th
import qepsilon as qe
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
qubit = qe.QubitLindbladSystem(n_qubits=1, batchsize=batchsize)
################################################
# define the terms in the Hamiltonian
################################################
sz_shift = qe.StaticPauliOperatorGroup(n_qubits=1, id="sz_shift", batchsize=batchsize, coef=-0.04*1000, requires_grad=True)
sz_shift.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz_shift)

sz1 = qe.LangevinNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_langevin", batchsize=batchsize, tau=100/1000, amp=0.01*1000, requires_grad=True)
sz1.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz1)

sz2 = qe.PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_periodic", batchsize=batchsize, tau=50/1000, amp=0.01*1000, requires_grad=True)
sz2.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz2)

sz3 = qe.PeriodicNoisePauliOperatorGroup(n_qubits=1, id="sz_noise_periodic_2", batchsize=batchsize, tau=100/1000, amp=0.01*1000, requires_grad=True)
sz3.add_operator('Z')
qubit.add_operator_group_to_hamiltonian(sz3)
################################################
# define the jump operators
################################################
sz_jump = qe.StaticPauliOperatorGroup(n_qubits=1, id="sz_jump", batchsize=batchsize, coef=np.sqrt(0.01*1000), requires_grad=True)
sz_jump.add_operator('Z')
qubit.add_operator_group_to_jumping(sz_jump)
 
################################################
# Define the operators we want to observe
################################################
obs_spin_sz = qe.StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sz", batchsize=batchsize, coef=1.0)
obs_spin_sz.add_operator('Z')
obs_spin_sx = qe.StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sx", batchsize=batchsize, coef=1.0)
obs_spin_sx.add_operator('X')
obs_spin_sy = qe.StaticPauliOperatorGroup(n_qubits=1, id="obs_spin_sy", batchsize=batchsize, coef=1.0)
obs_spin_sy.add_operator('Y')

################################################
#  data
################################################
## load csv data
sx_observed = np.load('../nine_modes/sx_tau0.10ps_9modes_dt10as.npy')
sy_observed = np.load('../nine_modes/sy_tau0.10ps_9modes_dt10as.npy')

t_observed = np.arange(sx_observed.shape[0]) / 1000  #ps
sx_observed = th.tensor(sx_observed)[:200]
sy_observed = th.tensor(sy_observed)[:200]
t_observed = th.tensor(t_observed)[:200]
################################################
#  train
################################################
nepoch = 200
dt = 0.0001
optimizer = th.optim.Adam([{'params': qubit.HamiltonianParameters(), 'lr': 0.03},
                          {'params': qubit.JumpingParameters(), 'lr': 0.03},
                          ])
sz_shift_history = []
damping_history = []
# ## scheduler
# scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
for epoch in range(nepoch):
    logging.info(f"Epoch={epoch}")
    for op in qubit.hamiltonian_operator_groups:
        try:
            logging.info(f'Hamiltonian: {op.id} Tau={op.tau.mean()}fs, Amp={op.amp.mean()}, Omega={op.omega.mean()}1/fs')
        except:
            try:
                logging.info(f'Hamiltonian: {op.id} Tau={op.tau.mean()}fs, Amp={op.amp.mean()}')
            except:
                logging.info(f'Hamiltonian: {op.id} coef={op.coef.mean()}')
    for op in qubit.jumping_operator_groups:
        try:
            logging.info(f'Jump: {op.id} Amp={op.amp.mean()}')
        except:
            logging.info(f'Jump: {op.id} Amp={op.coef.mean()}')
    sx_simulated = []
    sy_simulated = []
    ##### simulate #####
    nsteps = int(t_observed[-1] / dt) + 1
    observe_steps = [int(t / dt) for t in t_observed]
    dm = qubit.density_matrix
    qubit.reset()
    ## set the initial state
    qubit.rho = th.tensor([[0.01, 0.1], [0.1, 0.99]])
    # evolution
    for i in range(nsteps):
        if i in observe_steps:
            sx_simulated.append(qubit.observe(obs_spin_sx).mean())
            sy_simulated.append(qubit.observe(obs_spin_sy).mean())
        qubit.step(dt=dt)
    sx_simulated = th.stack(sx_simulated)
    sy_simulated = th.stack(sy_simulated)
    loss = ((sx_observed - sx_simulated) ** 2).mean() + ((sy_observed - sy_simulated) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()
    logging.info(f'Loss={loss.detach()}')
    logging.info(f"Data={sx_observed}")
    logging.info(f"Simulated={sx_simulated.detach()}")
    sz_shift_history.append(sz_shift.coef.detach().numpy().sum())
    damping_history.append(sz_jump.coef.detach().numpy().sum())

## two figures
fig, ax = plt.subplots(1,2,figsize=(8, 3))
ax[0].plot(t_observed, sx_simulated.detach().numpy(), markersize=0, linewidth=2, linestyle='dashed', label='Sx Simulated')
ax[0].plot(t_observed, sx_observed, markersize=0, linewidth=4, linestyle='solid', alpha=0.5, label='Sx Data')  
ax[0].plot(t_observed, sy_simulated.detach().numpy(), markersize=0, linewidth=2, linestyle='dashed', label='Sy Simulated')
ax[0].plot(t_observed, sy_observed, markersize=0, linewidth=4, linestyle='solid', alpha=0.5, label='Sy Data')  
ax[0].set_xlabel('Time [ms]')
ax[0].set_ylabel('<Sx>')
ax[0].legend()
ax[1].plot(sz_shift_history, label='sz_shift')
ax[1].plot(damping_history, label='damping')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Coef')
ax[1].legend()
plt.tight_layout()
plt.savefig('train_9mode_nonmarkov.png', dpi=200)

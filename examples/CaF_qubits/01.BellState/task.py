import warnings
import numpy as np
import torch as th
from qepsilon.utilities import XY8
from time import time
def RamseyScan_XY8_TwoQubits(system, dt: float, T: float, cycle_time: float, observe_at: list[float], loss_radius: float = 0.6):
    """
    Ramsey experiment with a XY8 dynamical decoupling sequence: scan the Ramsey contrast as a function of the free evolution time and the rotation angle.
    Args:
        system: the system object.
        dt: the time step size.
        T: the total time for the Ramsey experiment.
        cycle_time: the cycle time of the XY8 dynamical decoupling sequence.
        observe_at: the list of times to observe the Ramsey fringe.
        loss_radius: when the distance between a molecule and the trap center is larger than loss_radius, the molecule is considered lost.
    Returns:
        P00_list: the 1D tensor of |00> probabilities for observation times. Shape (len(observe_at)).
    """
    ## sanitary check and initiate the dynamical decoupling iterator
    if (cycle_time / dt) != int(cycle_time / dt):
        warnings.warn(f"The cycle time {cycle_time} is not an integer multiple of the time step size {dt}.")
    dd_iterator = XY8(cycle_time)
    ## 
    P00_list = []
    loss_list = []
    nsteps = int(T/dt)+1
    observe_steps = []
    for obs_t in observe_at:
        obs_t_cycle_end = np.round(obs_t/cycle_time) * cycle_time
        obs_step = int(np.round(obs_t_cycle_end / dt))
        observe_steps.append(obs_step)
    dm = system.density_matrix
    ## reset history of the system
    system.reset()
    # system.step_particles(3000) # 3ms thermalization
    ## set the initial state
    system.set_rho_by_config([0, 0])
    # first pi/2 rotation along x
    system.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
    # free evolution with dynamical decoupling
    dd_t, dd_direction, dd_phase = next(dd_iterator)
    dd_step = int(np.round(dd_t / dt))
    alive_flag = th.ones(system.nb, dtype=th.bool)
    t0 = time()
    for i in range(nsteps):
        if i == dd_step:
            system.rotate(direction=dd_direction, angle=dd_phase)
            op, coef = system._channel_group_dict["depol_channel"].sample()
            system.kraus_operate(op)
            dd_t, dd_direction, dd_phase = next(dd_iterator)
            dd_step = int(np.round(dd_t / dt))
        system.step(dt=dt, profile=False)
        if i in observe_steps:
            assert dd_iterator.idx == 0, f"observe before a cycle of dynamical decoupling sequence is finished. The index of the current DD step is {dd_iterator.idx}."
            # second pi/2 rotation along the direction u
            rho_t = dm.apply_unitary_rotation(system.rho, th.tensor([1.0,0,0]), np.pi/2)
            prob_00 = dm.observe_prob_by_config(rho_t, th.tensor([0,0]))
            ## check if any particle is lost
            tweezers_center = [x.center for x in system.particles._traps_dict.values()]
            pos = system.particles.get_positions()
            d0 =  pos[:,0,0] - tweezers_center[0][0] 
            d1 =  pos[:,1,0] - tweezers_center[1][0] 
            alive_flag = alive_flag & (th.abs(d0) < loss_radius) & (th.abs(d1) < loss_radius)
            alive_flag = alive_flag & th.isfinite(prob_00.to(device=alive_flag.device))
            n_dead = system.nb - alive_flag.sum()
            loss_list.append(n_dead/system.nb)
            P00_list.append((prob_00[alive_flag]).sum() / system.nb)
            print("observe t={}ms, wall_time={:.2f}s, P00={}, molecular loss={}%".format(i*dt/1000, time()-t0, P00_list[-1], loss_list[-1]*100))
    return th.stack(P00_list), th.stack(loss_list)

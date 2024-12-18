import warnings
import logging
import numpy as np
import torch as th
from qepsilon.utilities import XY8
############################################################
# Single-qubit Tasks
############################################################

def RamseyScan(system, dt: float, T: float, theta_list: list[float], observe_at: list[float]):
    """
    Vanilla Ramsey experiment: scan the Ramsey contrast as a function of the free evolution time and the rotation angle.
    Args:
        system: the system object.
        dt: the time step size.
        T: the total time for the Ramsey experiment.
        theta_list: the list of rotation angles.
        observe_at: the list of times to observe the Ramsey fringe.
    Returns:
        P0_list: the 2D tensor of Ramsey fringe probabilities for different rotation angles and observation times. Shape (len(observe_at), len(theta_list)).
    """
    P0_list = []
    nsteps = int(T / dt) + 1 
    observe_steps = [int(t / dt) for t in observe_at]
    dm = system.density_matrix
    system.reset()
    ## set the initial state
    system.set_rho_by_config([0])
    # first pi/2 rotation along x
    system.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
    # free evolution
    for i in range(nsteps):
        system.step(dt=dt)
        if i in observe_steps:
            for theta in theta_list:
                u = th.tensor([np.cos(theta), np.sin(theta), 0.0], dtype=th.float)
                # apply the second pi/2 rotation along the direction u
                rho_t = dm.apply_unitary_rotation(system.rho, u, np.pi/2)
                # observe the probability of being in the state |0>
                prob_0 = dm.observe_prob_by_config(rho_t, th.tensor([0]))
                P0_list.append(prob_0.mean())
    return th.stack(P0_list).reshape(len(observe_at), len(theta_list))
 
def RamseySpinEcho(system, dt: float, T: float, theta_list: list[float]):
    """
    Ramsey experiment with a spin echo: obtain the Ramsey fringe for a given free evolution time T and a list of rotation angles.
    Args:
        system: the system object.
        dt: the time step size.
        T: the total time for the Ramsey experiment.
        theta_list: the list of rotation angles.
    Returns:
        P0_list: the 1D tensor of Ramsey fringe probabilities for different rotation angles. Shape (len(theta_list)).
    """
    P0_list = []
    nsteps = int(T / dt) + 1 
    dm = system.density_matrix
    ## the rotation direction of the second pi/2 rotation
    system.reset()
    ## set the initial state
    system.set_rho_by_config([0])
    # first pi/2 rotation along x
    system.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
    # free evolution for half of the total time
    for i in range(nsteps//2):
        system.step(dt=dt)
    ## spin echo
    system.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi)
    # free evolution for the other half of the total time
    for i in range(nsteps//2, nsteps):
        system.step(dt=dt)
    ## scan the Ramsey fringe for a list of rotation angles, then observe the probability of being in the state |0>
    for theta in theta_list:
        u = th.tensor([np.cos(theta), np.sin(theta), 0.0], dtype=th.float)
        rho_t = dm.apply_unitary_rotation(system.rho, u, np.pi/2)
        prob_0 = dm.observe_prob_by_config(rho_t, th.tensor([0]))
        P0_list.append(prob_0.mean())
    return th.stack(P0_list)
 

def RamseyXY8(system, dt: float, T: float, cycle_time: float, theta_list: list[float]):
    """
    Ramsey experiment with a XY8 dynamical decoupling sequence: obtain the Ramsey fringe for a given free evolution time T and a list of rotation angles.
    Args:
        system: the system object.
        dt: the time step size.
        T: the total time for the Ramsey experiment.
        cycle_time: the cycle time of the XY8 dynamical decoupling sequence.
        theta_list: the list of rotation angles.
    Returns:
        P0_list: the 1D tensor of Ramsey fringe probabilities for different rotation angles. Shape (len(theta_list)).
    """
    ## sanitary check and initiate the dynamical decoupling iterator
    if (cycle_time / dt) != int(cycle_time / dt):
        warnings.warn(f"The cycle time {cycle_time} is not an integer multiple of the time step size {dt}.")
    dd_iterator = XY8(cycle_time)
    ## 
    P0_list = []
    T = np.round(T/cycle_time) * cycle_time
    nsteps = int(np.round(T / dt)) 
    dm = system.density_matrix
    ## reset history of the system
    system.reset()
    ## set the initial state
    system.set_rho_by_config([0])
    # first pi/2 rotation along x
    system.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
    # free evolution with dynamical decoupling
    dd_t, dd_direction, dd_phase = next(dd_iterator)
    dd_step = int(np.round(dd_t / dt))
    for i in range(nsteps):
        if i == dd_step:
            system.rotate(direction=dd_direction, angle=dd_phase)
            dd_t, dd_direction, dd_phase = next(dd_iterator)
            dd_step = int(np.round(dd_t / dt))
        system.step(dt=dt)
    # second pi/2 rotation along the direction u
    for theta in theta_list:
        u = th.tensor([np.cos(theta), np.sin(theta), 0.0], dtype=th.float)
        rho_t = dm.apply_unitary_rotation(system.rho, u, np.pi/2)
        prob_0 = dm.observe_prob_by_config(rho_t, th.tensor([0]))
        P0_list.append(prob_0.mean())
    return th.stack(P0_list)


def RamseyScan_XY8(system, dt: float, T: float, cycle_time: float, theta_list: list[float], observe_at: list[float]):
    """
    Ramsey experiment with a XY8 dynamical decoupling sequence: scan the Ramsey contrast as a function of the free evolution time and the rotation angle.
    Args:
        system: the system object.
        dt: the time step size.
        T: the total time for the Ramsey experiment.
        cycle_time: the cycle time of the XY8 dynamical decoupling sequence.
        theta_list: the list of rotation angles.
        observe_at: the list of times to observe the Ramsey fringe.
    Returns:
        P0_list: the 2D tensor of Ramsey fringe probabilities for different rotation angles and observation times. Shape (len(observe_at), len(theta_list)).
    """
    ## sanitary check and initiate the dynamical decoupling iterator
    if (cycle_time / dt) != int(cycle_time / dt):
        warnings.warn(f"The cycle time {cycle_time} is not an integer multiple of the time step size {dt}.")
    dd_iterator = XY8(cycle_time)
    ## 
    P0_list = []
    nsteps = int(T/dt)+1
    observe_steps = []
    for obs_t in observe_at:
        obs_t_cycle_end = np.round(obs_t/cycle_time) * cycle_time
        obs_step = int(np.round(obs_t_cycle_end / dt))
        observe_steps.append(obs_step)
    dm = system.density_matrix
    ## reset history of the system
    system.reset()
    ## set the initial state
    system.set_rho_by_config([0])
    # first pi/2 rotation along x
    system.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
    # free evolution with dynamical decoupling
    dd_t, dd_direction, dd_phase = next(dd_iterator)
    dd_step = int(np.round(dd_t / dt))
    for i in range(nsteps):
        if i == dd_step:
            system.rotate(direction=dd_direction, angle=dd_phase)
            system.kraus_operate(system._channel_group_dict["depol_channel"].sample())
            dd_t, dd_direction, dd_phase = next(dd_iterator)
            dd_step = int(np.round(dd_t / dt))
        system.step(dt=dt)
        if i in observe_steps:
            assert dd_iterator.idx == 0, f"observe before a cycle of dynamical decoupling sequence is finished. The index of the current DD step is {dd_iterator.idx}."
            # second pi/2 rotation along the direction u
            for theta in theta_list:
                u = th.tensor([np.cos(theta), np.sin(theta), 0.0], dtype=th.float)
                rho_t = dm.apply_unitary_rotation(system.rho, u, np.pi/2)
                prob_0 = dm.observe_prob_by_config(rho_t, th.tensor([0]))
                P0_list.append(prob_0.mean())
    return th.stack(P0_list).reshape(len(observe_at), len(theta_list))

############################################################
# Two-qubit Tasks
############################################################
def RamseyScan_XY8_TwoQubits(system, dt: float, T: float, cycle_time: float, observe_at: list[float]):
    """
    Ramsey experiment with a XY8 dynamical decoupling sequence: scan the Ramsey contrast as a function of the free evolution time and the rotation angle.
    Args:
        system: the system object.
        dt: the time step size.
        T: the total time for the Ramsey experiment.
        cycle_time: the cycle time of the XY8 dynamical decoupling sequence.
        observe_at: the list of times to observe the Ramsey fringe.
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
    for i in range(nsteps):
        if i == dd_step:
            system.rotate(direction=dd_direction, angle=dd_phase)
            system.kraus_operate(system._channel_group_dict["depol_channel"].sample())
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
            d0 = pos[:,0,0] - tweezers_center[0][0]
            d1 = pos[:,1,0] - tweezers_center[1][0]
            alive_flag = alive_flag & (th.abs(d0) < 0.6) & (th.abs(d1) < 0.6)
            alive_flag = alive_flag & th.isfinite(prob_00)
            n_dead = system.nb - alive_flag.sum()
            loss_list.append(n_dead/system.nb)
            logging.info(f"At time {i*dt/1000}ms. the total number of lost system is {n_dead}, the number of NaN density matrix is {system.nb - th.isfinite(prob_00).sum()}")
            P00_list.append((prob_00[alive_flag]).sum() / system.nb)
            logging.info(f"The probability of |00> is {P00_list[-1]}")
            # dm_two_terms = system.rho[:, 1, 2] + system.rho[:, 2, 1]
            # logging.info(f"The coherance of the density matrix is {dm_two_terms}")
            # logging.info("Probability of |00> is {}".format(prob_00.numpy()))
            # logging.info("density matrix is {}".format(system.rho.numpy()))
    return th.stack(P00_list), th.stack(loss_list)

"""
This file contains the particle class for the QEpsilon project.
"""

import torch as th
import numpy as np
import warnings
from qepsilon.utilities import Constants

class OpticalTweezer:
    def __init__(self, min_waist, wavelength, max_depth, center: th.Tensor, axis: th.Tensor=th.tensor([0.0, 0.0, 1.0]), on = True):
        """
        Initialize the optical tweezer.
        Args:
            min_waist: float, the minimum waist of the tweezer
            wavelength: float, the wavelength of the tweezer
            max_depth: float, the maximum depth of the tweezer
            center: th.Tensor, shape (3), the center of the tweezer
            axis: th.Tensor, shape (3), the axis of the tweezer
        """
        self.min_waist = min_waist
        self.wavelength = wavelength
        self.max_depth = max_depth
        self.zR = np.pi * self.min_waist**2 / self.wavelength
        self.center = center
        self.axis = axis / th.norm(axis)  ## normalize the axis
        self.on = on
        self.pot_engine = None

    def get_pot(self, positions: th.Tensor):
        """
        Get the potential at a given position.
        Args:
            positions: th.Tensor, shape (nbatch, n_particles, 3), the positions of the particles
        Returns:
            the potential at the given position, shape (nbatch)
        """
        R = positions - self.center[None, None,:]
        z = (R * self.axis).sum(dim=-1) # (nbatch, n_particles)
        r = R - z[:,:, None] * self.axis[None, None,:]
        r = th.norm(r, dim=-1) # (nbatch, n_particles)
        wz = self.min_waist * th.sqrt(1 + (z / self.zR)**2)
        pot = - self.max_depth * (self.min_waist / wz)**2 * th.exp(-2 * r**2 / wz**2)
        return pot
    
    def init_pot_engine(self, example_input: th.Tensor):
        center = self.center
        axis = self.axis
        min_waist = self.min_waist
        max_depth = self.max_depth
        zR = self.zR
        def pot_engine(positions: th.Tensor):
            """
            Get the potential at a given position.
            """
            R = positions - center[None, None,:]
            z = (R * axis).sum(dim=-1) # (nbatch, n_particles)
            r = R - z[:,:, None] * axis[None, None,:]
            r = th.norm(r, dim=-1) # (nbatch, n_particles)
            wz = min_waist * th.sqrt(1 + (z / zR)**2)
            pot = - max_depth * (min_waist / wz)**2 * th.exp(-2 * r**2 / wz**2)
            pot = pot.sum()
            return pot
        self.pot_engine = th.jit.trace(pot_engine, example_input)
        return
    
    def get_force(self, positions: th.Tensor):
        """
        Get the force at a given position.
        """
        input = positions.clone()
        input.requires_grad = True
        pot = self.pot_engine(input)
        force = - th.autograd.grad(pot, input)[0]
        return force
      
    # def get_force(self, positions: th.Tensor):
    #     """
    #     Get the force at a given position.
    #     """
    #     input = positions.clone()
    #     input.requires_grad = True
    #     pot = self.get_pot(input).sum()
    #     force = - th.autograd.grad(pot, input)[0]
    #     return force

    def get_info(self):
        return {
            "on": self.on,
            "center": self.center,
            "axis": self.axis,
            "min_waist": self.min_waist,
            "wavelength": self.wavelength,
            "max_depth": self.max_depth,
        }


class Particles(th.nn.Module):
    """
    This class represents particles in the QEpsilon project.
    """

    def __init__(self, n_particles: int, batchsize: int = 1, mass: float = 1.0, radial_temp: float = 1.0, axial_temp: float = 1.0, dt: float = 0.1, tau: float = None):
        super().__init__()
        self.nq = n_particles
        self.nb = batchsize
        self.positions = th.zeros(batchsize, n_particles, 3)
        self.velocities = th.zeros(batchsize, n_particles, 3)
        self.forces = th.zeros(batchsize, n_particles, 3)
        self.masses = th.ones(n_particles) * mass
        self.radial_temp = radial_temp
        self.axial_temp = axial_temp
        self._traps_dict = {}
        self.traj = []
        self.dt = dt
        ## parameters for Langevin dynamics
        if tau is None:
            self.tau = 100 * dt
        else:
            self.tau = tau  
        self.gamma = 1.0 / self.tau

    ###########################################################################
    # Methods for dealing with optical tweezers
    ###########################################################################
    def init_tweezers(self, id, min_waist, wavelength, max_depth, center, axis, on=True):
        """
        Initialize a tweezer.
        Args:
            id: int, the id of the tweezer
            min_waist: float, the minimum waist of the tweezer
            wavelength: float, the wavelength of the tweezer
            max_depth: float, the maximum depth of the tweezer
            center: th.Tensor, shape (3), the center of the tweezer
            axis: th.Tensor, shape (3), the axis of the tweezer
        """
        ## sanitary check
        if id in self._traps_dict:
            raise ValueError(f"Tweezer with id {id} already exists")
        self._traps_dict[id] = OpticalTweezer(min_waist, wavelength, max_depth, center, axis, on)
    
    def get_tweezer_by_id(self, id):
        return self._traps_dict[id]

    def remove_tweezer(self, id):
        del self._traps_dict[id]

    def turn_on_tweezer(self, id):
        self._traps_dict[id].on = True
    
    def turn_off_tweezer(self, id):
        self._traps_dict[id].on = False
 
    def get_trapping_info(self):
        infos = {}
        for id, trap in self._traps_dict.items():
            infos[id] = trap.get_info()
        return infos

    ###########################################################################
    # Methods for dealing with particles
    ###########################################################################
    def get_positions(self):
        return self.positions.clone().detach()
    
    def get_velocities(self):
        return self.velocities.clone().detach()

    def get_temperature(self):
        temp = (self.get_velocities()**2) * self.masses[None,:,None] / Constants.kb
        temp = temp.mean(1)
        return temp

    def get_forces(self):
        return self.forces.clone().detach()

    def get_trajectory(self):
        return th.stack(self.traj)
    
    def set_positions(self, positions: th.Tensor):
        if positions.shape == (self.nb, self.nq, 3):
            self.positions = positions
        elif positions.shape == (self.nq, 3):
            self.positions = positions.repeat(self.nb,1,1)
        else:
            raise ValueError(f"Positions must have shape ({self.nb}, {self.nq}, 3) or ({self.nq}, 3)")

    def set_positions_at_tweezer_center(self):
        self.positions = th.zeros_like(self.positions)
        if len(self._traps_dict) != self.nq:
            raise ValueError(f"Number of tweezers ({len(self._traps_dict)}) must be equal to the number of particles ({self.nq}) if setting positions at tweezer centers")
        for idx, trap in enumerate(self._traps_dict.values()):
            self.positions[:, idx, :] += trap.center.to(dtype=self.positions.dtype, device=self.positions.device)

    def set_gaussian_velocities(self, radial_temp: float = None, axial_temp: float = None):
        if radial_temp is None:
            radial_temp = self.radial_temp
        if axial_temp is None:
            axial_temp = self.axial_temp
        self.velocities = th.randn_like(self.positions)
        temp = th.tensor([self.radial_temp, self.radial_temp, self.axial_temp], dtype=self.positions.dtype, device=self.positions.device)
        self.velocities *= th.sqrt(Constants.kb * temp)[None,None,:]
        self.velocities *= th.sqrt(1 / self.masses[None,:,None])

    def set_velocities(self, velocities: th.Tensor):
        if velocities.shape != (self.nb, self.nq, 3):
            raise ValueError(f"Velocities must have shape ({self.nb}, {self.nq}, 3)")
        self.velocities = velocities

    def zero_forces(self):
        self.forces = th.zeros_like(self.positions).detach()

    def get_trapping_forces(self):
        """
        Get the trapping forces on the particles.
        """
        forces = th.zeros_like(self.positions.detach())
        for trap in self._traps_dict.values():
            if trap.pot_engine is None:
                trap.init_pot_engine(self.positions)
            if trap.on:
                forces += trap.get_force(self.positions)
        return forces
    
    def modify_forces(self, df: th.Tensor):
        """
        Modify the forces on the particles.
        """
        self.forces += df

    def reset(self):
        """
        Reset the particles to a thermal equilibrium state.
        """
        self.set_positions_at_tweezer_center()
        self.set_gaussian_velocities()
        self.equilibrate(nsteps=200, tau=100*self.dt)
        self.traj = []


    ###########################################################################
    # Methods for simulating dynamics
    ###########################################################################    
    def get_noise(self):
        noise = th.randn_like(self.positions)
        temp = th.tensor([self.radial_temp, self.radial_temp, self.axial_temp], dtype=self.positions.dtype, device=self.positions.device)
        noise *= th.sqrt(Constants.kb * temp)[None,None,:]
        noise *= th.sqrt(1 / self.masses[None,:,None])
        return noise

    def step_langevin(self, record_traj=False):
        """
        Isothermal Langevin dynamics. Update the positions and velocities by one time step with mid-point Langevin method.
        """
        dt = self.dt
        ## parameters for Langevin dynamics
        z1 = np.exp(-dt * self.gamma)
        z2 = np.sqrt(1 - np.exp(-2 * dt * self.gamma))
        ## initial values
        v0 = self.get_velocities()
        x0 = self.get_positions()
        ## update velocities for one step with gradient force
        v1 = v0 + self.forces * dt / self.masses[None,:,None]
        ## update positions for half step
        x1 = x0 + 0.5 * v1 * dt
        ## modify velocity with damping and random force
        noise = self.get_noise()
        v1 = z1 * v1 + z2 * noise
        ## update positions for half step
        x1 += 0.5 * v1 * dt
        ## wrap up
        self.set_positions(x1)
        self.set_velocities(v1)
        if record_traj:
            self.traj.append(x1.clone().detach())
        return
        
    def equilibrate(self, nsteps: int, tau: float = 1):
        """
        Equilibrate the particles with Langevin dynamics.
        """
        dt = self.dt
        system_damping = self.gamma * 1.0
        self.gamma = 1 / tau
        for _ in range(nsteps):
            self.zero_forces()
            self.modify_forces(self.get_trapping_forces())
            self.step_langevin(record_traj=False)
        self.gamma = system_damping
        return

    def step_adiabatic(self, record_traj=False):
        """
        Adiabatic dynamics. Update the positions and velocities by one time step with leapfrog method.
        """
        dt = self.dt
        ## initial values
        v0 = self.get_velocities()
        x0 = self.get_positions()
        ## update velocities for one step with gradient force
        v1 = v0 + self.forces * dt / self.masses[None,:,None]
        ## update positions for half step
        x1 = x0 + v1 * dt
        ## wrap up
        self.set_positions(x1)
        self.set_velocities(v1)
        if record_traj:
            self.traj.append(x1.clone().detach())
        return


class PathIntegralParticles(Particles):
    def __init__(self, n_particles: int, batchsize: int = 1, mass: float = 1.0, temperature: float = 1.0):
        super().__init__(n_particles, batchsize, mass)
        self.temperature = temperature
        self.spring_constant = None
        raise NotImplementedError("Path integral particles are not implemented yet")

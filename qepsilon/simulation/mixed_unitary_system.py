"""
This file contains the mixed classical-quantum unitary simulation class for the QEpsilon project.
"""

import numpy as np
import torch as th
from qepsilon.operator_group.spin_operators import spin_oscillators_interaction
from qepsilon.operator_group.tb_operators import tb_oscillators_interaction
from qepsilon.simulation.unitary_system import QubitUnitarySystem, TightBindingUnitarySystem
from qepsilon.system.particles import Particles
import logging
import warnings
from time import time as timer


class OscillatorQubitUnitarySystem(QubitUnitarySystem):
    """
    This class represents the states of mixed classical quantum system. 
    This class is not general since only (classical position * onsite number operator) type of classical-quantum coupling is supported.
    Implementation of other types of classical-quantum coupling is possible by modifying the `bind_oscillators_to_qubit` method and implementing the needed coupling operator as a OperatorGroup object.

    The classical degrees of freedom are harmonic oscillators, represented by a Particles object.
    The quantum degrees of freedom are represented by a QubitPureEnsemble object.
    """
    def __init__(self, n_qubits: int, batchsize: int, cls_dt: float = 0.1):
        """
        Args:
            n_qubits: int, the number of qubits.
            batchsize: int, the batch size.
            cls_dt: float, the time step of the classical oscillators.
        """
        super().__init__(n_qubits, batchsize)
        self._oscillator_dict = {}
        self.cls_dt = cls_dt

    ############################################################
    # Getters and setters 
    ############################################################

    def get_oscilator_by_id(self, id: str):
        if id in self._oscillator_dict:
            return self._oscillator_dict[id]
        else:
            raise ValueError(f"The oscillator with id {id} does not exist.")
        
    def get_all_oscillators(self):
        return [self.get_oscilator_by_id(id) for id in self._oscillator_dict]

    def reset(self, temp: float = None):
        """
        Reset the system. 
        In addition to the reset of the quantum system, the center-of-mass positions and momenta of the particles are also reset to arbitrary thermal states.
        """
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.reset()
        for oscillator in self.get_all_oscillators():
            oscillator['particles'].reset(temp=temp) 
        return
        
    ############################################################
    # Methods for adding classical oscillators to the system and binding them to the tight-binding system.
    ############################################################
    
    def add_classical_oscillators(self, id: str, nmodes: int, freqs: th.Tensor, masses: th.Tensor, couplings: th.Tensor, x0: th.Tensor = None, init_temp: float = None, tau: float = None, unit: str = 'pm_ps'):
        """
        Add onsite classical oscillators to the system. 
        Args:
            id: str, the id of the oscillator.
            nmodes: int, the number of modes of the oscillator.
            freqs: th.Tensor, the frequencies of the oscillators. shape: (nmodes,)
            masses: th.Tensor, the masses of the oscillators. shape: (nmodes,)
            couplings: th.Tensor, the couplings between the oscillators and the qubits. shape: (nmodes,)
            x0: th.Tensor, the initial positions of the oscillators. shape: (nmodes, ndim)
            tau: float, the relaxation time of the oscillators.
            unit: str, the unit of the oscillator motion.
        """
        ## sanity check
        if freqs.shape != (nmodes,):
            raise ValueError(f"The number of frequencies must be equal to the number of modes.")
        if masses.shape != (nmodes,):
            raise ValueError(f"The number of masses must be equal to the number of modes.")
        if couplings.shape != (nmodes,):
            raise ValueError(f"The number of couplings must be equal to the number of modes.")
        
        ## add classical oscillators as approximate bosonic modes
        oscillator = Particles(nmodes, batchsize=self.nb, ndim=1, mass=masses, dt=self.cls_dt, tau=tau, unit=unit)
        self._oscillator_dict[id] = {
            'nmodes': nmodes,
            'freqs': freqs,
            'masses': masses,
            'couplings': couplings,
            'coefs': couplings * freqs * th.sqrt(2 * masses * freqs),
            'binding_qubit':None,
            'binding_interaction':None,
            'particles': oscillator
        }
        if x0 is not None:
            oscillator.set_positions(x0)
        if init_temp is not None:
            if type(init_temp) == float:
                oscillator.set_gaussian_velocities(temp=init_temp)
            else:
                raise ValueError(f"The initial temperature must be a float.")
        logging.info(f"The oscillator with id {id} is added to the system. It has {nmodes} modes, with frequencies {freqs}, masses {masses}, and couplings {couplings}.")
        logging.info(f"It has not been bound to any qubit yet. Use the method `bind_oscillators_to_qubit` to bind it to a qubit.")
        return

    def bind_oscillators_to_qubit(self, qubit_idx: int, oscillators_id: str, requires_grad: bool = False):
        """
        Bind the oscillators to a qubit.
        Each oscillator is coupled to the qubit with interaction $g \omega \sqrt{2M\omega} x \hat{N}$. 
        Here $g$ is the coupling strength, $\omega$ is the frequency of the oscillator, $M$ is the mass of the oscillator, and $\hat{N}$ is the number operator of the qubit.
        """
        if qubit_idx >= self.nq:
            raise ValueError(f"The qubit index {qubit_idx} is out of range.")
        if qubit_idx < 0:
            raise ValueError(f"The qubit index {qubit_idx} is negative.")
        oscillators = self._oscillator_dict[oscillators_id]
        ## get the interaction operator
        epc_op = spin_oscillators_interaction(self.nq, id=f"site-{qubit_idx}_{oscillators_id}_epc", batchsize=self.nb, particles=oscillators['particles'], coef=oscillators['coefs'], requires_grad=requires_grad)
        pauli_sequence = ["I"] * self.nq
        pauli_sequence[qubit_idx] = "N"
        pauli_sequence = "".join(pauli_sequence)
        epc_op.add_operator(pauli_sequence)
        ## add the operator to the Hamiltonian
        self.add_operator_group_to_hamiltonian(epc_op)
        ## store the binding information in the oscillator dictionary
        if oscillators['binding_qubit'] is not None:
            raise ValueError(f"The oscillator with id {oscillators_id} is already bound to a qubit.")
        if oscillators['binding_interaction'] is not None:
            raise ValueError(f"The oscillator with id {oscillators_id} is already bound to a qubit.")
        oscillators['binding_qubit'] = qubit_idx
        oscillators['binding_interaction'] = epc_op
        logging.info(f"The oscillator with id {oscillators_id} is bound to site-{qubit_idx}.")
        return epc_op

    ############################################################
    # Integration of the system
    ############################################################

    def step_particles(self, temp: float = None, feedback: bool = True):
        """
        This function steps the thermal motino of the particles for a time step.
        There are two types of forces on the particles:
        1. The harmonic force from the harmonic trap.
        2. The binding force from the interaction between the oscillators and the qubit.
        """
        dt = self.cls_dt
        all_oscillators = self.get_all_oscillators()
        with th.no_grad():
            for oscillator in all_oscillators:
                qubit_idx = oscillator['binding_qubit']
                omegas = oscillator['freqs'].reshape(oscillator['nmodes'],1)
                ## zero the forces
                particles = oscillator['particles']
                particle_dim = particles.ndim
                if particle_dim > 1:
                    raise NotImplementedError(f"The Ehrenfest force is currently only implemented for one-dimensional classical oscillators.")
                particles.zero_forces()
                ## compute the harmonic force
                particles.modify_forces_by_harmonic_trap(omega=omegas)
                if feedback:
                    ## compute the non-adiabatic force from classical-quantum coupling
                    if oscillator['binding_interaction'].op_static is None:
                        ehrenfest_op, _ = oscillator['binding_interaction'].sample(dt=None)
                    else:
                        ehrenfest_op = oscillator['binding_interaction'].op_static
                    ehrenfest_op_exp = self.pure_ensemble.get_expectation(ehrenfest_op)
                    if ehrenfest_op_exp.shape != (self.nb,):
                        raise ValueError(f"The shape of the expectation value of the Ehrenfest operator must be (batchsize,).")
                    ehrenfest_op_exp = ehrenfest_op_exp.to(device=particles.positions.device)
                    exp_real = ehrenfest_op_exp.real
                    exp_imag = ehrenfest_op_exp.imag
                    epc_coef = oscillator['binding_interaction'].coef.detach().to(device=particles.positions.device)
                    ehrenfest_force = - epc_coef[None, :, None] * exp_real[:, None, None]
                    particles.modify_forces(ehrenfest_force)
                ## step the particles
                if temp is not None:
                    particles.step_langevin(record_traj=False, temp=temp)
                else:
                    particles.step_adiabatic(record_traj=False)
        return

    def step(self, dt: float, temp: float = None, set_buffer: bool = False, profile: bool = False, feedback: bool = True):
        """
        This function steps the system for a time step dt. Overrides the step function in the QubitUnitarySystem class.
        """
        if dt != self.cls_dt:
            raise NotImplementedError(f"The time step of the quantum system must be the same as the time step ({self.cls_dt}) of the classical oscillators.")
        if profile:
            t0 = timer()
            th.cuda.synchronize()
        self.step_particles(temp=temp, feedback=feedback)
        if profile:
            th.cuda.synchronize()
            t1 = timer()
            logging.info(f"The time taken for stepping the Ehrenfest dynamics of particles is {t1 - t0}s.")
        hamiltonian = self.step_hamiltonian(dt, set_buffer)
        if profile:
            th.cuda.synchronize()
            t2 = timer()
            logging.info(f"The time taken for stepping the Hamiltonian is {t2 - t1}s.")
        self.pse = self.step_pse(dt, hamiltonian, set_buffer)
        if profile:
            th.cuda.synchronize()
            t3 = timer()
            logging.info(f"The time taken for stepping the quantum states is {t3 - t2}s.")
        return self.pse



class OscillatorTightBindingUnitarySystem(TightBindingUnitarySystem):
    """
    This class represents the states of mixed classical quantum system. 

    The classical degrees of freedom are harmonic oscillators, represented by a Particles object.
    The quantum degrees of freedom are represented by a TightBindingPureEnsemble object, i.e. single-particle tight binding system.
    """
    def __init__(self, n_sites: int, batchsize: int, cls_dt: float = 0.1):
        """
        Args:
            n_sites: int, the number of sites.
            batchsize: int, the batch size.
            cls_dt: float, the time step of the classical oscillators.
        """
        super().__init__(n_sites, batchsize)
        self._oscillator_dict = {}
        self.cls_dt = cls_dt

    ############################################################
    # Getters and setters 
    ############################################################

    def get_oscilator_by_id(self, id: str):
        if id in self._oscillator_dict:
            return self._oscillator_dict[id]
        else:
            raise ValueError(f"The oscillator with id {id} does not exist.")
        
    def get_all_oscillators(self):
        return [self.get_oscilator_by_id(id) for id in self._oscillator_dict]

    def reset(self, temp: float = None):
        """
        Reset the system. 
        In addition to the reset of the quantum system, the center-of-mass positions and momenta of the particles are also reset to arbitrary thermal states.
        """
        for operator_group in self._hamiltonian_operator_group_dict.values():
            operator_group.reset()
        for oscillator in self.get_all_oscillators():
            oscillator['particles'].reset(temp=temp) 
        return
        
    ############################################################
    # Methods for adding classical oscillators to the system and binding them to the tight-binding system.
    ############################################################
    
    def add_classical_oscillators(self, id: str, nmodes: int, freqs: th.Tensor, masses: th.Tensor, couplings: th.Tensor, x0: th.Tensor = None, init_temp: float = None, tau: float = None, unit: str = 'pm_ps'):
        """
        Add onsite classical oscillators to the system. 
        Args:
            id: str, the id of the oscillator.
            nmodes: int, the number of modes of the oscillator.
            freqs: th.Tensor, the frequencies of the oscillators. shape: (nmodes,)
            masses: th.Tensor, the masses of the oscillators. shape: (nmodes,)
            couplings: th.Tensor, the couplings between the oscillators and the tight-binding sites. shape: (nmodes,)
            x0: th.Tensor, the initial positions of the oscillators. shape: (batchsize, nmodes, ndim) or (nmodes, ndim)
            tau: float, the relaxation time of the oscillators.
            unit: str, the unit of the oscillator motion.
        """
        ## sanity check
        if freqs.shape != (nmodes,):
            raise ValueError(f"The number of frequencies must be equal to the number of modes.")
        if masses.shape != (nmodes,):
            raise ValueError(f"The number of masses must be equal to the number of modes.")
        if couplings.shape != (nmodes,):
            raise ValueError(f"The number of couplings must be equal to the number of modes.")
        
        ## add classical oscillators as approximate bosonic modes
        oscillator = Particles(nmodes, batchsize=self.nb, ndim=1, mass=masses, dt=self.cls_dt, tau=tau, unit=unit)
        self._oscillator_dict[id] = {
            'nmodes': nmodes,
            'freqs': freqs,
            'masses': masses,
            'couplings': couplings,
            'coefs': couplings * freqs * th.sqrt(2 * masses * freqs),
            'binding_site':None,
            'binding_interaction':None,
            'particles': oscillator
        }
        if x0 is not None:
            oscillator.set_positions(x0)
        if init_temp is not None:
            if type(init_temp) == float:
                oscillator.set_gaussian_velocities(temp=init_temp)
            else:
                raise ValueError(f"The initial temperature must be a float.")
        logging.info(f"The oscillator with id {id} is added to the system. It has {nmodes} modes, with frequencies {freqs}, masses {masses}, and couplings {couplings}.")
        logging.info(f"It has not been bound to any quantum degrees of freedom yet. Use the method `bind_oscillators_to_tb` to bind it to a tight-binding system.")
        return oscillator

    def bind_oscillators_to_tb(self, site_idx: int, oscillators_id: str, requires_grad: bool = False):
        """
        Bind the oscillators to a tight-binding site.
        Each oscillator is coupled to the tight-binding site with interaction $g \omega \sqrt{2M\omega} x \hat{N}$. 
        Here $g$ is the coupling strength, $\omega$ is the frequency of the oscillator, $M$ is the mass of the oscillator, and $\hat{N}$ is the number operator of the tight-binding site.
        """
        if site_idx >= self.ns:
            raise ValueError(f"The tight-binding site index {site_idx} is out of range.")
        if site_idx < 0:
            raise ValueError(f"The tight-binding site index {site_idx} is negative.")
        oscillators = self._oscillator_dict[oscillators_id]
        ## get the interaction operator
        epc_op = tb_oscillators_interaction(self.ns, id=f"site-{site_idx}_{oscillators_id}_epc", batchsize=self.nb, particles=oscillators['particles'], coef=oscillators['coefs'], requires_grad=requires_grad)
        tb_sequence = ["X"] * self.ns
        tb_sequence[site_idx] = "N"
        tb_sequence = "".join(tb_sequence)
        epc_op.add_operator(tb_sequence)
        ## add the operator to the Hamiltonian
        self.add_operator_group_to_hamiltonian(epc_op)
        ## store the binding information in the oscillator dictionary
        if oscillators['binding_site'] is not None:
            raise ValueError(f"The oscillator with id {oscillators_id} is already bound to a tight-binding site.")
        if oscillators['binding_interaction'] is not None:
            raise ValueError(f"The oscillator with id {oscillators_id} is already bound to a tight-binding site.")
        oscillators['binding_site'] = site_idx
        oscillators['binding_interaction'] = epc_op
        logging.info(f"The oscillator with id {oscillators_id} is bound to site-{site_idx}.")
        return epc_op

    ############################################################
    # Integration of the system
    ############################################################

    def step_particles(self, temp: float = None, feedback: bool = True):
        """
        This function steps the thermal motino of the particles for a time step.
        There are two types of forces on the particles:
        1. The harmonic force from the harmonic trap.
        2. The binding force from the interaction between the oscillators and the tight-binding sites.
        """
        dt = self.cls_dt
        all_oscillators = self.get_all_oscillators()
        self.normalize()
        pse_site_prob = th.abs(self.pse)**2
        pse_site_prob = pse_site_prob.to(device=all_oscillators[0]['particles'].positions.device)
        for oscillator in all_oscillators:
            site_idx = oscillator['binding_site']
            omegas = oscillator['freqs'].reshape(oscillator['nmodes'],1)
            ## zero the forces
            particles = oscillator['particles']
            particle_dim = particles.ndim
            if particle_dim > 1:
                raise NotImplementedError(f"The Ehrenfest force is currently only implemented for one-dimensional classical oscillators.")
            particles.zero_forces()
            ## compute the harmonic force
            particles.modify_forces_by_harmonic_trap(omega=omegas)
            #################### HACK start ####################
            # compute the non-adiabatic force from classical-quantum coupling
            # if oscillator['binding_interaction'].op_static is None:
            #     ehrenfest_op, _ = oscillator['binding_interaction'].sample(dt=None)
            # else:
            #     ehrenfest_op = oscillator['binding_interaction'].op_static
            # ehrenfest_op_exp = self.pure_ensemble.get_expectation(ehrenfest_op)
            # if ehrenfest_op_exp.shape != (self.nb,):
            #     raise ValueError(f"The shape of the expectation value of the Ehrenfest operator must be (batchsize,).")
            # ehrenfest_op_exp = ehrenfest_op_exp.to(device=particles.positions.device).real
            # this is a hack to speed up the computation; works only for the case that the quantum operator is the onsite number operator
            # ehrenfest_op_exp = th.abs(self.pse[:, site_idx])**2
            # ehrenfest_op_exp = ehrenfest_op_exp.to(device=particles.positions.device)
            #################### HACK end ####################
            if feedback:
                # ehrenfest_force = - oscillator['coefs'][None, :, None] * ehrenfest_op_exp[:, None, None]
                epc_coef = oscillator['binding_interaction'].coef.detach().to(device=particles.positions.device)
                ehrenfest_force = - epc_coef[None, :, None] * pse_site_prob[:, [site_idx], None]
                particles.modify_forces(ehrenfest_force)
            ## step the particles
            if temp is not None:
                particles.step_langevin(record_traj=False, temp=temp)
            else:
                particles.step_adiabatic(record_traj=False)
        return

    def step(self, dt: float, temp: float = None, set_buffer: bool = False, profile: bool = False, feedback: bool = True):
        """
        This function steps the system for a time step dt. Overrides the step function in the TightBindingUnitarySystem class.
        """
        if dt != self.cls_dt:
            raise NotImplementedError(f"The time step of the quantum system must be the same as the time step ({self.cls_dt}) of the classical oscillators.")
        if profile:
            t0 = timer()
            th.cuda.synchronize()
        self.step_particles(temp=temp, feedback=feedback)
        if profile:
            th.cuda.synchronize()
            t1 = timer()
            logging.info(f"The time taken for stepping the Ehrenfest dynamics of particles is {t1 - t0}s.")
        hamiltonian = self.step_hamiltonian(dt, set_buffer)
        if profile:
            th.cuda.synchronize()
            t2 = timer()
            logging.info(f"The time taken for stepping the Hamiltonian is {t2 - t1}s.")
        self.pse = self.step_pse(dt, hamiltonian, set_buffer)
        if profile:
            th.cuda.synchronize()
            t3 = timer()
            logging.info(f"The time taken for stepping the quantum states is {t3 - t2}s.")
        return self.pse

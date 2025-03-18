"""
This file contains the mixed classical-quantum unitary simulation class for the QEpsilon project.
"""

import numpy as np
import torch as th
from qepsilon.operator_group.spin_operators import spin_oscillators_interaction
from qepsilon.simulation.unitary_system import QubitUnitarySystem
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

    def add_classical_oscillators(self, id: str, nmodes: int, freqs: th.Tensor, masses: th.Tensor, couplings: th.Tensor, tau: float = None, unit: str = 'pm_ps'):
        """
        Add onsite classical oscillators to the system. 
        Args:
            id: str, the id of the oscillator.
            nmodes: int, the number of modes of the oscillator.
            freqs: th.Tensor, the frequencies of the oscillators.
            masses: th.Tensor, the masses of the oscillators.
            couplings: th.Tensor, the couplings between the oscillators and the qubits.
            dt: float, the time step of the oscillator motion.
        """
        ## sanity check
        if freqs.shape != (nmodes,):
            raise ValueError(f"The number of frequencies must be equal to the number of modes.")
        if masses.shape != (nmodes,):
            raise ValueError(f"The number of masses must be equal to the number of modes.")
        if couplings.shape != (nmodes,):
            raise ValueError(f"The number of couplings must be equal to the number of modes.")
        ## add classical oscillators as approximate bosonic modes
        self._oscillator_dict[id] = {
            'nmodes': nmodes,
            'freqs': freqs,
            'masses': masses,
            'couplings': couplings,
            'coefs': couplings * freqs * th.sqrt(2 * masses * freqs),
            'binding_qubit':None,
            'binding_interaction':None,
            'particles': Particles(nmodes, batchsize=self.nb, ndim=1, mass=masses, dt=self.cls_dt, tau=tau, unit=unit)
        }
        logging.info(f"The oscillator with id {id} is added to the system. It has {nmodes} modes, with frequencies {freqs}, masses {masses}, and couplings {couplings}.")
        logging.info(f"It has not been bound to any qubit yet. Use the method `bind_oscillators_to_qubit` to bind it to a qubit.")
        return

    def get_oscilator_by_id(self, id: str):
        if id in self._oscillator_dict:
            return self._oscillator_dict[id]
        else:
            raise ValueError(f"The oscillator with id {id} does not exist.")
        
    def bind_oscillators_to_qubit(self, qubit_id: str, oscillators_id: str):
        """
        Bind the oscillators to a qubit.
        Each oscillator is coupled to the qubit with interaction $g \omega \sqrt{2M\omega} x \hat{N}$. 
        Here $g$ is the coupling strength, $\omega$ is the frequency of the oscillator, $M$ is the mass of the oscillator, and $\hat{N}$ is the number operator of the qubit.
        """
        if qubit_id not in self._qubit_dict:
            raise ValueError(f"The qubit with id {qubit_id} does not exist.")
        oscillators = self._oscillator_dict[oscillators_id]
        ## get the interaction operator
        epc_op = spin_oscillators_interaction(self.n_qubits, id=f"{qubit_id}_{oscillators_id}_epc", batchsize=self.nb, particles=oscillators['particles'], coef=oscillators['coefs'], requires_grad=False)
        pauli_sequence = "I" * self.n_qubits
        pauli_sequence[qubit_id] = "N"
        epc_op.add_operator(pauli_sequence)
        ## add the operator to the Hamiltonian
        self.add_operator_group_to_hamiltonian(epc_op)
        ## store the binding information in the oscillator dictionary
        if oscillators['binding_qubit'] is not None:
            raise ValueError(f"The oscillator with id {oscillators_id} is already bound to a qubit.")
        if oscillators['binding_interaction'] is not None:
            raise ValueError(f"The oscillator with id {oscillators_id} is already bound to a qubit.")
        oscillators['binding_qubit'] = qubit_id
        oscillators['binding_interaction'] = epc_op
        logging.info(f"The oscillator with id {oscillators_id} is bound to the qubit with id {qubit_id}.")
        return


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

    def step_particles(self):
        """
        This function steps the thermal motino of the particles for a time step.
        There are two types of forces on the particles:
        1. The harmonic force from the harmonic trap.
        2. The binding force from the interaction between the oscillators and the qubit.
        """
        dt = self.cls_dt
        all_oscillators = self.get_all_oscillators()
        for oscillator in all_oscillators:
            omegas = oscillator['freqs'].reshape(oscillator['nmodes'],1)
            ## zero the forces
            particles = oscillator['particles']
            particles.zero_forces()
            ## compute the harmonic force
            particles.modify_forces_by_harmonic_trap(omega=omegas)
            ## compute the non-adiabatic force from classical-quantum coupling
            ehrenfest_op = oscillator['binding_interaction'].sum_operators()
            ehrenfest_op_exp = self.pse.get_expectation(ehrenfest_op)
            ehrenfest_force = - oscillator['coefs'] * ehrenfest_op_exp 
            particles.modify_forces(ehrenfest_force)
            ## step the particles
            particles.step_langevin(record_traj=True)
        return

    def step(self, dt: float, profile: bool = False):
        """
        This function steps the system for a time step dt. Overrides the step function in the QubitUnitarySystem class.
        """
        if dt != self.cls_dt:
            raise NotImplementedError(f"The time step of the quantum system must be the same as the time step ({self.cls_dt}) of the classical oscillators.")
        if profile:
            t0 = timer()
            th.cuda.synchronize()
        self.step_particles()
        if profile:
            th.cuda.synchronize()
            t1 = timer()
            logging.info(f"The time taken for stepping the Ehrenfest dynamics of particles is {t1 - t0}s.")
        hamiltonian = self.step_hamiltonian(dt, set_buffer=True)
        if profile:
            th.cuda.synchronize()
            t2 = timer()
            logging.info(f"The time taken for stepping the Hamiltonian is {t2 - t1}s.")
        self.pse = self.step_pse(dt, hamiltonian)
        if profile:
            th.cuda.synchronize()
            t3 = timer()
            logging.info(f"The time taken for stepping the quantum states is {t3 - t2}s.")
        return self.pse

Basic Usage Examples
===================

This section covers fundamental examples to get you started with QEpsilon.

Two-Level System Decay
----------------------

This example demonstrates the basic usage of QEpsilon by simulating spontaneous emission from a two-level system.

.. code-block:: python

   import qepsilon as qe
   import numpy as np
   import matplotlib.pyplot as plt

   # Create a two-level system
   tls = qe.TLS()
   
   # System parameters
   omega = 1.0    # Energy splitting (GHz)
   gamma = 0.1    # Decay rate (GHz)
   
   # Define Hamiltonian and Lindblad operator
   H = omega * tls.sigma_z()
   L_decay = np.sqrt(gamma) * tls.sigma_minus()
   
   # Create the simulation
   sim = qe.LindbladSystem(hamiltonian=H, lindblad_operators=[L_decay])
   
   # Initial state (excited state)
   rho_0 = tls.excited_state()
   
   # Time evolution
   times = np.linspace(0, 20, 200)
   results = sim.evolve(rho_0, times)
   
   # Calculate population in excited state
   excited_pop = [np.real(np.trace(rho @ tls.sigma_plus() @ tls.sigma_minus())) 
                  for rho in results]
   
   # Plot results
   plt.figure(figsize=(8, 6))
   plt.plot(times, excited_pop, 'b-', linewidth=2, label='Excited state population')
   plt.plot(times, np.exp(-gamma * times), 'r--', linewidth=2, label='Exponential decay')
   plt.xlabel('Time (ns)')
   plt.ylabel('Population')
   plt.legend()
   plt.title('Two-Level System Spontaneous Emission')
   plt.grid(True, alpha=0.3)
   plt.show()

**Physics**: This example shows how a two-level system initially prepared in the excited state decays to the ground state through spontaneous emission. The population follows an exponential decay :math:`P(t) = e^{-\gamma t}`.

Harmonic Oscillator
-------------------

Simulation of a quantum harmonic oscillator with thermal noise.

.. code-block:: python

   import qepsilon as qe
   import numpy as np

   # Create a bosonic mode (harmonic oscillator)
   boson = qe.Boson(max_occupation=20)
   
   # System parameters
   omega = 1.0      # Oscillator frequency
   gamma = 0.05     # Damping rate
   n_th = 2.0       # Thermal occupation number
   
   # Hamiltonian
   H = omega * boson.number_operator()
   
   # Lindblad operators for thermal bath
   L_cooling = np.sqrt(gamma * (n_th + 1)) * boson.annihilation()
   L_heating = np.sqrt(gamma * n_th) * boson.creation()
   
   # Create simulation
   sim = qe.LindbladSystem(
       hamiltonian=H,
       lindblad_operators=[L_cooling, L_heating]
   )
   
   # Initial state (coherent state)
   alpha = 3.0  # Coherent state amplitude
   rho_0 = boson.coherent_state(alpha)
   
   # Time evolution
   times = np.linspace(0, 10, 100)
   results = sim.evolve(rho_0, times)
   
   # Calculate average occupation number
   occupation = [np.real(np.trace(rho @ boson.number_operator())) 
                 for rho in results]
   
   print(f"Initial occupation: {occupation[0]:.2f}")
   print(f"Final occupation: {occupation[-1]:.2f}")
   print(f"Thermal occupation: {n_th:.2f}")

**Physics**: The harmonic oscillator starts in a coherent state and evolves towards thermal equilibrium with the environment. The final occupation number approaches the thermal value :math:`n_{th}`.

Rabi Oscillations
-----------------

Demonstration of Rabi oscillations in a driven two-level system.

.. code-block:: python

   import qepsilon as qe
   import numpy as np

   # Create two-level system
   tls = qe.TLS()
   
   # System parameters
   omega_0 = 5.0    # Qubit frequency
   omega_d = 5.0    # Drive frequency (on-resonance)
   Omega = 0.5      # Rabi frequency
   
   # Time-dependent Hamiltonian
   def H_drive(t):
       return (omega_0/2) * tls.sigma_z() + \
              (Omega/2) * (np.cos(omega_d * t) * tls.sigma_x() + 
                          np.sin(omega_d * t) * tls.sigma_y())
   
   # Unitary evolution (no decoherence)
   sim = qe.UnitarySystem(hamiltonian=H_drive)
   
   # Initial state (ground state)
   rho_0 = tls.ground_state()
   
   # Time evolution
   t_rabi = 2 * np.pi / Omega  # Rabi period
   times = np.linspace(0, 3 * t_rabi, 300)
   results = sim.evolve(rho_0, times)
   
   # Calculate populations
   ground_pop = [np.real(np.trace(rho @ tls.ground_projector())) for rho in results]
   excited_pop = [np.real(np.trace(rho @ tls.excited_projector())) for rho in results]
   
   # Theoretical Rabi oscillations
   theory_excited = np.sin(Omega * times / 2)**2
   
   plt.figure(figsize=(10, 6))
   plt.plot(times/t_rabi, excited_pop, 'b-', linewidth=2, label='Simulation')
   plt.plot(times/t_rabi, theory_excited, 'r--', linewidth=2, label='Theory')
   plt.xlabel('Time / Rabi period')
   plt.ylabel('Excited state population')
   plt.legend()
   plt.title('Rabi Oscillations')
   plt.grid(True, alpha=0.3)
   plt.show()

**Physics**: Under resonant driving, the qubit population oscillates between ground and excited states at the Rabi frequency :math:`\Omega`. This is a fundamental process in quantum control.

Multiple Qubits
---------------

Example with two coupled qubits demonstrating entanglement generation.

.. code-block:: python

   import qepsilon as qe
   import numpy as np

   # Create two-qubit system
   spin = qe.SpinSystem(n_sites=2, spin=0.5)
   
   # System parameters
   omega1, omega2 = 1.0, 1.1  # Individual qubit frequencies
   J = 0.2                     # Coupling strength
   
   # Hamiltonian
   H = omega1 * spin.Sz(0) + omega2 * spin.Sz(1) + \
       J * (spin.Sx(0) @ spin.Sx(1) + spin.Sy(0) @ spin.Sy(1))
   
   # Unitary evolution
   sim = qe.UnitarySystem(hamiltonian=H)
   
   # Initial state (both qubits in ground state)
   rho_0 = spin.ground_state()
   
   # Apply π/2 pulse to first qubit to create superposition
   pulse = qe.UnitarySystem(hamiltonian=np.pi/4 * spin.Sx(0))
   rho_0 = pulse.evolve(rho_0, [1.0])[-1]
   
   # Time evolution under coupling
   times = np.linspace(0, 20, 200)
   results = sim.evolve(rho_0, times)
   
   # Calculate entanglement (concurrence)
   def concurrence(rho):
       # Simplified concurrence calculation for two qubits
       # (This is a basic implementation)
       return np.abs(rho[0,3] - rho[1,2])
   
   entanglement = [concurrence(rho.numpy()) for rho in results]
   
   plt.figure(figsize=(8, 6))
   plt.plot(times, entanglement, 'g-', linewidth=2)
   plt.xlabel('Time')
   plt.ylabel('Entanglement (Concurrence)')
   plt.title('Entanglement Generation Between Two Qubits')
   plt.grid(True, alpha=0.3)
   plt.show()

**Physics**: Two qubits coupled through an exchange interaction can generate entanglement when one is initially in a superposition state. The entanglement oscillates as the system evolves. 
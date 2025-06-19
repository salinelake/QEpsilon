Quick Start Guide
=================

This guide will help you get started with QEpsilon quickly.

Basic Concepts
-------------

QEpsilon is built around several key concepts:

* **Systems**: Physical quantum systems (spins, bosons, fermions)
* **Operators**: Quantum operators acting on the systems
* **Simulations**: Time evolution of quantum master equations
* **Tasks**: High-level simulation workflows

Your First Simulation
--------------------

Here's a simple example to get you started with a two-level system (TLS):

.. code-block:: python

   import qepsilon as qe
   import numpy as np

   # Create a two-level system
   tls = qe.TLS()
   
   # Define the system parameters
   omega = 1.0  # Energy splitting
   gamma = 0.1  # Decay rate
   
   # Create operators
   H = omega * tls.sigma_z()  # Hamiltonian
   L = np.sqrt(gamma) * tls.sigma_minus()  # Lindblad operator
   
   # Set up the simulation
   sim = qe.LindbladSystem(hamiltonian=H, lindblad_operators=[L])
   
   # Define initial state (excited state)
   rho_0 = tls.excited_state()
   
   # Time evolution
   times = np.linspace(0, 10, 100)
   results = sim.evolve(rho_0, times)
   
   print("Simulation completed!")

Working with Different Systems
-----------------------------

Spin Systems
~~~~~~~~~~~~

.. code-block:: python

   # Create a spin-1/2 system
   spin = qe.SpinSystem(n_sites=2, spin=0.5)
   
   # Heisenberg interaction
   J = 1.0
   H = J * (spin.Sx(0) @ spin.Sx(1) + 
            spin.Sy(0) @ spin.Sy(1) + 
            spin.Sz(0) @ spin.Sz(1))

Bosonic Systems
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a bosonic mode
   boson = qe.BosonSystem(n_modes=1, max_occupation=10)
   
   # Harmonic oscillator
   omega = 1.0
   H = omega * boson.number_operator(0)

Tight-Binding Systems
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a tight-binding chain
   tb = qe.TightBindingSystem(n_sites=4)
   
   # Hopping Hamiltonian
   t = 1.0
   H = -t * sum(tb.c_dag(i) @ tb.c(i+1) + tb.c_dag(i+1) @ tb.c(i) 
                for i in range(3))

Setting Up Master Equations
---------------------------

Lindblad Master Equation
~~~~~~~~~~~~~~~~~~~~~~~~

For open quantum systems with Markovian dynamics:

.. code-block:: python

   # Define system and environment
   system = qe.TLS()
   
   # Hamiltonian
   H = omega * system.sigma_z()
   
   # Lindblad operators for different processes
   L_decay = np.sqrt(gamma) * system.sigma_minus()  # Spontaneous emission
   L_dephasing = np.sqrt(gamma_phi) * system.sigma_z()  # Pure dephasing
   
   # Create the master equation
   sim = qe.LindbladSystem(
       hamiltonian=H,
       lindblad_operators=[L_decay, L_dephasing]
   )

Unitary Evolution
~~~~~~~~~~~~~~~~

For closed quantum systems:

.. code-block:: python

   # Unitary evolution (no decoherence)
   sim = qe.UnitarySystem(hamiltonian=H)

Time-Dependent Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Time-dependent driving
   def H_t(t):
       return omega * system.sigma_z() + A * np.cos(omega_drive * t) * system.sigma_x()
   
   sim = qe.LindbladSystem(hamiltonian=H_t, lindblad_operators=[L_decay])

Running Simulations
------------------

Basic Time Evolution
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initial state
   rho_0 = system.ground_state()
   
   # Time points
   times = np.linspace(0, 10, 1000)
   
   # Evolve
   results = sim.evolve(rho_0, times)
   
   # Extract observables
   population = [np.real(np.trace(rho @ system.sigma_z())) for rho in results]

GPU Acceleration
~~~~~~~~~~~~~~~

For large systems, use GPU acceleration:

.. code-block:: python

   import torch
   
   # Move to GPU
   sim = sim.to('cuda')
   rho_0 = rho_0.to('cuda')
   
   # Run on GPU
   results = sim.evolve(rho_0, times)

Next Steps
----------

* Explore the :doc:`examples/index` section for more detailed examples
* Check the :doc:`api/index` for complete API reference
* Read about the :doc:`theory` behind QEpsilon

For more advanced usage, see the examples in the ``examples/`` directory of the repository. 
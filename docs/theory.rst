Theoretical Background
=====================

This section provides the theoretical foundation for QEpsilon's quantum master equation approach.

Open Quantum Systems
--------------------

QEpsilon is designed to simulate open quantum systems - quantum systems that interact with their environment. Unlike closed quantum systems that evolve unitarily, open systems experience decoherence and dissipation due to environmental coupling.

The density matrix formalism is essential for describing open systems, as pure states generally evolve into mixed states due to environmental interactions.

The Lindblad Master Equation
----------------------------

The most general form of a Markovian quantum master equation is the Lindblad equation:

.. math::

   \frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)

Where:

* :math:`\rho(t)` is the system density matrix
* :math:`H` is the system Hamiltonian  
* :math:`L_k` are the Lindblad operators describing system-environment coupling
* :math:`[A,B] = AB - BA` is the commutator
* :math:`\{A,B\} = AB + BA` is the anticommutator

The first term describes unitary evolution, while the second term represents dissipative dynamics.

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~

The Lindblad operators :math:`L_k` represent different physical processes:

**Spontaneous Emission**
   For a two-level system: :math:`L = \sqrt{\gamma} \sigma_-`
   
   Where :math:`\gamma` is the emission rate and :math:`\sigma_- = |g\rangle\langle e|` lowers the system from excited to ground state.

**Pure Dephasing**  
   :math:`L = \sqrt{\gamma_\phi} \sigma_z`
   
   This destroys coherences without changing populations.

**Thermal Bath**
   For a harmonic oscillator:
   
   * Cooling: :math:`L_- = \sqrt{\gamma(n_{th}+1)} a`
   * Heating: :math:`L_+ = \sqrt{\gamma n_{th}} a^\dagger`
   
   Where :math:`n_{th}` is the thermal occupation number.

Mathematical Properties
~~~~~~~~~~~~~~~~~~~~~~

The Lindblad form ensures several important properties:

1. **Trace preservation**: :math:`\text{Tr}[\rho(t)] = 1` for all times
2. **Complete positivity**: The evolution is physically meaningful
3. **Hermiticity**: :math:`\rho^\dagger = \rho` is preserved

Unitary Evolution
----------------

For closed quantum systems, the master equation reduces to the von Neumann equation:

.. math::

   \frac{d\rho}{dt} = -i[H(t), \rho]

With solution:

.. math::

   \rho(t) = U(t) \rho(0) U^\dagger(t)

Where :math:`U(t)` is the time evolution operator satisfying:

.. math::

   i\hbar \frac{dU}{dt} = H(t) U(t)

Time-Dependent Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~

For time-dependent Hamiltonians, the evolution operator is:

.. math::

   U(t) = \mathcal{T} \exp\left(-\frac{i}{\hbar} \int_0^t H(t') dt' \right)

Where :math:`\mathcal{T}` is the time-ordering operator.

Numerical Implementation
-----------------------

Vectorization
~~~~~~~~~~~~

QEpsilon solves the master equation by vectorizing the density matrix. The Lindblad equation becomes:

.. math::

   \frac{d}{dt}|\rho\rangle\rangle = \mathcal{L} |\rho\rangle\rangle

Where :math:`|\rho\rangle\rangle` is the vectorized density matrix and :math:`\mathcal{L}` is the Liouvillian superoperator:

.. math::

   \mathcal{L} = -i(H \otimes I - I \otimes H^T) + \sum_k \left( L_k \otimes L_k^* - \frac{1}{2}(L_k^\dagger L_k \otimes I + I \otimes (L_k^\dagger L_k)^T) \right)

Integration Methods
~~~~~~~~~~~~~~~~~~

QEpsilon uses adaptive integration schemes to solve the master equation:

* **Runge-Kutta methods** for smooth dynamics
* **Exponential integrators** for stiff systems
* **GPU acceleration** via PyTorch for large systems

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

The computational cost scales as:

* **Memory**: :math:`O(d^2)` for a :math:`d`-dimensional Hilbert space
* **Time per step**: :math:`O(d^4)` for dense matrices, :math:`O(d^2)` for sparse systems

System Types
------------

Two-Level Systems (Qubits)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest quantum system with Hilbert space dimension 2. Operators are represented using Pauli matrices:

.. math::

   \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
   \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad  
   \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}

Harmonic Oscillators (Bosonic Systems)  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Infinite-dimensional systems truncated to finite occupation number. Creation and annihilation operators satisfy:

.. math::

   [a, a^\dagger] = 1

Number states: :math:`a^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle`, :math:`a|n\rangle = \sqrt{n}|n-1\rangle`

Spin Systems
~~~~~~~~~~~~

Multi-level systems with spin :math:`s`. Total dimension is :math:`2s+1`. Operators follow angular momentum algebra:

.. math::

   [S_i, S_j] = i\epsilon_{ijk} S_k

Tight-Binding Systems
~~~~~~~~~~~~~~~~~~~~

Fermionic systems with creation/annihilation operators satisfying anticommutation relations:

.. math::

   \{c_i, c_j^\dagger\} = \delta_{ij}, \quad \{c_i, c_j\} = 0

Applications
-----------

QEpsilon is applicable to various quantum systems:

**Quantum Optics**
   * Cavity QED
   * Laser dynamics  
   * Photon statistics

**Quantum Information**
   * Qubit decoherence
   * Quantum gates
   * Entanglement dynamics

**Condensed Matter**
   * Electron transport
   * Spin dynamics
   * Many-body systems

**Molecular Systems**
   * Excitation transfer
   * Vibrational dynamics
   * Chemical reactions

Advanced Topics
--------------

Non-Markovian Dynamics
~~~~~~~~~~~~~~~~~~~~~

QEpsilon can handle non-Markovian effects through:

* Time-dependent Lindblad operators
* Memory kernels
* Stochastic processes

Composite Systems
~~~~~~~~~~~~~~~~

Multiple subsystems can be coupled through:

* Direct Hamiltonian coupling
* Shared environmental modes
* Cascaded interactions

For more implementation details, see the :doc:`api/index` and :doc:`examples/index` sections. 
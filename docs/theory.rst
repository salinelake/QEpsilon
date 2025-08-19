Theoretical Background
======================

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

Unitary Evolution
-----------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For time-dependent Hamiltonians, the evolution operator is:

.. math::

   U(t) = \mathcal{T} \exp\left(-\frac{i}{\hbar} \int_0^t H(t') dt' \right)

Where :math:`\mathcal{T}` is the time-ordering operator.

Numerical Implementation
------------------------
 
Integration Methods
~~~~~~~~~~~~~~~~~~~

 
 
 
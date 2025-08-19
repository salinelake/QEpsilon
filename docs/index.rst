QEpsilon Documentation
======================

Welcome to QEpsilon's documentation!

QEpsilon is a Python package for modeling open quantum systems. QEpsilon is designed to minimize the effort required to build data-driven quantum master equations of open quantum systems and to perform time evolution of the master equation. Applications of QEpsilon span from artificial quantum systems to condensed matter systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   api/index
   examples/molecular_qubits
   examples/quantum_transport

Highlighted Features
--------------------

* **General and flexible**: Supporting the parameterization and simulation of spin, bosonic, tight-binding systems and their combinations.
* **GPU support**: Making it efficient to simulate relatively large quantum systems (~20 spins or several bosonic modes).
* **Highly modularized**: Easy to implement many-body Hamiltonian with mixed-type (e.g. spin-boson) operators.

The Quantum Master Equation
---------------------------

The quantum master equation modeled by QEpsilon is:

.. math::

   \frac{d}{dt} \rho(t) = -i[H_{\epsilon}(t), \rho(t)] + \sum_{k} \gamma_k \left( L_k \rho(t) L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho(t)\}\right)

Where:

- :math:`\rho(t)` is the density matrix of the system
- :math:`H(t)` is the time-dependent system Hamiltonian
- :math:`L_k` are the Lindblad operators describing the system-environment coupling

:math:`H_{\epsilon}(t) = H_0 + H_c(t) + \sum_{j=1}^{M} f_j(\epsilon(t)) S_j` is a linear combination of the
static system Hamiltonian :math:`H_0`, the external control :math:`H_c(t)`, and perturbing Hermitian operators :math:`S_j`. :math:`f_j(\epsilon(t))` is a
scalar function of the multidimensional, classical dynamical processes :math:`\epsilon(t)` that encodes information about the environment. 
The classical dynamics of :math:`\epsilon(t)`, described by parameterized Markovian equations of motion, can be optimized together with other system parameters (such as :math:`\gamma_k`) through chain rules, and the behavior of :math:`\rho_{\epsilon}(t)` can match time-series data of
the system. QEpsilon provides a flexible framework to do such optimization.



Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
QEpsilon Documentation
=====================

Welcome to QEpsilon's documentation!

QEpsilon is a Python package for modeling open quantum systems. QEpsilon is designed to minimize the effort required to build data-driven quantum master equations of open quantum systems and to perform time evolution of the master equation. Applications of QEpsilon span from artificial quantum systems (qubits) to molecular systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   architecture
   api/index
   examples/index
   theory

Highlighted Features
-------------------

* **General and flexible**: Supporting the parameterization and simulation of spin, fermionic, bosonic systems and their combinations.
* **GPU and sparse linear algebra support**: Making it efficient to simulate relatively large quantum systems (~20 spins or several bosonic modes).
* **Highly modularized**: Easy to implement many-body operators.

The Quantum Master Equation
---------------------------

The quantum master equation modeled by QEpsilon is:

.. math::

   \frac{d}{dt} \rho(t) = -i[H(t), \rho(t)] + \sum_{k} \left( L_k \rho(t) L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho(t)\}\right)

Where:

- :math:`\rho(t)` is the density matrix of the system
- :math:`H(t)` is the system Hamiltonian
- :math:`L_k` are the Lindblad operators describing the system-environment coupling

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
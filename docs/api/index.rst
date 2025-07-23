API Reference
=============

This section contains the complete API reference for QEpsilon.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   system
   simulation
   operator_basis
   operator_group
   utilities

Overview
--------

QEpsilon's API is organized into several modules:

* :doc:`system` - Physical quantum systems (density matrices, pure states, particles)
* :doc:`simulation` - Time evolution engines (Lindblad, unitary, mixed systems)
* :doc:`operator_basis` - Basic operator definitions (TLS, bosons, tight-binding)
* :doc:`operator_group` - Collections of operators (spins, fermions, bosons)
* :doc:`utilities` - Helper functions and utilities

Quick Access
------------

Most commonly used classes:

* :class:`qepsilon.system.DensityMatrix` - Density matrix representation
* :class:`qepsilon.system.PureEnsemble` - Pure state ensemble
* :class:`qepsilon.simulation.LindbladSystem` - Lindblad master equation evolution
* :class:`qepsilon.simulation.UnitarySystem` - Unitary time evolution
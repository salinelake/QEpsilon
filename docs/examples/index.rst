Examples
========

This section contains detailed examples demonstrating various features of QEpsilon.

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   basic_usage

Available Example Categories
----------------------------

QEpsilon comes with several categories of examples:

**Basic Usage**
   Simple examples to get you started with the basic functionality.

**Quantum Systems**
   Examples showing different types of quantum systems (spins, bosons, fermions).

**Open Systems**
   Examples demonstrating open quantum system dynamics with decoherence.

**Advanced Features**
   More complex examples showcasing advanced capabilities like GPU acceleration and large system simulations.

Running the Examples
--------------------

All examples can be found in the ``examples/`` directory of the QEpsilon repository. Each example includes:

* A detailed README explaining the physics
* Well-commented Python scripts
* Jupyter notebooks for interactive exploration
* Expected output and visualization scripts

To run an example:

.. code-block:: bash

   cd examples/basic_example/
   python run_example.py

Or open the corresponding Jupyter notebook:

.. code-block:: bash

   jupyter notebook example.ipynb

Example Datasets
----------------

Some examples use experimental data for comparison:

* **CaF Qubits**: Ramsey interference experiments
* **Rubrene Crystal**: Organic semiconductor dynamics
* **Organic Semiconductor**: Charge transfer processes

These examples demonstrate how QEpsilon can be used to model real experimental systems and compare with measured data. 
Tutorial: Simulation
====================
 
How to simulate a physical system?
----------------------------------

A physical system is described by its quantum/classical state, its Hamiltonian, and an equation of motion governing the dynamics of the state.

We have learned how to define a quantum/classical state, and how to define operators in :doc:`tutorial_basics`.

Here, we will learn how to assemble these components into a physical system, and how to simulate the dynamics of the system with chosen equation of motions.

In QEpsilon, we can simulate two types of physical systems:

(1) An open quantum system. 
The system state is described by a density matrix, and the equation of motion is a Lindblad equation with time-independent or time-dependent Hamiltonian.
The coefficient of the time-dependent terms in the Hamiltonian is decided by classical processes. The classical processes can be related to a classical particle system.

(2) A closed quantum system. 
The system state is described by a pure state, and the equation of motion is a Schrodinger equation with time-independent or time-dependent Hamiltonian.
The coefficient of the time-dependent terms in the Hamiltonian is decided by classical processes. The classical processes can be related to a classical particle system.

The simulation of these systems are implemented in the `simulation` module.

**Lindblad-based simulation**

The Lindblad-based simulation is implemented in the `LindbladSystem` class.
It is initialized with the number of states of the system, and the batchsize.
Upon initialization, a `DensityMatrix` object is created to store the density matrix of the system.

`LindbladSystem` has subclasses implemented with a few helper functions for convenience.
For example, `QubitLindbladSystem` is a subclass of `LindbladSystem` that is initialized with the number of qubits and the batchsize.
It has a method `set_rho_by_config` to set the density matrix by a configuration vector, 
and a method `rotate` to apply a unitary rotation on a selected subset of qubits.

A one-qubit system can be initialized as (the next three code blocks should be run in sequence):

.. code-block:: python

   from qepsilon import QubitLindbladSystem
   tls_lindblad = QubitLindbladSystem(n_qubits=1, batchsize=1)

The `LindbladSystem` has a method `add_operator_group_to_hamiltonian` to add an `OperatorGroup` to the Hamiltonian of the system.
The `LindbladSystem` has a method `add_operator_group_to_jumping` to add an `OperatorGroup` to the set of jumping operators of the system.

For example, we can add a static sigma-x operator to the Hamiltonian of the one-qubit system, and a static sigma-z operator to the jumping operators of the system.

.. code-block:: python

   from qepsilon import StaticPauliOperatorGroup
   ## add a static sigma-x operator to the Hamiltonian of the one-qubit system
   ham_0 = StaticPauliOperatorGroup(n_qubits=1, id="sigma_x", batchsize=1, coef=1.0, requires_grad=False)
   ham_0.add_operator('X')
   tls_lindblad.add_operator_group_to_hamiltonian(ham_0)  

   ## add a static sigma-z operator to the jumping operators of the system
   jump_0 = StaticPauliOperatorGroup(n_qubits=1, id="jump_0", batchsize=1, coef=2.0, requires_grad=False)
   jump_0.add_operator('Z')
   tls_lindblad.add_operator_group_to_jumping(jump_0)

Note that, the damping coefficient of the jumping operators has been absorbed into the coefficient of the OperatorGroup.
In the example above, `coef=2` is specified for the `jump_0` OperatorGroup, which is associated with the one-body operator :math:`L_0=2\sigma^z`.
This addes to the Lindblad equation the dissipator :math:`(L_0 \rho L_0^\dagger -\frac{1}{2}\{ L_0^\dagger L_0, \rho \})`, 
which is equivalent to :math:`4(\sigma^z \rho \sigma^z  -\frac{1}{2}\{ \sigma^z \sigma^z, \rho \})` in the standard Lindblad form.
This convention adopted by QEpsilon should be kept in mind when adding jumping operators to the system.

Having defined the Hamiltonian and the jumping operators of the system, we can now simulate the dynamics of the system.

.. code-block:: python

      import numpy as np
      import torch as th
      ## set the initial state to |0>
      tls_lindblad.set_rho_by_config([0])
      print('The initial density matrix of the one-qubit system is:', tls_lindblad.rho)
      # apply a pi/2 pulse along x
      tls_lindblad.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
      print('The density matrix after the pi/2 pulse is:', tls_lindblad.rho)
      # free evolution
      for i in range(100):
         tls_lindblad.step(dt=0.01)
      # apply a pi/2 pulse along x
      tls_lindblad.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
      # normalize the density matrix
      tls_lindblad.normalize()
      print('The density matrix after the free evolution and the second pi/2 pulse is:', tls_lindblad.rho)



**Schrodinger-based simulation**

The Schrodinger-based simulation is implemented in the `UnitarySystem` class.
It is initialized with the number of states of the system, and the batchsize.
Upon initialization, a `PureStatesEnsemble` object is created to store the pure states ensemble of the system.

`UnitarySystem` has subclasses implemented with a few helper functions for convenience.
For example, `QubitUnitarySystem` is a subclass of `UnitarySystem` that is initialized with the number of qubits and the batchsize.
It has a method `set_pse_by_config` to set the pure states ensemble by a configuration vector, 
and a method `rotate` to apply a unitary rotation on a selected subset of qubits. 
`TightBindingUnitarySystem` is also a subclass of `UnitarySystem`. It is initialized with the number of sites and the batchsize.

Similar to the `LindbladSystem`, a `UnitarySystem` has a method `add_operator_group_to_hamiltonian` to add an `OperatorGroup` to the Hamiltonian of the system.
Apparently, `UnitarySystem` does not need a method dealing with jumping operators.
 
A one-qubit system can be initialized and simulated as:

.. code-block:: python

   from qepsilon import QubitUnitarySystem
   from qepsilon import StaticPauliOperatorGroup

   tls_unitary = QubitUnitarySystem(n_qubits=1, batchsize=1)
   ## add a static sigma-x operator to the Hamiltonian of the one-qubit system
   ham_0 = StaticPauliOperatorGroup(n_qubits=1, id="sigma_x", batchsize=1, coef=1.0, requires_grad=False)
   ham_0.add_operator('X')
   tls_unitary.add_operator_group_to_hamiltonian(ham_0)  
   tls_unitary.set_pse_by_config([0])
   print('The initial pure states ensemble of the one-qubit system is:', tls_unitary.pse)
   # apply a pi/2 pulse along x
   tls_unitary.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
   print('The pure states ensemble after the pi/2 pulse is:', tls_unitary.pse)
   # free evolution
   for i in range(100):
      tls_unitary.step(dt=0.01)
   # apply a pi/2 pulse along x
   tls_unitary.rotate(direction=th.tensor([1.0,0,0]), angle=np.pi/2)
   # normalize the pure states ensemble
   tls_unitary.normalize()
   print('The normalized pure states ensemble after the free evolution and the second pi/2 pulse is:', tls_unitary.pse)

 
    
GPU Acceleration
-----------------

For large systems, such as those containing more than 10 qubits, use GPU acceleration. This requires you to have a GPU-enabled machine, 
with CUDA and a GPU-enabled PyTorch installed.

In QEpsilon, enabling GPU acceleration is as simple as moving the system to the GPU.

First, let us test how long it takes to simulate the dynamics of a 10-qubit system for 100 time steps.

.. code-block:: python

   from qepsilon import QubitUnitarySystem
   from qepsilon import StaticPauliOperatorGroup
   import time
   import torch as th

   if th.cuda.is_available():
      print("CUDA is available")
   else:
      print("CUDA is not available")

   nq = 12
   nbatch = 10
   tls_unitary = QubitUnitarySystem(n_qubits=nq, batchsize=nbatch)
   ham_0 = StaticPauliOperatorGroup(n_qubits=nq, id="sigma_x", batchsize=nbatch, coef=1.0, requires_grad=False)
   ham_0.add_operator('X'*nq)
   tls_unitary.add_operator_group_to_hamiltonian(ham_0)  
   tls_unitary.set_pse_by_config([0]*nq)
   print(tls_unitary.pse.shape)
   ## tls_unitary.to('cuda')
   start_time = time.time()
   for i in range(10):
      tls_unitary.step(dt=0.01)

   ## th.cuda.synchronize()
   end_time = time.time()
   print(f"Time taken: {end_time - start_time} seconds")

Here, we are simulating in parallel 10 copies of a 12-qubit system.
Using one CPU core from a AMD EPYC 7763 processor, the output is:

.. code-block::

   CUDA is not available
   torch.Size([10, 4096])
   Time taken: 9.977923393249512 seconds

Now let us run the same code on one NVIDIA A100 GPU. 
Uncomment the line ``tls_unitary.to('cuda')`` and the line ``th.cuda.synchronize()`` in the code above, and run the code again. The output is:

.. code-block::

   CUDA is available
   torch.Size([10, 4096])
   Time taken: 0.36470770835876465 seconds

For this specific example, the speedup is about 30x. And all you need to do is move the system to the GPU with a single line of code.
GPU-acceleration is especially useful for simulation with a batchsize larger than 1, which is needed when the Hamiltonian is time-dependent and stochastic.


Next Steps
-----------

* See the :doc:`tutorial_training` for how to train a data-informed mixed quantum-classical dynamics model
* Check the :doc:`api/index` for complete API reference
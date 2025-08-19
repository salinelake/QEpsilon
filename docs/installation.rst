Installation
============

Requirements
------------

QEpsilon requires Python 3.8 or later and the following dependencies:

* numpy >= 1.24.0
* torch >= 2.4.0

Installation from Source
------------------------

The recommended way to install QEpsilon is from source:

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd QEpsilon

2. Install in development mode:

   .. code-block:: bash

      pip install -e .
 
Verifying Installation
----------------------

To verify that QEpsilon is installed correctly, you can run:

.. code-block:: python

   import qepsilon
   print("QEpsilon installed successfully!")
 
GPU Support
-----------

QEpsilon leverages PyTorch for GPU acceleration. To use GPU features:

1. Ensure you have a CUDA-compatible GPU
2. Install PyTorch with CUDA support before you install QEpsilon. For example, if your GPU supports CUDA 12.8, you can run:

   .. code-block:: bash

      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

For more information on PyTorch installation with CUDA, see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_. 
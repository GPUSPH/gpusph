========
 GPUSPH
========

What is it
==========

This repository holds the source code for GPUSPH_, the first implementation
of weakly-compressible Smoothed Particle Hydrodynamics (WCSPH) to run fully
on Graphic Processing Units (GPU), using NVIDIA CUDA.

.. _GPUSPH: http://gpusph.org

Quick start guide
=================

Run ``make`` followed by ``make test`` to compile and run the default test
problem. You can see a list of available test problems using ``make
list-problems`` and run any of them with ``make $problem && ./GPUSPH`` where
``$problem`` is the selected problem.

Requirements
============

GPUSPH requires a recent version of the NVIDIA CUDA SDK (7.5 or higher,
preferably 8.0 or higher), and a compatible host compiler. Please consult
`the NVIDIA CUDA documentation
<https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_
for further information.

Further information can be found on `the project's website <http://gpusph.org/documentation>`_.

Contributing
============

If you wish to contribute to GPUSPH, please consult CONTRIBUTING_.

.. _CONTRIBUTING: CONTRIBUTING.rst

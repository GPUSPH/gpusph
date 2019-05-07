========================
 Contributing to GPUSPH
========================

Introduction
============

Until version 4, the only publicly available development branches of GPUSPH
have been ``master`` (holding the current released version, with a most a couple of
trivial bug fixes) and ``candidate`` (holding the feature-complete, in-testing
release candidate for the upcoming version). The main development branch, ``next``,
was only accessible to partners directly involved with GPUSPH.

Between version 4 and version 5, the ``next`` development branch has been made
public as well, to improve opportunities for collaboration without the
difficulties of basing one's work on a obsolescent code base.

Methodology
===========

The GPUSPH development history is tracked with git_. Before you get started,
you may want to get familiar with the ``git`` tool itself; an excellent resource
in this regard is the freely available `Pro Git`_ book.

.. _git: https://git-scm.com
.. _Pro Git: https://git-scm.com/book/

GPUSPH follows a centralized development model with feature branches. There is
`a main repository (hosted on GitHub) <https://github.com/GPUSPH/gpusph>`_,
with reference branches (``master``, ``candidate``, ``next``).

The ``next`` branch is the main development branch. New features
should be developed in independent branches forked off the current ``next``
branch. They will be merged back into the ``next`` branch on completion.

The ``candidate`` branch represent a feature-freeze release-candidate for the
upcoming version of GPUSPH. When the ``next`` branch is considered ready for
release, it will be merged into ``candidate``, which undergoes more thorough
testing and may receive appropriate bug-fixes.

When ready, the ``candidate`` branch is promoted to ``master``, and a new
GPUSPH release is tagged. If further bugfixes are necessary, they are applied
to ``master`` directly, which is then merged back into ``candidate`` and into
``next``.

Communication
=============

The GitHub issue tracker is available to submit specific issues about building
and running GPUSPH, as well as to present any feature request.

For more general discussion about GPUSPH, its use and development, please join
`the GPUSPH Discourse Forum <https://gpusph.discourse.group>`_.

The GPUSPH developers can also be found on IRC, ``#gpusph`` channel on the
FreeNode network.

Reading the source
====================

If you wish to contribute, it's better to start by getting familiar with
the source code itself, starting from the way it's laid out on disk.


Repository layout
-----------------

The repository top-level is thus structured:

    ``src/``
        Holds the main source directory.
    ``options/``
        Configuration-specific header files are stored here.
    ``scripts/``
        Holds auxiliary scripts to e.g. find the main GPU architecture or
        analyze/compare result files.
    ``docs/``
        Holds the documentation.
    ``data_files/``
        Created by ``scripts/get-data-files``, holds the support data files for
        the sample test cases.
    ``build/``, ``dist/``
        Object files and final executable will be stored here.
    ``tests/``
        By default, test cases will store their data in a subdirectory of this.

The ``src`` directory is thus structured:

    ``geometries/``
        Classes describing geometrical objects (cubes, cones, cylinders, etc).
    ``writers/``
        Classes implementing the ``Writer`` interface, to store simulation results on disk.
    ``cuda/``
        CUDA implementation of the simulation framework and its engines.
    ``problems/``
        Sample test cases
    ``problems/user/``
        User test cases


Main program flow
-----------------

The program entrypoint can be found in ``src/main.cc``, which takes care of
instantiating the ``Problem`` requested by the user at build time, as well as
the simulation engine itself (``GPUSPH``, implemented in ``src/GPUSPH.cc``)
and, if appropriate, the network manager for multi-node support.

The user-specified test-case subclasses ``Problem`` and defines the simulation
framework parameters (SPH formulation, viscous and boundary model, etc) as
well as the domain geometry and filling.

GPUSPH uses a “dumb worker” approach: the main engine ``GPUSPH`` gives
commands to one or more ``GPUWorker``, which each worker running in its own thread
and managing a separate device (GPU). ``GPUSPH`` itself takes care of
all initializations and host allocation, whereas all aspects of device management
are handled by the ``GPUWorker`` instances which, once initialized, enter
their own main loop, waiting for commands from ``GPUSPH`` and mapping these
to appropriate computational kernel invocations. The sequences of commands to
be executed during a simulation are define by ``Integrator`` instances.
Available integrators can be found under ``src/integrators``.

The computational kernel themselves are provided by the simulation framework,
which is a collection of engines taking care of separate parts of the
simulation (boundary conditions, forces computation, integration, etc) and is
instantiated by the ``Problem`` subclass implementing the user-specified
test-case.



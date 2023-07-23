.. tightbinder documentation master file, created by
   sphinx-quickstart on Sat Jul 11 18:18:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
======================

tightbinder is a Python library for electronic structure calculations based on the Slater-Koster approximation for tight-binding models. It provides all the necessary tools to build, 
modify and characterize any crystalline or disordered material.

The construction of a Slater-Koster model relies on the definition of a configuration file, which contains all the information needed to build the model.
Namely, one has to specify completely the crystalline structure, i.e. the Bravais vectors and motif, and the electronic structure which amounts to the
orbitals, onsite energies and hoppings between orbitals. The configuration file then fully characterizes the Slater-Koster models and constitutes the starting point for the majority of the calculations done
with the library. Alternatively, it is also possible to define custom models that can still leverage the capabilities of the package. 

Some of the features of the library are:

* Construction of Slater-Koster tight-binding models up to :math:`d` orbitals, with intraatomic spin-orbit coupling.
* Methods to modify the system as desired: construction of supercells, finite systems, introduction of vacancies or impurities, amorphization of the lattice,
  application of electric or magnetic fields.
* Complete topological characterization of materials: evolution of Wannier charge centers, :math:`\mathbb{Z}_2` invariant, Chern number and marker, and spatial entanglement spectrum.
* Transport calculations in two-terminal devices based on the Landauer-Buttiker formalism.
* Computation of observables such as the band structure, expected value of the spin components, density of states or local density of states, as well as the plotting routines for
  the corresponding quantities.
* Predefined models (e.g. Haldane or BHZ) and ability to define custom ones.


.. figure:: ../../paper_plot.png
   :align: center 
   :width: 800

   Characterization of Bi(111) using the library.


Contributing
===================
The library is still under development as new features and optimizations are added. Therefore, any kind of contribution is welcome, from completing the documentation, bug fixing, adding new features or reporting bugs of the library.
In general, any contribution should stick to the `PEP style guidelines <https://peps.python.org/pep-0008/>`_. If in doubt, do not hesitate to contact!



.. toctree::
   :hidden: 

   Home <self>
   Installation <install>
   Configuration files <config>
   Tutorial <tutorial/index>
   Examples <examples/index>
   Materials <materials/index>
   Documentation <api/_autosummary/tightbinder>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

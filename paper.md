---
title: 'tightbinder: A Python package for semi-empirical tight-binding models of crystalline and disordered solids'
tags:
  - Python
  - condensed matter physics
  - electronic structure
  - tight-binding
  - Slater-Koster
authors:
  - name: Alejandro José Uría-Álvarez
    orcid: 0000-0001-6668-7333
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Departamento de Física de la Materia Condensada, Universidad Autónoma de Madrid
   index: 1
date: 22 July 2023
bibliography: paper.bib

---

# Summary

`tightbinder` is a Python package for Slater-Koster, semi-empirical tight-binding
calculations of the electronic structure of solids. Tight-binding models are ubiquitous 
in condensed matter physics, since they provide an inexpensive description of electrons in materials.
Although the package can be used in principle for any kind of material (metals and insulators), 
it originates in the context of topological phases of matter. Since the prediction of topological insulators [@topological_insulators],
there has been a huge effort understanding and characterizing topological materials, resulting
in a complete classification of any crystalline system [@vergniory]. However, not so much is known in the context
of disordered solids. This is the aim of the library: to enable numerical studies of 
crystalline and disordered materials to identify possible topological systems where the usual
techniques are not useful. In any case, it also serves as a general purpose tight-binding framework, 
due to the modular approach taken in its construction.


# Statement of need

The determination of the band structure of a solid is the starting point for any calculation in condensed matter physics. 
This amounts to determining the hopping amplitudes $t^{\alpha\beta}_{ij}$ of the electronic Hamiltonian:
$$H=\sum_{ij,\alpha\beta}t^{\alpha\beta}_{ij}c^{\dagger}_{i\alpha}c_{j\beta}$$
where the indices $i,j$ run over atomic positions, and the indices $\alpha, \beta$ run over orbitals. $c^{\dagger}_{i\alpha}$ ($c_{i\alpha}$) are creation (annihialtion)
operators of electrons at atom $i$ and orbital $\alpha$. There exist several techniques to address this problem, 
varying in degrees of sophistication and scope. The most established method is density-functional theory (DFT) [@DFT_review], 
which provides an accurate description of the electronic structure, usually at the cost of slower computations. Tight-binding
models are as equally popular since they constitute a quick and inexpensive way to model systems, although by contruction, 
they are restricted to simpler, effective description of the bands. Slater-Koster tight-binding models [@SlaterKoster] provide a middle ground
since they allow to give a more accurate description of the material based on empirical considerations, while still being simple to compute.

In principle, one would resort to the most accurate methods available to predict properties of the material. 
However, DFT might become too expensive computationally if the amount of iteration needed to perform the numerical experients is too high. 
This is precisely the role of tight-binding models: supposed that the model captures the key features of the material, it can be used instead to describe the solid,
as long as the desired properties depend on those relevant features. In general, this approach allows for a qualitative 
exploration of the materials, while one should look for first principles calculations when seeking quantitive results. 

Currently, there are several tight-binding packages available, such as PyBinding [@pybinding], Pyqula, PythTB, Kwant [@kwant] and PySKTB [@pysktb]
as well as $\mathbb{Z}_2$Pack [@z2pack] for the computation of topological invariants. Of those libraries, only PySKTB was
designed to build Slater-Koster models, while the rest require specifying directly the hopping amplitudes. On top of
that, they are usually oriented towards crystalline structures, lacking tools for the description of disorder. `tightbinder`
provides all the standard functionality expected from a tight-binding code, focusing specifically on Slater-Koster models,
which can be used to describe realistic amorphous materials. 
It also provides tools for the topological characterization of solids, similar to those of $\mathbb{Z}_2Pack$ but
integrated within the API. These features give the library its own identity within the landscape of tight-binding frameworks, not
necessarily competing but providing alternative tools and a different perspective.


# Features 

The band structure is the spectrum of the Hamiltonian of the system,
which is obtained by computing and diagonalizing its matrix representation.
Therefore, the library is mainly built using linear algebra operations from the common Python libraries for scientific computing, i.e.
`NumPy` [@numpy], `SciPy` [@scipy] and `matplotlib` [@matplotlib] for visualization of the results.
`tightbinder` focuses on providing the necessary routines and classes to build, modify
and compute properties of empirical tight-binding models. The main features of the
library are:

- Determination of Slater-Koster tight-binding models with matrix elements involving up to $d$ orbitals, 
with intraatomic spin-orbit coupling. One can specify hoppings up to the nth-nearest neighbours, and use atoms from
different chemical species with different numbers of electrons.
- Configuration file based description of the model of the solid: Using a standarized
format for configuration files, one can specify all the relevant parameters of the model,
from the lattice vectors to the Slater-Koster hopping amplitudes. 
- Two main classes defined, one to describe crystalline Slater-Koster models,
and another one for amorphous solids.
There are also predefined models, and the possibility of defining custom models which inherit from a base `System` class.
- Methods and routines to modify systems: once they are built, there are methods to modify
the size or the boundaries of the solid, as well as routines to introduce different
forms of disorder and external fields.
- Topology identification: Computation of the $\mathbb{Z}_2$ invariant of time-reversal topological insulators 
and of the entanglement spectrum from a specified cut of the system.
- Computation of observables from the eigenstates of the system, e.g. bands, expected value of the spin,
density of states (also available using the kernel polynomial method), localization. 
There are plotting routines available for the different quantities.
- Fitting of the Slater-Koster parameters (or any user-defined model parameter) to reproduce
some given energy bands, usually from DFT calculations. 

![Characterization of Bi(111) with the library: (a) Band structure of a zigzag nanoribbon, with the edge bands highlighted in green. (b) Evolution of the Wannier charge centers (WCC). (c) The topological invariant can be obtained algorithmically from the WCC, allowing to compute the topological phase diagram as a function of the spin-orbit coupling. (d) Probability density of an edge state. (e) Transmission as a function of energy on an armchair nanoribbon.](paper_plot.png)

The basic workflow starts with the preparation of a configuration file, where we set all the parameters relative
to the material we want to describe. This is, the crystalographic information and then the details of the Slater-Koster model,
which imply specifying which orbitals participate for each chemical species, and the corresponding SK amplitudes.
With the configuration prepared, the model is initialized simply passing the parsed configuration to the class constructor.
From here, one can perform transformations of the base model, or directly obtain its spectrum and then perform
some postprocessing. `tightbinder` has already been valuable for one previous work [@uria], and continues to be used in the 
research of topological amorphous materials. We hope that more researchers will benefit from the package in their study of topological disordered solids.

The library provides a stable API, but is still under development to incorporate new functionality. Future plans include
additional routines to extract information from the models such as the pair distribution function $g(r)$, and rewriting
core parts of the library in C++ to improve performance. For an up-to-date list of features, we recommend visiting the documentation
website, where we will also provide a changelist for each new version. 



# Acknowledgements

AJUA acknowledges financial support from Spanish MICIN through Grant No. PRE2018-086552.

# References
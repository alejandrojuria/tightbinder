---
title: 'tightbinder: A Python package for semi-empirical tight-binding models of crystalline and disordered solids'
tags:
  - Python
  - condensed matter physics
  - electronic structure
  - tight-binding
  - Slater-Koster
authors:
  - name: Alejandro José Uría
    orcid: 0000-0001-6668-7333
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Departamento de Física de la Materia Condensada, Universidad Autónoma de Madrid
   index: 1
date: 24 October 2022
bibliography: paper.bib

---

# Summary

`tightbinder` is a Python package for Slater-Koster, semi-empirical tight-binding
calculations of the electronic structure of solids. Tight-binding models are ubiquitous 
in condensed matter physics, since they provide an inexpensive description of materials.
Although it can be used for metals and insulators, the package appears in the context
of the identificacion of topological phases of matter. Since the discovery of topological insulators,
there has been a huge effort understanding and characterizing topological materials, resulting
in the full classification of any cristalline system. However, not so much is known in the context
of disordered solids. This is precisely the purpose of this library: to allow numerical studies of 
crystalline and disordered materials, to identify possible topological systems. Although it was
designed with a particular objective, it can still be employed as a standard tight-binding framework, due
to the modular approach taken in its construction.


# Statement of need

The determination of the band structure of a solid is the starting point for any calculation in condensed matter physics. 
This amounts to determining the hopping amplitudes $t^{\alpha\beta}_{ij}$ of the electronic Hamiltonian:
$$H=\sum_{ij,\alpha\beta}t^{\alpha\beta}_{ij}c^{\dagger}_{i\alpha}c_{j\beta}$$
where the indices $i,j$ sum over lattice positions and orbitals, and $c^{\dagger}_i$ ($c_i$) are creation (annihialtion)
operators of electrons at position $i$. There exist several techniques to address this problem, 
varying in degrees of sophistication and scope. The most established method is density-functional theory (DFT) [@DFT_review], 
which provides an accurate description of the electronic structure, usually at the cost of slower computations. Tight-binding
models are as equally popular since they constitute a quick and inexpensive way to model systems, although by contruction, 
they are restricted to simpler, effective description of the bands. Slater-Koster tight-binding models [@SlaterKoster] provide a middle ground
since they allow to give a more accurate description of the material based on empirical considerations, while still being simple to compute.

Soemtimes, it makes sense to use the most accurate methods available to predict properties of the material. 
However, using DFT might become too expensive computationally if the amount of iteration needed to perform the numerical experients is too high. 
This is precisely the role of tight-binding models: supposed that the model captures the key features of the material, it can be used instead to describe the solid,
as long as the desired properties depend on those relevant features. In general, this approach allows for a qualitative 
exploration of the materials, while one should look for first principles calculations when seeking quantitive results. 

Currently, there are several tight-binding packages available, such as PyBinding, Pyqula, PythTB, Kwant and PySKTB
as well as $\mathbb{Z}_2Pack$ for computation of topological invariants. Of those libraries, only PySKTB was
designed to build Slater-Koster models, while the rest require specifying directly the hopping amplitudes. On top of
that, they are usually oriented towards crystalline structures, lacking tools for the description of disorder. `tightbinder`
provides all the standard functionality expected from a tight-binding code, focusing specifically on Slater-Koster models,
which are used additionaly to describe realistic amorphous materials. 
It also provides tools for the topological characterization of solids, similar to those of $\mathbb{Z}_2Pack$ but
integrated within the API. These features give the library its own identity within the landscape of tight-binding frameworks, not
necessarily competing but providing alternative tools and a different perspective.


# Features

The determination of the band structure requires obtaining the spectrum of the Hamiltonian of the system,
which in practice is equivalent to computing and diagonalizing its matrix representation.
Therefore, the library is mainly built using linear algebra operations from the common Python libraries for scientific computing, i.e.
`NumPy`, `SciPy` and `matplotlib` for visualization of the results.
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
and of the entanglement spectrum on a specified cut of the system.
- Computation of observables from the eigenstates of the system, e.g. bands, expected value of the spin,
density of states (also available using the kernel polynomial method), localization. 
There are plotting routines available for the different quantities.
- Fitting of the Slater-Koster parameters (or any user-defined model parameter) to reproduce
some given energy bands, usually from DFT calculations. 

The library is still under development. Future plans include electronic transport calculations in two dimensions,
additional routines to extract information from the models such as the pair distribution function $g(r)$, and rewriting
core parts of the library in C++ to improve performance.


# Usage






is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge support 

# References
<div align="center">
  
  ![logo](images/logo.png)
  [![Documentation Status](https://readthedocs.org/projects/tightbinder/badge/?version=latest)](https://tightbinder.readthedocs.io/en/latest/?badge=latest)
  [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/alejandrojuria/tightbinder/issues)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  ![PyPI - Downloads](https://img.shields.io/pypi/dm/tightbinder)

  
  
  
</div>

tightbinder is a Python library for electronic structure calculations based on the Slater-Koster approximation for tight-binding models. It provides all the necessary tools to build, 
modify and characterize any crystalline or disordered material.

The construction of a Slater-Koster model relies on the definition of a configuration file, which contains all the information needed to build the model.
Namely, one has to specify completely the crystalline structure, i.e. the Bravais vectors and motif, and the electronic structure which amounts to the
orbitals, onsite energies and hoppings between orbitals. The configuration file then fully characterizes the Slater-Koster models and constitutes the starting point for the majority of the calculations done
with the library. Alternatively, it is also possible to define custom models that can still leverage the capabilities of the package. 

The features of the library are:
* Construction of **Slater-Koster** tight-binding models up to $d$ orbitals, with intraatomic **spin-orbit coupling**. 
* Construction of **amorphous** Slater-Koster models, with hoppings modified accordingly (power or exponential law) with respect to the crystalline model.
* Methods to **modify** the system as desired: construction of supercells, finite systems, introduction of vacancies or impurities, amorphization of the lattice,
  application of electric or magnetic fields.
* Complete **topological characterization** of materials: evolution of Wannier charge centers, $\mathbb{Z}_2$ invariant, Berry curvature, Chern number and marker, and spatial entanglement spectrum.
* **Transport** calculations in **two-terminal** devices based on the Landauer-Buttiker formalism (conductance and transmission).
* Computation of **observables** such as the band structure, expected value of the spin components, density of states (either using Green's functions or the KPM), local density of states, as well as the plotting routines for
  the corresponding quantities.
* **Predefined** models (e.g. Haldane or BHZ) and ability to define custom ones.
* **Fitting** module to obtain the Slater-Koster parameters of any user-defined model from some input band structure.
* Additional classes for building models (define model bond by bond instead of relying on the Slater-Koster constructor, or stackings of 2d layers) (under development).

![Features](images/paper_plot.png)

For a complete description of the capabilities of the package, we refer to the [documentation](https://tightbinder.readthedocs.io/en/latest/) where several usage examples can be found together with the full API 
reference.


## Installation

Usage of a **virtual environment** is recommended in general to avoid conflicts between packages.

To install the latest version of the package, simply run:
```bash
pip install tightbinder
```

Alternatively, you can clone the repository to get the most up-to-date version of the package. 
```
git clone https://github.com/alejandrojuria/tightbinder.git
```

From the root folder of the repository, install the library (which will automatically install the required dependencies):
```
cd {path}/tightbinder
pip install .
```

## Documentation
The documentation can be accessed [online](https://tightbinder.readthedocs.io/en/latest/). To build it, you must have installed GNU Make and the library itself. Additional requirements to build documentation can be installed by specifying docs qualifier when installing tightbinder
```
cd {path}/tightbinder
pip install ".[docs]"
```
Then, ```cd docs/``` and then, run ```make html``` to build the documentation. It will be created in ```docs/build/html```, and can be accessed through ```index.html```.

## Contributing
The library is still under development as new features and optimizations are added. Therefore, any kind of contribution is welcome, from completing the documentation, bug fixing, adding new features or reporting bugs of the library.
In general, any contribution should stick to the [PEP style guidelines](https://peps.python.org/pep-0008/). If in doubt, do not hesitate to contact!




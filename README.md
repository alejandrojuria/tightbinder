# tight-binder
## Introduction
General purpose tight-binding code for electronic structure calculations based on the Slater-Koster approximation.
The code is yet to be finished: so far the modules include the strictly necessary routines to compute band structures
without additional information. 
It is designed to allow band structure calculations of alloys up to two atomic species (provided one gives the corresponding SK amplitudes).

The idea behind the program is to allow calculations simply using the configuration file, without any need to fiddle with the code (although that option is always available).
Some examples are provided (cube.txt, chain.txt) which show the parameters needed to run a simulation.

* Last Update: Added spin-orbit coupling up to d orbitals

## Installation
Usage of a **virtual environment** is recommended to avoid conflicts, specially since this package is still in development so
it will experiment changes periodically.

* From within the root folder of the repository, install the required packages:
```
$ cd {path}/tightbinder
$ pip install -r requirements.txt
```
* Then install the tightbinder package
``` 
$ pip install .
```
* You can use the application from within the repository, using the ```bin/app.py``` program in the following fashion:
``` 
$ python bin/app.py {config_file} 
```
Or since the library is installed, create your own scripts. For now, usage of the ```app.py``` program is advised.

### Documentation
To generate the documentation, you must have installed GNU Make previously. To do so, simply ``` $ cd docs/source``` and 
run ```$ make html```. The documentation will then be created in ```docs/build/html```.

## Examples
The folder ```examples/``` contains some basic cases to test that the program is working correcly.
* One-dimensional chain (1 orbital):
To run the example do ```$ python bin/app.py examples/chain.txt ```

This model is analytically solvable, its band dispersion relation is:
<img src="https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cvarepsilon%28k%29%20%3D%20%5Cvarepsilon_0%20-%202t%5Ccos%28ka%29"/> 

![alt text](screenshots/test_chain_band.png)

* Bi(111) bilayer
To run it: ```$python bin/app.py examples/bi(111).txt```
In this case we use a four-orbital model (s, px, py and pz). Since we are modelling a real material, we need to input some valid Slater-Koster coefficients as well as the spin-orbit coupling amplitude. These are given in [1, 2].

The resulting band structure is:
![alt text](screenshots/bi(111)_w_soc.png)

Bi(111) bilayers are known to be topological insulators. To confirm this, one can use the routines provided in the ```topology``` module to calculate its 
<img src="http://latex.codecogs.com/svg.latex?\mathbb{Z}_2" title="http://latex.codecogs.com/svg.latex?\mathbb{Z}_2"/> invariant.

To do so, we can compute its hybrid Wannier centre flow, which results to be:
![alt text](screenshots/wcc_flow_bi(111).png)

The crossing of the red dots indicates that the material is topological. For more complex cases, there is a routine implemented to automatize the counting of crossings, based on [3].

## Workroad
The future updates will be:
- [x] hamiltonian.py: Module for inititializing and solving the Hamiltonian of the system given in the config. file
- [x] topology.py: This module will include routines for computing topological invariants of the system.
  (19/12/20) Z2 invariant routines added. It remains to fix routines related to Chern invariant.
- [ ] disorder.py: Module with routines to introduce disorder in the system such as vacancies or impurities

A working GUI might be done in the future

## References




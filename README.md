# tight-binder
General purpose tight-binding code for electronic structure calculations based on the Slater-Koster approximation.
The code is yet to be finished: so far only the module for configuration file parsing (fileparse.py) has been finished. 
The code is designed to allow band structure calculations of alloys up to two atomic species (provided one gives the corresponding SK amplitudes).

The workroad is the following:
- hamiltonian.py: Module for inititializing and solving the Hamiltonian of the system given in the config. file
- topology.py: This module will include routines for computing topological invariants of the system
- disorder.py: Module with routines to introduce disorder in the system such as vacancies or impurities

A working GUI might be done in the future

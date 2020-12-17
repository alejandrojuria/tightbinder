# Implementation of System class, which incorporates both Hamiltonian and Crystal classes.
# Specifically, System includes the Crystal as a class attribute.
# The reasoning behind this, is to make available the crystal specific routines and attributes to the system,
# while being explicit about its ambit and without the crystal being a separate entity.
# As for the Hamiltonian, its functionality is directly available from System due to the inheritance.
# Base Hamiltonian class implementation. It will be used for the other hamiltonians to derive
# from it
# System has the basic functionality for the other models to derive from it.

import numpy as np
from crystal import Crystal
import result
import sys


class System(Crystal):
    def __init__(self, system_name=None, bravais_lattice=None, motif=None, crystal=None):
        if crystal is None:
            super().__init__(bravais_lattice, motif)
        else:
            assert type(crystal) == Crystal
            if bravais_lattice is not None or motif is not None:
                print("Warning: Provided values for Bravais lattice or motif are disregarded" +
                      "in presence of crystal object")
            super().__init__(crystal.bravais_lattice, crystal.motif)
        self.system_name = system_name
        self._norbitals = None
        self._basisdim = None

    @property
    def norbitals(self):
        return self._norbitals

    @norbitals.setter
    def norbitals(self, norbitals):
        assert type(norbitals) == int
        self._norbitals = norbitals

    # To be overwritten by specific implementations of System
    def hamiltonian_k(self, k):
        pass

    def solve(self, kpoints):
        """ Diagonalize the Hamiltonian to obtain the band structure and the eigenstates """
        nk = len(kpoints)
        eigen_energy = np.zeros([self._basisdim, nk])
        eigen_states = []
        for n, k in enumerate(kpoints):
            hamiltonian = self.hamiltonian_k(k)
            results = np.linalg.eigh(hamiltonian)
            eigen_energy[:, n] = results[0]
            eigen_states.append(results[1])

        return result.Result(eigen_energy, np.array(eigen_states), kpoints)


class FrozenClass:
    """ Class to enforce immutable attributes """
    _is_frozen = False

    def __setattr__(self, key, value):
        if self._is_frozen:
            raise Exception("Predefined model can not be modified")
        super().__setattr__(key, value)

    def _freeze(self):
        self._is_frozen = True



# Base Hamiltonian class implementation. It will be used for the other hamiltonians to derive
# from it

import numpy as np
import result
import sys


class BaseHamiltonian:
    def __init__(self):
        self.dimension = None
        self.dimensionality = None
        self.bravais_lattice = None
        self.system_name = None
        self.motif = None
        self.configuration = None

    def hamiltonian_k(self, k):
        pass

    def _init_configuration(self):
        if self.configuration is not None:
            print("Error: Cannot overwrite system configuration (given by file)")
            sys.exit(1)
        self.configuration = {"System name": self.system_name,
                              "Dimensionality": self.dimensionality,
                              "Bravais lattice": self.bravais_lattice,
                              "Motif": self.motif}

    def solve(self, kpoints):
        """ Diagonalize the Hamiltonian to obtain the band structure and the eigenstates """
        nk = len(kpoints)
        eigen_energy = np.zeros([self.dimension, nk])
        eigen_states = []
        for n, k in enumerate(kpoints):
            hamiltonian = self.hamiltonian_k(k)
            results = np.linalg.eigh(hamiltonian)
            eigen_energy[:, n] = results[0]
            eigen_states.append(results[1])

        return result.Result(self.configuration, eigen_energy, np.array(eigen_states), kpoints)



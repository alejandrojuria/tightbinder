# Definition of Result class to handle diagonalization results from Hamiltonian and observable calculation

import matplotlib.pyplot as plt
import numpy as np


class Result:
    def __init__(self, configuration, eigen_energy=None, eigen_states=None, kpoints=None):
        self.configuration = configuration
        self.eigen_energy = eigen_energy
        self.eigen_states = eigen_states
        self.kpoints = kpoints

    def __simplify_kpoints(self):
        """ Routine to reduce the k-points of the mesh from 3d to the corresponding dimension,
         for later graphical representation """

        dimension = self.configuration['Dimensionality']
        new_kpoints = np.zeros([len(self.kpoints), dimension])
        non_zero_index = np.nonzero(self.kpoints[0])
        if dimension == 1:
            for i, kpoint in enumerate(self.kpoints):
                new_kpoints[i, :] = kpoint[non_zero_index]

        elif dimension == 2:
            pass

        else:
            pass

        return new_kpoints

    def plot(self):
        plt.figure()
        if self.configuration['Dimensionality'] == 1:
            kpoints = self.__simplify_kpoints()
            for eigen_energy_k in self.eigen_energy:
                plt.plot(kpoints, eigen_energy_k, 'g-')

        plt.title(f'{self.configuration["System name"]} band structure')
        plt.xlabel(r'k ($A^{-1}$)')
        plt.ylabel(r'$\epsilon$ $(eV)$')

        plt.show()

    def write_bands_to_file(self):
        pass

    def write_states_to_file(self):
        pass





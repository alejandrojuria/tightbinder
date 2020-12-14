# Definition of Result class to handle diagonalization results from Hamiltonian and observable calculation

import matplotlib.pyplot as plt
import numpy as np


class Result:
    def __init__(self, eigen_energy=None, eigen_states=None, kpoints=None):
        self.eigen_energy = eigen_energy
        self.eigen_states = eigen_states
        self.kpoints = kpoints

    def __simplify_kpoints(self):
        """ Routine to reduce the k-points of the mesh from 3d to the corresponding dimension,
         for later graphical representation
         DEPRECATED """

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

    def plot(self, title=''):
        """ Method to plot bands from diagonalization in the whole Brillouin zone """

        plt.figure()
        if self.configuration['Dimensionality'] == 1:
            kpoints = self.__simplify_kpoints()
            for eigen_energy_k in self.eigen_energy:
                plt.plot(kpoints, eigen_energy_k, 'g-')

        plt.title(title + 'band structure')
        plt.xlabel(r'k ($A^{-1}$)')
        plt.ylabel(r'$\epsilon$ $(eV)$')

        plt.show()

    def plot_along_path(self, labels, title=''):
        """ Method to plot the bands along a path in reciprocal space, normally along high symmetry points """
        nk = len(self.kpoints)
        x_points = np.arange(0, nk)
        plt.figure()
        x_ticks = []
        number_of_paths = len(labels) - 1
        for n, label in enumerate(labels):
            x_ticks.append((nk - 1)/number_of_paths*n)
        for eigen_energy_k in self.eigen_energy:
            plt.plot(x_points, eigen_energy_k, 'g-')

        plt.xticks(x_ticks, labels)
        plt.ylabel(r'$\epsilon (eV)$')
        plt.title(title + " band structure")

        plt.show()

    def write_bands_to_file(self):
        pass

    def write_states_to_file(self):
        pass





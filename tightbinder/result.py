# Definition of Result class to handle diagonalization results from Hamiltonian and observable calculation

import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from utils import condense_vector


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

    def plot_bands(self, title=''):
        """ Method to plot bands from diagonalization in the whole Brillouin zone """

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        nk = len(self.kpoints[:, 0])
        size = int(math.sqrt(nk))
        x = self.kpoints[:, 0].reshape(size, size)
        y = self.kpoints[:, 1].reshape(size, size)
        z = self.kpoints[:, 2].reshape(size, size)
        for i in range(len(self.eigen_energy[:, 0])):
            en = self.eigen_energy[i, :].reshape(size, size)
            ax.plot_surface(x, y, en)

        plt.title(title + 'band structure')
        plt.xlabel(r'k ($A^{-1}$)')
        plt.ylabel(r'$\epsilon$ $(eV)$')

        plt.show()

    def plot_along_path(self, labels, title='', filling=None):
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

        if filling is not None:
            fermi_energy = self.calculate_fermi_level(filling)
            plt.axhline(y=fermi_energy, linewidth=2, color='r', label=r"$E_F$")

        plt.legend()
        plt.xlim([min(x_points), max(x_points)])

    def plot_spectrum(self, title=''):
        """ Routine to plot all the eigenvalues coming from the Bloch Hamiltonian diagonalization
        in an ordered way.
         Specially suitable for open systems, although it can be used for periodic systems as well."""

        all_eigenvalues = self.eigen_energy.reshape(-1, 1)
        all_eigenvalues = np.sort(all_eigenvalues)

        plt.plot(all_eigenvalues, 'g+')
        plt.title(f"Spectrum of {title}")
        plt.ylabel(r"$\varepsilon (eV)$")
        plt.xlabel("n")
        plt.show()

    def write_bands_to_file(self):
        pass

    def write_states_to_file(self):
        pass

    # --------------- Utilities ---------------
    @staticmethod
    def is_edge_state(state, norbitals, edge_indices, penetration=0.1):
        orbital_probabilities = np.abs(state) ** 2
        amplitudes = condense_vector(orbital_probabilities, norbitals)
        edge_density = np.sum(amplitudes[edge_indices])

        is_edge_state = False
        if edge_density > (1 - penetration):
            is_edge_state = True

        return is_edge_state

    def identify_edge_states(self, crystal, penetration=0.1):
        """ Routine to identify edge states according to a given penetration parameter """
        if self.eigen_states.shape[0] != 1:
            print("Error: identify_edge_states can only be used with OBC, exiting...")
            sys.exit(1)
        edge_indices = crystal.identify_edges()
        norbitals = len(self.eigen_states[0][:, 0])//len(crystal.motif)
        edge_states = []
        for n, state in enumerate(self.eigen_states[0].T):
            if self.is_edge_state(state, norbitals, edge_indices, penetration):
                edge_states.append(n)

        return edge_states

    def calculate_fermi_level(self, filling):
        """ Routine to compute the Fermi level energy according to the given filling """
        filling *= self.eigen_energy.shape[1]
        all_energies = self.eigen_energy.reshape(-1)
        all_energies = np.sort(all_energies)
        fermi_energy = all_energies[filling - 1]

        return fermi_energy

    def calculate_gap(self, filling):
        """ Routine to compute the gap of a material based on its Fermi level/filling """
        valence_band = self.eigen_energy[filling - 1, :]
        conduction_band = self.eigen_energy[filling, :]
        gap = np.min(conduction_band - valence_band)

        return gap






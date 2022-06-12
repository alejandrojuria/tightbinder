# Definition of Result class to handle diagonalization results from Hamiltonian and observable calculation

import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from .utils import condense_vector, scale_array


class Spectrum:
    def __init__(self, eigen_energy=None, eigen_states=None, kpoints=None, system=None):
        self.eigen_energy = eigen_energy
        self.eigen_states = eigen_states
        self.kpoints = kpoints
        self.system = system

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

    def plot_along_path(self, labels, title='', filling=None, fermi_level=False,
                        edge_states=False, ax=None, y_values=[], fontsize=10):
        """ Method to plot the bands along a path in reciprocal space, normally along high symmetry points """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if y_values and len(y_values) != 2:
            raise ValueError("y_values must be [y_min, y_max]")

        nk = len(self.kpoints)
        x_points = np.arange(0, nk)
        x_ticks = []
        number_of_paths = len(labels) - 1

        for n, label in enumerate(labels):
            xpos = (nk - 1)/number_of_paths*n
            x_ticks.append(xpos)
        for eigen_energy_k in self.eigen_energy:
            ax.plot(x_points, eigen_energy_k, 'g-', linewidth=3)

        if edge_states and filling is not None:
            edge_states_indices = [int(filling) + i for i in range(-2, 2)]
            for state in edge_states_indices:
                ax.plot(x_points, self.eigen_energy[state], 'r-')

        if fermi_level and filling is not None:
            fermi_energy = self.calculate_fermi_level(filling)
            ax.axhline(y=fermi_energy, linewidth=2, color='r', label=r"$E_F$")

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_ylabel(r'$\epsilon$ (eV)', fontsize=fontsize)
        ax.tick_params('y', labelsize=fontsize)
        if title != '':
            ax.set_title(title + " band structure", fontsize=fontsize)
        ax.set_xlim([min(x_points), max(x_points)])
        if y_values:
            ax.yaxis.set_ticks(np.arange(y_values[0], y_values[1] + 1, 1))
            ax.set_ylim(y_values)

        ax.grid(linestyle='--')


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
        """ TODO write_bands_to_file"""
        pass

    def write_states_to_file(self):
        """ TODO write_states_to_file"""
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

    def calculate_ipr(self):
        """ Method to compute the inverse participation ratio (IPR) from the eigenvectors.
         :returns: list with all ipr for all states """
        eigen_states = np.copy(self.eigen_states.reshape(self.system.basisdim, -1).T)
        ipr_array = []
        for eigenvector in eigen_states:
            state = State(eigenvector, self.system)
            ipr_array.append(state.compute_ipr())

        return ipr_array

    def calculate_average_ipr(self, states):
        """ Method to compute the average IPR associated to a group of states
         :param states: list with the eigenvectors of the states
         :returns: Average IPR of the given states """
        average_ipr = 0
        for eigenvector in states:
            state = State(eigenvector, self.system)
            average_ipr += state.compute_ipr()

        average_ipr /= len(states)
        return average_ipr

    def calculate_specific_occupation(self, atoms_indices, states=None):
        """ Method to compute the occupation for all the states in the spectrum
         on the specified atoms """
        if states is None:
            states = np.copy(self.eigen_states.reshape(self.system.basisdim, -1).T)
        occupations = []
        for eigenvector in states:
            state = State(eigenvector, self.system)
            occupation = state.compute_specific_occupation(atoms_indices)
            occupations.append(occupation)

        return occupations

    @staticmethod
    def plot_quantity(array, name=None, sort=False, ax=None):
        """ Method to plot an array of some quantity for each state
        in the spectrum as a bar plot """
        x = np.arange(0, len(array))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if sort:
            array = np.sort(array)
        ax.bar(x, array, width=1)
        ax.set_title(fr"{name}")
        ax.set_ylabel(f"{name}")
        ax.set_xlabel("States")


class State:
    def __init__(self, eigenvector, system):
        self.eigenvector = eigenvector
        self.motif = system.motif
        self.norbitals = system.norbitals
        self.hoppings = system.bonds

        self.amplitude = self.atomic_amplitude()

    def atomic_amplitude(self):
        """ Method to obtain the probability amplitude corresponding to each atom.
         Returns:
             array amplitude [len(eigenvector)/norbitals] """
        amplitude = np.abs(self.eigenvector) ** 2
        return condense_vector(amplitude, self.norbitals)

    def plot_amplitude(self, ax=None, title=None, linewidth=3):
        """ Method to plot the atomic amplitude of the state on top of the crystalline positions"""
        amplitude = np.array(self.atomic_amplitude())
        scaled_amplitude = scale_array(amplitude, factor=30)

        if ax is None:
            fig = plt.figure(figsize=(5, 6))
            ax = fig.add_subplot(111)

        atoms = np.array(self.motif)[:, :3]
        for n, bond in enumerate(self.hoppings):
            x0, y0 = atoms[bond[0], :2]
            xneigh, yneigh = atoms[bond[1], :2]
            ax.plot([x0, xneigh], [y0, yneigh], "-k", linewidth=1.5)
        ax.scatter(atoms[:, 0], atoms[:, 1],
                   c="royalblue", alpha=0.9,
                   s=scaled_amplitude)
        ax.set_xlim([np.min(atoms[:, 0]), np.max(atoms[:, 0])])
        ax.set_ylim([np.min(atoms[:, 1]), np.max(atoms[:, 1])])
        ax.set_xticks([0, 29])
        ax.set_yticks([0, 29])
        if title is not None:
            ax.set_title(title)
        ax.axis('off')
        ax.axis('tight')
        ax.axis('equal')

    def compute_ipr(self):
        """ Method to compute the inverse participation ratio (IPR) for the state """
        squared_amplitude = self.amplitude ** 2
        ipr = np.sum(squared_amplitude)

        return ipr

    def compute_specific_occupation(self, atoms_indices):
        """ Method to compute which percentage of the atomic amplitude is located
         at the given list of atoms """
        occupation = np.sum(self.amplitude[atoms_indices])
        return occupation

    def compute_spin_projection(self, axis):
        """ TODO compute_spin_projection
        Method to compute the expected value of any of the three spin operators Sx, Sy or Sz.
        All calculations are done supposing that the original atomic basis for the tight-binding is written
        using the z projection of the spin (if the calculation is spinful).
        Parameters:
            char axis: 'x', 'y' or 'z'
        Returns:
             float spin  """

        axis_list = ["x", "y", "z"]
        if axis not in axis_list:
            raise KeyError("Axis must be x, y or z")

        pass






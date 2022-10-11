# Definition of Result class to handle diagonalization results from Hamiltonian and observable calculation

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import numpy as np
import math
import sys
from .utils import condense_vector, scale_array
from .system import System
from typing import List

class Spectrum:
    """
    The Spectrum class is designed to store the results from the Hamiltonian diagonalization, 
    and to perform manipulations on the eigenvectors and eigenvalues and extract information
    about the system.
    """

    def __init__(self, eigen_energy: np.ndarray = None, eigen_states: np.ndarray = None, kpoints: np.ndarray = None, 
                system: System = None) -> None:

        self.eigen_energy = eigen_energy
        self.eigen_states = eigen_states
        self.kpoints = kpoints
        self.system = system

    def __simplify_kpoints(self) -> np.ndarray:
        """ 
        Routine to reduce the k-points of the mesh from 3d to the corresponding dimension,
        for later graphical representation
        DEPRECATED 
        """

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

    def plot_bands(self, title: str = '') -> None:
        """ 
        Method to plot bands from diagonalization in the whole Brillouin zone.
        :param title: Title for plot. Defaults to empty string.
        """

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

    def plot_along_path(self, labels: List[str], title: str = '', edge_states: bool = False, rescale: bool = True,
                        ax: Axes = None, e_values: List[float] = [], fontsize: float = 10) -> None:
        """ 
        Method to plot the bands along a path in reciprocal space, normally along high symmetry points.
        :param labels: Labels of the High Symmetry Points of the path.
        :param title: Title of the plot. Defaults to empty string.
        :param edge_states: Boolean parameter to toggle on or off edge bands in a different color. 
        Edge bands are defined as those immediately above and below the Fermi level. Defaults to False.
        :param rescale: Boolean to rescale the energy spectrum to the Fermi energy. I.e. highest occupied
        state has energy zero. Defaults to True.
        :param ax: Axes object from matplotlib to plot bands there. Useful for figures with subplots.
        :param e_values: List with two values, [e_min, e_max] to show bands only in that energy range.
        :param fontsize: Adjusts size of lines and text.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if e_values and len(e_values) != 2:
            raise ValueError("e_values must be [e_min, e_max]")

        if rescale:
            self.rescale_bands_to_fermi_level()

        nk = len(self.kpoints)
        x_points = np.arange(0, nk)
        x_ticks = []
        number_of_paths = len(labels) - 1

        for n, label in enumerate(labels):
            xpos = (nk - 1)/number_of_paths*n
            x_ticks.append(xpos)
        
        for eigen_energy_k in self.eigen_energy:
            ax.plot(x_points, eigen_energy_k, 'g-', linewidth=3)

        if edge_states and self.system.filling is not None:
            edge_states_indices = [int(self.system.filling) + i for i in range(-2, 2)]
            for state in edge_states_indices:
                ax.plot(x_points, self.eigen_energy[state], 'r-')

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_ylabel(r'$\epsilon$ (eV)', fontsize=fontsize)
        ax.tick_params('y', labelsize=fontsize)
        if title != '':
            ax.set_title(title + " band structure", fontsize=fontsize)
        ax.set_xlim([min(x_points), max(x_points)])
        if e_values:
            ax.yaxis.set_ticks(np.arange(np.round(e_values[0]), np.round(e_values[1]) + 1, 1))
            ax.set_ylim(e_values)

        # ax.grid(linestyle='--')

    def plot_bands_w_atomic_occupation(self, labels, atom_indices, title='', rescale=True, 
                                       ax=None, e_values=[], fontsize=10):
    
        """ 
        Method to plot bands as function of k, but also as a colormap to show edge occupation of 
        each state. 
        :param labels: Labels of the High Symmetry Points of the path.
        :param atom_indices: List of indices of atoms where we want to measure the occupation.
        :param title: Title of the plot. Defaults to empty string.
        :param edge_states: Boolean parameter to toggle on or off edge bands in a different color. 
        Edge bands are defined as those immediately above and below the Fermi level. Defaults to False.
        :param rescale: Boolean to rescale the energy spectrum to the Fermi energy. I.e. highest occupied
        state has energy zero. Defaults to True.
        :param ax: Axes object from matplotlib to plot bands there. Useful for figures with subplots.
        :param e_values: List with two values, [e_min, e_max] to show bands only in that energy range.
        :param fontsize: Adjusts size of lines and text.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if e_values and len(e_values) != 2:
            raise ValueError("e_values must be [y_min, y_max]")

        if rescale:
            self.rescale_bands_to_fermi_level()

        nk = len(self.kpoints)
        x_points = np.arange(0, nk)
        x_ticks = []
        number_of_paths = len(labels) - 1

        for n, _ in enumerate(labels):
            xpos = (nk - 1)/number_of_paths*n
            x_ticks.append(xpos)
        
        occupation_matrix = self.calculate_occupation(atom_indices)            
        
        cmap = get_cmap("viridis")
        segments = []
        colors = []
        for i, eigen_energy_k in enumerate(self.eigen_energy):
            for j in x_points[:-1]:
                segments.append([(x_points[j], eigen_energy_k[j]), (x_points[j + 1], eigen_energy_k[j + 1])])
                colors.append(cmap(occupation_matrix[i, j]))

        lines = LineCollection(segments, colors=colors, linewidth=3)
        ax.add_collection(lines)
        print(np.max(occupation_matrix))

        # im = ax.pcolormesh(x, y, z, cmap='viridis')
        # fig.colorbar(im, ax=ax)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_ylabel(r'$\epsilon$ (eV)', fontsize=fontsize)
        ax.tick_params('y', labelsize=fontsize)
        if title != '':
            ax.set_title(title + " band structure", fontsize=fontsize)
        ax.set_xlim([min(x_points), max(x_points)])
        if e_values:
            ax.yaxis.set_ticks(np.arange(np.round(e_values[0]), np.round(e_values[1]) + 1, 1))
            ax.set_ylim(e_values)

        # ax.grid(linestyle='--')

    def plot_spectrum(self, title=''):
        """ 
        Routine to plot all the eigenvalues coming from the Bloch Hamiltonian diagonalization
        in an ordered way.
        Specially suitable for open systems, although it can be used for periodic systems as well.
        """

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

    def identify_edge_states(self, system, penetration=0.1):
        """ Routine to identify edge states according to a given penetration parameter """
        if self.eigen_states.shape[0] != 1:
            print("Error: identify_edge_states can only be used with OBC, exiting...")
            sys.exit(1)
        edge_indices = system.identify_edges()
        edge_indices = system.find_lowest_coordination_atoms()
        norbitals = len(self.eigen_states[0][:, 0])//len(system.motif)
        edge_states = []
        for n, state in enumerate(self.eigen_states[0].T):
            if self.is_edge_state(state, norbitals, edge_indices, penetration):
                edge_states.append(n)

        return edge_states

    def sort_states_by_edge_occupation(self, system):
        """ Method to compute the edge occupation of each state, and  """

    def calculate_fermi_energy(self, filling):
        """ Routine to compute the Fermi level energy according to the given filling """
        filling *= self.eigen_energy.shape[1]
        all_energies = self.eigen_energy.reshape(-1)
        all_energies = np.sort(all_energies)
        fermi_energy = all_energies[int(filling) - 1]

        return fermi_energy

    def rescale_bands_to_fermi_level(self):
        """ Routine to set the Fermi energy to zero """
        fermi_energy = self.calculate_fermi_energy(self.system.filling)
        print(self.system.filling)
        self.eigen_energy -= fermi_energy

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

    def calculate_occupation(self, atom_indices: List[int]) -> np.ndarray:
        """ 
        Method to compute the occupation for all the states in the spectrum
        on the specified atoms
        :param atom_indices: List with the indices of the atoms where we want to obtain the occupation.
        :return: Matrix with the occupations of the states, with the shape of eigen_energy.
        """
        
        occupations = np.zeros(self.eigen_energy.shape)
        for kIndex, k_eigenstates in enumerate(self.eigen_states):
            for i, eigenstate in enumerate(k_eigenstates.T):
                state = State(eigenstate, self.system)
                occupations[i, kIndex] = state.compute_specific_occupation(atom_indices)

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

        amplitudes = np.abs(self.eigenvector) ** 2
        reduced_vector = []
        counter = 0
        for atom_index in range(self.motif.shape[0]):
            species = self.motif[atom_index, 3]
            orbitals = self.norbitals[int(species)]
            atom_amplitude = amplitudes[counter:counter + orbitals]
            reduced_vector.append(np.sum(atom_amplitude))
            counter += orbitals

        return np.array(reduced_vector)

    def plot_amplitude(self, ax=None, title=None, linewidth=3, bonds=True):
        """ Method to plot the atomic amplitude of the state on top of the crystalline positions"""
        amplitude = np.array(self.atomic_amplitude())
        scaled_amplitude = scale_array(amplitude, factor=30)

        if ax is None:
            fig = plt.figure(figsize=(5, 6))
            ax = fig.add_subplot(111)

        atoms = np.array(self.motif)[:, :3]
        if bonds:
            for n, bond in enumerate(self.hoppings):
                x0, y0 = atoms[bond[0], :2]
                xneigh, yneigh = atoms[bond[1], :2]
                ax.plot([x0, xneigh], [y0, yneigh], "-k", linewidth=1.)
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






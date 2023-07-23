"""
Definition of Result class to handle diagonalization results from Hamiltonian and observable calculation.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import numpy as np
import math
import sys
from .utils import condense_vector, scale_array
from .system import System
from typing import List, Union

class Spectrum:
    """
    The Spectrum class is designed to store the results from the Hamiltonian diagonalization, 
    and to perform manipulations on the eigenvectors and eigenvalues and extract information
    about the system.

    :param eigen_energy: Array (n, nk) with the energies for each kpoint stored as columns.
    :param eigen_states: Array (nk, n, n) with the eigenvectors for each kpoint.
    :param kpoints: Array with kpoints where the Bloch Hamiltonian was evaluated.
    :param system: System used in the calculation.
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
                        ax: Axes = None, e_values: List[float] = [], fontsize: float = 10, linewidth: float = 2,
                        edgecolor = "red") -> None:
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
        :param linewidth: Linewidth. Defaults to 2.
        :param fontsize: Adjusts size of lines and text.
        :param edgecolor: Color of the edge bands if edge_states=True.
        """

        # First get distances of relevant high symmetry points
        points = [self.system.high_symmetry_points[point] for point in labels]
        path_distances = [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
        cummulative_path_weights = [0] 
        cummulative_path_weights += [np.sum(path_distances[:i + 1])/np.sum(path_distances) for i in range(len(labels) - 1)]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if e_values and len(e_values) != 2:
            raise ValueError("e_values must be [e_min, e_max]")

        if rescale:
            self.rescale_bands_to_fermi_level()

        nk = len(self.kpoints)
        x_points = np.arange(0, nk, dtype=float)
        x_ticks = []
        number_of_paths = len(labels) - 1

        label_points = []
        for n, label in enumerate(labels):
            xpos = (nk - 1)*cummulative_path_weights[n]
            x_ticks.append(xpos)
            factor = int((nk - 1)/number_of_paths)
                    
        x_points = [0]
        for n, point in enumerate(x_ticks[:-1]):
            next_point = x_ticks[n + 1]
            path_points = np.linspace(point, next_point, factor + 1, endpoint=True)
            x_points += list(path_points[1:])
                
        for eigen_energy_k in self.eigen_energy:
            ax.plot(x_points, eigen_energy_k, 'k-', linewidth=linewidth)

        if edge_states and self.system.filling is not None:
            edge_states_indices = [int(self.system.filling) + i for i in range(-2, 2)]
            for state in edge_states_indices:
                ax.plot(x_points, self.eigen_energy[state], '-', linewidth=linewidth, c=edgecolor)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_ylabel(r'$E$ (eV)', fontsize=fontsize)
        ax.tick_params('y', labelsize=fontsize)
        if title != '':
            ax.set_title(title + " band structure", fontsize=fontsize)
        ax.set_xlim([min(x_points), max(x_points)])
        if e_values:
            ax.yaxis.set_ticks(np.arange(np.round(e_values[0]), np.round(e_values[1]) + 1, 1))
            ax.set_ylim(e_values)

        ax.grid(linestyle='dashed', axis="x")

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
            raise ValueError("e_values must be [e_min, e_max]")

        if rescale:
            self.rescale_bands_to_fermi_level()

        nk = len(self.kpoints)
        x_points = np.arange(0, nk)
        x_ticks = []
        number_of_paths = len(labels) - 1

        for n, _ in enumerate(labels):
            xpos = (nk - 1)/number_of_paths*n
            x_ticks.append(xpos)
        
        # Sort by increasing occupation to avoid visual artifacts when plotting
        occupation_matrix = self.calculate_occupation(atom_indices)

        colormap = "viridis"
        cmap = get_cmap(colormap)
        segments = []
        colors = []
        for i, eigen_energy_k in enumerate(self.eigen_energy):
            for j in x_points[:-1]:
                segments.append([(x_points[j], eigen_energy_k[j]), (x_points[j + 1], eigen_energy_k[j + 1])])
                colors.append(cmap(occupation_matrix[i, j]))

        lines = LineCollection(segments[::-1], colors=colors[::-1], linewidth=2)
        ax.add_collection(lines)
        print(f"Max occupation: {np.max(occupation_matrix)}")

        cbar = plt.colorbar(lines, ax=ax, extend="both")
        cbar.set_label("Edge occupation", fontsize=fontsize)
        lines.set_cmap(colormap)
        cbar.ax.tick_params(labelsize=fontsize) 

        ax.set_xticks(x_ticks)
        for i, label in enumerate(labels):
            if label == "G":
                labels[i] = r"$\Gamma$"
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_ylabel(r'$\varepsilon$ (eV)', fontsize=fontsize)
        ax.tick_params('y', labelsize=fontsize)
        if title != '':
            ax.set_title(title + " band structure", fontsize=fontsize)
        ax.set_xlim([min(x_points), max(x_points)])
        if e_values:
            ax.yaxis.set_ticks(np.arange(np.round(e_values[0]), np.round(e_values[1]) + 1, 1))
            ax.set_ylim(e_values)
        

        # ax.grid(linestyle='--')

    def plot_spectrum(self, title: str = '',  ax: Axes = None, fontsize: int = 10) -> None:
        """ 
        Routine to plot all the eigenvalues coming from the Bloch Hamiltonian diagonalization
        in an ordered way.
        Specially suitable for open systems, although it can be used for periodic systems as well.

        :param title: Optional title for the plot.
        :param ax: Matplotlib Axes object to the plot.
        :param fontsize: Fontsize for plot labels. Defaults to 10.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        all_eigenvalues = self.eigen_energy.reshape(-1, 1)
        all_eigenvalues = np.sort(all_eigenvalues)

        ax.plot(all_eigenvalues, 'g+')
        if title != '':
            ax.set_title(f"Spectrum of {title}", fontsize=fontsize)
        else:
            ax.set_title("Spectrum", fontsize=fontsize)
        ax.set_ylabel(r"$\varepsilon (eV)$", fontsize=fontsize)
        ax.set_xlabel("n", fontsize=fontsize)
        ax.tick_params('both', labelsize=fontsize)

    def write_bands_to_file(self, file: str):
        """ TODO write_bands_to_file"""
        pass

    def write_states_to_file(self, file: str):
        """ TODO write_states_to_file"""
        pass

    # --------------- Utilities ---------------
    @staticmethod
    def is_edge_state(state: np.ndarray, norbitals: int, edge_indices: List[int], penetration: float = 0.1) -> bool:
        """
        Routine to check if an eigenstate can be regarded as edge state (as in localized at the edge).
        TODO: norbitals should be a list with the orbitals for all species.

        :param state: Array with coefficients of eigenstate.
        :param norbitals: Number of orbitals per atom. NB: Must be changed to list.
        :param edge_indices: List with indices of atoms in the edge.
        :param penetration: Maximum probability allowed for the state to be in non-edge atoms.
            Defaults to 0.1
        :return: True or False.
        """
        
        orbital_probabilities = np.abs(state) ** 2
        amplitudes = condense_vector(orbital_probabilities, norbitals)
        edge_density = np.sum(amplitudes[edge_indices])

        is_edge_state = False
        if edge_density > (1 - penetration):
            is_edge_state = True

        return is_edge_state

    def identify_edge_states(self, system: System, penetration: float = 0.1) -> List[int]:
        """ 
        Routine to identify edge states according to a given penetration parameter. To
        be used with OBC only.

        :param system: System of interest.
        :param penetration: Maximum probability allowed for the states to be in non-edge atoms.
            Defaults to 0.1
        :return: List of states located at the edge.
        """

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

    def calculate_fermi_energy(self, filling: int, bottom: bool = True) -> float:
        """ 
        Routine to compute the Fermi level energy according to the given filling .
        Fermi energy is set systematically in-between the last occupied state and the first
        unoccupied state (midgap for insulators and approximately last filled for metals).
        
        :param filling: Total number of electrons in the system.
        :param bottom: Whether to set Fermi energy at maximum of valence band. Defaults to True.
        :return: Value of Fermi energy.
        """
        
        filling = int(filling*self.eigen_energy.shape[1])
        all_energies = self.eigen_energy.reshape(-1)
        all_energies = np.sort(all_energies)
        if bottom:
            fermi_energy = all_energies[filling - 1]
        else:
            fermi_energy = (all_energies[filling - 1] + all_energies[filling])/2

        return fermi_energy

    def rescale_bands_to_fermi_level(self) -> None:
        """ 
        Routine to set the Fermi energy to zero. 
        """
        
        fermi_energy = self.calculate_fermi_energy(self.system.filling)
        self.eigen_energy -= fermi_energy

    def calculate_gap(self, filling: float) -> float:
        """ 
        Routine to compute the gap of a material based on its Fermi level/filling. 
        
        :param filling: Value of filling.
        :return: Gap value.
        """
        
        valence_band = self.eigen_energy[filling - 1, :]
        conduction_band = self.eigen_energy[filling, :]
        gap = np.min(conduction_band - valence_band)

        return gap

    def calculate_ipr(self) -> List[float]:
        """ 
        Method to compute the inverse participation ratio (IPR) from the eigenvectors.

        :return: list with all ipr for all states 
        """
        
        eigen_states = np.copy(self.eigen_states.reshape(self.system.basisdim, -1).T)
        ipr_array = []
        for eigenvector in eigen_states:
            state = State(eigenvector, self.system)
            ipr_array.append(state.compute_ipr())

        return ipr_array

    def calculate_average_ipr(self, states: Union[np.ndarray, list]) -> List[float]:
        """ 
        Method to compute the average IPR associated to a group of states.
        
        :param states: list with the eigenvectors of the states.
        :return: Average IPR of the given states. 
        """
        
        average_ipr = 0
        for eigenvector in states:
            state = State(eigenvector, self.system)
            average_ipr += state.compute_ipr()

        average_ipr /= len(states)
        return average_ipr

    def calculate_occupation(self, atom_indices: List[int]) -> np.ndarray:
        """ 
        Method to compute the occupation for all the states in the spectrum
        on the specified atoms.

        :param atom_indices: List with the indices of the atoms where we want to obtain the occupation.
        :return: Matrix with the occupations of the states, with the shape of eigen_energy.
        """
        
        occupations = np.zeros(self.eigen_energy.shape)
        for kIndex, k_eigenstates in enumerate(self.eigen_states):
            for i, eigenstate in enumerate(k_eigenstates.T):
                state = State(eigenstate, self.system)
                occupations[i, kIndex] = state.compute_occupation(atom_indices)

        return occupations

    @staticmethod
    def plot_quantity(array: Union[list, np.ndarray], name: str = None, sort: bool = False, ax: Axes = None) -> None:
        """ 
        Method to plot an array of some quantity for each state
        in the spectrum as a bar plot 
        
        :param array: Array with values to be plotted.
        :param name: Optional title for the plot. Defaults to None.
        :param sort: If True, sorts the values of array. Defaults to False.
        :param ax: Axes object to plot in.
        """
        
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
    """
    The State class is defined to separate part of the functionality of the Spectrum class
    to the eigenstates themselves.
    
    :param eigenvector: Array with coefficients of eigenvector.
    :param system: System object.
    """

    def __init__(self, eigenvector: np.ndarray, system: System):
        self.eigenvector = eigenvector
        self.motif       = system.motif
        self.norbitals   = system.norbitals
        self.hoppings    = system.bonds
        self.system      = system
        self.amplitude   = self.atomic_amplitude()

    def atomic_amplitude(self) -> None:
        """ 
        Method to obtain the probability amplitude corresponding to each atom.
        :return: Array with the probability of finding the electron on each atom (i.e. ordered by atoms like motif).
        """

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

    def plot_amplitude(self, ax: Axes = None, title: str = None, linewidth: int = 1, bonds: bool = True, factor: int = 200):
        """ 
        Method to plot the atomic amplitude of the state on top of the crystalline positions.
        
        :param ax: Axes object to plot the amplitude. Defaults to None.
        :param title: Plot title. Defaults to None.
        :param linewidth: Change width of bonds if present. Defaults to 1.
        :param bonds: Toggle on/off bonds between atoms. Defaults to True.
        :param factor: To scale the amplitude up or down.
        """
        
        amplitude = np.array(self.atomic_amplitude())
        scaled_amplitude = amplitude/np.max(amplitude)
        size_adjustment = len(amplitude)/2 - np.power((1 - scaled_amplitude), 4)
        scaled_amplitude[scaled_amplitude < 1/len(scaled_amplitude)] = 0

        if ax is None:
            fig = plt.figure(figsize=(5, 6))
            ax = fig.add_subplot(111)

        atoms = np.array(self.motif)[:, :3]
        if bonds:
            self.system.plot_wireframe(ax=ax, linewidth=linewidth)

        ax.scatter(atoms[:, 0], atoms[:, 1], c=scaled_amplitude,
                   cmap="plasma", alpha=1, s=scaled_amplitude*factor)
        ax.set_xlim([np.min(atoms[:, 0]), np.max(atoms[:, 0])])
        ax.set_ylim([np.min(atoms[:, 1]), np.max(atoms[:, 1])])
        ax.set_xticks([0, 29])
        ax.set_yticks([0, 29])
        if title is not None:
            ax.set_title(title)
        ax.axis('off')
        ax.axis('tight')
        ax.axis('equal')

    def compute_ipr(self) -> float:
        """ 
        Method to compute the inverse participation ratio (IPR) for the state. 
        
        :return: IPR of state.
        """
        
        squared_amplitude = self.amplitude ** 2
        ipr = np.sum(squared_amplitude)

        return ipr

    def compute_occupation(self, atoms_indices: List[int]) -> float:
        """ 
        Method to compute which percentage of the atomic amplitude is located
        at the given list of atoms. 
        
        :param atom_indices: List of indices of atoms where we want to compute the occupation.
        """
        
        occupation = np.sum(self.amplitude[atoms_indices])
        return occupation

    def compute_spin_projection(self, axis: str = "z"):
        """ 
        TODO compute_spin_projection.
        Method to compute the expected value of any of the three spin operators Sx, Sy or Sz.
        All calculations are done supposing that the original atomic basis for the tight-binding is written
        using the z projection of the spin (if the calculation is spinful).
        
        :param axis: 'x', 'y' or 'z'.
        :return: Expected value of spin along the given direction.
        """

        axis_list = ["x", "y", "z"]
        if axis not in axis_list:
            raise KeyError("Axis must be x, y or z")

        pass






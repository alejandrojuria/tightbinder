""" 
Implementation of System class.

The System class extendes the Crystal class, and it 
has the basic functionality for the other models to derive from it.
"""

from __future__ import annotations
from typing import Tuple, Union, List
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import sys
from itertools import product

from tightbinder.crystal import Crystal


class System(Crystal):
    """ 
    System class provides all basic functionality to create any type of model. This class
    serves as a base class for the actual models to inherit from. By itself it does not solve
    any system as it lacks specific implementation of the Hamiltonian, which has to be provided
    in the actual model implementation.

    :param system_name: Name of the system. Defaults to None.
    :param bravais_lattice: List with bravais lattice basis. Defaults to None.
    :param motif: List of atoms with their positions and their species. Defaults to None.
    :param crystal: Crystal object. Defaults to None.
    """
    
    def __init__(self, system_name: str = None, bravais_lattice: Union[list, np.ndarray] = None, 
                 motif: Union[list, np.ndarray] = None, crystal: Crystal = None):
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
        self._filling = None
        self._boundary = None
        self.bonds = []
        self._unit_cell_list = None
        self.hamiltonian = None
        self.first_neighbour_distance = None
        self._matrix_type = "dense"
        # TODO add flag for initialize_hamiltonian

    # ####################################################################################
    # #################################### Properties ####################################
    # ####################################################################################

    @property
    def norbitals(self):
        return self._norbitals

    @norbitals.setter
    def norbitals(self, norbitals):
        assert type(norbitals) == list
        self._norbitals = norbitals

    @property
    def filling(self):
        return self._filling

    @filling.setter
    def filling(self, filling):
        if filling < 0:
            print("Error: filling attribute must be between 0 and norbitals")
        else:
            self._filling = filling

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary not in ["PBC", "OBC"]:
            print('Error: Incorrect boundary option')
            sys.exit(1)
        self._boundary = boundary

    @property
    def basisdim(self):
        return self._basisdim

    @basisdim.setter
    def basisdim(self, basisdim):
        if basisdim < self.natoms:
            raise ValueError(f"basisdim has to be at least equal to the number of atoms (motif, {self.natoms})")
        self._basisdim = basisdim

    @property
    def matrix_type(self):
        return self._matrix_type

    @matrix_type.setter
    def matrix_type(self, type):
        if type not in ["dense", "sparse"]:
            raise ValueError("Invalid matrix type. Must be either dense or sparse.")
        self._matrix_type = type
    
    @property
    def unit_cell_list(self):
        if self._unit_cell_list is None:
            return []
        for cell in self._unit_cell_list:
            return self._unit_cell_list



    # ####################################################################################
    # ################################# Bonds/Neighbours #################################
    # ####################################################################################

    def add_bond(self, initial: int, final: int, cell: Union[list, tuple, np.ndarray] = (0., 0., 0.), neigh: str = '1') -> None:
        """ 
        Method to add a hopping between two atoms of the motif.
        NB: The hopping has a specified direction, from initial to final. The HSince the final
        Hamiltonian is computed taking into account hermiticity, it is not necessary to specify the hopping
        in the other direction. (TODO: Change this, no longer working like this).

        :param initial: Index of initial atom of the bond.
        :param final: Index of final atom of the bond.
        :param cell: Array with cell containing the final atom. Defaults to [0., 0., 0.].
        :param neigh: String to denote if the bond corresponds to first neighbour, second, etc. 
            Defaults to '1'.
        """

        hopping_info = [initial, final, cell, neigh]
        self.bonds.append(hopping_info)

    def add_bonds(self, initial: List[int], final: List[int], cells: Union[list, np.ndarray] = None, neigh: List[str] = None) -> None:
        """ 
        Same method as add_hopping but we input a list of hoppings at once.

        :param initial: List of indices of initial atom of the bond.
        :param final: List of indices of final atom of the bond.
        :param cell:  List of cell vectors containing the final atom. Defaults to None, which
            corresponds to the origin cell.
        :param neigh: List of strings to denote if the bonds corresponds to first neighbour, second, etc respectively.
            Defaults to None, which corresponds to all being '1'.
        """

        if len(initial) != len(final):
            raise ValueError("Initial and final lists do not have same size")
        if cells is None:
            cells = np.zeros([len(initial), 3])
        if neigh is None:
            neigh = ['1']*len(initial)
        else:
            self.boundary = "PBC"
            cells = np.array(cells)
        for bond in zip(initial, final, cells, neigh):
            self.add_bond(bond[0], bond[1], bond[2], bond[3])

    def compute_neighbour_distances(self, nni: int) -> List[float]:
        """ 
        Method to compute neighbour distance up to neighbours nni.

        :param nni: Maximum neighbour distance to consider. nni=5 mean that it
            will find first, second up to fifth neighbours distances.
        :return: List of neighbour distances.
        """

        if self.boundary == "OBC":
            near_cells = np.array([[0.0, 0.0, 0.0]])
        else:
            near_cells = generate_near_cells(self.bravais_lattice, nni)

        neigh_distance = np.array([])
        atoms = np.kron(np.array(self.motif)[:, :3], np.ones((near_cells.shape[0], 1))) + np.kron(np.ones((len(self.motif), 1)), near_cells)
        for reference_atom in self.motif:
            distance = np.linalg.norm(atoms - reference_atom[:3], axis=1)
            distance = np.unique(distance)[1:] # Remove 0 from distances
            neigh_distance = np.concatenate((neigh_distance, distance))
        
        neigh_distance = neigh_distance.round(2)
        neigh_distance = np.unique(neigh_distance)[:nni]

        return neigh_distance

    def find_first_neighbours(self) -> list:
        """ 
        Method to find the first neighbours of the atoms of the motif.
        :return: List of the corresponding bonds. 
        """

        if self.boundary == "OBC":
            near_cells = np.array([[0.0, 0.0, 0.0]])
        else:
            near_cells = generate_near_cells(self.bravais_lattice)
        first_neigh_distance = self.compute_first_neighbour_distance(near_cells)
        eps = 1E-2
        atoms = np.copy(self.motif)
        bonds = []
        for n, reference_atom in enumerate(self.motif):
            for cell in near_cells:
                distance = np.linalg.norm(atoms[:, :3] + cell - reference_atom[:3], axis=1)
                neigh_atoms_indices_max = np.where(distance <= first_neigh_distance + eps)[0]
                neigh_atoms_indices_min = np.where(first_neigh_distance - eps < distance)[0]
                neigh_atoms_indices = np.intersect1d(neigh_atoms_indices_max, neigh_atoms_indices_min)
                for i in neigh_atoms_indices:
                    bonds.append([n, i, cell])
        
        return bonds

    def find_neighbours(self, mode: str = "minimal", nn: int = 1, r: float = None) -> None:
        """ 
        From a list of atoms (motif), it sets the list of bonds within the solid.
        I.e.: Atom list -> List of neighbours/atom.
        By default it will look for the minimal distance between atoms to determine first neighbours.
        For amorphous systems the option radius is available to determine neighbours within a given radius R.
        Boundary conditions can also be set, either PBC (default) or OBC.

        :param mode: Search mode, can be either 'minimal' or 'radius'. Defaults to 'minimal'.
        :param nn: Next neighbour, used to specify up to which neighbour there are hoppings if mode='minimal'
        :param r: Value for radius sphere to detect neighbours if mode='radius' 
        """

        if self.bonds is not None:
            self.bonds = []
        eps = 1E-2
        if mode is "radius":
            if r is None:
                raise Exception("Error: Search mode is radius but no r given, exiting...")
            # TODO: Check if this is necessary
            # min_cell_size = np.min(np.linalg.norm(self.bravais_lattice, axis=1))
            # nn = int(r // min_cell_size + 1)

        elif mode is "minimal" and r is not None:
            print("Search mode is minimal but a radius was given (will not be used)")

        # Prepare unit cells to loop over depending on boundary conditions
        if self.boundary == "OBC":
            near_cells = np.array([[0.0, 0.0, 0.0]])
        else:
            near_cells = generate_near_cells(self.bravais_lattice, nn)

        # Determine neighbour distances up to nn
        neigh_distances = self.compute_neighbour_distances(nn)
        self.first_neighbour_distance = neigh_distances[0]
        # print(f"Neighbour distances: {neigh_distances}")
        if mode == "radius" and r < self.first_neighbour_distance:
            print("Warning: Radius smaller than first neighbour distance")

        # Look for neighbours
        # First in mode='minimal'
        atoms = np.copy(self.motif)
        index = 0
        if mode == 'minimal':
            for n, reference_atom in enumerate(self.motif):
                for cell in near_cells:
                    distance = np.linalg.norm(atoms[:, :3] + cell - reference_atom[:3], axis=1)
                    for nn, nn_distance in enumerate(neigh_distances):
                        neigh_atoms_indices_max = np.where(distance <= nn_distance + eps)[0]
                        neigh_atoms_indices_min = np.where(nn_distance - eps < distance)[0]

                        neigh_atoms_indices = np.intersect1d(neigh_atoms_indices_max, neigh_atoms_indices_min)
                        
                        for i in neigh_atoms_indices:
                            self.add_bond(n, i, cell, str(nn + 1))


        # Then in mode='radius'
        else:
            for n, reference_atom in enumerate(self.motif):
                for cell in near_cells:
                    distance = np.linalg.norm(atoms[:, :3] + cell - reference_atom[:3], axis=1)
                    neigh_atoms_indices_max = np.where(distance <= r + eps)[0]
                    neigh_atoms_indices_min = np.where(eps < distance)[0]
                    neigh_atoms_indices = np.intersect1d(neigh_atoms_indices_max, neigh_atoms_indices_min)
                    for i in neigh_atoms_indices:
                        self.add_bond(n, i, cell)

        print("Done")

    def find_disconnected_atoms(self) -> List[int]:
        """ 
        Method to find which atoms do not have neighbours, i.e. they are disconnected from
        the lattice.

        :return: List of indices of disconnected atoms.
        """
        
        all_bonds = self.__reconstruct_all_bonds()
        initial_atoms = [initial_atom for initial_atom, _, _ in all_bonds]
        disconnected_atoms = [atom for atom in range(self.natoms) if atom not in initial_atoms]

        return disconnected_atoms

    def remove_disconnected_atoms(self):
        """ 
        Method to remove disconnected atoms from the motif. Based on the method remove_atoms.
        """
        
        disconnected_atoms = self.find_disconnected_atoms()
        self.remove_atoms(disconnected_atoms)

    def __reconstruct_all_bonds(self) -> list:
        """ 
        Method to obtain all neighbours for all atoms since the find_neighbours method
        only computes one-way hoppings (i.e. i->j and not j->i).
        NB: Not necessary in principle since we now compute all explicitly.
        
        :return: List with all bonds.
        """
        
        all_bonds = self.bonds[:]
        for bond in self.bonds:
            initial, final, cell, nn = bond
            all_bonds.append([final, initial, cell, nn])

        return all_bonds

    @staticmethod
    def atom_coordination_number(index: int, bonds: list) -> int:
        """ 
        Method to find the coordination number for a specific atom in the crystal. 
        
        :param index: Index of atom whose coordination number we want to compute.
        :param bonds: List of indices within the solid.
        :return: Number of neighbours.
        """
        
        neighbours = np.where(bonds[:, 0] == index)[0]
        return len(neighbours)

    def coordination_number(self) -> Tuple[float, List[int]]:
        """ 
        Method to find the coordination number of the solid. Returns coordination number of solid,
        and a list with the different individual coordinations numbers found within the solid. 
        
        :return: Coordination number, and list of all individual coordination numbers found.
        """
        
        # bonds = self.__reconstruct_all_bonds()
        bonds = np.array(self.bonds, dtype=object)
        coordination = 0
        coordination_list = []
        for index in range(self.natoms):
            coordination_number = self.atom_coordination_number(index, bonds)
            coordination += coordination_number
            if coordination_number not in coordination_list:
                coordination_list.append(coordination_number)

        coordination_list = np.sort(coordination_list)
        coordination /= self.natoms
        return coordination, coordination_list

    def find_lowest_coordination_atoms(self, include_first_neighbours: bool = False) -> List[int]:
        """
        Routine to determine the atoms of the solid with the lowest coordination atoms.
        TODO: Define parameter to specify storing atoms with coordinations lower than
        one specified.
        
        :param include_first_neighbours: Includes in the list of atoms, also the first
            neighbours of those found originally. Defaults to False.
        :return: List of indices of atoms.
        """

        # bonds = self.__reconstruct_all_bonds()
        bonds = np.array(self.bonds, dtype=object)
        if bonds.size == 0:
            raise RuntimeError("Model must be initialized first")

        coordination, coordination_list = self.coordination_number()
        atoms = []
        for index in range(self.natoms):
            atom_coordination = self.atom_coordination_number(index, bonds)
            if atom_coordination < coordination_list[-1]:
                atoms.append(index)

        if include_first_neighbours:
            for atom in np.copy(atoms):
                neighbours = np.where(bonds[:, 0] == atom)[0]
                for index in neighbours:
                    neighbour = bonds[index][1]
                    if neighbour not in atoms:
                        atoms.append(neighbour)
        return atoms

    def compute_first_neighbour_distance(self, near_cells: Union[list, np.ndarray] = None) -> float:
        """
        Routine to compute the first neighbour distance.
        DEPRECATED since there is compute_neighbour_distances which works up to n neighbour distances.
        
        :param near_cells: List of unit cells to loop over. Defaults to None.
        :return: First neighbour distance.
        """

        if near_cells is None:
            near_cells = generate_near_cells(self.bravais_lattice)

        neigh_distance = 1E100
        fixed_atom = self.motif[0][:3]
        for atom, cell in product(self.motif, near_cells):
            distance = np.linalg.norm(atom[:3] + cell - fixed_atom)
            if 1E-4 < distance < neigh_distance:
                neigh_distance = distance

        self.first_neighbour_distance = neigh_distance

        return neigh_distance

    def _determine_connected_unit_cells(self) -> None:
        """ 
        Method to calculate which unit cells connect with the origin from the neighbour list. 
        """

        unit_cell_list = [[0.0, 0.0, 0.0]]
        if self.boundary == "PBC":
            for bond in self.bonds:
                unit_cell = list(bond[2])
                if unit_cell not in unit_cell_list:
                    unit_cell_list.append(unit_cell)

        self._unit_cell_list = unit_cell_list

    def identify_boundary(self, alpha: float = 0.4, ndim: int = 2, verbose: bool = False, corner_atoms: bool = False) -> np.ndarray:
        """ 
        Method to obtain the indices of the outermost atoms of the motif, taking into account
        boundary conditions. NB: Currently works only in 2D

        :param alpha: This parameters tunes the shape of the boundary. For alpha=0,
            the boundary is a convex hull; as alpha increases the boundary becomes more
            concave. Default value is 0.4
        :param ndim: Dimension of the expected boundary. Defaults to 2.
        :param verbose: Boolean parameter to print indices of boundary atoms. Defaults to False.
        :param cornes_atoms: Boolean to toggle detection of atoms that might be shared by the 
            periodic boundary and the open boundary.
        :return: Indices of atoms in the boundary.
        :raises AssertionError: Crystal dimension different from two raises error. Also 
            raises if bonds are not computed.
        """

        # Requires having computed the bonds previously
        if not self.bonds:
            raise AssertionError("Bonds must be computed first")
        bonds = self.bonds

        edges = self.identify_motif_edges(alpha, ndim)
        if not edges:
            import copy
            # If there are no edges, is because the system is effectively unidimensional, 
            # meaning that the Delaunay triangulation fails. Then consider to supercell to
            # be able to triangulate.

            bigger_system = copy.deepcopy(self).supercell(n1=2)
            bigger_system.initialize_hamiltonian()
            bonds = bigger_system.bonds

            edges = bigger_system.identify_motif_edges(alpha, ndim)

        boundary_atoms = [edge[i] for edge in edges for i in range(2)]
        boundary_atoms = np.unique(boundary_atoms)

        # Remove boundary atoms from the periodic boundary
        # First compute atoms in the periodic boundary
        periodic_boundary_atoms = []
        for bond in bonds:
            cell = bond[2]
            if np.linalg.norm(cell) != 0:
                periodic_boundary_atoms.append(bond[0])
                periodic_boundary_atoms.append(bond[1])
        periodic_boundary_atoms = np.unique(periodic_boundary_atoms)      
        
        boundary_atoms = np.setdiff1d(boundary_atoms, periodic_boundary_atoms)

        # Detect atoms in the corner between periodic and finite boundary
        if corner_atoms:
            corner_atoms = []
            for atom in boundary_atoms:
                for bond in bonds:
                    if bond[0] == atom and bond[1] in periodic_boundary_atoms:
                        corner_atoms.append(bond[1])

            corner_atoms = np.unique(corner_atoms)
            boundary_atoms = np.concatenate([boundary_atoms, corner_atoms])

        atoms = []
        for atom in boundary_atoms:
            if atom in range(len(self.motif)):
                atoms.append(atom)
        atoms = np.unique(atoms)

        if verbose:
            print(f"Atoms in boundary: {atoms}")

        if len(atoms) == 0:
            print("Warning: No boundary atoms found")

        return np.array(atoms, dtype=int)
    
    def plot_wireframe(self, ax: plt.Axes = None, linewidth: float = 1, alpha: float = 0.5, color="black") -> None:
        """
        Method to plot a wireframe of the system according to the detected bonds.

        :param ax: Matplotlib axes object. If None, a new Axis object is created.
        :param linewidth: Linewidth of the wireframe. Defaults to 1.
        :param alpha: Sets wireframe transparency. Defaults to 0.5
        :param color: Wireframe color. Defaults to black.
        """

        if not self.bonds:
            raise LookupError("plot_wireframe requires having computed the bonds of the system.")

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        for n, bond in enumerate(self.bonds):

            x0, y0 = self.motif[bond[0], :2]
            xneigh, yneigh = self.motif[bond[1], :2]
            ax.plot([x0, xneigh], [y0, yneigh], "-", linewidth=linewidth, alpha=alpha, color=color)


    # ####################################################################################
    # ############################### System modifications ###############################
    # ####################################################################################

    def supercell(self, update: bool = True, **ncells: dict) -> System:
        """ 
        Routine to generate a supercell for a system using the number of cells
        along each Bravais vectors specified in ncells.

        :param update: Boolean parameter to update all of Crystal class atributes.
            Defaults to True, only set to false when used inside other routines (reduce)
        :param dict: Specify number of cells in some direction: n1, n2 and/or n3.
        :return: Modified System object. 
        """

        if len(ncells) == 0:
            print("Error: Reduce method must be called with at least one parameter (nx, ny or nz), exiting...")
            sys.exit(1)
        key_to_index = {"n1": 0, "n2": 1, "n3": 2}
        for key in ncells.keys():
            if key not in ["n1", "n2", "n3"]:
                print("Error: Invalid input (must be n1, n2 or n3), exiting...")
                sys.exit(1)
            if key_to_index[key] > self.ndim:
                print(f"Error: Axis {key} to reduce along not present (higher than system dimension)")

            new_motif = self.motif
            for n in range(1, ncells[key]):
                motif_copy_displaced = np.copy(self.motif)
                motif_copy_displaced[:, :3] += n * self.bravais_lattice[key_to_index[key]]
                new_motif = np.append(new_motif, motif_copy_displaced, axis=0)

            # Update Bravais lattice and motif
            self.bravais_lattice[key_to_index[key]] *= ncells[key]
            self.motif = new_motif
        
        if update:
            self.update()

        return self

    def reduce(self, **ncells: dict) -> System:
        """ 
        Routine to reduce the dimensionality of the System object along the specified
        directions, by repeating unit cells along those directions until a given size
        (number of unit cells) is reached. Thus we make the original system finite along those
        directions.

        :param ncells: Number of cells of direction(s) along which we want to reduce the
            dimensionality of the system, n1, n2 and/or n3.
        :return: Modified System object.
        """

        key_to_index = {"n1": 0, "n2": 1, "n3": 2}
        self.supercell(update=False, **ncells)
        indices = [index for index in list(range(self.ndim))
                   if index not in [key_to_index[key] for key in ncells.keys()]]
        if not indices:
            self.bravais_lattice = None
            self.boundary = "OBC"
        else:
            self.bravais_lattice = self.bravais_lattice[indices]

        return self

    def ribbon(self, width: int, orientation: str = "horizontal", periodic: bool = False) -> System:
        """ 
        Routine to generate a ribbon for a 2D crystal. It is designed to create rectangular ribbons,
        based on a rectangular lattice. Therefore there must exist a combination of basis vectors such that
        we can obtain a rectangular supercell. Otherwise the method returns an error. 

        :param width: Width of the ribbon (number of cells).
        :param orientation: 'horizontal' or 'vertical'. Defaults to 'horizontal'.
        :param periodic: Boolean to keep the ribbon periodic along the direction it is built. Defauts to False.
            Turning this one would correspond to a nanotube instead of nanoribbon (finite).
        :return: Modified System object.
        """
        
        if orientation not in ["horizontal", "vertical"]:
            raise ValueError("Wrong orientation provided, must be either vertical or horizontal, exiting...")
        
        if self.ndim != 2:
            print(f"Ribbons can not be generated for {self.ndim}D structures")
            sys.exit(1)
        else:
            mesh_points = []
            for _ in range(self.ndim):
                mesh_points.append(list(range(-1, 2)))

            combinations = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.ndim)
            rectangular_basis = np.zeros([2, 3])
            all_possible_atoms = []
            motif = []

            # Calculate rectangular lattice basis vectors
            for n, m in combinations:
                vector = n * self.bravais_lattice[0] + m * self.bravais_lattice[1]
                for atom in self.motif:
                    new_atom = np.copy(atom)
                    new_atom[:3] += vector
                    all_possible_atoms.append(new_atom)

                if n == m == 0:
                    continue
                vector_angle = np.arctan2(vector[1], vector[0])

                if abs(vector_angle - 0.0) < 1E-10:
                    rectangular_basis[0, :] = vector
                elif abs(vector_angle - np.pi/2) < 1E-10:
                    rectangular_basis[1, :] = vector

            if not np.linalg.norm(rectangular_basis, axis=1).any():
                print("Error: Could not generate a rectangular lattice, use reduce method directly. Exiting...")
                sys.exit(1)

            # Calculate new motif
            for i, atom in enumerate(all_possible_atoms):
                if (0 <= atom[0] < np.linalg.norm(rectangular_basis[0, :])) and \
                   (0 <= atom[1] < np.linalg.norm(rectangular_basis[1, :])):
                    motif.append(atom)

            # Update system attributes
            self.bravais_lattice = rectangular_basis
            self.motif = motif

            # Reduce system dimensionality
            if not periodic:
                if orientation == "horizontal":
                    self.reduce(n1=width)
                else:
                    self.reduce(n2=width)
            else:
                if orientation == "horizontal":
                    self.supercell(n1=width)
                else:
                    self.supercell(n2=width)


            return self
        
    def extend_copy(self, vector: List[list, np.ndarray], n: int = 1) -> System:
        """
        Method to extend the system by attaching a copy of the motif displaced by some input vector.

        :param vector: Vector by which we displace the copy of the system.
        :param n: Number of copies to attach. The n-th copy will be displaced by n*vector. Defaults to 1.
        :return: Extended system.
        """

        displaced_motif = np.copy(self.motif)
        for _ in range(n):
            displaced_motif[:, :3] += np.array(vector)
            self.motif = np.concatenate((self.motif, displaced_motif))  

        return self
    
    def rotate(self, angle: float = 90) -> System:
        """
        Method to rotate the whole system in the xy plane by some angle.

        :param angle: Value of rotation angle in degrees. Defaults to 90 degrees.
        :return: Rotated system.
        """

        rotation_matrix = np.array([[np.cos(np.pi*angle/180), -np.sin(np.pi*angle/180), 0],
                                    [np.sin(np.pi*angle/180),  np.cos(np.pi*angle/180), 0],
                                    [                      0,                        0, 1]])

        for n in range(self.natoms):
            self.motif[n, :3] = rotation_matrix @ self.motif[n, :3]

        rotated_bravais = np.zeros(self.bravais_lattice.shape)
        for n in range(self.ndim):
            rotated_bravais[n] = rotation_matrix @ self.bravais_lattice[n]
        self.bravais_lattice = rotated_bravais

        return self

    def restrict_lattice2rectangle(self) -> None:
        """
        TO BE IMPLEMENTED YET (is it really necessary?).
        """
        
        atoms = []
        mesh_points = []
        for i in range(self.ndim):
            mesh_points.append(list(range(-1, 2)))

        combinations = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.ndim)
        for n, m in combinations:
            vector = n*self.bravais_lattice[0] + m*self.bravais_lattice[1]
            for position in self.motif:
                atom_position = vector + position


    # ####################################################################################
    # ################################### Hamiltonian ####################################
    # ####################################################################################

    def initialize_hamiltonian(self):
        """ 
        Generic implementation of initialization of the Hamiltonian. To be overwritten
        by specific implementations of System. 
        """
        
        print("initialize_hamiltonian has to be implemented by child class")

    def hamiltonian_k(self, k: Union[list, np.ndarray]) -> np.ndarray:
        """ 
        Generic implementation of hamiltonian_k H(k). To be overwritten
        by specific implementations of System. 
        """
        
        print("Has to be implemented by child class")

    def solve(self, kpoints: list = None, neigval: int = None):
        """ 
        Diagonalize the Hamiltonian to obtain the band structure and the eigenstates. 
        
        :param kpoints: List of kpoints where we want to diagonalize the Bloch Hamiltonian.
            Defaults to None, which corresponds to [[0., 0., 0.]] for both PBC and OBC.
        :param neigval: Number of eigenvalues to compute if using sparse matrices.
            Defaults to None, which in turn is set to the filling.
        :return: Spectrum object.
        """

        # Import result here to avoid circular import
        from . import result
        
        if kpoints is None:
            kpoints = np.array([[0., 0., 0.]])
        if neigval is None and self.matrix_type == "sparse":
            neigval = int(self.filling)

        nk = len(kpoints)
        eigen_energy = np.zeros([self.basisdim, nk])
        eigen_states = []

        if self.matrix_type == "dense":
            solver = lambda h: np.linalg.eigh(h)
        else:
            solver = lambda h: scipy.sparse.linalg.eigsh(h, k=neigval)

        for n, k in enumerate(kpoints):
            hamiltonian = self.hamiltonian_k(k)
            results = solver(hamiltonian)
            eigen_energy[:, n] = results[0]
            eigen_states.append(results[1])

        return result.Spectrum(eigen_energy, np.array(eigen_states), kpoints, self)


class FrozenClass:
    """ 
    Class to enforce immutable attributes. Used for predefined models.
    """
    
    _is_frozen = False

    def __setattr__(self, key, value):
        if self._is_frozen:
            raise Exception("Predefined model can not be modified")
        super().__setattr__(key, value)

    def _freeze(self):
        self._is_frozen = True


def generate_all_combinations(ndim: int, n: int = 1) -> np.ndarray:
    """ 
    Auxiliary routine to generate an array of combinations of possible neighbouring
    unit cells. 
    
    :param ndim: Number of axis to make the combinations.
    :param int: Values on each axis go from [-n, n]. Defaults to 1.
    :return: Array with vectors of length ndim and combinations.
    """
    
    mesh_points = []
    for i in range(ndim):
        mesh_points.append(list(range(-n, n+1)))
    mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, ndim)

    return mesh_points


def generate_half_combinations(ndim: int) -> np.ndarray:
    """ 
    Auxiliary routine to generate an array of combinations of possible neighbouring
    unit cells. NB: It only generates half of them, since we are going to use hermiticity to
    generate the Hamiltonian. 
    
    :param ndim: Number of axis to make the combinations.
    :return: Half the combinations.
    """
    
    all_combinations = generate_all_combinations(ndim)

    # Eliminate vectors that are the inverse of others
    points = []
    for point in all_combinations:
        if list(-point) not in points:
            points.append(list(point))

    return np.array(points)


def generate_near_cells(bravais_lattice: Union[list, np.ndarray], n: int = 1, half: bool = False) -> np.ndarray:
    """ 
    Auxiliary routine to generate the Bravais vectors corresponding to unit cells
    neighbouring the origin one. NB: It only generates half of them, since we are going to use hermiticity to
    generate the Hamiltonian. 
    
    :param bravais_lattice: Basis vectors of the Bravais lattice.
    :param n: Values on each axis go from [-n, n]. Defaults to 1.
    :param half: Generate only half the combinations. Defaults to False.
    :return: Array with combinations of Bravais basis vectors.
    """
    
    if bravais_lattice is None:
        return np.array([[0., 0., 0.]])
        
    ndim = len(bravais_lattice)
    if not half:
        mesh_points = generate_all_combinations(ndim, n)
    else:
        mesh_points = generate_half_combinations(ndim, n)
    near_cells = np.zeros([len(mesh_points), 3])
    for n, coefficients in enumerate(mesh_points):
        cell_vector = np.array([0.0, 0.0, 0.0])
        for i, coefficient in enumerate(coefficients):
            cell_vector += (coefficient * bravais_lattice[i])
        near_cells[n, :] = cell_vector

    return near_cells




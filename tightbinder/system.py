# Implementation of System class, which incorporates both Hamiltonian and Crystal classes.
# Specifically, System includes the Crystal as a class attribute.
# The reasoning behind this, is to make available the crystal specific routines and attributes to the system,
# while being explicit about its ambit and without the crystal being a separate entity.
# As for the Hamiltonian, its functionality is directly available from System due to the inheritance.
# Base Hamiltonian class implementation. It will be used for the other hamiltonians to derive
# from it
# System has the basic functionality for the other models to derive from it.

from .crystal import Crystal
from .result import Spectrum
import numpy as np
import sys
from multiprocessing import cpu_count, Pool
from itertools import product


num_cores = cpu_count()


class System(Crystal):
    """ System class provides all basic functionality to create any type of model. This class
     serves as a base class for the actual models to inherit from. By itself it does not solve
     any system as it lacks specific implementation of the Hamiltonian, which has to be provided
     in the actual model implementation. """
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
        self._filling = None
        self._boundary = None
        self.bonds = []
        self._unit_cell_list = None
        self.hamiltonian = None
        self.first_neighbour_distance = None
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
        if filling > 1 or filling < 0:
            print("Error: filling attribute must be between 0 and 1")
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
    # ####################################################################################
    # ################################# Bonds/Neighbours #################################
    # ####################################################################################

    def add_bond(self, initial, final, cell=(0., 0., 0.)):
        """ Method to add a hopping between two atoms of the motif.
        NB: The hopping has a specified direction, from initial to final. Since the final
        Hamiltonian is computed taking into account hermiticity, it is not necessary to specify the hopping
        in the other direction.
         Parameters:
             int initial, final: Indices of the atoms in the motif
             array cell: Bravais vector connecting the cells of the two atoms. Defaults to zero """
        hopping_info = [initial, final, cell]
        self.bonds.append(hopping_info)

    def add_bonds(self, initial, final, cells=None):
        """ Same method as add_hopping but we input a list of hoppings at once.
        Parameters:
             list hoppings: list of size nhop
             list initial, final: list of indices
             matrix cells: Each row denotes the Bravais vector connecting two cells. Defaults to None """
        if len(initial) != len(final):
            raise ValueError("Initial and final lists do not have same size")
        if cells is None:
            cells = np.zeros([len(initial), 3])
        else:
            self.boundary = "PBC"
            cells = np.array(cells)
        for bond in zip(initial, final, cells):
            self.add_bond(bond[0], bond[1], bond[2])

    def find_neighbours(self, mode="minimal", r=None):
        """ Given a list of atoms (motif), it returns a list in which each
        index corresponds to a list of atoms that are first neighbours to that index
        on the initial atom list.
        I.e.: Atom list -> List of neighbours/atom.
        By default it will look for the minimal distance between atoms to determine first neighbours.
        For amorphous systems the option radius is available to determine neighbours within a given radius R.
        Boundary conditions can also be set, either PBC (default) or OBC.
        :param mode: Search mode, can be either 'minimal' or 'radius'. Defaults to 'minimal'.
        :param r: Value for radius sphere to detect neighbours
        """
        if self.bonds is not None:
            self.bonds = []
        eps = 1E-2
        if mode is "radius" and r is None:
            raise Exception("Error: Search mode is radius but no r given, exiting...")
        elif mode is "minimal" and r is not None:
            print("Search mode is minimal but a radius was given (will not be used)")

        # Prepare unit cells to loop over depending on boundary conditions
        if self.boundary == "OBC":
            near_cells = np.array([[0.0, 0.0, 0.0]])
        else:
            near_cells = generate_near_cells(self.bravais_lattice)

        # Determine neighbour distance from one fixed atom
        neigh_distance = self.compute_first_neighbour_distance(near_cells)
        if mode == "radius" and r < neigh_distance:
            print("Warning: Radius smaller than first neighbour distance")
        elif mode == "minimal":
            r = neigh_distance

        # Determine list of neighbours for each atom of the motif
        index = 0
        atoms = np.copy(self.motif)
        for n, reference_atom in enumerate(self.motif):
            for cell in near_cells:
                distance = np.linalg.norm(atoms[index:, :3] + cell - reference_atom[:3], axis=1)
                neigh_atoms_indices_max = np.where(distance <= r + eps)[0]
                if mode == "minimal":
                    neigh_atoms_indices_min = np.where(r - eps < distance)[0]
                else:
                    neigh_atoms_indices_min = np.where(eps < distance)[0]
                neigh_atoms_indices = np.intersect1d(neigh_atoms_indices_max, neigh_atoms_indices_min)
                for i in neigh_atoms_indices:
                    self.add_bond(n, i + index, cell)
            index += 1

        print("Done")

    def find_disconnected_atoms(self):
        """ Method to find which atoms do not have neighbours, i.e. they are disconnected from
         te he lattice """
        all_bonds = self.__reconstruct_all_bonds()
        initial_atoms = [initial_atom for initial_atom, _, _ in all_bonds]
        disconnected_atoms = [atom for atom in range(self.natoms) if atom not in initial_atoms]

        return disconnected_atoms

    def remove_disconnected_atoms(self):
        """ Method to remove disconnected atoms from the motif """
        disconnected_atoms = self.find_disconnected_atoms()
        self.remove_atoms(disconnected_atoms)

    def __reconstruct_all_bonds(self):
        """ Method to obtain all neighbours for all atoms since the find_neighbours method
         only computes one-way hoppings (i.e. i->j and not j->i) """
        all_bonds = self.bonds[:]
        for bond in self.bonds:
            initial, final, cell = bond
            all_bonds.append([final, initial, cell])

        return np.array(all_bonds)

    @staticmethod
    def atom_coordination_number(index, bonds):
        """ Method to find the coordination number for a specific atom in the crystal """
        neighbours = np.where(bonds[:, 0] == index)[0]

        return len(neighbours)

    def coordination_number(self):
        """ Method to find the coordination number of the solid """
        bonds = self.__reconstruct_all_bonds()
        coordination = 0
        for index in range(self.natoms):
            coordination += self.atom_coordination_number(index, bonds)

        coordination /= self.natoms
        return coordination

    def find_lowest_coordination_atoms(self):
        bonds = self.__reconstruct_all_bonds()
        coordination = self.coordination_number()
        atoms = []
        for index in range(self.natoms):
            atom_coordination = self.atom_coordination_number(index, bonds)
            if coordination >= 3 and atom_coordination <= 3:
                atoms.append(index)
            elif coordination < 3 and atom_coordination <= 2:
                atoms.append(index)

        return atoms

    def compute_first_neighbour_distance(self, near_cells=None):
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

    def _determine_connected_unit_cells(self):
        """ Method to calculate which unit cells connect with the origin from the neighbour list """

        unit_cell_list = [[0.0, 0.0, 0.0]]
        if self.boundary == "PBC":
            for bond in self.bonds:
                unit_cell = list(bond[2])
                if unit_cell not in unit_cell_list:
                    unit_cell_list.append(unit_cell)

        self._unit_cell_list = unit_cell_list

    # ####################################################################################
    # ############################### System modifications ###############################
    # ####################################################################################

    def supercell(self, update=True, **ncells):
        """ Routine to generate a supercell for a system using the number of cells
         along each Bravais vectors specified in ncells.
         Input:
         update: Boolean parameter to update all of Crystal class atributes.
         Defaults to True, only set to false when used inside other routines (reduce)
         int nx, ny and/or nz """

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

        self.basisdim = self.norbitals * len(self.motif)

        return self

    def reduce(self, **ncells):
        """ Routine to reduce the dimensionality of the System object along the specified
         directions, by repeating unit cells along those directions until a given size
         (number of unit cells) is reached. Thus we make the original system finite along those
         directions.
         Input: int n1, n2 or n3 """

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

    def ribbon(self, width, orientation="horizontal"):
        """ Routine to generate a ribbon for a 2D crystal. It is designed to create rectangular ribbons,
        based on a rectangular lattice. Therefore there must exist a combination of basis vectors such that
        we can obtain a rectangular supercell. Otherwise the method returns an error. """
        if self.ndim != 2:
            print(f"Ribbons can not be generated for {self.ndim}D structures")
            sys.exit(1)
        else:
            mesh_points = []
            for i in range(self.ndim):
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
            if orientation == "horizontal":
                self.reduce(n1=width)
            elif orientation == "vertical":
                self.reduce(n2=width)
            else:
                print("Error: Wrong orientation provided, must be either vertical or horizontal, exiting...")
                sys.exit(1)

            return self

    def stack(self, height, layers=1):
        """ Method to stack layers of a two-dimensional material vertically.
        The returned system is still two-dimensional.
        :param height: Distance in the z axis between two contiguous layers, measured
        from one fixed atom of the motif. Units are those used in the lattice.
        :param layers: Number of layers to stack. Defaults to 1. """
        if self.ndim != 2:
            raise Exception("Error: System dimension must be 2 to stack, exiting...")
        if layers <= 0 or type(layers) != int:
            raise Exception("Error: Layers must be a positive integer")
        if height < 0:
            raise Exception("Error: Height must be strictly positive ")

        motif = np.copy(self.motif)
        for layer in range(1, layers + 1):
            displaced_layer = np.copy(motif)
            displaced_layer[:, :3] += np.array([0., 0., -height*layer])
            motif = np.append(motif, displaced_layer, axis=0)

        self.motif = motif

        return self

    def _restrict_lattice2rectangle(self, vectors):
        """ TO BE IMPLEMENTED YET (is it really necessary?) """
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
        """ Generic implementation of initialization of the Hamiltonian. To be overwritten
            by specific implementations of System """
        raise Exception("Has to be implemented by child class")

    def hamiltonian_k(self, k):
        """ Generic implementation of hamiltonian_k H(k). To be overwritten
            by specific implementations of System """
        raise Exception("Has to be implemented by child class")

    def solve(self, kpoints=None):
        """ Diagonalize the Hamiltonian to obtain the band structure and the eigenstates """
        if kpoints is None:
            kpoints = np.array([[0., 0., 0.]])

        nk = len(kpoints)
        eig_num = 1000
        eigen_energy = np.zeros([self.basisdim, nk])
        eigen_states = []

        for n, k in enumerate(kpoints):
            hamiltonian = self.hamiltonian_k(k)
            results = np.linalg.eigh(hamiltonian)
            eigen_energy[:, n] = results[0]
            eigen_states.append(results[1])

        return Spectrum(eigen_energy, np.array(eigen_states), kpoints, self)


class FrozenClass:
    """ Class to enforce immutable attributes """
    _is_frozen = False

    def __setattr__(self, key, value):
        if self._is_frozen:
            raise Exception("Predefined model can not be modified")
        super().__setattr__(key, value)

    def _freeze(self):
        self._is_frozen = True


def search_neighbour(reference_atom, i, atom, cell, radius):
    """ Auxiliary routine to parallelize search for all hoppings (neighbours) within a given
     system"""
    eps = 1E-14
    distance = np.linalg.norm(atom[:3] + cell - reference_atom[:3])
    if distance <= radius:
        print(i, cell)
        if not np.array_equal(reference_atom, atom) and not np.array_equal(cell, [0, 0, 0]):
            return [i, cell]
        else:
            return


def generate_all_combinations(ndim):
    """ Auxiliary routine to generate an array of combinations of possible neighbouring
     unit cells. """
    mesh_points = []
    for i in range(ndim):
        mesh_points.append(list(range(-1, 2)))
    mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, ndim)

    return mesh_points


def generate_half_combinations(ndim):
    """ Auxiliary routine to generate an array of combinations of possible neighbouring
     unit cells. NB: It only generates half of them, since we are going to use hermiticity to
     generate the Hamiltonian """
    all_combinations = generate_all_combinations(ndim)

    # Eliminate vectors that are the inverse of others
    points = []
    for point in all_combinations:
        if list(-point) not in points:
            points.append(list(point))

    return points


def generate_near_cells(bravais_lattice, half=False):
    """ Auxiliary routine to generate the Bravais vectors corresponding to unit cells
     neighbouring the origin one. NB: It only generates half of them, since we are going to use hermiticity to
     generate the Hamiltonian """
    ndim = len(bravais_lattice)
    if not half:
        mesh_points = generate_all_combinations(ndim)
    else:
        mesh_points = generate_half_combinations(ndim)
    near_cells = np.zeros([len(mesh_points), 3])
    for n, coefficients in enumerate(mesh_points):
        cell_vector = np.array([0.0, 0.0, 0.0])
        for i, coefficient in enumerate(coefficients):
            cell_vector += (coefficient * bravais_lattice[i])
        near_cells[n, :] = cell_vector

    return near_cells





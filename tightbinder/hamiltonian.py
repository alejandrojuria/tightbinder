# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

import numpy as np
import sys
import math
import cmath
import itertools
from system import System
from crystal import Crystal
# --------------- Constants ---------------
PI = 3.14159265359
EPS = 0.001 # !!!! To be changed, depends on the precision of the crystal vectors


class SKModel(System):
    def __init__(self, configuration=None, mode='minimal', r=None, boundary="PBC"):
        if configuration is not None:
            super().__init__(system_name=configuration["System name"],
                             bravais_lattice=configuration["Bravais lattice"],
                             motif=configuration["Motif"])
        else:
            super().__init__()

        """ Specific attributes of SKModel """
        self.configuration = configuration
        self.hamiltonian = None
        self.neighbours = None
        self.spin_orbit_hamiltonian = None
        self.__unit_cell_list = None
        if mode not in ['minimal', 'radius']:
            print('Error: Incorrect mode')
            sys.exit(1)
        self.__mode = mode
        self.__r = r
        if boundary not in ["PBC", "OBC"]:
            print('Error: Incorrect boundary option')
            sys.exit(1)
        self.__boundary = boundary
        self.__spin_blocks = None
        self.__zeeman = None
        self.ordering = None

    # --------------- Methods ---------------
    def first_neighbours(self):
        """ Given a list of atoms (motif), it returns a list in which each
        index corresponds to a list of atoms that are first neighbours to that index
        on the initial atom list.
        I.e.: Atom list -> List of neighbours/atom.
        By default it will look for the minimal distance between atoms to determine first neighbours.
        For amorphous systems the option radius is available to determine neighbours within a given radius R.
        Boundary conditions can also be set, either PBC (default) or OBC."""

        # Prepare unit cells to loop over depending on boundary conditions
        if self.__boundary == "PBC":
            mesh_points = []
            for i in range(self.ndim):
                mesh_points.append(list(range(-1, 2)))
            mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.ndim)

            near_cells = np.zeros([len(mesh_points), 3])
            for n, coefficients in enumerate(mesh_points):
                cell_vector = np.array([0.0, 0.0, 0.0])
                for i, coefficient in enumerate(coefficients):
                    cell_vector += (coefficient * self.bravais_lattice[i])
                near_cells[n, :] = cell_vector

        elif self.__boundary == "OBC":
            near_cells = np.array([0.0, 0.0, 0.0])

        # Determine neighbour distance from one fixed atom
        neigh_distance = 1E100
        fixed_atom = self.motif[0][:3]
        for cell in near_cells:
            for atom in self.motif:
                distance = np.linalg.norm(atom[:3] + cell - fixed_atom)
                if distance < neigh_distance and distance != 0: neigh_distance = distance

        # Determine list of neighbours for each atom of the motif
        if self.__mode == "minimal":
            neighbours_list = []
            for n, reference_atom in enumerate(self.motif):
                neighbours = []
                for cell in near_cells:
                    for i, atom in enumerate(self.motif):
                        distance = np.linalg.norm(atom[:3] + cell - reference_atom[:3])
                        if abs(distance - neigh_distance) < EPS: neighbours.append([i, cell])
                neighbours_list.append(neighbours)

        elif self.__mode == "radius":
            if self.__r is None:
                print('Radius not defined in "radius" mode, exiting...')
                sys.exit(1)
            elif self.__r < neigh_distance:
                print("Warning: Radius smaller than first neighbour distance")

            neighbours_list = []
            for n, reference_atom in enumerate(self.motif):
                neighbours = []
                for cell in near_cells:
                    for i, atom in enumerate(self.motif):
                        distance = np.linalg.norm(atom[0] + cell - reference_atom[0])
                        if distance <= self.__r: neighbours.append([i, cell])
                neighbours.append(neighbours)

        self.neighbours = neighbours_list

    def __hopping_amplitude(self, position_diff, *orbitals):
        """ Routine to compute the hopping amplitude from one atom to another depending on the
        participating orbitals, following the Slater-Koster approxiamtion """

        initial_orbital = orbitals[0][0]
        initial_species = orbitals[0][1]
        final_orbital = orbitals[0][2]
        final_species = orbitals[0][3]

        initial_orbital_type = initial_orbital[0]
        final_orbital_type = final_orbital[0]

        amplitudes = self.configuration['SK amplitudes']

        possible_orbitals = {'s': 0, 'p': 1, 'd': 2}
        if possible_orbitals[initial_orbital_type] > possible_orbitals[final_orbital_type]:
            position_diff = np.array(position_diff)*(-1)
            orbitals = [final_orbital, final_species, initial_orbital, initial_species]
            hopping = self.__hopping_amplitude(position_diff, orbitals)
            return hopping

        amplitudes = np.array(amplitudes)
        # Mixing of amplitudes on case of having different species
        # So far the mixing if equivalent (beta = 0.5)
        beta = 0.5
        if initial_species != final_species:
            effective_amplitudes = (amplitudes[initial_species] + amplitudes[final_species]) * beta
        else:
            effective_amplitudes = amplitudes[initial_species]

        direction_cosines = position_diff / np.linalg.norm(position_diff)
        direction_cosines = {'x': direction_cosines[0], 'y': direction_cosines[1], 'z': direction_cosines[2]}
        [l, m, n] = direction_cosines.values()
        (Vsss, Vsps, Vpps, Vppp, Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd) = effective_amplitudes
        special_orbital = True if final_orbital == 'dx2-y2' or final_orbital == 'd3z2-r2' else False

        # Start checking the different possibilities
        if initial_orbital == "s":
            if final_orbital == "s":
                hopping = Vsss
            elif final_orbital_type == "p":
                coordinate_final = final_orbital[1]
                hopping = direction_cosines[coordinate_final] * Vsps
            elif final_orbital_type == "d" and not special_orbital:
                first_coordinate = final_orbital[1]
                second_coordinate = final_orbital[2]
                hopping = math.sqrt(3) * direction_cosines[first_coordinate] * direction_cosines[second_coordinate] * Vsds
            elif final_orbital == "dx2-y2":
                hopping = math.sqrt(3)/2.*(l*l - m*m)*Vsds
            elif final_orbital == "d3z2 - r2":
                hopping = (n*n - (l*l + m*m)/2.)*Vsds

        elif initial_orbital_type == "p":
            coordinate_initial = initial_orbital[1]
            if final_orbital_type == "p":
                coordinate_final = final_orbital[1]
                hopping = direction_cosines[coordinate_initial]*direction_cosines[coordinate_final]*(Vpps - Vppp)
                if coordinate_final == coordinate_initial:
                    hopping += Vppp
            elif final_orbital_type == "d" and not special_orbital:
                first_coordinate = final_orbital[1]
                second_coordinate = final_orbital[2]
                hopping = (direction_cosines[coordinate_initial]*direction_cosines[first_coordinate]
                           * direction_cosines[second_coordinate]*(math.sqrt(3)*Vpds - 2*Vpdp))
                if coordinate_initial == first_coordinate:
                    hopping += direction_cosines[second_coordinate]*Vpdp
                elif coordinate_initial == second_coordinate:
                    hopping += direction_cosines[first_coordinate]*Vpdp
            elif final_orbital == "dx2-y2":
                hopping = direction_cosines[coordinate_initial]*(l*l - m*m)*(math.sqrt(3)/2.*Vpds - Vpdp)
                if coordinate_initial == 'x':
                    hopping += l*Vpdp
                elif coordinate_initial == 'y':
                    hopping += -m*Vpdp
            elif final_orbital == "d3z2-r2":
                hopping = direction_cosines[coordinate_initial]*(n*n - (l*l + m*m)/2.)*Vpds
                if coordinate_initial == 'z':
                    hopping += math.sqrt(3)*n*(l*l + m*m)*Vpdp
                else:
                    hopping -= math.sqrt(3)*direction_cosines[coordinate_initial]*n*n*Vpdp

        elif initial_orbital_type == "d" and initial_orbital not in ["dx2-y2, d3r2-z2"]:
            coordinate_initial_first = initial_orbital[1]
            coordinate_initial_second = initial_orbital[2]
            for coordinate in ["x", "y", "z"]:  # Determine non present coordinate
                if coordinate not in [coordinate_initial_first, coordinate_initial_second]:
                    coordinate_initial_third = coordinate
            if not special_orbital:
                coordinate_final_first = final_orbital[1]
                coordinate_final_second = final_orbital[2]
                [l_aux, m_aux, n_aux] = ([direction_cosines[coordinate_initial_first],
                                          direction_cosines[coordinate_initial_second],
                                          direction_cosines[coordinate_initial_third]])
                hopping = (3.*l_aux*m_aux *
                           direction_cosines[coordinate_final_first]*direction_cosines[coordinate_final_second]*Vdds)
                if initial_orbital == final_orbital:
                    hopping += (l_aux**2 + m_aux**2 - 4*l_aux**2*m_aux**2)*Vddp + (n_aux**2 + l_aux**2*m_aux**2)*Vddd
                else:
                    non_repeated_coordinates = list({coordinate_initial_first, coordinate_initial_second} -
                                                    {coordinate_final_first, coordinate_final_second})
                    repeated_coordinate = list({coordinate_initial_first, coordinate_initial_second} &
                                               {coordinate_final_first, coordinate_final_second})[0]
                    m_aux = direction_cosines[repeated_coordinate]
                    [l_aux, n_aux] = [direction_cosines[non_repeated_coordinates[0]],
                                      direction_cosines[non_repeated_coordinates[1]]]
                    hopping += l_aux*n_aux*(1 - 4*m_aux**2)*Vddp + l_aux*n_aux*(m_aux**2 - 1)*Vddd
            elif final_orbital == "dx2-y2":
                hopping = (direction_cosines[coordinate_initial_first]*direction_cosines[coordinate_initial_second] *
                           (l*l - m*m)*(3./2*Vdds - 2*Vddp + Vddd/2.))
                if initial_orbital == "yz":
                    hopping += (direction_cosines[coordinate_initial_first]*direction_cosines[coordinate_initial_second] *
                                (-Vddp + Vddd/2.))
                elif initial_orbital == "zx":
                    hopping -= (direction_cosines[coordinate_initial_first]*direction_cosines[coordinate_initial_second] *
                                (-Vddp + Vddd/2.))
            elif final_orbital == "d3z2-r2":
                hopping = math.sqrt(3)*(direction_cosines[coordinate_initial_first] *
                                        direction_cosines[coordinate_initial_second]*(n*n - (l*l + m*m)/2))*Vdds
                if initial_orbital == 'xy':
                    hopping += -2*l*m*n*n*Vddp + (l*m*(1 + n*n)/2)*Vddd
                else:
                    hopping += (direction_cosines[coordinate_initial_first]*direction_cosines[coordinate_initial_second] *
                                ((l*l + m*m - n*n)*Vddp - (l*l + m*m)/2.*Vddd))

            elif initial_orbital == "dx2-y2":
                if final_orbital == "dx2-y2":
                    hopping = 3./4*(l*l - m*m)**2*Vdds + (l*l + m*m - (l*l - m*m)**2)*Vddp + (n*n + (l*l - m*m)**2/4)*Vddd
                elif final_orbital == "d3z2-r2":
                    hopping = math.sqrt(3)*((l*l - m*m)*(n*n - (l*l + m*m)/2))*Vdds/2 + n*n*(m*m - l*l)*Vddp + ((1 + n*n) *
                              (l*l - m*m)/4)*Vddd
            elif initial_orbital == "d3z2-r2":
                hopping = (n*n - (l*l - m*m)/2)**2*Vdds + 3*n*n*(l*l + m*m)*Vddp + 3./4*(l*l + m*m)**2*Vddd

        return hopping

    def __transform_orbitals_to_string(self):
        """ Method to transform the orbitals list from logical form back to string form, to be used in the
         hamiltonian routine """

        possible_orbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'd3z2-r2']
        orbitals_string = []
        orbitals = self.configuration['Orbitals']
        for i, orbital in enumerate(orbitals):
            if orbital:
                orbitals_string.append(possible_orbitals[i])

        self.norbitals = len(orbitals_string)
        return orbitals_string

    def __create_atomic_orbital_basis(self):
        """ Method to calculate the Cartesian product between the motif list and the orbitals list
         to get the standard atomic-orbital basis |i,\alpha>. The ordering is as written in the ket: first
         the atomic position, then the orbital. For fixed atom, we iterate over the possible orbitals """

        basis = []
        orbitals = self.__transform_orbitals_to_string()
        for element in itertools.product(self.motif, orbitals):
            basis.append(element)

        self._basisdim = len(basis)
        return basis

    def __extend_onsite_vector(self):
        """ Routine to extend the onsite energies from the original list to
         a list the same length as the orbitals one, with the corresponding onsite energy
         per index """

        onsite_energies = self.configuration['Onsite energy']
        orbitals = self.__transform_orbitals_to_string()

        onsite_list = []
        onsite_full_list = []
        reduced_orbitals = []
        onsite_orbital_dictionary = {}
        count = 0
        for element in range(self.configuration['Species']):
            for orbital in orbitals:
                if orbital[0] not in reduced_orbitals:
                    reduced_orbitals.append(orbital[0])
                    onsite_orbital_dictionary.update({orbital[0]: onsite_energies[element][count]})
                    count += 1
            for orbital in orbitals:
                onsite_list.append(onsite_orbital_dictionary[orbital[0]])
            onsite_full_list.append(onsite_list)

        return onsite_full_list

    def __determine_connected_unit_cells(self):
        """ Method to calculate which unit cells connect with the origin from the neighbour list """

        neighbours_list = self.neighbours
        unit_cell_list = [[0.0, 0.0, 0.0]]
        for neighbour_list in neighbours_list:
            for neighbour in neighbour_list:
                unit_cell = list(neighbour[1])
                if unit_cell not in unit_cell_list:
                    unit_cell_list.append(unit_cell)

        self.__unit_cell_list = unit_cell_list

    def __initialize_spin_orbit_coupling(self):
        """ Method to initialize the whole spin-orbit coupling matrix corresponding to the
         orbitals that participate in the tight-binding model"""

        dimension_block = int(len(self.configuration["Orbitals"]))
        # We hardcode the whole spin-orbit hamilonian up to d orbitals
        spin_orbit_hamiltonian = np.zeros([dimension_block*2, dimension_block*2], dtype=np.complex_)
        p_orbital_beginning = 1
        d_orbital_beginning = 4
        # p orbitals
        spin_orbit_hamiltonian[p_orbital_beginning, p_orbital_beginning + 1] = -1j
        spin_orbit_hamiltonian[p_orbital_beginning, dimension_block + p_orbital_beginning + 2] = 1
        spin_orbit_hamiltonian[p_orbital_beginning + 1, dimension_block + p_orbital_beginning + 2] = -1j
        spin_orbit_hamiltonian[p_orbital_beginning + 2, dimension_block + p_orbital_beginning] = -1
        spin_orbit_hamiltonian[p_orbital_beginning + 2, dimension_block + p_orbital_beginning + 1] = 1j
        spin_orbit_hamiltonian[dimension_block + p_orbital_beginning, dimension_block + p_orbital_beginning + 1] = 1j

        # d orbitals
        spin_orbit_hamiltonian[d_orbital_beginning, d_orbital_beginning + 3] = 2j
        spin_orbit_hamiltonian[d_orbital_beginning, dimension_block + d_orbital_beginning + 1] = -1j
        spin_orbit_hamiltonian[d_orbital_beginning, dimension_block + d_orbital_beginning + 2] = 1
        spin_orbit_hamiltonian[d_orbital_beginning + 1, d_orbital_beginning + 2] = -1j
        spin_orbit_hamiltonian[d_orbital_beginning + 1, dimension_block + d_orbital_beginning] = 1j
        spin_orbit_hamiltonian[d_orbital_beginning + 1, dimension_block + d_orbital_beginning + 3] = -1j
        spin_orbit_hamiltonian[d_orbital_beginning + 1, dimension_block + d_orbital_beginning + 4] = math.sqrt(3)
        spin_orbit_hamiltonian[d_orbital_beginning + 2, dimension_block + d_orbital_beginning] = -1
        spin_orbit_hamiltonian[d_orbital_beginning + 2, dimension_block + d_orbital_beginning + 3] = 1j
        spin_orbit_hamiltonian[d_orbital_beginning + 2, dimension_block + d_orbital_beginning + 4] = -1j*math.sqrt(3)
        spin_orbit_hamiltonian[d_orbital_beginning + 3, dimension_block + d_orbital_beginning + 1] = 1
        spin_orbit_hamiltonian[d_orbital_beginning + 3, dimension_block + d_orbital_beginning + 2] = 1j
        spin_orbit_hamiltonian[d_orbital_beginning + 4, dimension_block + d_orbital_beginning + 1] = -math.sqrt(3)
        spin_orbit_hamiltonian[d_orbital_beginning + 4, dimension_block + d_orbital_beginning + 2] = 1j*math.sqrt(3)
        spin_orbit_hamiltonian[dimension_block + d_orbital_beginning, dimension_block + d_orbital_beginning + 3] = -2j
        spin_orbit_hamiltonian[dimension_block + d_orbital_beginning + 1,
                               dimension_block + d_orbital_beginning + 2] = 1j

        spin_orbit_hamiltonian += np.conj(spin_orbit_hamiltonian.T)

        self.spin_orbit_hamiltonian = self.configuration['Spin-orbit coupling']*spin_orbit_hamiltonian

    def __spin_orbit_h(self):
        """ Method to obtain the actual spin-orbit hamiltonian that corresponds to the orbitals
         that participate in the model. Generate spin blocks to append later to the global hamiltonian.
          Ordering is: self.spin_blocks=[up up, up down, down up, down down] """

        orbitals = self.configuration["Orbitals"]
        npossible_orbitals = len(orbitals)
        orbitals_indices = np.array([index for index, orbital in enumerate(orbitals) if orbital])

        self.spin_blocks = []

        for indices in itertools.product([0, 1], [0, 1]):
            self.spin_blocks.append(self.spin_orbit_hamiltonian[
                                        np.ix_(orbitals_indices + indices[0]*npossible_orbitals,
                                               orbitals_indices + indices[1]*npossible_orbitals)])

        if self.ordering == "atomic":
            spin_block_size = len(orbitals_indices)
            spin_orbit_hamiltonian = np.zeros([spin_block_size*2,
                                               spin_block_size*2], dtype=np.complex_)
            print(self.natoms)
            print(spin_block_size)
            for n, indices in enumerate(itertools.product([0, 1], [0, 1])):
                spin_orbit_hamiltonian[indices[0] * spin_block_size:
                                            (indices[0] + 1)*spin_block_size,
                                            indices[1] * spin_block_size:
                                            (indices[1] + 1) * spin_block_size] = self.spin_blocks[n]
                self.spin_orbit_hamiltonian = np.kron(np.eye(self.natoms, self.natoms), spin_orbit_hamiltonian)

        else:
            for n, spin_block in enumerate(self.spin_blocks):
                self.spin_blocks[n] = np.kron(np.eye(self.natoms, self.natoms), spin_block)

    def initialize_hamiltonian(self):
        """ Routine to initialize the hamiltonian matrices which describe the system. """

        orbitals = self.__transform_orbitals_to_string()
        basis = self.__create_atomic_orbital_basis()

        print('Computing first neighbours...\n')
        self.first_neighbours()
        self.__determine_connected_unit_cells()

        hamiltonian = []
        for _ in self.__unit_cell_list:
            hamiltonian.append(np.zeros(([self._basisdim, self._basisdim]), dtype=np.complex_))

        onsite_energies = self.__extend_onsite_vector()

        for n, atom in enumerate(self.motif):
            species = int(atom[3])  # To match list beginning on zero

            hamiltonian_atom_block = np.diag(np.array(onsite_energies[species]))
            hamiltonian[0][self.norbitals*n:self.norbitals*(n+1),
                           self.norbitals*n:self.norbitals*(n+1)] = hamiltonian_atom_block

        for i, atom in enumerate(basis):
            atom_position = atom[0][:3]
            species = int(atom[0][3])
            orbital = atom[1]
            atom_index = int(i/self.norbitals)
            for neighbour in self.neighbours[atom_index]:
                neigh_index = neighbour[0]
                neigh_unit_cell = list(neighbour[1])
                neigh_position = self.motif[neigh_index][:3]
                neigh_species = int(self.motif[neigh_index][3])
                for j, neigh_orbital in enumerate(orbitals):
                    position_difference = -np.array(atom_position) + np.array(neigh_position) + np.array(neigh_unit_cell)
                    orbital_config = [orbital, species, neigh_orbital, neigh_species]
                    h_cell = self.__unit_cell_list.index(neigh_unit_cell)
                    hamiltonian[h_cell][i, neigh_index*self.norbitals + j] += self.__hopping_amplitude(
                                                                                                position_difference,
                                                                                                orbital_config)
        # Check spinless or spinful model and initialize spin-orbit coupling
        if self.configuration['Spin']:
            self.norbitals = self.norbitals * 2
            self._basisdim = self._basisdim * 2

            for index, cell in enumerate(self.__unit_cell_list):
                if self.ordering == "atomic":
                    aux_hamiltonian = np.zeros([self._basisdim, self._basisdim], dtype=np.complex_)
                    for n in range(self.natoms):
                        for m in range(self.natoms):
                            atom_block = np.kron(np.eye(2, 2),
                                                 hamiltonian[index][n * self.norbitals//2:(n + 1) * self.norbitals//2,
                                                                    m * self.norbitals//2:(m + 1) * self.norbitals//2])
                            aux_hamiltonian[n*self.norbitals:(n+1)*self.norbitals,
                                            m*self.norbitals:(m+1)*self.norbitals] += atom_block
                    hamiltonian[index] = aux_hamiltonian
                else:
                    hamiltonian[index] = np.kron(np.eye(2, 2), hamiltonian[index])

            if self.configuration['Spin-orbit coupling'] != 0:
                self.__initialize_spin_orbit_coupling()
                self.__spin_orbit_h()

                if self.ordering == "atomic":
                    hamiltonian[0] += self.spin_orbit_hamiltonian
                else:
                    for block, indices in enumerate(itertools.product([0, 1], [0, 1])):
                        hamiltonian[0][indices[0] * self._basisdim//2:(indices[0] + 1) * self._basisdim//2,
                                       indices[1] * self._basisdim//2:(indices[1] + 1) * self._basisdim//2] += self.spin_blocks[block]

                self.__zeeman_term(0.0)
                hamiltonian[0] += self.__zeeman

        self.hamiltonian = hamiltonian

    def __zeeman_term(self, intensity):
        """ Routine to incorporate a Zeeman term to the Hamiltonian """
        zeeman_h = np.kron(np.array([[1, 0], [0, -1]]), np.eye(self._basisdim//2)*intensity)

        self.__zeeman = zeeman_h

    def hamiltonian_k(self, k):
        """ Add the k dependency of the Bloch Hamiltonian through the complex exponential
         and adds the spin-orbit term in case it is present """

        hamiltonian_k = np.zeros([self._basisdim, self._basisdim], dtype=np.complex_)
        for cell_index, cell in enumerate(self.__unit_cell_list):
            hamiltonian_k += self.hamiltonian[cell_index] * cmath.exp(1j*np.dot(k, cell))

        return hamiltonian_k

    # ---------------------------- IO routines ----------------------------
    def export_model(self, filename):
        """ Routine to write the Hamiltonian matrices calculated to a file """
        with open(filename, "w") as file:
            # Write ndim, natoms, norbitals, ncells and bravais lattice basis vectors
            file.write(str(self.ndim) + '\t' + str(self.natoms) + '\t'
                       + str(self.norbitals) + '\t'
                       + str(len(self.__unit_cell_list)) + '\n')
            np.savetxt(file, self.bravais_lattice)

            # Write motif atoms
            for atom in self.motif:
                np.savetxt(file, [atom[0:3]])

            # Write Bloch Hamiltonian matrices containing hopping to different unit cells
            for i, matrix in enumerate(self.hamiltonian):
                np.savetxt(file, [self.__unit_cell_list[i]])
                np.savetxt(file, matrix, fmt='%.10f')
                file.write("#\n")

    @classmethod
    def import_model(cls, filename):
        """ Routine to read the Hamiltonian matrix from a file """
        it = 0
        with open(filename, "r") as file:
            line = file.readline().split()
            ndim, natoms, norbitals, ncells = [int(num) for num in line]
            basisdim = norbitals * natoms
            bravais_lattice = []
            for i in range(ndim):
                line = file.readline().split()
                if len(line) != 3:
                    print("Unexpected line found, exiting...")
                    sys.exit(1)
                bravais_lattice.append([float(num) for num in line])

            motif = []
            for i in range(natoms):
                line = file.readline().split()
                if len(line) != 3:
                    print("Unexpected line found, exiting...")
                    sys.exit(1)
                motif.append([float(num) for num in line])

            unit_cell_list = []
            hamiltonian = []
            hamiltonian_matrix = np.zeros([basisdim, basisdim], dtype=np.complex_)
            for line in file.readlines():
                line = line.split()
                if len(line) == 3:
                    unit_cell_list.append([float(num) for num in line])
                elif line[0] == "#":
                    it = 0
                    hamiltonian.append(hamiltonian_matrix)
                    hamiltonian_matrix = np.zeros([basisdim, basisdim], dtype=np.complex_)
                else:
                    hamiltonian_matrix[it, :] = [complex(num) for num in line]
                    it += 1

        if ncells != len(unit_cell_list):
            print("Mismatch between number of Bloch matrices provided and declared, exiting...")
            sys.exit(1)

        model = cls()
        model.system_name = filename
        model.ndim = ndim
        model.norbitals = norbitals
        model.natoms = natoms
        model._basisdim = basisdim
        model.__unit_cell_list = unit_cell_list
        model.bravais_lattice = bravais_lattice
        model.hamiltonian = hamiltonian

        return model


if __name__ == '__main__':
    pass
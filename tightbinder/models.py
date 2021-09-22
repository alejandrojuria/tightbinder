# Module with all the models declarations, from the Slater-Koster tight-binding model
# to toy models such as the BHZ model or Wilson fermions.

from .system import System, FrozenClass
from .crystal import Crystal
import numpy as np
import sys
import math
import cmath
import itertools

# Module-level variables
sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=np.complex_)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=np.complex_)
sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=np.complex_)

# --------------- Constants ---------------
PI = 3.14159265359


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
        self._unit_cell_list = None
        if mode not in ['minimal', 'radius']:
            print('Error: Incorrect mode')
            sys.exit(1)
        self.__mode = mode
        self.__r = r

        self.boundary = boundary
        self.__spin_blocks = None
        self.__zeeman = None
        self.ordering = None

    # --------------- Methods ---------------
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
        onsite_full_list = []

        for element in range(self.configuration['Species']):
            onsite_list = []
            reduced_orbitals = []
            onsite_orbital_dictionary = {}
            count = 0
            for orbital in orbitals:
                if orbital[0] not in reduced_orbitals:
                    reduced_orbitals.append(orbital[0])
                    onsite_orbital_dictionary.update({orbital[0]: onsite_energies[element][count]})
                    count += 1
            for orbital in orbitals:
                onsite_list.append(onsite_orbital_dictionary[orbital[0]])
            onsite_full_list.append(onsite_list)

        return onsite_full_list

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
        self.find_neighbours()
        self._determine_connected_unit_cells()

        hamiltonian = []
        for _ in self._unit_cell_list:
            hamiltonian.append(np.zeros(([self._basisdim, self._basisdim]), dtype=np.complex_))

        onsite_energies = self.__extend_onsite_vector()

        for n, atom in enumerate(self.motif):
            species = int(atom[3])  # To match list beginning on zero

            hamiltonian_atom_block = np.diag(np.array(onsite_energies[species])/2)
            hamiltonian[0][self.norbitals*n:self.norbitals*(n+1),
                           self.norbitals*n:self.norbitals*(n+1)] = hamiltonian_atom_block

        for bond in self.bonds:
            initial_atom_index, final_atom_index, cell = bond
            initial_atom = self.motif[initial_atom_index][:3]
            initial_atom_species = self.motif[initial_atom_index][3]
            final_atom = self.motif[final_atom_index][:3]
            final_atom_species = self.motif[final_atom_index][3]
            for i, initial_orbital in enumerate(orbitals):
                for j, final_orbital in enumerate(orbitals):
                    position_difference = np.array(final_atom) + np.array(cell) - np.array(initial_atom)
                    orbital_config = [initial_orbital, initial_atom_species, final_orbital, final_atom_species]
                    h_cell = self._unit_cell_list.index(list(cell))
                    hopping_amplitude = self.__hopping_amplitude(position_difference, orbital_config)
                    hamiltonian[h_cell][initial_atom_index * self.norbitals + i,
                                        final_atom_index * self.norbitals + j] += hopping_amplitude

        # Check spinless or spinful model and initialize spin-orbit coupling
        if self.configuration['Spin']:
            self.norbitals = self.norbitals * 2
            self._basisdim = self._basisdim * 2

            for index, cell in enumerate(self._unit_cell_list):
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

            # Add Zeeman term
            self.__zeeman_term(1E-7)
            hamiltonian[0] += self.__zeeman

            if self.configuration['Spin-orbit coupling'] != 0:
                self.__initialize_spin_orbit_coupling()
                self.__spin_orbit_h()

                if self.ordering == "atomic":
                    hamiltonian[0] += self.spin_orbit_hamiltonian
                else:
                    for block, indices in enumerate(itertools.product([0, 1], [0, 1])):
                        hamiltonian[0][indices[0] * self._basisdim//2:(indices[0] + 1) * self._basisdim//2,
                                       indices[1] * self._basisdim//2:(indices[1] + 1) * self._basisdim//2] += self.spin_blocks[block]

        self.hamiltonian = hamiltonian

    def __zeeman_term(self, intensity):
        """ Routine to incorporate a Zeeman term to the Hamiltonian """
        if self.ordering == "atomic":
            zeeman = np.kron(np.array([[1, 0], [0, -1]]), np.eye(self.norbitals//2))
            zeeman_h = np.kron(np.eye(self.natoms), zeeman) * intensity
        else:
            zeeman_h = np.kron(np.array([[1, 0], [0, -1]]), np.eye(self._basisdim // 2) * intensity)

        self.__zeeman = zeeman_h

    def hamiltonian_k(self, k):
        """ Add the k dependency of the Bloch Hamiltonian through the complex exponential
         and adds the spin-orbit term in case it is present """

        hamiltonian_k = np.zeros([self._basisdim, self._basisdim], dtype=np.complex_)
        for cell_index, cell in enumerate(self._unit_cell_list):
            hamiltonian_k += self.hamiltonian[cell_index] * cmath.exp(1j*np.dot(k, cell))

        hamiltonian_k = (hamiltonian_k + np.transpose(np.conjugate(hamiltonian_k)))
        return hamiltonian_k

    # ---------------------------- IO routines ----------------------------
    def export_model(self, filename):
        """ Routine to write the Hamiltonian matrices calculated to a file """
        with open(filename, "w") as file:
            # Write ndim, natoms, norbitals, ncells and bravais lattice basis vectors
            file.write(str(self.ndim) + '\t' + str(self.natoms) + '\t'
                       + str(self.norbitals) + '\t'
                       + str(len(self._unit_cell_list)) + '\n')
            np.savetxt(file, self.bravais_lattice)

            # Write motif atoms
            for atom in self.motif:
                np.savetxt(file, [atom[0:3]])

            # Write Bloch Hamiltonian matrices containing hopping to different unit cells
            for i, matrix in enumerate(self.hamiltonian):
                np.savetxt(file, [self._unit_cell_list[i]])
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
        model._unit_cell_list = unit_cell_list
        model.bravais_lattice = bravais_lattice
        model.hamiltonian = hamiltonian

        return model


class BHZ(System, FrozenClass):
    """
    Implementation of generalized BHZ model. The model is based on a
    2D square lattice of side a=1, and is a four band model. The model takes three parameters:
    g = mixing term
    u = sublattice potential
    c = coupling operator "amplitude"
    """

    def __init__(self, g, u, c):
        super().__init__(system_name="BHZ model",
                         crystal=Crystal([[1, 0, 0], [0, 1, 0]],
                                         motif=[[0, 0, 0, 0]]))
        self.norbitals = 4
        self.filling = 0.5
        self.basisdim = self.norbitals
        self.g = g
        self.u = u
        self.c = c

        self._freeze()

    def hamiltonian_k(self, k):
        global sigma_x, sigma_y, sigma_z
        coupling = self.c * sigma_y

        id2 = np.eye(2)
        kx, ky = k[0], k[1]

        hamiltonian = (np.kron(id2, (self.u + np.cos(kx) + np.cos(ky)) * sigma_z + np.sin(ky) * sigma_y) +
                       np.kron(sigma_z, np.sin(kx) * sigma_x) + np.kron(sigma_x, coupling) +
                       self.g * np.kron(sigma_z, sigma_y) * (np.cos(kx) + np.cos(7 * ky) - 2))

        return hamiltonian


class WilsonFermions2D(System, FrozenClass):
    """
    Implementation of Wilson-fermions model. This model takes a 2D square lattice of side a, it is
    a four band model.
    The hamiltonian takes the following two parameters:
    t = hopping amplitude
    m = mass term
    All parameters default to 1 if not specified otherwise
    """

    def __init__(self, side=1, t=1, m=1):
        super().__init__(system_name="Wilson-fermions 2D",
                         crystal=Crystal([[side, 0, 0], [0, side, 0]],
                                         motif=[[0, 0, 0, 0]]))
        self.system_name = "Wilson-fermions model"
        self.norbitals = 4
        self._basisdim = self.norbitals * 1
        self.filling = 0.5

        self.a = side
        self.t = t
        self.m = m

        self._freeze()

    def hamiltonian_k(self, k):
        global sigma_x, sigma_y, sigma_z

        alpha_x = np.kron(sigma_x, sigma_x)
        alpha_y = np.kron(sigma_x, sigma_y)
        beta = np.kron(sigma_z, np.eye(2, dtype=np.complex_))

        hamiltonian = self.t * (np.sin(k[0] * self.a) * alpha_x + np.sin(k[1] * self.a) * alpha_y) + (
                np.cos(k[0] * self.a) + np.cos(k[1] * self.a) + self.m - 3) * beta

        return hamiltonian


class WilsonFermions3D(System, FrozenClass):
    """
    Implementation of Wilson-fermions model. This model takes a 3D square lattice of side a, it is
    a four band model.
    The hamiltonian takes the following two parameters:
    t = hopping amplitude
    m = mass term
    All parameters default to 1 if not specified otherwise
    """

    def __init__(self, side=1, t=1, m=1):
        super().__init__(system_name="Wilson-fermions 3D",
                         crystal=Crystal([[side, 0, 0], [0, side, 0], [0, 0, side]],
                                         motif=[[0, 0, 0, 0]]))
        self.num_orbitals = 4
        self._basisdim = self.num_orbitals * self.natoms

        self.a = side
        self.t = t
        self.m = m

        self._freeze()

    def hamiltonian_k(self, k):
        global sigma_x, sigma_y, sigma_z

        alpha_x = np.kron(sigma_x, sigma_x)
        alpha_y = np.kron(sigma_x, sigma_y)
        alpha_z = np.kron(sigma_x, sigma_z)
        beta = np.kron(sigma_z, np.eye(2, dtype=np.complex_))

        hamiltonian = (self.t * (np.sin(k[0] * self.a) * alpha_x + np.sin(k[1] * self.a) * alpha_y +
                                 np.sin(k[2] * self.a) * alpha_z) +
                       (np.cos(k[0] * self.a) + np.cos(k[1] * self.a) + np.cos(k[2] * self.a) +
                        self.m - 3) * beta)

        return hamiltonian


class WilsonAmorphous(System):
    """
    Implementation of 3D Wilson-fermion model in real-space. This models is a generalization of the
    standard Wilson-fermion model, which uses a cubic unit cell. Instead, this allows for any spatial
    distribution of the atoms, and since the Hamiltonian is built in real-space any number of atoms
    can be considered.
    Parameters:
    side: first-neighbour distance in the crystalline system
    t: Hopping parameter
    m: Mass term
    r: cutoff distance for neighbour interaction
    All parameters default to 1 if not specified otherwise
    """

    def __init__(self, side=1, t=1, m=1, r=1.1):
        super().__init__(system_name="Wilson Amorphous model",
                         crystal=Crystal([[side, 0, 0], [0, side, 0], [0, 0, side]],
                                         motif=[[0, 0, 0, 0]]))

        self.filling = 0.5
        self.norbitals = 4
        self._basisdim = self.norbitals * len(self.motif)
        self.boundary = "PBC"

        self.a = side
        self.t = t
        self.m = m
        self.r = r
        self.parameters = {"C": self.t, "a": self.a, "M": self.m}

    @staticmethod
    def _hopping_matrix(initial_position, final_position, parameters):
        """ Computes hopping matrix according to Wilson-fermion model for two
          given atomic positions.
          :param initial_position: Array 1x3
          :param final_position: Array 1x3
          :param parameters: Dictionary{C, a, M}"""

        x, y, z = final_position - initial_position
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi = math.atan2(y, x)
        theta = math.acos(z / r)

        hopping = np.zeros([4, 4], dtype=np.complex_)
        np.fill_diagonal(hopping, [1, 1, -1, -1])
        offdiag_block = np.array([[-1j * math.cos(theta), -1j * cmath.exp(-1j * phi) * math.sin(theta)],
                                  [-1j * cmath.exp(1j * phi) * math.sin(theta), 1j * math.cos(theta)]],
                                 dtype=np.complex_)
        hopping[0:2, 2:4] = offdiag_block
        hopping[2:4, 0:2] = offdiag_block
        hopping *= 0.5
        hopping *= parameters["C"] * math.exp(1 - r / parameters["a"])

        return hopping

    def initialize_hamiltonian(self, find_bonds=True):
        """ Routine to initialize the matrices that compose the Bloch Hamiltonian """
        self._basisdim = self.natoms * self.norbitals
        if find_bonds:
            print("Computing neighbours...")
            self.find_neighbours(mode="radius", r=self.r)
        self._determine_connected_unit_cells()

        hamiltonian = []
        for _ in self._unit_cell_list:
            hamiltonian.append(np.zeros([self._basisdim, self._basisdim], dtype=np.complex_))

        hamiltonian_atom_block = np.diag(np.array([-3 + self.m, -3 + self.m,
                                                   3 - self.m, 3 - self.m])*0.5)
        for n, atom in enumerate(self.motif):
            hamiltonian[0][self.norbitals * n:self.norbitals * (n + 1),
                           self.norbitals * n:self.norbitals * (n + 1)] = hamiltonian_atom_block

        for bond in self.bonds:
            initial_atom_index, final_atom_index, cell = bond
            initial_atom = self.motif[initial_atom_index][:3]
            final_atom = self.motif[final_atom_index][:3]
            neigh_position = np.array(final_atom) + np.array(cell)
            h_cell = self._unit_cell_list.index(list(cell))
            hamiltonian[h_cell][4 * initial_atom_index:4 * (initial_atom_index + 1),
                                4 * final_atom_index:4 * (final_atom_index + 1)] = \
                self._hopping_matrix(initial_atom, neigh_position, self.parameters)

        self.hamiltonian = hamiltonian

    def hamiltonian_k(self, k):
        """
        Routine to evaluate the Bloch Hamiltonian at a given k point. It adds the k dependency of the Bloch Hamiltonian
        through the complex exponentials.

        :param k: k vector (Array 1x3)
        :param conditions: defaults to PBC. Can be either PBC or OBC
        :return: Bloch Hamiltonian matrix
        """

        dimension = len(self.hamiltonian[0])
        hamiltonian_k = np.zeros((dimension, dimension), dtype=np.complex_)
        for cell_index, cell in enumerate(self._unit_cell_list):
            hamiltonian_k += (self.hamiltonian[cell_index] * cmath.exp(1j * np.dot(k, cell)))

        hamiltonian_k = (hamiltonian_k + np.transpose(np.conjugate(hamiltonian_k)))
        return hamiltonian_k


class RSmodel(System):
    """ Class to construct toy models, in which one sets the hoppings manually. This models
     are by default OBC; by setting one hopping between different unit cells it automatically becomes
     PBC. """
    def __init__(self, system_name=None, bravais_lattice=None, motif=None):
        super().__init__(system_name=system_name, bravais_lattice=bravais_lattice, motif=motif)
        self.hoppings = None
        self.boundary = "OBC"

    def add_hopping(self, hopping, initial, final, cell=(0., 0., 0.)):
        """ Method to add a hopping between two atoms of the motif.
        NB: The hopping has a specified direction, from initial to final. Since the final
        Hamiltonian is computed taking into account hermiticity, it is not necessary to specify the hopping
        in the other direction.
         Parameters:
             complex hopping
             int initial, final: Indices of the atoms in the motif
             array cell: Bravais vector connecting the cells of the two atoms. Defaults to zero """

        assert type(initial) == int, "initial must be an integer"
        assert type(final)   == int, "initial must be an integer"
        assert type(hopping) != str, "hopping must be a complex number"

        self.add_bond(initial, final, cell)
        bond_index = len(self.bonds) - 1
        self.hoppings.append([bond_index, hopping])

    def add_hoppings(self, hoppings, initial, final, cells=None):
        """ Same method as add_hopping but we input a list of hoppings at once.
        Parameters:
             list hoppings: list of size nhop
             list initial, final: list of indices
             matrix cells: Each row denotes the Bravais vector connecting two cells. Defaults to None """
        if len(hoppings) != len(initial) or len(initial) != len(final):
            raise ValueError("Provided list must have the same length")

        if cells is None:
            cells = np.zeros([len(hoppings), 3])
        else:
            self.boundary = "PBC"
            cells = np.array(cells)
        for n, hopping in enumerate(hoppings):
            self.add_hopping(hopping, initial[n], final[n], cells[n, :])

    def initialize_hamiltonian(self):
        """ Method to set up the matrices that compose the Hamiltonian, either the Bloch Hamiltonian
        or the real-space one """
        unit_cell_list = [[0., 0., 0.]]
        hamiltonian = []
        for _ in range(self.unit_cell_list):
            hamiltonian.append(np.zeros((self.natoms, self.natoms), dtype=np.complex))
        for hopping in self.hoppings:
            bond_index, amplitude = hopping
            bond = self.bonds[bond_index]
            initial_atom_index, final_atom_index, cell = bond


def bethe_lattice(z=3, depth=3, length=1):
    """ Routine to generate a Bethe lattice
     :param z: Coordination number. Defaults to 3
     :param depth: Number of shells of the lattice (central atom is depth 0)
     :param length: Float to specify length of bonds
     :return: Motif: list of all atoms' positions """

    motif_previous_shell = [[0., 0., 0., 0]]
    motif = [[0., 0., 0., 0]]
    cell = [0., 0., 0.]
    bonds = []
    previous_angles = [0]
    bond_length = length
    atom_index = 1
    for i in range(1, depth + 1):
        natoms_in_shell = z*(z - 1)**(i - 1)
        angles = np.linspace(0, 2*np.pi, natoms_in_shell + 1)
        angle_between = angles[1] - angles[0]
        angles += angle_between/2 + previous_angles[0]
        for angle in angles[:-1]:
            atom = bond_length*np.array([np.cos(angle), np.sin(angle), 0, 0])
            motif.append(atom)
            distance = np.linalg.norm(atom[:3] - np.array(motif_previous_shell)[:, :3], axis=1)
            neighbour = np.where(distance == np.min(distance))[0][0]
            bonds.append([atom_index, neighbour, cell])
            atom_index += 1

        motif_previous_shell = np.copy(motif)
        bond_length += length
        previous_angles = angles

    return motif, bonds







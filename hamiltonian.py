# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

import numpy as np
import sys
import math
import itertools

# --------------- Constants ---------------
PI = 3.14159265359
EPS = 1E-16

# --------------- Routines ---------------


def first_neighbours(motif, bravais_lattice, mode="minimal", r=None, boundary="PBC"):
    """ Given a list of atoms (motif), it returns a list in which each
    index corresponds to a list of atoms that are first neighbours to that index
    on the initial atom list.
    I.e.: Atom list -> List of neighbours/atom.
    By default it will look for the minimal distance between atoms to determine first neighbours.
    For amorphous systems the option radius is available to determine neighbours within a given radius R.
    Boundary conditions can also be set, either PBC (default) or OBC."""

    dimension = len(bravais_lattice)
    bravais_lattice = np.array(bravais_lattice)

    # Prepare unit cells to loop over depending on boundary conditions
    if boundary == "PBC":
        mesh_points = []
        for i in range(dimension):
            mesh_points.append(list(range(-1, 2)))
        mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, dimension)

        near_cells = np.zeros([len(mesh_points), 3])
        for n, coefficients in enumerate(mesh_points):
            cell_vector = np.array([0.0, 0.0, 0.0])
            for i, coefficient in enumerate(coefficients):
                cell_vector += (coefficient * bravais_lattice[i])
            near_cells[n, :] = cell_vector

    elif boundary == "OBC":
        near_cells = np.array([0.0, 0.0, 0.0])
    else:
        print('Incorrect boundary argument, exiting...')
        sys.exit(1)

    # Determine neighbour distance from one fixed atom
    neigh_distance = 1E100
    fixed_atom = motif[0][0]
    for cell in near_cells:
        for atom in motif:
            distance = np.linalg.norm(atom[0] + cell - fixed_atom)
            if distance < neigh_distance and distance != 0: neigh_distance = distance

    # Determine list of neighbours for each atom of the motif
    if mode == "minimal":
        neighbours_list = []
        for n, reference_atom in enumerate(motif):
            neighbours = []
            for cell in near_cells:
                for i, atom in enumerate(motif):
                    distance = np.linalg.norm(atom[0] + cell - reference_atom[0])
                    if abs(distance - neigh_distance) < EPS: neighbours.append([i, cell])
            neighbours_list.append(neighbours)

    elif mode == "radius":
        if r is None:
            print('Radius not defined in "radius" mode, exiting...')
            sys.exit(1)
        elif r < neigh_distance:
            print("Warning: Radius smaller than first neighbour distance")

        neighbours_list = []
        for n, reference_atom in enumerate(motif):
            neighbours = []
            for cell in near_cells:
                for i, atom in enumerate(motif):
                    distance = np.linalg.norm(atom[0] + cell - reference_atom[0])
                    if distance <= r: neighbours.append([i, cell])
            neighbours.append(neighbours)
    else:
        print('Incorrect mode option. Exiting... ')
        sys.exit(1)

    return neighbours_list


def hopping_amplitude(position_diff, amplitudes, *orbitals):
    """ Routine to compute the hopping amplitude from one atom to another depending on the
    participating orbitals, following the Slater-Koster approxiamtion """

    initial_orbital = orbitals[0]
    initial_species = orbitals[1]
    final_orbital = orbitals[2]
    final_species = orbitals[3]

    initial_orbital_type = initial_orbital[0]
    final_orbital_type = final_orbital[0]

    possible_orbitals = {'s': 0, 'p': 1, 'd': 2}
    if possible_orbitals[initial_orbital_type] > possible_orbitals[final_orbital_type]:
        position_diff = np.array(position_diff)
        hopping_amplitude(-position_diff, amplitudes, final_orbital, final_species, initial_orbital, initial_species)

    amplitudes = np.array(amplitudes)
    # Mixing of amplitudes on case of having different species
    # So far the mixing if equivalent (beta = 0.5)
    beta = 0.5
    if initial_orbital != final_species:
        effective_amplitudes = (amplitudes[initial_species] + amplitudes[final_species]) * beta
    else:
        effective_amplitudes = amplitudes[initial_species]
    direction_cosines = position_diff / np.linalg.norm(position_diff)
    direction_cosines = {'x': direction_cosines[0], 'y': direction_cosines[1], 'z': direction_cosines[2]}
    [l, m, n] = direction_cosines.values()
    (Vsss, Vsps, Vpps, Vppp, Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd) = amplitudes
    special_orbital = True if final_orbital == 'dx2-y2' or final_orbital == 'd3z2-r2' else False

    # Start checking the different possibilities
    if initial_orbital == "s":
        if final_orbital == "s":
            hopping = Vsss
        elif final_orbital_type == "p":
            coordinate_initial = final_orbital[1]
            hopping = direction_cosines[coordinate_initial] * Vsps
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


def transform_orbitals_to_string(orbitals):
    """ Method to transform the orbitals list from logical form back to string form, to be used in the
     hamiltonian routine """

    possible_orbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'd3z2-r2']
    orbitals_string = []
    for i, orbital in enumerate(orbitals):
        if orbital:
            orbitals_string.append(possible_orbitals[i])

    return orbitals_string


def create_atomic_orbital_basis(motif, orbitals):
    """ Method to calculate the Cartesian product between the motif list and the orbitals list
     to get the standard atomic-orbital basis |i,\alpha>. The ordering is as written in the ket: first
     the atomic position, then the orbital. For fixed atom, we iterate over the possible orbitals """

    basis = []
    for element in itertools.product(motif, orbitals):
        basis.append(element)

    return basis


def initialize_hamiltonian(k, configuration):
    """ Routine to initialize the hamiltonian matrix which describes the system. """

    orbitals = transform_orbitals_to_string(configuration['Orbitals'])
    basis = create_atomic_orbital_basis(configuration['Motif'], orbitals)

    dimension_block = len(orbitals)

    hamiltonian_block = np.zeros([dimension_block, dimension_block], dtype=np.complex_)
    #for atom in basis:





    return None


if __name__ == '__main__':

    hopping_amplitude([1,1,1], [1,1,1,1,1,1,1,1,1,1], 's', 1, 'dxy', 1)

# Module for topology-related routines. It incorporates routines to compute the Wilson loop spectra,
# as well as the topological invariants derived from it. Local markers such as the Bott index or the
# local Chern number are implemented as well.

import numpy as np
import matplotlib.pyplot as plt
import sys


def __wilson_loop(system, path):
    """ Routine to calculate the wilson loop matrix for a system along a given path. Returns a matrix
     of size Nocc x Nocc """

    result = system.solve(path)

    filling = int(system.filling * system.basisdim)
    wilson_loop = np.eye(filling)
    for i in range(len(path) - 1):
        overlap_matrix = np.dot(np.conj(np.transpose(result.eigen_states[i][:, :filling])),
                                result.eigen_states[i + 1][:, :filling])
        wilson_loop = np.dot(wilson_loop, overlap_matrix)

    overlap_matrix = np.dot(np.conj(np.transpose(result.eigen_states[i + 1][:, :filling])),
                            result.eigen_states[0][:, :filling])
    wilson_loop = np.dot(wilson_loop, overlap_matrix)

    return wilson_loop


def __check_orthogonality(matrix):
    epsilon = 1E-16
    difference = np.dot(np.conjugate(np.transpose(matrix)), matrix)
    difference = (np.conj(difference) * difference)
    difference[difference < epsilon] = 0
    if np.linalg.norm(difference) < epsilon:
        is_orthonormal = True
    else:
        is_orthonormal = False

    return is_orthonormal


def __generate_path(k_fixed, system, nk):
    """ Routine to generate a path in the Brilloin zone with one of the k components fixed """
    basis_vector = system.reciprocal_basis[0]
    cell_side = system.reciprocal_basis[0][0]
    path = []
    # k_list = np.linspace(-cell_side/2, cell_side/2, nk + 1)
    for i in range(nk + 1):
        # basis_vector = np.array([k_list[i], 0, 0])
        kpoint = basis_vector*i/nk + k_fixed
        # kpoint = basis_vector + k_fixed
        path.append(kpoint)

    return path


def __extract_wcc_from_wilson_loop(wilson_loop):
    """ Routine to calculate the Wannier charge centres from the Wilson loop matrix.
     NOTE: We truncate the eigenvalues up to a given precision to avoid numerical error due to floating point
     when computing the midpoints """
    eigval = np.linalg.eigvals(wilson_loop)
    eigval = (np.angle(eigval)/np.pi).round(decimals=5)

    eigval[eigval == -1] = 1  # Enforce interval (-1, 1], NOTE: it breaks partial polarization!
    eigval = np.sort(eigval)

    return eigval


def calculate_wannier_centre_flow(system, number_of_points, additional_k=None, nk_subpath=50):
    """ Routine to compute the evolution of the Wannier charge centres through Wilson loop calculation """

    print("Computing Wannier centre flow...")
    wcc_results = []
    # cell_side = crystal.high_symmetry_points
    # k_fixed_list = np.linspace(-cell_side/2, cell_side/2, number_of_points)
    for i in range(-number_of_points//2, number_of_points//2 + 1):
        k_fixed = system.reciprocal_basis[1]*i/number_of_points
        if additional_k is not None:
            k_fixed += additional_k
        # k_fixed = np.array([0, k_fixed_list[i], 0])
        path = __generate_path(k_fixed, system, nk_subpath)
        wilson_loop = __wilson_loop(system, path)
        wcc = __extract_wcc_from_wilson_loop(wilson_loop)

        wcc_results.append(wcc)

    return np.array(wcc_results)


# ---------------------------- Chern number ----------------------------
def calculate_polarization(wcc):
    """ Routine to compute the polarization from the WCC """
    polarization = np.sum(wcc)
    return polarization


def calculate_chern_number(wcc_flow):
    """ Routine to compute the first Chern number from the WCC flow """
    wcc_k0 = wcc_flow[0, :]
    wcc_k2pi = wcc_flow[-1, :]

    chern_number = calculate_polarization(wcc_k0) - calculate_polarization(wcc_k2pi)
    chern_number = 0 if chern_number < 1E-16 else chern_number
    return chern_number


# ---------------------------- Z2 invariant ----------------------------
def __find_wcc_midpoint_gap(wcc):
    """ Routine to find the midpoint of the biggest gap between WCCs """
    # wcc_extended = np.append(wcc, wcc[0] + 2)
    wcc_extended = np.append(wcc[-1] - 2, wcc)
    gaps = []
    for i in range(len(wcc_extended) - 1):
        gaps.append(wcc_extended[i+1] - wcc_extended[i])
    biggest_gap = np.max(gaps)
    position_gap = gaps.index(biggest_gap)

    midpoint_gap = biggest_gap/2 + wcc_extended[position_gap]
    return midpoint_gap, position_gap


def calculate_wcc_gap_midpoints(wcc_flow):
    """ Routine to compute the midpoint of the biggest gap for the WCC flow, and
     their relative positions to the WCC """
    midpoints, positions = [], []
    for wcc in wcc_flow:
        gap, position = __find_wcc_midpoint_gap(wcc)
        midpoints.append(gap)
        positions.append(position)

    return midpoints, positions


def __directed_triangle(phi1, phi2, phi3):
    """ Routine to determine sign of a directed triangle. Used to determine
     whether the midpoint jump over the WCC bands. """
    if abs(phi1 - phi2) < 1E-14:
        g = 1
    else:
        g = np.sin(phi2 - phi1) + np.sin(phi3 - phi2) + np.sin(phi1 - phi3)
    return g


def calculate_z2_invariant(wcc_flow):
    """ Routine to compute the z2 invariant from the number of jumps the
     max gap midpoint does across different WCC bands """
    nk = len(wcc_flow[:, 0])
    num_crosses = 0
    for i in range(nk//2, nk - 1):
        wcc = wcc_flow[i, :]
        next_wcc = wcc_flow[i + 1, :]
        midpoint, position = __find_wcc_midpoint_gap(wcc)
        next_midpoint, next_position = __find_wcc_midpoint_gap(next_wcc)
        sign = 1
        for position in next_wcc:
            triangle = __directed_triangle(np.pi*midpoint, np.pi*next_midpoint, np.pi*position)
            sign *= np.sign(triangle)
        if sign == 1:
            crosses = 0
        else:
            crosses = 1
        num_crosses += crosses

    invariant = num_crosses % 2
    return invariant


def plot_wannier_centre_flow(wcc_flow, show_midpoints=False, title=''):
    """ Routine to plot the WCC evolution """

    plt.figure()
    for wcc in wcc_flow.T:
        plt.plot(wcc, 'ob', fillstyle='none')

    if show_midpoints:
        midpoints, _ = calculate_wcc_gap_midpoints(wcc_flow)
        plt.plot(midpoints, 'Dr')

    plt.title('WCC flow ' + title)
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\hat{x}_n$')
    plt.xticks([0, wcc_flow.shape[0] - 1], [r'$-\pi$', r'$\pi$'])


def plot_polarization_flow(wcc_flow):
    """ Routine to plot the polarization flow used to calculate the Chern number """
    polarization_array = []
    for wcc in wcc_flow:
        polarization_array.append(calculate_polarization(wcc))

    plt.figure()
    plt.plot(polarization_array)


# ---------------------------- Entanglement entropy ----------------------------
def __truncate_eigenvectors(eigenvectors, sector, system):
    """ Routine to truncate the eigenvectors according to the filling and the
    atoms that are on a specific partition, accounting for orbitals and spin structure """
    filling = int(system.filling * system.norbitals * len(system.motif))
    basisdim = len(system.motif) * system.norbitals
    if system.ordering == "atomic" or system.configuration["Spin"] == "False":
        sector = sector * system.norbitals
        orbitals = np.copy(sector)
        for n in range(1, system.norbitals):
            orbitals = np.concatenate((orbitals, sector + n))
        orbitals = np.sort(orbitals)

    else:
        sector = sector * (system.norbitals//2)
        orbitals = np.copy(sector)
        orbitals = np.concatenate((orbitals, sector + basisdim//2))
        for n in range(1, system.norbitals//2):
            orbitals = np.concatenate((orbitals, sector + n))
            orbitals = np.concatenate((orbitals, sector + basisdim//2 + n))
        orbitals = np.sort(orbitals)

    truncated_eigenvectors = eigenvectors[orbitals]
    truncated_eigenvectors = truncated_eigenvectors[:, :filling]

    return truncated_eigenvectors


def __density_matrix(truncated_eigenvectors):
    """ Routine to calculate the one-particle reduced density matrix for the half-system. """
    density_matrix = np.dot(truncated_eigenvectors, np.conj(truncated_eigenvectors.T))

    return density_matrix


def __specify_partition(system, plane):
    """ Routine to specify a real-space partition of the atoms of the system based on a given
    plane.
    Input: plane Array [A,B,C,D] <-> Ax + By + Cz + D = 0
    Returns: indices of atoms on a side Array """
    sector = []
    plane_normal_vector = plane[:3]
    plane_coefficient = plane[3]

    for i, atom in enumerate(system.motif):
        atom_position = atom[:3]
        side = np.dot(atom_position, plane_normal_vector) - plane_coefficient
        if np.sign(side) == -1:
            sector.append(i)

    if not sector or len(sector) == len(system.motif):
        print("Error: Could not find atoms with specified plane, exiting...")
        sys.exit(1)

    return np.array(sector)


def entanglement_spectrum(system, plane, kpoints=None):
    """ Routine to obtain the eigenvalues from the correlation/density matrix, which is directly
     related with the entangled Hamiltonian. Should be computable for both PBC and OBC """
    if kpoints is None and system.boundary != "OBC":
        print("Error: kpoints argument must be given when using PBC. Exiting...")
        sys.exit(1)
    if system.boundary == "OBC":
        if kpoints is not None:
            print("Warning: kpoints argument given but system uses OBC")
            print("Defaulting kpoints to origin...")
        kpoints = [[0., 0., 0.]]

    nk = len(kpoints)
    sector = __specify_partition(system, plane)
    results = system.solve(kpoints)

    spectrum = np.zeros([len(sector) * system.norbitals, nk])
    for i in range(len(kpoints)):
        truncated_eigenvectors = __truncate_eigenvectors(results.eigen_states[i],
                                                         sector, system)
        density_matrix = __density_matrix(truncated_eigenvectors)
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)

        spectrum[:, i] = eigenvalues

    return spectrum


def entanglement_entropy(spectrum):
    """ Routine to compute the entanglement entropy from the entanglement spectrum """
    entropy = 0
    for eigval in spectrum.T:
        entropy -= np.dot(eigval, np.log(eigval)) + np.dot(1 - eigval, np.log(1 - eigval))
    entropy /= len(spectrum.T)

    return entropy


def write_entanglement_spectrum_to_file(spectrum, file, n=None, shift=0):
    """ Routine to write the entanglement spectrum to a text file. If n is given,
      only n eigenvalues are written to file.
      The algorithm is as follows: first we search for the eigenvalue that is closer to 0.5, and then
      we write the n/2 eigenvalues to its left and right. """

    print("Writing entanglement spectrum to file...")
    eigenvalues = np.sort(spectrum.reshape(-1,))
    if n > len(eigenvalues):
        print("Error: n is bigger than overall number of eigenvalues")
        sys.exit(1)
    if n is None:
        n = len(eigenvalues)

    for i, eigval in enumerate(eigenvalues):
        if eigval <= 0.5 and eigenvalues[i+1] > 0.5:
            central_index = i
            break

    with open(file, "w") as textfile:
        for i in range(central_index - n//2 + shift, central_index + n//2 + shift):
            textfile.write(f"{eigenvalues[i]}\n")


def plot_entanglement_spectrum(spectrum, system):
    """ Routine to plot the entanglement spectrum as a function of k.
     CAREFUL: This routine is not made to deal with any set of kpoints, rather it is intended
     to be used with a set of kpoints along a given line (1D). """
    if system.boundary == "OBC":
        plt.plot(spectrum, 'g+')
    else:
        for entanglement_band in spectrum:
            plt.plot(entanglement_band, 'g+')

    plt.title(f"Entanglement spectrum of {system.system_name}")
    plt.xlabel("kpoints")
    plt.ylabel("Eigenvalues")

    plt.show()














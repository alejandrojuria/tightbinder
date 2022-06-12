# Module for topology-related routines. It incorporates routines to compute the Wilson loop spectra,
# as well as the topological invariants derived from it. Local markers such as the Bott index or the
# local Chern number are implemented as well.

from multiprocessing.sharedctypes import Value
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
    eigval = (np.angle(eigval)/np.pi).round(decimals=3)

    # eigval[eigval == -1] = 1  # Enforce interval (-1, 1], NOTE: it breaks partial polarization!

    # Extra care to ensure that the eigenvalues are not doubled, either at -1 or 1
    if np.max(np.abs(eigval)) == 1:
        one_indices = np.where(eigval == 1)[0]
        if len(one_indices) == 2:
            eigval[one_indices[0]] = -1
        elif len(one_indices) == 0:
            minus_one_indices = np.where(eigval == -1)[0]
            eigval[minus_one_indices[0]] = 1

    eigval = np.sort(eigval)

    return eigval


def calculate_wannier_centre_flow(system, number_of_points, additional_k=None, nk_subpath=50):
    """ Routine to compute the evolution of the Wannier charge centres through Wilson loop calculation """

    print("Computing Wannier centre flow...")
    if system.filling is None:
        raise ValueError('Filling must be specified to compute WCC')
    wcc_results = []
    # cell_side = crystal.high_symmetry_points
    # k_fixed_list = np.linspace(-cell_side/2, cell_side/2, number_of_points)
    for i in range(-number_of_points, number_of_points + 1):
        k_fixed = system.reciprocal_basis[1]*i/(2*number_of_points)
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
    gaps = np.array(gaps).round(decimals=3)
    biggest_gap = np.max(gaps)
    position_gap = np.where(gaps == biggest_gap)[0][0]

    midpoint_gap = biggest_gap/2 + wcc_extended[position_gap]
    midpoint_gap = round(midpoint_gap, 3)

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


def count_crossings(wcc, midpoint, next_midpoint):
    """ Routine to compute the number of WCC that lie between one midpoint and the
     next one """
    higher_midpoint = max(midpoint, next_midpoint)
    lower_midpoint = min(midpoint, next_midpoint)
    crossings = 0
    for center in wcc:
        if lower_midpoint <= center < higher_midpoint:
            crossings += 1

    return crossings


def calculate_z2_invariant(wcc_flow):
    """ Routine to compute the z2 invariant from the number of jumps the
     max gap midpoint does across different WCC bands """
    nk = len(wcc_flow[:, 0])
    num_crosses = 0
    for i in range(int(nk/2), nk - 1):
        wcc = np.copy(wcc_flow[i, :])
        next_wcc = wcc_flow[i + 1, :]
        midpoint, _ = __find_wcc_midpoint_gap(wcc)
        next_midpoint, _ = __find_wcc_midpoint_gap(next_wcc)
        num_crosses += count_crossings(next_wcc, midpoint, next_midpoint)

        # Do same trick as in find midpoints to ensure all crosses are correctly found
        # next_wcc = np.append(next_wcc[-1] - 2, next_wcc)
        # for position in next_wcc:
        #    triangle = __directed_triangle(np.pi*midpoint, np.pi*next_midpoint, np.pi*position)
        #    sign *= np.sign(triangle)
        # if sign == 1:
        #    crosses = 0
        # else:
        #    crosses = 1
        # num_crosses += crosses

    invariant = num_crosses % 2
    return invariant


def plot_wannier_centre_flow(wcc_flow, show_midpoints=False, ax=None, title=None,
                             fontsize=10):
    """ Routine to plot the WCC evolution """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for wcc in wcc_flow.T:
        ax.plot(wcc, 'ob', fillstyle='none')

    if show_midpoints:
        midpoints, _ = calculate_wcc_gap_midpoints(wcc_flow)
        ax.plot(midpoints, 'Dr')

    if title is not None:
        ax.set_title('WCC flow ' + title)
    ax.set_xlabel(r'$k_y$', fontsize=fontsize)
    ax.set_ylabel(r'$\hat{x}_n$', fontsize=fontsize)
    ax.set_xticks([0, wcc_flow.shape[0]/2 - 1/2, wcc_flow.shape[0] - 1])
    ax.set_xticklabels([r'$\pi$', "0", r'$\pi$'], fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_xlim([0, wcc_flow.shape[0] - 1])
    ax.set_ylim(bottom=-1, top=1)


def calculate_chern(wannier_centre_flow):
    """ TODO calculate_chern fix broken computation """
    """ Routine to compute explicitly the first Chern number from the WCC flow calculation """
    polarization_flow = np.sum(wannier_centre_flow, axis=1)
    plt.plot(polarization_flow)
    plt.show()


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


def specify_partition_plane(system, plane):
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


def specify_partition_shape(system, shape, center, r):
    """ Routine to determine which atoms are inside of a partition of the system.
        :param system: System object to extract the motif positions
        :param shape: Either 'square' or 'circle'. Shape of the partition we are considering.
        :param center: np.array or list with the coordinates of the center of the shape
        :param r: radius of the shape; for a square it corresponds to its side
        :return partition: Indices of atoms referred to the motif ordering which are
        inside the specified partition """

    """ TODO specify_partition_shape: 'square' shape not implemented """
    available_shapes = ["square", "circle"]
    if shape not in available_shapes:
        raise ValueError("shape must be either square or circle")

    partition = []
    for n, atom in enumerate(system.motif):
        atom_position = atom[:3]
        if np.linalg.norm(atom_position - np.array(center)) < r:
            partition.append(n)

    return np.array(partition)


def entanglement_spectrum(system, partition, kpoints=None):
    """ Routine to obtain the eigenvalues from the correlation/density matrix, which is directly
     related with the entangled Hamiltonian. Should be computable for both PBC and OBC """
    if kpoints is None and system.boundary != "OBC":
        raise Exception("Error: kpoints argument must be given when using PBC. Exiting...")
    if system.boundary == "OBC":
        if kpoints is not None:
            print("Warning: kpoints argument given but system uses OBC")
            print("Defaulting kpoints to origin...")
        kpoints = [[0., 0., 0.]]

    print("Computing entanglement spectrum...")
    print("Diagonalizing system...")
    nk = len(kpoints)
    results = system.solve(kpoints)

    spectrum = np.zeros([len(partition) * system.norbitals, nk])
    print("Computing density matrices...")
    for i in range(len(kpoints)):
        truncated_eigenvectors = __truncate_eigenvectors(results.eigen_states[i],
                                                         partition, system)

        density_matrix = __density_matrix(truncated_eigenvectors)
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)

        spectrum[:, i] = eigenvalues

    return spectrum


def entanglement_entropy(spectrum):
    """ TODO entanglement_entropy fix log(0) """
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
        if eigval <= 0.5 < eigenvalues[i + 1]:
            central_index = i
            break

    with open(file, "w") as textfile:
        for i in range(central_index - n//2 + shift, central_index + n//2 + shift):
            textfile.write(f"{eigenvalues[i]}\n")


def plot_entanglement_spectrum(spectrum, system, ax=None,
                               fontsize=10, title=None, markersize=5, color="b"):
    """ Routine to plot the entanglement spectrum as a function of k.
     CAREFUL: This routine is not made to deal with any set of kpoints, rather it is intended
     to be used with a set of kpoints along a given line (1D). """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    if system.boundary == "OBC" or spectrum.shape[1] == 1:
        ax.plot(spectrum, 'o', c=color, markersize=markersize)
        ax.set_xlim(0, len(spectrum))
    else:
        for entanglement_band in spectrum:
            ax.plot(entanglement_band, 'o', c=color)
        ax.set_xlim(0, len(spectrum.T) - 1)
        ax.set_xticks([0, len(spectrum.T)/2 - 1/2, len(spectrum.T) - 1])
        ax.set_xticklabels([r"-$\pi/a$", "0", r"$\pi/a$"], fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize, direction="in")

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    # ax.set_title(f"Entanglement spectrum of {system.system_name}", fontsize=15)
    ax.set_xlabel("n", fontsize=fontsize)
    ax.set_ylabel(rf"$\xi_n$", fontsize=fontsize)














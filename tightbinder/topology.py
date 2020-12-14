# Module for topology-related routines. It incorporates routines to compute the Wilson loop spectra,
# as well as the topological invariants derived from it. Local markers such as the Bott index or the
# local Chern number are implemented as well.

import numpy as np
import matplotlib.pyplot as plt


def __wilson_loop(system, path, filling):
    """ Routine to calculate the wilson loop matrix for a system along a given path. Returns a matrix
     of size Nocc x Nocc """

    result = system.solve(path)

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
    eigval = np.sort(np.angle(eigval)/np.pi).round(decimals=5)

    eigval[eigval == -1] = 1  # Enforce interval (-1, 1], NOTE: it breaks partial polarization!

    return eigval


def calculate_wannier_centre_flow(system, filling, number_of_points, additional_k=None):
    """ Routine to compute the evolution of the Wannier charge centres through Wilson loop calculation """

    wcc_results = []
    # cell_side = crystal.high_symmetry_points
    # k_fixed_list = np.linspace(-cell_side/2, cell_side/2, number_of_points)
    for i in range(-number_of_points//2, number_of_points//2 + 1):
        k_fixed = system.reciprocal_basis[1]*i/number_of_points
        if additional_k is not None:
            k_fixed += additional_k
        # k_fixed = np.array([0, k_fixed_list[i], 0])
        path = __generate_path(k_fixed, system, nk=100)
        wilson_loop = __wilson_loop(system, path, filling)
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













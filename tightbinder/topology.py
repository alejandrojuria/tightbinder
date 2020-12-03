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
        #print(__check_orthogonality(result.eigen_states[i]))
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
    #print(difference)
    #print(np.trace(difference), len(matrix))
    if np.linalg.norm(difference) < epsilon:
        is_orthonormal = True
    else:
        is_orthonormal = False

    return is_orthonormal


def __generate_path(k_fixed, crystal, nk):
    """ Routine to generate a path in the Brilloin zone with one of the k components fixed """
    basis_vector = crystal.reciprocal_basis[0]
    cell_side = crystal.reciprocal_basis[0][0]
    path = []
    k_list = np.linspace(-cell_side/2, cell_side/2, nk + 1)
    for i in range(nk + 1):
        # basis_vector = np.array([k_list[i], 0, 0])
        kpoint = basis_vector*i/nk + k_fixed
        # kpoint = basis_vector + k_fixed
        path.append(kpoint)

    return path


def __extract_wcc_from_wilson_loop(wilson_loop):
    """ Routine to calculate the Wannier charge centres from the Wilson loop matrix """
    eigval = np.linalg.eigvals(wilson_loop)
    print(eigval)
    eigval = np.angle(eigval)/np.pi
    print(eigval)
    print("---------------")

    return eigval


def calculate_wannier_centre_flow(system, crystal, filling, number_of_points):
    """ Routine to compute the evolution of the Wannier charge centres through Wilson loop calculation """

    wcc_results = []
    cell_side = crystal.high_symmetry_points
    k_fixed_list = np.linspace(-cell_side/2, cell_side/2, number_of_points)
    for i in range(-number_of_points//2, number_of_points//2 + 1):
        # k_fixed = crystal.reciprocal_basis[1]*i/number_of_points
        k_fixed = np.array([0, k_fixed_list[i], 0])
        path = __generate_path(k_fixed, crystal.reciprocal_basis, nk=100)
        wilson_loop = __wilson_loop(system, path, filling)
        wcc = __extract_wcc_from_wilson_loop(wilson_loop)

        wcc_results.append(wcc)

    return np.array(wcc_results)


def calculate_polarization(wcc):
    """ Routine to compute the polarization from the WCC """
    polarization = np.sum(wcc)
    return polarization


def calculate_chern_number(wcc_flow):
    """ Routine to compute the first Chern number from the WCC flow """
    wcc_k0 = wcc_flow[0, :]
    wcc_k2pi = wcc_flow[-1, :]

    chern_number = calculate_polarization(wcc_k0) - calculate_polarization(wcc_k2pi)
    return chern_number


def plot_wannier_centre_flow(wcc_results):
    """ Routine to plot the WCC evolution """
    plt.figure()
    for wcc in wcc_results.T:
        plt.plot(wcc, 'ob', fillstyle='none')

    plt.title('WCC flow')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\hat{x}_n$')
    plt.xticks([0, wcc_results.shape[0]], ['0', r'$\pi$'])
    plt.show()













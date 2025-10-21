"""
Topology-related routines. 

It incorporates routines to compute the Wilson loop spectra,
as well as the topological invariants derived from it. Local markers such as the
local Chern number are implemented as well.
"""

from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy as sp
from matplotlib.axes import Axes
from io import TextIOWrapper

from tightbinder.system import System
from tightbinder.result import Spectrum

def __wilson_loop(system: System, path: list) -> np.ndarray:
    """ 
    Routine to calculate the wilson loop matrix for a system along a given path. Returns a matrix
    of size Nocc x Nocc.

    :param system: System used to compute the Wilson loop.
    :param list: List of kpoints corresponing to closed path in BZ to compute Wilson loop.
    :return: Wilson loop matrix corresponding to given closed loop.
    """

    result = system.solve(path)

    filling = system.filling
    wilson_loop = np.eye(filling)
    for i in range(len(path) - 1):
        overlap_matrix = np.dot(np.conj(np.transpose(result.eigen_states[i][:, :filling])),
                                result.eigen_states[i + 1][:, :filling])
        wilson_loop = np.dot(wilson_loop, overlap_matrix)

    overlap_matrix = np.dot(np.conj(np.transpose(result.eigen_states[i + 1][:, :filling])),
                            result.eigen_states[0][:, :filling])
    wilson_loop = np.dot(wilson_loop, overlap_matrix)

    return wilson_loop


def __check_orthogonality(matrix: np.ndarray) -> bool:
    """
    Private method to check whether the columns of a given matrix are orthonormal between them 
    (using complex scalar product).

    :param matrix: Matrix to check orthogonality.
    :return: True if orthogonal, else False.
    """
    
    epsilon = 1E-16
    difference = np.dot(np.conjugate(np.transpose(matrix)), matrix)
    difference = (np.conj(difference) * difference)
    difference[difference < epsilon] = 0
    if np.linalg.norm(difference) < epsilon:
        is_orthonormal = True
    else:
        is_orthonormal = False

    return is_orthonormal


def __generate_path(k_fixed: np.ndarray, system: System, nk: int) -> np.ndarray:
    """ 
    Routine to generate a closed path in the Brilloin zone with one of the k components fixed.

    :param k_fixed: Array corresponding to kpoint. Size (1, 3)
    :param system: System whose reciprocal lattice we use to generate a path.
    :param nk: Number of kpoints in path to generate.
    :return: Array with kpoints of path, size (nk, 3)
    """
    
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


def __extract_wcc_from_wilson_loop(wilson_loop: np.ndarray) -> np.ndarray:
    """ 
    Routine to calculate the Wannier charge centres from the Wilson loop matrix.
    NOTE: We truncate the eigenvalues up to a given precision to avoid numerical error due to floating point
    when computing the midpoints.

    :param wilson_loop: Matrix of the Wilson loop.
    :return: Array with eigenvalues of Wilson loop. 
    """

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


def calculate_wannier_centre_flow(system: System, number_of_points: int, full_BZ: bool = False, 
                                  additional_k: np.ndarray = None, nk_subpath: int = 50, 
                                  refine_mesh: bool = True, pos_tol: float = 5E-2, max_iterations: int = 20) -> np.ndarray:
    """ 
    Routine to compute the evolution of the Wannier Charge Centres (WCC) through Wilson loop calculation.
    TODO: Add parameter to toggle on/off using half/full BZ.

    :param system: System used to compute the WCC evolution.
    :param number_of_points: Number of points where the WCC are computed. 
    :param full_BZ: Boolean to toggle WCC calculation on full BZ instead of half the BZ. Useful
        for Chern number calculations (full_BZ=True). Defaults to False (for Z2 TIs).
    :param additional_k: Array corresponding to kpoints to displace the closed paths by that vector.
    :param nk_subpath: Number of points for each closed path for which we compute the Wilson loop. Defaults to 50.
    :param refine_mesh: Boolean to toggle mesh refinement. Defaults to True.
    :param pos_tol: Value of the maximum difference allowed between two consecutive WCCs. Used as criteria for mesh
        refinement.
    :param max_iterations: Maximum number of times the refinement loop is allowed to be repeated. Usually, if the
        algorithm times out it means that the Z2 calculation can't converge (due to band crossings). 
        Defaults to 20.
    :return: Array with the WCCs evaluated at each kpoint.
    """

    print("Computing Wannier centre flow...")
    if system.filling is None:
        raise ValueError('Filling must be specified to compute WCC')

    all_wcc = []
    fixed_kpoints = []

    # First generate uniform mesh in BZ
    if full_BZ:
        factor = 1
    else:
        factor = 2
    for i in range(0, number_of_points + 1):
        k_fixed = system.reciprocal_basis[1]*i/(factor*number_of_points)
        if additional_k is not None:
            k_fixed += additional_k
        fixed_kpoints.append(k_fixed)

    # Compute WCC on uniform mesh
    for k_fixed in fixed_kpoints:
        path = __generate_path(k_fixed, system, nk_subpath)
        wilson_loop = __wilson_loop(system, path)
        wcc = __extract_wcc_from_wilson_loop(wilson_loop)

        all_wcc.append(wcc)
    
    # Now check if WCC evolution is smooth. If not, refine mesh where needed
    if refine_mesh:
        refined_mesh_count = 0
        converged = False
        n_iterations = 0
        while not converged:
            print(f"iteration {n_iterations}")
            n_iterations += 1
            added_points_iteration = 0
            all_wcc_copy = np.copy(all_wcc)
            kpoints_copy = np.copy(fixed_kpoints)
            for i, wcc in enumerate(all_wcc_copy[:-1]):
                next_wcc = all_wcc_copy[i + 1]

                if __detect_wcc_position_difference(wcc, next_wcc, pos_tol):
                    new_kpoint = (kpoints_copy[i] + kpoints_copy[i + 1])/2
                    path = __generate_path(new_kpoint, system, nk_subpath)
                    wilson_loop = __wilson_loop(system, path)
                    new_wcc = __extract_wcc_from_wilson_loop(wilson_loop)

                    all_wcc.insert(i + added_points_iteration + 1, new_wcc)
                    fixed_kpoints.insert(i + added_points_iteration + 1, new_kpoint)
                    added_points_iteration += 1

            refined_mesh_count += added_points_iteration
            if (added_points_iteration == 0 or n_iterations == max_iterations):
                converged = True
        
        print(f"Invariant required computing {refined_mesh_count} extra points")

    return np.array(all_wcc)


def __detect_wcc_position_difference(wcc: np.ndarray, next_wcc: np.ndarray, pos_tol: float) -> bool:
    """ 
    Private routine to check the difference in positions of contiguous WCC. If there are not centers
    such that the difference is below the tolerance. Used for the mesh refinement algorithm in 
    calculate_wannier_centre_flow. 

    :param wcc: Array with the WCCs corresponding to one path.
    :param next_wcc: Array with the WCCs of the consecutive path.
    :param pos_tol: Maximum value allowed for the difference between WCCs of consecutive paths.
    :return: True if the difference between WCCs is higher than pos_tol. 
    """

    # Forward pass
    next_wcc_copy = np.append([next_wcc[-1] - 2], next_wcc)
    next_wcc_copy = np.append(next_wcc_copy, [next_wcc[0] + 2])
    for center in wcc:
        wcc_diff = np.abs(center - next_wcc_copy)
        if np.min(wcc_diff) > pos_tol:
            return True

    # Backward pass
    wcc_copy = np.append([wcc[-1] - 2], wcc)
    wcc_copy = np.append(wcc_copy, [wcc[0] + 2])
    for center in next_wcc:
        wcc_diff = np.abs(center - wcc_copy)
        if np.min(wcc_diff) > pos_tol:
            return True
    
    return False

# ---------------------------- Chern number ----------------------------
def calculate_polarization(wcc: np.ndarray) -> float:
    """ 
    Routine to compute the polarization from the WCC. Note that it requires computing
    the WCCs in the whole BZ, and not half BZ (which is the default calculation). 
    Useful for Chern insulators. TODO: Has to be fixed.
    
    :param wcc: Array with the WCCs for all paths.
    :return: Polarization. 
    """

    polarization = np.sum(wcc)
    return polarization


def calculate_chern_number(wcc_flow: np.ndarray) -> float:
    """ 
    Routine to compute the first Chern number from the WCC flow. Useful for Chern insulators.
    NB: This routine gives an estimate of the Chern number (non quantized), and is more precise 
    the more kpoints there are in the calculation.

    :param wcc_flow: Array with the WCCs for all paths.
    :return: Value of Chern number.
    """

    wcc_k0 = wcc_flow[0, :]
    wcc_k2pi = wcc_flow[-2, :]

    chern_number = calculate_polarization(wcc_k0) - calculate_polarization(wcc_k2pi)
    chern_number = 0 if chern_number < 1E-16 else chern_number
        
    return chern_number / 2.


# ---------------------------- Z2 invariant ----------------------------
def __find_wcc_midpoint_gap(wcc: np.ndarray) -> Tuple[float, float]:
    """ 
    Routine to find the midpoint of the biggest gap between WCCs. 
    
    :param wcc: Array with WCCs corresponding to a single path.
    :return: Position of midpoint of biggest gap among the WCCs, and index within WCC array. 
    """

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
    midpoint_gap = midpoint_gap + 2 if midpoint_gap < -1 else midpoint_gap

    return midpoint_gap, position_gap


def calculate_wcc_gap_midpoints(wcc_flow: np.ndarray) -> Tuple[list, list]:
    """ 
    Routine to compute the midpoint of the biggest gap for the WCC flow, and
    their relative positions to the WCC.
    
    :param wcc_flow: Array with the WCCs for all paths.
    :return: Position and index of midpoints fort he WCCs in each path.
    """

    midpoints, positions = [], []
    for wcc in wcc_flow:
        gap, position = __find_wcc_midpoint_gap(wcc)
        midpoints.append(gap)
        positions.append(position)

    return midpoints, positions


def __directed_triangle(phi1: float, phi2: float, phi3: float) -> float:
    """ 
    Routine to determine sign of a directed triangle. Used to determine
    whether the midpoint jump over the WCC bands. Deprecated over direct counting
    of crossed WCCs.

    :param phi1: First angle.
    :param phi2: Second angle.
    :param phi3: Third angle.
    :return: Orientation of the angle.
    """

    if abs(phi1 - phi2) < 1E-14:
        g = 1
    else:
        g = np.sin(phi2 - phi1) + np.sin(phi3 - phi2) + np.sin(phi1 - phi3)
    return g


def count_crossings(wcc: np.ndarray, midpoint: float, next_midpoint: float) -> int:
    """ 
    Routine to compute the number of WCC that lie between one midpoint and the
    next one. Used to determine the Z2 invariant.
    
    :param wcc: Array with WCCs for one closed path.
    :param midpoints: Value of midpoint of the current WCC array.
    :param next_midpoints: Value of midpoint of the consecutive array of WCCs.
    :return: Number of WCCs crossed when going from the present midpoint to the next one.
    """
    
    higher_midpoint = max(midpoint, next_midpoint)
    lower_midpoint = min(midpoint, next_midpoint)
    crossings = 0
    for center in wcc:
        if lower_midpoint <= center < higher_midpoint:
            crossings += 1

    return crossings


def calculate_z2_invariant(wcc_flow: np.ndarray) -> int:
    """ 
    Routine to compute the z2 invariant from the number of jumps the
    max gap midpoint does across different WCC bands. 
    
    :param wcc_flow: Array with the WCC for all paths.
    :return: Value of Z2 invariant.
    """
    
    nk = len(wcc_flow[:, 0])
    num_crosses = 0
    for i in range(nk - 1):
        wcc = np.copy(wcc_flow[i, :])
        next_wcc = wcc_flow[i + 1, :]

        midpoint, _ = __find_wcc_midpoint_gap(wcc)
        next_midpoint, _ = __find_wcc_midpoint_gap(next_wcc)
        num_crosses += count_crossings(next_wcc, midpoint, next_midpoint)

    invariant = num_crosses % 2
    return invariant


def plot_wannier_centre_flow(wcc_flow: np.ndarray, show_midpoints: bool = True, ax: Axes = None, title: str = None,
                             fontsize: int = 10, symmetric: bool = False, full_BZ: bool = False) -> None:
    """ 
    Routine to plot the WCC evolution. 
    
    :param wcc_flow: Array with the WCCs for all paths.
    :param show_midpoints: Boolean to toggle on showing the midpoint for each WCC. Defaults to True.
    :param ax: Matplotlib Axes object to plot the WCC evolution on. Defaults to None.
    :param title: Name of the plot. Defaults to empty string.
    :param fontsize: Value to change the fontsize in the plot. Defaults to 10.
    :param symmetric: Boolean to plot the WCC over the whole BZ instead of half of it.
    :param full_BZ: Boolean to plot WCC over complete BZ. Defaults to False.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for wcc in wcc_flow.T:
        if symmetric:
            wcc = np.concatenate((wcc[::-1][:-1], wcc), axis=0)
        ax.plot(wcc, 'ob', fillstyle='none')

    if show_midpoints:
        midpoints, _ = calculate_wcc_gap_midpoints(wcc_flow)
        if symmetric:
            midpoints = np.concatenate((midpoints[::-1][:-1], midpoints))
        ax.plot(midpoints, 'Dr', label="Midpoints")

    if title is not None:
        ax.set_title('WCC flow ' + title)
    ax.set_xlabel(r'$k_y$', fontsize=fontsize)
    ax.set_ylabel(r'$\hat{x}_n$', fontsize=fontsize)
    # ax.set_xticks([0, wcc_flow.shape[0]/2 - 1/2, wcc_flow.shape[0] - 1])
    if symmetric:
        ax.set_xticks([0, wcc_flow.shape[0] - 1, 2*wcc_flow.shape[0] - 2])
        ax.set_xticklabels([r'$-\pi$', r"0", r'$\pi$'], fontsize=fontsize)
    else:
        ax.set_xticks([0, wcc_flow.shape[0] - 1])
        if not full_BZ:
            ax.set_xticklabels([r"0", r'$\pi$'], fontsize=fontsize)
        else:
            ax.set_xticklabels([r"-$\pi$", r'$\pi$'], fontsize=fontsize)

    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_xlim([0, wcc_flow.shape[0] - 1])
    ax.set_ylim(bottom=-1, top=1)


def plot_polarization_flow(wcc_flow: np.ndarray) -> None:
    """ 
    Routine to plot the polarization flow used to calculate the Chern number. 
    
    :param wcc_flow: Array with the WCCs for all paths.
    """
    
    polarization_array = []
    for wcc in wcc_flow:
        polarization_array.append(calculate_polarization(wcc))

    plt.figure()
    plt.plot(polarization_array)


def chern_marker(system: System, results: Spectrum, ef: float = 0) -> np.ndarray:
    """
    Routine to evaluate the Chern marker at every position of the lattice.
    Must be used with systems with OBC or PBC at k=0.

    :param system: System to compute the Chern marker.
    :param results: Spectrum object with the results from the diagonalization.
    :param ef: Fermi energy, defaults to 0 (for particle-hole symmetric systems).
    :return: Array with the Chern marker evaluated at each atomic position.
    """

    
    P = np.diag((results.eigen_energy < ef).reshape(-1))
    Q = np.eye(system.basisdim) - P
    x = np.zeros([system.basisdim, system.basisdim])
    y = np.zeros([system.basisdim, system.basisdim])
    it = 0
    for n, atom in enumerate(system.motif):
        norbitals = system.norbitals[int(atom[3])]
        for i in range(norbitals):
            x[it, it] = atom[0]
            y[it, it] = atom[1]
            it += 1

    eigvec = results.eigen_states[0]

    x = eigvec.T.conj() @ x @ eigvec
    y = eigvec.T.conj() @ y @ eigvec

    PxP, PyP = P @ x @ P, P @ y @ P

    C = 2 * np.pi * 1j * (PxP @ PyP - PyP @ PxP)

    # PxQ, PyQ, QxP, QyP = P @ x @ Q, P @ y @ Q, Q @ x @ P, Q @ y @ P
    # C1 = 2 * np.pi * np.imag(QxP @ PyQ - QyP @ PxQ)
    # C2 = -2 * np.pi * np.imag(PxQ @ QyP - PyQ @ QxP)
    # C = (C1 + C2)/2
    C = eigvec @ C @ eigvec.T.conj()
    C = np.diag(C).real
    
    # Finally sum over orbitals
    # it = 0
    # trC = []
    # for n in range(system.natoms):
    #     species = int(system.motif[n, 3])
    #     norbitals = int(system.norbitals[species])
    #     summed_chern = np.sum(C[it : it + norbitals])
    #     trC.append(summed_chern)
    #     it += norbitals
    
    return C

    return C

# ---------------------------- Entanglement entropy ----------------------------
def __truncate_eigenvectors(eigenvectors: np.ndarray, sector: np.ndarray, system: System) -> np.ndarray:
    """ 
    Routine to truncate the eigenvectors according to the filling and the
    atoms that are on a specific partition, accounting for orbitals and spin structure.
    
    :param eigenvectors: Matrix with eigenvectors as columns.
    :param sector: 1d array of indices of atoms where the eigenvectors are evaluated.
    :param system: System used (necessary to extract orbital information and filling).
    :return: Eigenvectors with entries corresponding only to the specified sector,
        and truncated with the filling.
    """

    orbitals = []
    counter = 0
    for n in range(system.natoms):
        norb = system.norbitals[int(system.motif[n, 3])]
        if n in sector:
            for _ in range(0, norb):
                orbitals.append(counter)
                counter += 1
        else:
            counter += norb

    truncated_eigenvectors = eigenvectors[orbitals]
    truncated_eigenvectors = truncated_eigenvectors[:, :system.filling]

    return truncated_eigenvectors


def __density_matrix(truncated_eigenvectors: np.ndarray) -> np.ndarray:
    """ 
    Private routine to calculate the one-particle reduced density matrix for the half-system. 
    
    :param truncated_eigenvectors: Matrix with truncated eigenvectors.
    :return: Density matrix.
    """
    
    
    density_matrix = np.dot(truncated_eigenvectors, np.conj(truncated_eigenvectors.T))

    return density_matrix


def specify_partition_plane(system: System, plane: list) -> np.ndarray:
    """ 
    Routine to specify a real-space partition of the atoms of the system based on a given
    plane.
    
    :param system: System whose atoms we want to separate into two partitions.
    :param plane: List with coefficients of the plane generating the partition. 
        Array [A,B,C,D] <-> Ax + By + Cz = D
    :return: Array with indices of atoms on a side. """

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


def specify_partition_shape(system: System, shape: str, center: np.ndarray, r: float) -> np.ndarray:
    """ 
    Routine to determine which atoms are inside of a partition of the system.
    
    :param system: System object to extract the motif positions.
    :param shape: Either 'square' or 'circle'. Shape of the partition we are considering.
    :param center: Array with the coordinates of the center of the shape.
    :param r: Radius of the shape; for a square it corresponds to its side.
    :return partition: Indices of atoms referred to the motif ordering which are
        inside the specified partition.
    :return: Array with indices of atoms on a side. """

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


def entanglement_spectrum(system: System, partition: np.ndarray, kpoints: np.ndarray = None) -> np.ndarray:
    """ 
    Routine to obtain the eigenvalues from the correlation/density matrix, which is directly
    related with the entangled Hamiltonian. Should be computable for both PBC and OBC.
    
    :param system: System where we want to compute the the entanglement spectrum.
    :param partition: Array with the indices of the atoms in the partition where we want to 
        compute the entanglement spectrum.
    :param kpoints: Array of kpoints where we evaluate the entanglement spectrum. Defaults to None
        which is [0., 0., 0.] for both OBC and PBC.
    :return: Entanglement spectrum evaluated at each kpoint (if given).
    """
    
    if kpoints is None and system.boundary != "OBC":
        print('PBC: Defaulting kpoints to origin...')
        kpoints = [[0., 0., 0.]]
    if system.boundary == "OBC":
        if kpoints is not None:
            print("Warning: kpoints argument given but system uses OBC")
            print("OBC: Defaulting kpoints to origin...")
        kpoints = [[0., 0., 0.]]

    print("Computing entanglement spectrum...")
    print("Diagonalizing system...")
    nk = len(kpoints)
    results = system.solve(kpoints)

    basisdim = np.sum([system.norbitals[int(system.motif[atom, 3])] for atom in partition])
    spectrum = np.zeros([basisdim, nk])
    print("Computing density matrices...")
    for i in range(len(kpoints)):
        truncated_eigenvectors = __truncate_eigenvectors(results.eigen_states[i],
                                                         partition, system)

        density_matrix = __density_matrix(truncated_eigenvectors)
        eigenvalues, _ = np.linalg.eigh(density_matrix)

        spectrum[:, i] = eigenvalues

    return spectrum


def entanglement_entropy(spectrum: np.ndarray) -> float:
    """ 
    TODO entanglement_entropy fix log(0). 
    Routine to compute the entanglement entropy from the entanglement spectrum.

    :param spectrum: Array with the entanglement spectrum.
    :return: Value of entanglement entropy.
    """
    
    entropy = 0
    for eigval in spectrum.T:
        entropy -= np.dot(eigval, np.log(eigval)) + np.dot(1 - eigval, np.log(1 - eigval))
    entropy /= len(spectrum.T)

    return entropy


def write_entanglement_spectrum_to_file(spectrum: np.ndarray, file: TextIOWrapper, n: int = None, shift: int = 0) -> None:
    """ 
    Routine to write the entanglement spectrum to a text file. If n is given,
    only n eigenvalues are written to file.
    The algorithm is as follows: first we search for the eigenvalue that is closer to 0.5, and then
    we write the n/2 eigenvalues to its left and right. 
    
    :param spectrum: Array with the entanglement spectrum.
    :param file: Pointer to the file to write the entanglement spectrum.
    """

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


def plot_entanglement_spectrum(spectrum: np.ndarray, system: System, ax: Axes = None,
                               fontsize: int = 10, title: str = None, markersize: int = 5, color: str = "b") -> None:
    """ 
    Routine to plot the entanglement spectrum as a function of k.
    CAREFUL: This routine is not made to deal with any set of kpoints, rather it is intended
    to be used with a set of kpoints along a given line (1D). 
    
    :param spectrum: Array with the entanglement spectrum.
    :param system: System where we compute the entanglement spectrum.
    :param ax: Matplotlib Axes object to plot in. Defaults to None.
    :param fontsize: Value to change the fontsize of the plot. Defaults to 10.
    :param title: Title of the plot. Defaults to empty string.
    :param markersize: Size of the markers of the plot. Defaults to 5.
    :param color: Color of the markers. Defaults to blue.
    """
    
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    if system.boundary == "OBC" or len(spectrum.shape) == 1 or spectrum.shape[1] == 1:
        ax.plot(spectrum, 'o', c=color, markersize=markersize)
        ax.set_xlim(0, len(spectrum))
        ax.set_xlabel(r"$n$", fontsize=fontsize)
    else:
        for entanglement_band in spectrum:
            ax.plot(entanglement_band, 'o', c=color)
        ax.set_xlim(0, len(spectrum.T) - 1)
        ax.set_xticks([0, len(spectrum.T)/2 - 1/2, len(spectrum.T) - 1])
        ax.set_xticklabels([r"-$\pi/a$", "0", r"$\pi/a$"], fontsize=fontsize)
        ax.set_xlabel(r"$k_y$", fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize, direction="in")

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    # ax.set_title(f"Entanglement spectrum of {system.system_name}", fontsize=15)
    
    ax.set_ylabel(rf"$\xi_n$", fontsize=fontsize)



# ---------------------- Quantum geometry ----------------------
def berry_curvature(model: System, nk: int, band_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculation of the Berry curvature of a single band using the Fukui method.
    Only valid for 2D systems.
    Note that this routine is only valid for single bands without degeneracies, for
    non-abelian Berry curvatures one should use the generalized method.

    :param model: System from which we want to obtain the Berry curvature.
    :param nk: Number of k points along one axis of the BZ.
    :param band_index:
    :returns: List of kpoints and field strenght of Berry connection.
    """

    if model.ndim != 2:
        raise AttributeError("Model must be two-dimensional.")
    
    kpoints = model.brillouin_zone_mesh([nk, nk])
    
    unit_displacements = []
    for i in range(model.ndim):
        unit_displacements.append(model.reciprocal_basis[i]/nk)

    curvature = []
    for kpoint in kpoints:
        local_kpoints = [kpoint, kpoint + unit_displacements[0], 
                         kpoint + unit_displacements[0] + unit_displacements[1], 
                         kpoint + unit_displacements[1], kpoint]
        eigvec = model.solve(local_kpoints).eigen_states

        links = []
        for i in range(len(local_kpoints) - 1):
            link = np.vdot(eigvec[i][:, band_index], eigvec[i + 1][:, band_index])
            link /= np.abs(link)
            links.append(link)

        strength = -1j * np.log(np.prod(links))
        curvature.append(strength)


    return kpoints, np.array(curvature)













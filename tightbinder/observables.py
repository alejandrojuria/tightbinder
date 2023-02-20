# Module containing routines for computation of observables

import sys
from tightbinder.result import Spectrum
from tightbinder.system import System
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from typing import Tuple, List
from matplotlib.axes import Axes
import scipy.sparse as sp

def _retarded_green_function(w: float, e: float, delta: float) -> complex:
    """ 
    Routine to compute the retarded Green function, to be used to obtain
    the DOS.

    :param w: Value of frequency.
    :param e: Value of energy.
    :param delta: Value of broadening.
    :return: Retarded Green's function evaluated at w, e. 
    """
    
    green = 1./(w - e + 1j*delta)
    return green


def dos(result: Spectrum, energy: float = None, delta: float = 0.1, npoints: int = 200) -> Tuple[List[float], List[float]]:
    """ 
    Routine to compute the density of states from a Result object.

    :param result: Result object.
    :param energy: Value of energy where DoS is computed. If None, then the DoS is computed
        for the whole energy window of the spectrum.
    :param delta: Delta broadening, defaults to 0.01. 
    :param npoints: Number of energy points to use if sampling whole DoS.
    :return: Returns duple with values of dos, and energies where was computed.
    """
    
    nkpoints = result.eigen_energy.shape[1]
    eigval = result.eigen_energy.reshape(-1, )
    if not energy:
        emin, emax = np.min(result.eigen_energy)*1.5, np.max(result.eigen_energy)*1.5
        energies = np.linspace(emin, emax, npoints, endpoint=True)
    else:
        energies = [energy]

    energies = np.array(energies)
    dos = []
    for energy in energies:
        density = (np.sum(-np.imag(_retarded_green_function(energy, eigval, delta))/np.pi)
                /(nkpoints*result.system.unit_cell_area))
        dos.append(density)

    return dos, energies


def dos_kpm(system: System, energy: float = None, npoints: int = 200, nmoments: int = 30, r: int = 10) -> Tuple[List[float], List[float]]:
    """ 
    Routine to compute the density of states using the Kernel Polynomial Method.
    Intended to be used with supercells and k=0.

    :param system: System whose density of states we want to compute.
    :param energy: Value of energy where we want to obtain the DoS. If None, DoS is computed for all
        energy window of spectrum.
    :param npoints: Number of energy points to consider if computing full DoS. Defaults to 200.
    :param nmoments: Number of moments to use in the KPM expansion. Defaults to 30.
    :param r: Number of samples to use in the stochastic evaluation of the KPM traces.
        Defaults to 10.
    :return:
    """

    if system.matrix_type != "sparse":
        print("Warning: KPM computations are intended to be run with sparse matrices (faster and less memory usage)")

    print("Computing DOS using the KPM...")
    h = system.hamiltonian_k([[0., 0., 0.]])
    if system.matrix_type == "dense":
        h = sp.bsr_matrix(h)
    # h spectrum has to be between -1 and 1
    h_norm = sp.linalg.norm(h, ord=np.inf)
    h /= h_norm
    
    if not energy:
        emin, emax = -1, 1
        energies = np.linspace(emin, emax, npoints, endpoint=True)[1:-1]
    else:
        energies = [energy/h_norm]

    energies = np.array(energies)
    moments = compute_dos_momentum(nmoments, h, r)
    jackson = jackson_kernel(nmoments)
    moments = moments * jackson
    densities = []
    for energy in energies:
        polynomials = chebyshev_polynomial_values(nmoments, energy)
        density = moments[0] * polynomials[0] + 2*np.sum(moments[1:] * polynomials[1:])
        density = density/(np.pi*np.sqrt(1 - energy**2))
        densities.append(density)

    energies = energies * h_norm
    return densities, energies


def jackson_kernel(nmoments: int) -> List[float]:
    """ 
    Routine to calculate the Jackson kernel to improve the KPM calculations. 
    
    :param nmoments: Number of moments of KPM expansions.
    :return: Jackson kernel for n=0, ..., nmoments - 1
    """

    g = [((nmoments - n + 1)*np.cos(np.pi*n/(nmoments + 1)) +
          np.sin(np.pi*n/(nmoments + 1))/np.tan(np.pi*n/(nmoments + 1)))/(nmoments + 1) for n in range(1, nmoments)]
    g = [1] + g  # n = 0 gives NaN instead of 1
    
    return g


def chebyshev_polynomial_values(n: int, e: float) -> np.ndarray:
    """ 
    Chebyshev polynomial of order n on value e. 
    
    :param n: Maximum order of Chebyshev polynomial.
    :param e: Value where the polynomials is evaluated (-1 < e < 1).
    :return: List with all polynomials from order 0 to n - 1 evaluated on e.
    """
    
    polynomials = [1, e]
    if n >= 2:
        for i in range(2, n):
            polynomials.append(2*e*polynomials[i - 1] - polynomials[i - 2]) 
    return np.array(polynomials)


def compute_dos_momentum(n: int, h: np.ndarray, r: int) -> np.ndarray:
    """ 
    Routine to compute up to the n-th momentum associated to a Chebyshev polynomial expansion
    of the density of states. Instead of computing the whole trace of h, we perform the stochastic
    evaluation of the trace over a small set of random vectors. 
    
    :param n: Total number of momentum computed for DoS calculation.
    :param h: Hamiltonian, usually on k=0 (intended for supercell calculations).
    :param r: Number of samples on stochastic ealuation of trace.
    :return: Array with all momentum.
    """

    d = h.shape[0]

    # Generate sample of random vectors and normalize them
    vectors = np.random.uniform(-1, 1, (r, d))
    norm = np.sqrt(np.diag(np.dot(vectors, vectors.T)))
    vectors = vectors / norm[:, None]
    vectors = vectors.T

    # Evaluate each momentum
    momentum = []
    operators = [sp.eye(h.shape[0], h.shape[1]).dot(vectors), h.dot(vectors)]
    momentum.append(np.sum(vectors.T.dot(operators[0]).diagonal())/(r*d))
    momentum.append(np.sum(vectors.T.dot(operators[1]).diagonal())/(r*d))
    for i in range(2, n):
        operator = 2*h.dot(operators[1]) - operators[0]
        mu = np.sum(vectors.T.dot(operator).diagonal())/(r*d)
        momentum.append(mu)
        operators[0] = operators[1]
        operators[1] = operator

    return np.array(momentum)


def plot_dos(result: Spectrum, npoints: int = 100, delta: float = 0.3, ax: Axes = None) -> None:
    """ 
    Routine to plot the density of states using the retarded Green function. 
    
    :param result: Spectrum object containing the eigenenergies.
    :param npoints: Number of points used to sample the DoS. Defaults to 100.
    :param delta: Broadening of the states. Defaults to 0.3.
    :param ax: Axes object to plot the DoS in it. Defaults to None.
    """

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    density, energies = dos(result, delta=delta, npoints=npoints)

    ax.plot(energies, density)
    ax.set_xlabel(r'E (eV)')
    ax.set_ylabel(r'DOS')

    return density, energies


def plot_dos_kpm(system: System, npoints: int = 200, nmoments: int = 100, r: int = 10, ax: Axes = None) -> None:
    """
    Routine to plot the density of states using the KPM. 
    
    :param system: System whose DoS we want to obtain and plot.
    :param npoints: Number of points used to sample the DoS. Defaults to 200.
    :param nmoments: Number of moments in KPM expansion. Defaults to 100.
    :param r: Number of samples for stochastic evaluation of trace in KPM. Defaults to 100.
    :param ax: Optional Axes object to draw the plot in.
    """

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    dos, energies = dos_kpm(system, npoints=npoints, nmoments=nmoments, r=r)

    ax.plot(energies, dos)
    ax.set_title('DOS (KPM)')
    ax.set_xlabel(r'E (eV)')
    ax.set_ylabel(r'DOS')

    return dos, energies


def fermi_energy(dos: list, energy: list, system: System) -> float:
    """ 
    Routine to compute the Fermi energy from the density of states
    integrating up to the number of electrons in the system. It is based on 
    trapezoidal integration of the DoS.
    
    :param dos: List with density of states values.
    :param energy: List with energies where the DoS are evaluated.
    :param system: System object (used to provide filling and basisdim).
    :return: Fermi energy.
    """

    # First normallize to basisdim (maximum number of states)
    area = np.trapz(dos, energy)
    dos = dos / area * system.basisdim
    # Then search
    for i in range(len(dos)):
        nstates = np.trapz(dos[:i], energy[:i])
        if nstates > system.filling:
            break
    
    return energy[i - 1]


def compute_projector_momentum(nmoments: int, energy: int) -> np.ndarray:
    """ 
    Routine to compute the momentum from the KPM expansion of the ground state projector. 
    
    :param nmoments: Number of momentum we want to compute.
    :param energy: Energy up to which states are considered.
    :return: List of moments.
    """

    first_moment = [1 - np.arccos(energy)/np.pi]
    moments = [-2*np.sin(m*np.arccos(energy))/(m*np.pi) for m in np.arange(1, nmoments)]
    moments = first_moment + moments

    return np.array(moments)

def ground_state_projector(nmoments: int, h: np.ndarray, energy: int) -> np.ndarray:
    """ 
    Routine to compute the ground state projector using the KPM. Note that this calculation
    is done using sparse arrays, since it results in faster computation.
    
    :param nmoments: Number of moments in the expansion.
    :param h: Bloch Hamiltonian, usually at k=0 for supercells.
    :param energy: Energy up to which the states are considered. To compute the ground state
        projector, this should be the Fermi energy.
    :return: Projector.
    """

    h_norm = np.linalg.norm(h)
    h /= h_norm
    h = sp.bsr_matrix(h)
    print("Cast to sparse done")
    energy /= h_norm

    print("Computing moments...")
    moments = compute_projector_momentum(nmoments, energy)
    jackson = jackson_kernel(nmoments)
    moments = moments * jackson
    operators = [sp.eye(h.shape[0], h.shape[1], format='csr'), h]
    projector = moments[0]*operators[0] + moments[1]*operators[1]
    for i in range(2, nmoments):
        print(f"Moment: {i}")
        operator = 2*h.dot(operators[1]) - operators[0]
        projector += moments[i] * operator
        operators[0] = operators[1]
        operators[1] = operator

    return projector.toarray()


def restricted_density_matrix(system: System, partition: list, nmoments: int = 100, npoints: int = 200, 
                              nmoments_dos: int = 300, r: int = 10) -> np.ndarray:
    """ 
    Routine to compute the one-particle density matrix, restricted to one spatial partition of the system. First it estimates the Fermi
    energy from the DOS computed with the KPM, and then uses it to compute the reduced density matrix. 
    
    :param system: System whose reduced density matrix we want to compute.
    :param partition: List of indices of atoms where the reduced density matrix is computed.
    :param nmoments: Number of moments in the KPM expansion. Defaults to 100.
    :param npoints: Number of used to sample the density of states. Defaults to 200.
    :param nmoments_dos: Number of moments in the KPM expansion of the DoS. Defaults to 300.
    :param r: Number of samples in the stochastic evaluation of the trace. Defaults to 10.
    :return: Reduced density matrix.
    """

    if system.matrix_type != "sparse":
        print("Warning: KPM computations are intended to be run with sparse matrices (faster and less memory usage)")
    
    h = system.hamiltonian_k([[0., 0., 0.]])
    if system.matrix_type == "dense":
        h = sp.bsr_matrix(h)

    h_norm = sp.linalg.norm(h)
    h /= h_norm
    
    # First compute fermi energy from dos
    dos, energy = dos_kpm(system, npoints=npoints, nmoments=nmoments_dos, r=r)
    efermi = fermi_energy(dos, energy, system)/h_norm

    orbitals = []
    counter = 0
    for n in range(system.natoms):
        norb = system.norbitals[int(system.motif[n, 3])]
        if n in partition:
            for _ in range(0, norb):
                orbitals.append(counter)
                counter += 1
        else:
            counter += norb
    
    orbital_cols = np.arange(len(orbitals))
    orbital_values = np.ones(len(orbitals))
    orbital_matrix = sp.csr_matrix((orbital_values, (orbitals, orbital_cols)), shape=(system.basisdim, len(orbitals)))

    moments = compute_projector_momentum(nmoments, efermi)
    jackson = jackson_kernel(nmoments)
    moments = moments * jackson
    operators = [orbital_matrix, h.dot(orbital_matrix)]

    density_matrix = moments[0]*operators[0] + moments[1]*operators[1]
    for i in range(2, nmoments):
        print(f"Moment: {i}")
        operator = 2*h.dot(operators[1]) - operators[0]
        density_matrix += moments[i] * operator
        operators[0] = operators[1]
        operators[1] = operator

    density_matrix = orbital_matrix.transpose().dot(density_matrix).toarray()

    print(f"Fermi energy: {efermi}")
    return density_matrix


def transmission(system: System, left_boundary: list, right_boundary: list, minE: float, maxE: float, npoints: int = 100, t: float = 1, delta: float = 1E-4):
    """
    Function to compute the transmission function T(E) of a system. One has to specify the boundaries
    where the leads are connected. Only two terminal setups are allowed. The leads are simulated
    assuming s orbitals, and a square lattice. The system is also assumed to have the shape of a rectangle to setup
    the connection to the leads.

    :param system: System to compute transmission
    :param left_boundary: List of indices of atoms of the system where the left lead is connected
    :param right_boundary: Indices where the right lead connects.
    :param minE: Minimum value of energy window where the transmission is computed.
    :param maxE: Maximum value of energy window.
    :param npoints: Sampling of energy window. Defaults to 100.
    :param t: Value of leads hopping.
    :param delta: Value of broadening used in Green's functions.
    :return: List of transmission values and energy window.
    """

    device_hamiltonian, block_lead_L, block_lead_R = __attach_leads(system, left_boundary, right_boundary, t)
    device_dim = device_hamiltonian[0].shape[0]
    energies = np.linspace(minE, maxE, npoints)
    currents = []
    extended_lead_selfenergy_L = sp.lil_matrix((device_dim, device_dim), dtype=np.complex_)
    extended_lead_selfenergy_R = sp.lil_matrix((device_dim, device_dim), dtype=np.complex_)

    for energy in energies:
        lead_selfenergy_L = __lead_selfenergy(energy, block_lead_L, t, delta)  
        
        coupling_L = 1j*(lead_selfenergy_L - lead_selfenergy_L.transpose().conjugate())   
        extended_lead_selfenergy_L[:lead_selfenergy_L.shape[0], :lead_selfenergy_L.shape[0]] = coupling_L

        lead_selfenergy_R = __lead_selfenergy(energy, block_lead_R, t, delta)
        coupling_R = 1j*(lead_selfenergy_R - lead_selfenergy_R.transpose().conjugate())
        extended_lead_selfenergy_R[(device_dim - lead_selfenergy_R.shape[0]):, (device_dim - lead_selfenergy_R.shape[0]):] = coupling_R

        device_green = __device_green_function(energy, device_hamiltonian, block_lead_L, block_lead_R, t, delta)
        
        G = (extended_lead_selfenergy_L @ device_green @ extended_lead_selfenergy_R @ device_green.transpose().conjugate()).trace()
        print("G: ", G)
        currents.append(G)

    return (currents, energies)


def __attach_leads(system: System, left_boundary: list, right_boundary: list, t: float):
    """ 
    Private method to extend the Hamiltonian to include a part of the leads. It assumes that the system has
    rectangle shape. The leads will have the dimension of the number of atoms in the corresponding boundary.

    :param system: System to attach leads to.
    :param left_boundary: List of indices of atoms where we attach the left lead.
    :param right_boundary: List of atoms to attach right lead.
    :param t: Hopping amplitude of lead.
    :return: ([Hd],Hl,Hr) [Hd] = Fock matrices of Hamiltonian of device with leads attached.
        Vl = Left lead matrices. Vr = Right lead matrices.
    """

    if not left_boundary or not right_boundary:
        raise ValueError("Boundaries list must not be empty.")


    left_coupling = __lead_system_coupling(system, left_boundary, t)
    right_coupling = __lead_system_coupling(system, right_boundary, t)
    leftnatoms = len(left_boundary)
    rightnatoms = len(right_boundary)

    left_lead_block_h = sp.lil_matrix((leftnatoms, leftnatoms), dtype=np.complex_)
    try:
        left_lead_block_h.setdiag(t, 1)
        left_lead_block_h.setdiag(t, -1)
    except ValueError as e:
        pass
    right_lead_block_h = sp.lil_matrix((rightnatoms, rightnatoms), dtype=np.complex_)
    try:
        right_lead_block_h.setdiag(t, 1)
        right_lead_block_h.setdiag(t, -1)
    except ValueError as e:
        pass
    left_lead_block_h = left_lead_block_h.tocsc()
    right_lead_block_h = right_lead_block_h.tocsc()

    device_dim = system.basisdim + leftnatoms + rightnatoms
    device_hamiltonian = [sp.lil_matrix((device_dim, device_dim), dtype=np.complex_) for _ in range(len(system.hamiltonian))]
    for i, matrix in enumerate(system.hamiltonian):
        device_hamiltonian[i][leftnatoms:leftnatoms + system.basisdim, 
                              leftnatoms:leftnatoms + system.basisdim] = matrix
    
    device_hamiltonian[0][:leftnatoms, :leftnatoms] = left_lead_block_h
    device_hamiltonian[0][:leftnatoms, leftnatoms:leftnatoms + system.basisdim] = left_coupling
    device_hamiltonian[0][leftnatoms:leftnatoms + system.basisdim, :leftnatoms] = left_coupling.transpose()

    device_hamiltonian[0][leftnatoms + system.basisdim:, leftnatoms + system.basisdim:] = right_lead_block_h
    device_hamiltonian[0][leftnatoms + system.basisdim:, leftnatoms:leftnatoms + system.basisdim] = right_coupling
    device_hamiltonian[0][leftnatoms:leftnatoms + system.basisdim, leftnatoms + system.basisdim:] = right_coupling.transpose()

    for i in range(len(device_hamiltonian)):
        device_hamiltonian[i] = device_hamiltonian[i].tocsc()

    return (device_hamiltonian, left_lead_block_h, right_lead_block_h)


def __lead_system_coupling(system: System, boundary: list, t: float):
    """
    Private routine to define the coupling matrix between the system and one lead.

    :param system: System to attach leads to.
    :param boundary: List of indices of atoms where we attach the lead.
    :param t: Hopping amplitude of lead.
    :return: Coupling matrix between beginning of lead and system
    """

    boundary_positions = system.motif[boundary, :]
    boundary_positions[:, 3] = boundary
    boundary_positions = boundary_positions[boundary_positions[:, 1].argsort()]

    coupling_matrix = sp.lil_matrix((len(boundary), system.basisdim), dtype=np.complex_)

    index_to_matrix_array = [0]
    for index in range(system.natoms - 1):
        species = int(system.motif[index, 3])
        index_to_matrix_array.append(index_to_matrix_array[-1] + int(system.norbitals[species]))

    for i, row in enumerate(boundary_positions):
        index = int(row[3])
        matrix_index = index_to_matrix_array[index]
        species = int(system.motif[index, 3])
        norbitals = int(system.norbitals[species])
        coupling_matrix[i, matrix_index:matrix_index + norbitals] = t

    coupling_matrix = coupling_matrix.tocsc()

    return coupling_matrix


def __lead_selfenergy(energy: float, lead_unit_cell_block: sp.csr_matrix, t: float, delta: float = 1E-7, threshold: float = 1E-4, mixing: float = 0.6):
    """
    Routine to compute the selfenergy of one leaf at a given energy. 

    :param energy: Energy where we evaluate the selfenergy.
    :param lead_unit_cell_block: Hamiltonian block corresponding to the unit cell of the lead.
    :param t: Value of hopping of lead.
    :param delta: Imaginary part of lead. Usually infinitesimal, its sign defined retarded or advanced lead. Defaults to 1E-7.
    :param threshold: Threshold to stop self-consistency of selfenergy. Defaults to 1E-4.
    :param mixing: Mixing parameter to achieve convergence faster.
    :return: Selfenergy matrix evaluated at energy+i*delta
    """

    selfenergy = np.zeros(lead_unit_cell_block.shape, dtype=np.complex_)

    notConverged = True
    old_selfenergy = np.zeros(lead_unit_cell_block.shape, dtype=np.complex_)
    old2_selfenergy = np.zeros(lead_unit_cell_block.shape, dtype=np.complex_)
    while notConverged:
        old2_selfenergy = old_selfenergy
        old_selfenergy = selfenergy
        new_selfenergy = t**2*np.linalg.inv((energy + 1j*delta)*np.eye(lead_unit_cell_block.shape[0]) - lead_unit_cell_block - selfenergy)
        selfenergy = old_selfenergy * (1 - mixing) + mixing * new_selfenergy
        maxNorm = np.abs(selfenergy - old_selfenergy).max()
        maxNorm2 = np.abs(selfenergy - old2_selfenergy).max()
        if maxNorm < threshold and maxNorm2 < threshold:
            notConverged = False
    
    return selfenergy


def __device_green_function(energy: float, device_hamiltonian: list, lead_unit_cell_block_L: sp.csr_matrix, lead_unit_cell_block_R: sp.csr_matrix, t: float, delta: float = 1E-7):
    """
    Routine to compute the Green's function of the device at energy + i*delta.

    :param energy: Energy where we evaluate the Green's function.
    :param device_hamiltonian: List with the Fock matrices of the device.
    :param lead_unit_cell_block_L: Block matrix to compute selfenergy of left lead.
    :param lead_unit_cell_block_R: Block matrix to compute selfenergy of right lead.
    :param t: Value of hopping of leads.
    :param delta: Broadening. Defaults to 1E-7.
    :return: Green's function of the device evaluated at energy + i*delta.
    """

    device_dim = device_hamiltonian[0].shape[0]
    
    lead_selfenergy_L = __lead_selfenergy(energy, lead_unit_cell_block_L, t, delta=delta)
    extended_lead_selfenergy_L = sp.lil_matrix((device_dim, device_dim), dtype=np.complex_)
    extended_lead_selfenergy_L[:lead_selfenergy_L.shape[0], :lead_selfenergy_L.shape[0]] = lead_selfenergy_L

    lead_selfenergy_R = __lead_selfenergy(energy, lead_unit_cell_block_R, t, delta=delta)
    extended_lead_selfenergy_R= sp.lil_matrix((device_dim, device_dim), dtype=np.complex_)
    extended_lead_selfenergy_R[(device_dim - lead_selfenergy_R.shape[0]):, (device_dim - lead_selfenergy_R.shape[0]):] = lead_selfenergy_R

    extended_lead_selfenergy_L = extended_lead_selfenergy_L.tocsc()
    extended_lead_selfenergy_R = extended_lead_selfenergy_R.tocsc()

    device_h = sp.csc_matrix((device_dim, device_dim), dtype=np.complex_)
    for h in device_hamiltonian:
        device_h += h

    green_device = sp.linalg.inv((energy + 1j*delta)*sp.eye(device_dim, format="csc") - device_h - extended_lead_selfenergy_L - extended_lead_selfenergy_R)

    return green_device.tocsr()


def ldos(result: Spectrum, atom_index: int, energy: float, delta: float = 1E-4):
    """
    Routine to compute the local density of states at a given energy.

    :param result: Spectrum object with the results from the diagonalization.
    :param atom_index: Index of the atom where we want to compute the LDOS.
    :param energy: Value of energy where we evaluate the LDOS.
    :param delta: Value of energy broadening. Defaults to 1E-4.
    :return: Value of LDOS.
    """

    pass


def integrated_ldos(result: Spectrum, atom_index: int, minE: float, maxE: float):
    """
    Routine to compute the local density of states integrated on a specified energy window.

    :param result: Spectrum object with the result of the diagonalization of the system.
    :param atom_index: Index of the atom where we evaluate the integrated LDOS.
    :param minE: Minimum value of energy window.
    :param maxE: Maximum value of energy window.
    :return: Integrated LDOS evaluated on atom_index.
    """

    index_to_matrix_array = [0]
    for index in range(result.system.natoms - 1):
        species = int(result.system.motif[index, 3])
        index_to_matrix_array.append(index_to_matrix_array[-1] + result.system.norbitals[species])

    init_index = int(index_to_matrix_array[atom_index])
    species_index = int(result.system.motif[atom_index, 3])
    final_index = init_index + int(result.system.norbitals[species_index])

    ildos = 0
    for j in range(result.eigen_energy.shape[1]):
        for i in range(result.eigen_energy.shape[0]):
            eigval = result.eigen_energy[i, j]
            if eigval > maxE or eigval < minE:
                continue
            eigvec = result.eigen_states[j, :, i]
            eigvec = eigvec[init_index:final_index]
            ildos += np.vdot(eigvec, eigvec)
    
    return ildos



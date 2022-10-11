# Module containing routines for computation of observables

import sys
from tightbinder.result import Spectrum
from tightbinder.system import System
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from typing import Tuple, List

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


def dos(result: Spectrum, energy: float = None, delta: float = 0.1, npoints: int = 200) -> Tuple(List[float], List[float]):
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


def dos_kpm(system: System, energy: float = None, npoints: int = 200, nmoments: int = 30, r: int = 10) -> float:
    """ 
    Routine to compute the density of states using the Kernel Polynomial Method.
    Intended to be used with supercells and k=0. 
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
    """ Routine to calculate the Jackson kernel to improve the KPM calculations """
    g = [((nmoments - n + 1)*np.cos(np.pi*n/(nmoments + 1)) +
          np.sin(np.pi*n/(nmoments + 1))/np.tan(np.pi*n/(nmoments + 1)))/(nmoments + 1) for n in range(1, nmoments)]
    g = [1] + g  # n = 0 gives NaN instead of 1
    
    return g


def chebyshev_polynomial_values(n: int, e: float) -> np.ndarray:
    """ Chebyshev polynomial of order n on value e. """
    polynomials = [1, e]
    if n >= 2:
        for i in range(2, n):
            polynomials.append(2*e*polynomials[i - 1] - polynomials[i - 2]) 
    return np.array(polynomials)


def compute_dos_momentum(n: int, h, r: int) -> np.ndarray:
    """ Routine to compute the n-th momentum associated to a Chebyshev polynomial expansion
    of the density of states. Instead of computing the whole trace of h, we perform the stochastic
    evaluation of the trace over a small set of random vectors. """
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


def plot_dos(result: Spectrum, npoints=100, delta=0.3, ax=None) -> None:
    """ Routine to plot the density of states using the retarded Green function """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    density, energies = dos(result, delta=delta, npoints=npoints)

    ax.plot(energies, density)
    ax.set_xlabel(r'E (eV)')
    ax.set_ylabel(r'DOS')

    return density, energies


def plot_dos_kpm(system: System, npoints=200, nmoments: int = 100, r: int = 10, ax=None):
    """ Routine to plot the density of states using the retarded Green function """
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
    """ Routine to compute the Fermi energy from the density of states
    integrating up to the number of electrons in the system """

    # First normallize to basisdim (maximum number of states)
    area = np.trapz(dos, energy)
    dos = dos / area * system.basisdim
    # Then search
    for i in range(len(dos)):
        nstates = np.trapz(dos[:i], energy[:i])
        if nstates > system.filling:
            break
    
    return energy[i - 1]


def compute_projector_momentum(nmoments: int, energy: int):
    """ Routine to compute the momentum from the KPM expansion of the ground state projector """

    first_moment = [1 - np.arccos(energy)/np.pi]
    moments = [-2*np.sin(m*np.arccos(energy))/(m*np.pi) for m in np.arange(1, nmoments)]
    moments = first_moment + moments

    return np.array(moments)

def ground_state_projector(nmoments: int, h: np.ndarray, energy: int):
    """ Routine to compute the ground state projector using the KPM """

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


def restricted_density_matrix(system: System, partition: list, nmoments: int = 100, npoints: int = 200, nmoments_dos: int = 300, r: int = 10):
    """ Routine to compute the one-particle density matrix, restricted to one spatial partition of the system. First it estimates the Fermi
    energy from the DOS computed with the KPM, and then uses it to compute the reduced density matrix """

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

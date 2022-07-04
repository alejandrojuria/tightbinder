# Module containing routines for computation of observables

import sys
from tightbinder.result import Spectrum
from tightbinder.system import System
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp

def _retarded_green_function(w: float, e: float, delta: float) -> complex:
    """ Routine to compute the retarded Green function, to be used to obtain
    the DOS """
    green = 1./(w - e + 1j*delta)
    return green


def dos(result: Spectrum, energy: float = None, delta: float = 0.1, npoints: int = 200) -> list[float]:
    """ Routine to compute the density of states from a Result object.
    :param result: Result object
    :param delta: Delta broadening, defaults to 0.01 """
    nkpoints = result.eigen_energy.shape[1]
    eigval = result.eigen_energy.reshape(-1, )
    if not energy:
        emin, emax = np.min(result.eigen_energy), np.max(result.eigen_energy)
        energies = np.linspace(emin, emax, npoints, endpoint=True)
    else:
        energies = [energy]

    energies = np.array(energies)
    dos = (np.sum(-np.imag(_retarded_green_function(energies, eigval, delta))/np.pi)
                /(nkpoints*result.system.unit_cell_area))

    return dos, energies


def dos_kpm(system: System, energy: float = None, npoints: int = 200, nmoments: int = 30, r: int = 10) -> float:
    """ Routine to compute the density of states using the Kernel Polynomial Method.
    Intended to be used with supercells and k=0. """

    h = system.hamiltonian_k([[0., 0., 0.]])
    # h spectrum has to be between -1 and 1
    h_norm = np.linalg.norm(h, ord=np.inf)
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


def jackson_kernel(nmoments: int) -> list[float]:
    """ Routine to calculate the Jackson kernel to improve the KPM calculations """
    g = [((nmoments - n + 1)*np.cos(np.pi*n/(nmoments + 1)) +
          np.sin(np.pi*n/(nmoments + 1))/np.tan(np.pi*n/(nmoments + 1)))/(nmoments + 1) for n in range(1, nmoments)]
    g = [1] + g  # n = 0 gives NaN instead of 1
    
    return g


def chebyshev_polynomial_operators(n: int, h: np.ndarray) -> list[np.ndarray]:
    """ Chebyshev polynomial of order n of an operator h """
    
    polynomials = [np.eye(h.shape[0], h.shape[1]), h]
    if n >= 2:
        for i in range(2, n):
            polynomials.append(2*np.dot(h, polynomials[i - 1]) - polynomials[i - 2]) 
    return polynomials


def chebyshev_polynomial_values(n: int, e: float) -> np.ndarray:
    """ Chebyshev polynomial of order n on value e. """
    polynomials = [1, e]
    if n >= 2:
        for i in range(2, n):
            polynomials.append(2*e*polynomials[i - 1] - polynomials[i - 2]) 
    return np.array(polynomials)


def compute_dos_momentum(n: int, h: np.ndarray, r: int) -> np.ndarray:
    """ Routine to compute the n-th momentum associated to a Chebyshev polynomial expansion
    of the density of states. Instead of computing the whole trace of h, we perform the stochastic
    evaluation of the trace over a small set of random vectors. """
    d = h.shape[0]

    # Generate sample of random vectors and normalize them
    vectors = np.random.uniform(-1, 1, (r, d))
    norm = np.sqrt(np.diag(np.dot(vectors, vectors.T)))
    vectors = vectors / norm[:, None]

    polynomials = [np.eye(h.shape[0], h.shape[1]), h]
    if n >= 2:
        for i in range(2, n):
            polynomials.append(2*np.dot(h, polynomials[i - 1]) - polynomials[i - 2]) 

    # Evaluate each momentum
    momentum = []
    operators = [np.eye(h.shape[0], h.shape[1]), h]
    momentum.append(np.sum(np.dot(vectors, np.dot(operators[0], vectors.T)))/(r*d))
    momentum.append(np.sum(np.dot(vectors, np.dot(operators[1], vectors.T)))/(r*d))
    for i in range(2, n):
        operator = 2*np.dot(h, operators[1]) - operators[0]
        mu = np.sum(np.diag(np.dot(vectors, np.dot(operator, vectors.T))))/(r*d)
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

    first_moment = [1 - 1./np.arccos(energy)]
    moments = [-2*np.sin(m*np.arccos(energy))/(m*np.pi) for m in np.arange(1, nmoments)]
    moments = first_moment + moments

    return np.array(moments)

def ground_state_projector(nmoments: int, h: np.ndarray, energy: int):
    """ Routine to compute the ground state projector using the KPM """

    h_norm = np.linalg.norm(h)
    h /= h_norm
    energy /= h_norm

    moments = compute_projector_momentum(nmoments, energy)
    jackson = jackson_kernel(nmoments)
    moments = moments * jackson
    operators = [np.eye(h.shape[0], h.shape[1]), h]
    projector = moments[0]*operators[0] + moments[1]*operators[1]
    for i in range(2, nmoments):
        operator = 2*np.dot(h, operators[1]) - operators[0]
        projector += moments[i] * operator
        operators[0] = operators[1]
        operators[1] = operator

    return projector


def restricted_density_matrix(system: System, partition: list, nmoments: int = 100, r: int = 10):
    """ Routine to compute the one-particle density matrix, restricted to"""
    
    h = system.hamiltonian_k([[0., 0., 0.]])

    # Compute first fermi energy from DOS
    print("Computing DOS...")
    dos, energy = dos_kpm(system, nmoments=nmoments, r=r)
    efermi = fermi_energy(dos, energy, system)
    print(f"Fermi energy at: {efermi}")

    # Compute then ground state projector
    print("Computing ground state projector...")
    projector = ground_state_projector(nmoments, h, efermi)

    # Finally restrict projector to desired positions
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

    density_matrix = projector[np.ix_(orbitals, orbitals)]

    return density_matrix

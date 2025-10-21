"""
Module containing routines for computation of observables.
"""
 

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Union, List, Tuple
from copy import deepcopy

from tightbinder.result import Spectrum
from tightbinder.system import System
from tightbinder.models import SlaterKoster


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
                /(nkpoints))
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
    h = system.hamiltonian_k([0., 0., 0.])
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
    densities = np.array(densities).real * h_norm
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


class TransportDevice:
    def __init__(self, system: System, left_lead: Union[List, np.ndarray], right_lead: Union[List, np.ndarray],
                 period: Union[List, float], mode: str = "default") -> None:
        """
        Constructor of TransportDevice class to model a transport setup.

        :param system: System to compute transmission
        :param left_lead: Unit cell of left lead. Array or list where each row contains the position
            and chemical species of each atom of the lead.
        :param right_lead: Same as left lead.
        :param period: Distance between consecutive lead unit cells. If given one value, it is used
            for both right and left leads. If given a list with two values, the first one specifies left
            period, and second one the right period.
        :param mode: Either "default" or "direct". Default mode uses the SK configuration to establish the
            system-leads bonds, the intra-lead and the lead-lead bonds. Intended to be used with crystalline
            or quasicrystalline situations (neighbour based search).
            Direct mode instead connects the leads to the system straight: each atom of the lead has only
            one bond (the first found) to the system. The intra-lead and lead-lead bonds are determined 
            using the SK configuration. Intended to be used with amorphous systems (radius based search).
        """
         
        
        if not left_lead.size or not right_lead.size:
            raise ValueError("Must provide a non-empty list for leads")

        if type(period) != list:
            period = [period, period]
        else:
            if len(period) != 2:
                raise ValueError("period can hold only two values, [left_period, right_period]")
        
        if mode not in ["default", "direct"]:
            raise ValueError("Invalid mode. Must be either 'default' or 'direct'.")
        
        self.system = system
        self.left_lead = left_lead
        self.right_lead = right_lead
        self.period = period
        self.mode = mode
        self.bonds = []
        
        device, left_lead, right_lead = self.__attach_leads()

        self.device_h = device
        self.left_lead_h = left_lead
        self.right_lead_h = right_lead


    def transmission(self, minE: float, maxE: float, npoints: int = 100, 
                    delta: float = 1E-7, method: str = "LS") -> Tuple[List, List]:
        """
        Function to compute the transmission function T(E) of a system. One has to specify the unit
        cell of each lead in terms of the positions and chemical species. Only two terminal setups are allowed. 
        The lead-system coupling and the lead-lead coupling are computed using the corresponding Slater-Koster amplitudes.

        :param minE: Minimum value of energy window where the transmission is computed.
        :param maxE: Maximum value of energy window.
        :param npoints: Sampling of energy window. Defaults to 100.
        :param delta: Value of broadening used in Green's functions. Defaults to 1E-7.
        :param method: Method to convergence the self-energies of the leads. Options are 'LS' (Lopez Sancho method)
            and 'LM' (Linear Mixing). Defaults to 'LS'.
        :return: List of transmission values and energy window.
        """


        device_dim = self.device_h.shape[0]
        energies = np.linspace(minE, maxE, npoints)
        currents = []
        extended_lead_selfenergy_L = sp.lil_matrix((device_dim, device_dim), dtype=np.complex128)
        extended_lead_selfenergy_R = sp.lil_matrix((device_dim, device_dim), dtype=np.complex128)

        for energy in energies:
            lead_selfenergy_L = self.__lead_selfenergy(energy, self.left_lead_h[0], self.left_lead_h[1].transpose().conjugate(), delta, method=method)  
            coupling_L = 1j*(lead_selfenergy_L - lead_selfenergy_L.transpose().conjugate())   
            extended_lead_selfenergy_L[:lead_selfenergy_L.shape[0], :lead_selfenergy_L.shape[0]] = coupling_L

            lead_selfenergy_R = self.__lead_selfenergy(energy, self.right_lead_h[0], self.right_lead_h[1], delta, method=method)
            coupling_R = 1j*(lead_selfenergy_R - lead_selfenergy_R.transpose().conjugate())
            extended_lead_selfenergy_R[(device_dim - lead_selfenergy_R.shape[0]):, (device_dim - lead_selfenergy_R.shape[0]):] = coupling_R

            device_green = self.__device_green_function(energy, self.device_h, lead_selfenergy_L, lead_selfenergy_R, delta)

            T = (extended_lead_selfenergy_L @ device_green.transpose().conjugate() @ extended_lead_selfenergy_R @ device_green).trace()
            currents.append(T.real)

        return (currents, energies)


    def conductance(self, delta: float = 1E-7, method: str = "LS") -> float:
        """
        Routine to compute the conductance at zero temperature in equilibrium.

        :param delta: Value of broadening used in Green's functions. Defaults to 1E-7.
        :param method: Method to convergence the self-energies of the leads. Options are 'LS' (Lopez Sancho method)
            and 'LM' (Linear Mixing). Defaults to 'LS'.
        :return: Value of conductance.
        """

        self.system.initialize_hamiltonian()
        spectrum = self.system.solve()
        fermi_energy = spectrum.calculate_fermi_energy(self.system.filling)
        G = self.transmission(fermi_energy, fermi_energy, 1, delta, method)

        return G[0][0]
            

    def __find_direct_bonds(self, first_group: Union[List, np.ndarray], second_group: Union[List, np.ndarray], r: float) -> List:
        """
        Private method to determine the bonds from a first group of atoms to a second. 
        For each atom of the first group, it determines only the first bond to an atom of the
        second group. To be used by the direct mode of the transmission routine.

        :param first_group: List of atomic positions.
        :param second_group: List of atomic positions.
        :param r: Cutoff for bond search.
        :return: List of direct bonds from first group to second group of atoms.
        """

        first_group = np.array(first_group)
        second_group = np.array(second_group)
        if self.system.bravais_lattice is not None:
            cells = [self.system.bravais_lattice[0], np.array([0., 0., 0.]), -self.system.bravais_lattice[0]]
        else:
            cells = [np.array([0., 0., 0.])]
        bonds = []
        it = 0
        for i, atom in enumerate(first_group):
            for cell in cells:
                final_atoms = np.copy(second_group)[:, :3] + cell - atom[:3] 
                distances = np.linalg.norm(final_atoms, axis=1)
                indices = np.argsort(distances)
                for index in indices:
                    if distances[index] < r:
                        bonds.append([i, index, cell, "1"])
        
        return bonds


    def __attach_leads(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Private method to extend the Hamiltonian to include a part of the leads. It assumes that the system has
        rectangle shape. 

        :param system: System to attach leads to.
        :param left_leaf: List of positions and chemical species of atoms of the left lead.
        :param right_boundary: List of positions and chemical species of atoms of the right lead.
        :param period: Distance between consecutive lead unit cells. If given one value, it is used
            for both right and left leads. If given a list with two values, the first one specifies left
            period, and second one the right period.
        :param mode: Lead-system connection mode. Either "default" or "direct".
        :return: (Hd,[Hl, Vl],[Hr, Vr]) Hd = Hamiltonian of device with leads attached.
            [Hl, Vl] = Left lead hamiltonian and coupling. [Hr, Vr] = Right lead hamiltonian and coupling.
        """


        max_nn = max(int(value) for value in self.system.configuration["SKAmplitudes"].keys())
        device_motif = np.concatenate((self.left_lead, self.system.motif, self.right_lead))
        model_device = deepcopy(self.system)
        model_device.matrix_type = "sparse"
        model_device.motif = device_motif
        if self.mode == "default":
            model_device.initialize_hamiltonian()
        else:
             # Bonds within left lead unit cell
            left_lead = deepcopy(self.system)
            left_lead.motif = self.left_lead
            left_lead.find_neighbours(mode="minimal", nn=max_nn)
            fnn = left_lead.compute_first_neighbour_distance()

            # First obtain bonds between left lead and system
            disp = len(self.left_lead)
            left_bonds = self.__find_direct_bonds(self.left_lead, self.system.motif, fnn * 1.3)
            left_bonds = [[i, j + disp, cell, nn] for [i, j, cell, nn] in left_bonds]
            reversed_bonds = [[j, i, -cell, nn] for [i, j, cell, nn] in left_bonds]
            left_bonds = left_bonds + reversed_bonds + left_lead.bonds

            # Bonds within right lead
            right_lead = deepcopy(self.system)
            right_lead.motif = self.right_lead
            right_lead.find_neighbours(mode="minimal", nn=max_nn)
            fnn = right_lead.compute_first_neighbour_distance()

            # Obtain bonds between right lead and system
            disp = len(self.left_lead) + len(self.system.motif)
            right_bonds = self.__find_direct_bonds(self.right_lead, self.system.motif, fnn * 1.3)
            right_bonds = [[i + disp, j + len(self.left_lead), cell, nn] for [i, j, cell, nn] in right_bonds]
            reversed_bonds = [[j, i, -cell, nn] for [i, j, cell, nn] in right_bonds]
            right_bonds = right_bonds + reversed_bonds + [[i + disp, j + disp, cell, nn] for [i, j, cell, nn] in right_lead.bonds]

            self.system.initialize_hamiltonian()
            system_bonds = [[i + len(self.left_lead), j + len(self.left_lead), cell, nn] for [i, j, cell, nn] in self.system.bonds]
            model_device.bonds = system_bonds + left_bonds + right_bonds
            model_device.initialize_hamiltonian(find_bonds=False)

        device_hamiltonian = sp.csc_matrix(model_device.hamiltonian[0].shape, dtype=np.complex128)
        for h in model_device.hamiltonian:
            device_hamiltonian += h

        self.bonds.append(model_device.bonds)

        left_lead_H  = self.__lead_hamiltonian(self.left_lead, -self.period[0])
        right_lead_H = self.__lead_hamiltonian(self.right_lead, self.period[1])

        return (device_hamiltonian, left_lead_H, right_lead_H)


    def __lead_hamiltonian(self, lead: np.ndarray, period: float) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
        """
        Routine to attach together two lead unit cells to extract the cell Hamiltonian,
        and the coupling between cells.

        :param lead: Array with atomic positions and species of the lead.
        :param period: Norm of Bravais vector of the lead.
        :return: [H, V] Hamiltonian of unit cell and coupling between unit cells.2
        """

        displaced_lead = np.copy(lead)
        displaced_lead[:, :3] += np.array([period, 0, 0])
        if period > 0:
            left_lead_motif = np.concatenate((lead, displaced_lead))
        else:
            left_lead_motif = np.concatenate((displaced_lead, lead))
        model_lead = deepcopy(self.system)
        model_lead.motif = left_lead_motif
        model_lead.matrix_type = "sparse"
        if self.mode == "direct":
            max_nn = max(int(value) for value in self.system.configuration["SK amplitudes"].keys())
            model_lead.find_neighbours(mode="minimal", nn=max_nn)
            findBonds = False
        else:
            findBonds = True
        model_lead.initialize_hamiltonian(find_bonds=findBonds)
        lead_total_hamiltonian = sp.csc_matrix(model_lead.hamiltonian[0].shape, dtype=np.complex128)
        for h in model_lead.hamiltonian:
            lead_total_hamiltonian += h

        lead_dim = model_lead.basisdim // 2
        lead_coupling = lead_total_hamiltonian[:lead_dim, lead_dim:]
        lead_hamiltonian = lead_total_hamiltonian[:lead_dim, :lead_dim]

        self.bonds.append(model_lead.bonds)

        return (lead_hamiltonian, lead_coupling)


    def __lead_selfenergy(self, energy: float, lead_hamiltonian: sp.csr_matrix, lead_coupling: sp.csr_matrix, delta: float = 1E-7, 
                          threshold: float = 1E-10, mixing: float = 0.5, method: str = "LS"):
        """
        Routine to compute the selfenergy of one leaf at a given energy. 

        :param energy: Energy where we evaluate the selfenergy.
        :param lead_hamiltonian: Hamiltonian block corresponding to the unit cell of the lead.
        :param lead_coupling: Coupling between lead unit cells.
        :param delta: Imaginary part of lead. Usually infinitesimal, its sign defined retarded or advanced lead. Defaults to 1E-7.
        :param threshold: Threshold to stop self-consistency of selfenergy. Defaults to 1E-4.
        :param mixing: Mixing parameter to achieve convergence faster.
        :param method: Method to convergence the self-energies of the leads. Options are 'LS' (Lopez Sancho method)
            and 'LM' (Linear Mixing). Defaults to 'LS'.
        :return: Selfenergy matrix evaluated at energy+i*delta
        """

        notConverged = True
        identity = np.eye(lead_hamiltonian.shape[0])

        if method == "LS":
            s = lead_hamiltonian.todense()
            e = lead_hamiltonian.todense()
            alpha = lead_coupling.todense()
            beta = alpha.transpose().conjugate()
            
            while notConverged:
                g = np.linalg.inv((energy + 1j*delta)*identity - e)
                factor = alpha @ g @ beta
                s += factor
                e += factor + beta @ g @ alpha
                alpha = alpha @ g @ alpha
                beta = beta @ g @ beta
                norm = np.linalg.norm(alpha)
                if norm < threshold:
                    notConverged = False

            g = np.linalg.inv((energy + 1j*delta)*identity - s)
            selfenergy = lead_coupling @ g @ lead_coupling.transpose().conjugate()

        elif method == "LM":
            selfenergy = sp.csc_matrix(lead_hamiltonian.shape, dtype=np.complex128)

            old_selfenergy = sp.csc_matrix(lead_hamiltonian.shape, dtype=np.complex128)
            old2_selfenergy = sp.csc_matrix(lead_hamiltonian.shape, dtype=np.complex128)
            identity = sp.eye(lead_hamiltonian.shape[0], format="csc")
            while notConverged:
                old2_selfenergy = old_selfenergy
                old_selfenergy = selfenergy
                new_selfenergy = (lead_coupling @ 
                                sp.linalg.inv((energy + 1j*delta)*identity - lead_hamiltonian - selfenergy) @ 
                                lead_coupling.transpose().conjugate())
                selfenergy = old_selfenergy * (1 - mixing) + mixing * new_selfenergy 
                maxNorm = sp.linalg.norm(selfenergy - old_selfenergy)
                maxNorm2 = sp.linalg.norm(selfenergy - old2_selfenergy)
                if maxNorm < threshold and maxNorm2 < threshold:
                    notConverged = False
        
        else:
            raise ValueError("method must be either 'LS' or 'LM'.")

        return sp.csc_matrix(selfenergy)


    # def __lead_selfenergy(energy: float, lead_hamiltonian: sp.csr_matrix, lead_coupling: sp.csr_matrix, delta: float = 1E-7, threshold: float = 1E-3, mixing: float = 0.5):
    #     """
    #     Routine to compute the selfenergy of one leaf at a given energy. Implements a Pulay mixer to achieve self-consistency.

    #     :param energy: Energy where we evaluate the selfenergy.
    #     :param lead_hamiltonian: Hamiltonian block corresponding to the unit cell of the lead.
    #     :param lead_coupling: Coupling between lead unit cells.
    #     :param delta: Imaginary part of lead. Usually infinitesimal, its sign defined retarded or advanced lead. Defaults to 1E-7.
    #     :param threshold: Threshold to stop self-consistency of selfenergy. Defaults to 1E-4.
    #     :param mixing: Mixing parameter to achieve convergence faster.
    #     :return: Selfenergy matrix evaluated at energy+i*delta
    #     """

    #     selfenergy = sp.csc_matrix(lead_hamiltonian.shape, dtype=np.complex128)
    #     residuals, selfenergies = [], []

    #     notConverged = True
    #     old_selfenergy = sp.csc_matrix(lead_hamiltonian.shape, dtype=np.complex128)
    #     identity = sp.eye(lead_hamiltonian.shape[0], format="csc")
    #     n = 3
    #     first_iterations = 10
    #     counter = 0

    #     while notConverged:
            
    #         new_selfenergy = (lead_coupling @ 
    #                           sp.linalg.inv((energy + 1j*delta)*identity - lead_hamiltonian - selfenergy) @ 
    #                           lead_coupling.transpose().conjugate())
            
            
    #         # #  Pulay mixing
    #         # Store first n elements
    #         selfenergies.append(selfenergy)
    #         residuals.append(new_selfenergy - selfenergy)
    #         if counter < first_iterations:
    #             if counter >= n:
    #                 selfenergies.pop(0)
    #                 residuals.pop(0)
    #             selfenergy = mixing * selfenergy + (1 - mixing) * new_selfenergy
    #             norm = sp.linalg.norm(residuals[-1])
    #         # After storing first iterations, perform mixing
    #         else:
    #             selfenergies.pop(0)
    #             residuals.pop(0)
            
    #             residual_matrix = np.zeros([len(residuals) + 1, len(residuals) + 1], dtype=np.complex128)
    #             for i in range(len(residuals)):
    #                 ri = residuals[i].transpose().conjugate()
    #                 for j in range(len(residuals) - i):
    #                     rj = residuals[j]
    #                     if j == 0:
    #                         residual_matrix[i, i + j] = (ri @ rj).trace() / 2
    #                     else:
    #                         residual_matrix[i, i + j] = (ri @ rj).trace()
    #             residual_matrix += np.conjugate(np.transpose(residual_matrix))
    #             residual_matrix[:, -1] = 1
    #             residual_matrix[-1, :] = 1
    #             residual_matrix[-1, -1] = 0

    #             independent_vector = np.zeros(len(residuals) + 1)
    #             independent_vector[-1] = 1

    #             coefs = np.linalg.solve(residual_matrix, independent_vector)

    #             selfenergy = sp.csc_matrix(lead_hamiltonian.shape, dtype=np.complex128)
    #             for i, coef in enumerate(coefs[:-1]):
    #                 selfenergy += coef  * (selfenergies[i] + 0.5 * residuals[i])

    #             norm = np.sqrt(abs(coefs[-1]))
                    
    #         counter += 1
    #         if norm < threshold:
    #             notConverged = False
                
        
    #     return selfenergy


    def __device_green_function(self, energy: float, device: list, left_self_energy: sp.csc_matrix, right_self_energy: sp.csc_matrix, delta: float = 1E-7):
        """
        Routine to compute the Green's function of the device at energy + i*delta.

        :param energy: Energy where we evaluate the Green's function.
        :param device: List with the Fock matrices of the device.
        :param left_self_energy: Self-energy of left lead.
        :param right_self_energy: Self-energy of right lead.
        :param delta: Broadening. Defaults to 1E-7.
        :return: Green's function of the device evaluated at energy + i*delta.
        """

    
        device_dim = device.shape[0]
        
        extended_lead_selfenergy_L = sp.lil_matrix((device_dim, device_dim), dtype=np.complex128)
        extended_lead_selfenergy_L[:left_self_energy.shape[0], :left_self_energy.shape[0]] = left_self_energy

        extended_lead_selfenergy_R= sp.lil_matrix((device_dim, device_dim), dtype=np.complex128)
        extended_lead_selfenergy_R[(device_dim - right_self_energy.shape[0]):, (device_dim - right_self_energy.shape[0]):] = right_self_energy

        extended_lead_selfenergy_L = extended_lead_selfenergy_L.tocsc()
        extended_lead_selfenergy_R = extended_lead_selfenergy_R.tocsc()

        green_device = sp.linalg.inv((energy + 1j*delta)*sp.eye(device_dim, format="csc") - device - extended_lead_selfenergy_L - extended_lead_selfenergy_R)

        return green_device


    def visualize_device(self, ax: Axes = None, pbc: bool = False, ) -> None:
        """
        Routine to visualize the device. Intended to be used only with two dimensional (planar) systems.

        :param ax: Axis object to plot the device on.
        :param pbc: Boolean to toggle 3d plot of the device in case the system and the leads have PBC.
        """

        if ax is None:
            if pbc and self.system.boundary == "PBC":
                ax = plt.figure().add_subplot(projection='3d')
                # fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
            elif pbc and self.system.boundary == "OBC":
                    print("Warning: plot set to periodic but system is finite. Plot will be 2d.")
            else:
                fig, ax = plt.subplots(1, 1)


        # First plot bonds
        displaced_left_motif = np.copy(self.left_lead)
        displaced_left_motif[:, :3] -= [self.period[0], 0, 0]

        displaced_right_motif = np.copy(self.right_lead)
        displaced_right_motif[:, :3] += [self.period[1], 0, 0]

        left_lead = np.concatenate((displaced_left_motif, self.left_lead))
        right_lead = np.concatenate((self.right_lead, displaced_right_motif))

        system_motif = self.system.motif

        # Map into cilinder. z component is disregarded.
        if pbc:
            L = self.system.bravais_lattice[0][1]
            r = L/(2*np.pi)
            for n, atom in enumerate(left_lead):
                y = atom[1]
                left_lead[n, 1] = r*np.cos(2*np.pi*y/L)
                left_lead[n, 2] = r*np.sin(2*np.pi*y/L)
            for n, atom in enumerate(right_lead):
                y = atom[1]
                right_lead[n, 1] = r*np.cos(2*np.pi*y/L)
                right_lead[n, 2] = r*np.sin(2*np.pi*y/L)
        
        if not pbc:
            # Then plot atoms
            for atom in system_motif:
                ax.scatter(atom[0], atom[1], c="mediumseagreen", edgecolors="black", linewidths=1)
            
            for atom in self.left_lead:
                ax.scatter(atom[0], atom[1], c="orange", edgecolors="black", linewidths=1)
                ax.scatter(atom[0] - self.period[0], atom[1], c="moccasin", edgecolors="black", linewidths=1)

            for atom in self.right_lead:
                ax.scatter(atom[0], atom[1], c="orange", edgecolors="black", linewidths=1)
                ax.scatter(atom[0] + self.period[0], atom[1], c="moccasin", edgecolors="black", linewidths=1)
            
            total_motif = np.concatenate((self.left_lead, self.system.motif, self.right_lead), axis=0)
            for bond in self.bonds[0]:
                rx = [total_motif[bond[0], 0], total_motif[bond[1], 0]]
                ry = [total_motif[bond[0], 1], total_motif[bond[1], 1]]
                ax.plot(rx, ry, "k-", zorder=-1)
            
            for bond in self.bonds[1]:
                rx = [left_lead[bond[0], 0], left_lead[bond[1], 0]]
                ry = [left_lead[bond[0], 1], left_lead[bond[1], 1]]
                ax.plot(rx, ry, "k-", zorder=-1)
            
            for bond in self.bonds[2]:
                rx = [right_lead[bond[0], 0], right_lead[bond[1], 0]]
                ry = [right_lead[bond[0], 1], right_lead[bond[1], 1]]
                ax.plot(rx, ry, "k-", zorder=-1)

            ax.axis('equal')

        else:
            for atom in system_motif:
                ax.scatter(atom[0], r*np.cos(2*np.pi*atom[1]/L), r*np.sin(2*np.pi*atom[1]/L), c="mediumseagreen", edgecolors="black", linewidths=1)
            
            for atom in self.left_lead:
                ax.scatter(atom[0], r*np.cos(2*np.pi*atom[1]/L), r*np.sin(2*np.pi*atom[1]/L), c="orange", edgecolors="black", linewidths=1)
                ax.scatter(atom[0] - self.period[0], r*np.cos(2*np.pi*atom[1]/L), r*np.sin(2*np.pi*atom[1]/L), c="moccasin", edgecolors="black", linewidths=1)

            for atom in self.right_lead:
                ax.scatter(atom[0], r*np.cos(2*np.pi*atom[1]/L), r*np.sin(2*np.pi*atom[1]/L), c="orange", edgecolors="black", linewidths=1)
                ax.scatter(atom[0] + self.period[0], r*np.cos(2*np.pi*atom[1]/L), r*np.sin(2*np.pi*atom[1]/L), c="moccasin", edgecolors="black", linewidths=1)
            
            total_motif = np.concatenate((self.left_lead, self.system.motif, self.right_lead), axis=0)
            for bond in self.bonds[0]:
                rx = [total_motif[bond[1], 0], total_motif[bond[0], 0]]
                ry_org = [total_motif[bond[1], 1], total_motif[bond[0], 1]]
                ry = [r*np.cos(2*np.pi*value/L) for value in ry_org]
                rz = [r*np.sin(2*np.pi*value/L) for value in ry_org]
                ax.plot(rx, ry, rz, "k-", zorder=-1)
            
            for bond in self.bonds[1]:
                rx = [left_lead[bond[1], 0], left_lead[bond[0], 0]]
                ry = [left_lead[bond[1], 1], left_lead[bond[0], 1]]
                rz = [left_lead[bond[1], 2], left_lead[bond[0], 2]]
                ax.plot(rx, ry, rz, "k-", zorder=-1)
            
            for bond in self.bonds[2]:
                rx = [right_lead[bond[1], 0], right_lead[bond[0], 0]]
                ry = [right_lead[bond[1], 1], right_lead[bond[0], 1]]
                rz = [right_lead[bond[1], 2], right_lead[bond[0], 2]]
                ax.plot(rx, ry, rz, "k-")

            ax.view_init(elev=0)
            try:
                ax.set_aspect('equal')
            except NotImplementedError as e:
                print("Plotting the device in 3d requires matpltolib 3.7.0 or higher.")
                raise(e)
            ax.set_axis_off()



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



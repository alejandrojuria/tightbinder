# Module with routines to introduce disorder in a system such as vacancies
# or impurities. All this routines are intended to be used with supercells, although
# no explicit check is done on that.
# There can be found routines of two types:
# - Random: Unless one fixes the seed (np.seed), each call to this routines
# will have a different effect upon the system it is being used on
# - Listing: Instead of generating pseudorandom numbers to select the target atoms,
# alternatively we can specify which atoms

from multiprocessing.sharedctypes import Value
import numpy as np
from tightbinder.models import SlaterKoster
from tightbinder.system import System
import sys
import random


# ----------------------------- Random routines -----------------------------
def introduce_vacancies(system: System, probability=0.5):
    """ Routine to introduce vacancies in a system, i.e. to remove atoms. It is a
     statistical method, meaning that each atom in the motif has a probability of being removed
     or not. Thus each call to this method would generate a different structure.
     :param system: Instance of System class or subclass derived from it
      NB: Strictly speaking, this can be used on Crystal class as well
     :param probability: Parameter to specify probability of removing an atom;
            defaults to 0.5
    """
    prob_array = np.random.uniform(0, 1, system.natoms)
    remaining_atoms = np.where(prob_array > probability)
    system.motif = system.motif[remaining_atoms]

    return system


def introduce_impurities(system: System, energy=2, probability=0.5):
    """ Routine to introduce impurities in the system. These impurities are implemented
     via a change in the on-site energies of the randomly selected atoms. This routine
     randomly chooses atoms according to the specified probability, meaning that each call
     will generate a different distribution.
     NB: This routine will override ALL on-site energies for the selected atoms,
     in a multi-orbital scenario.
     NB2: Current implementation requires already having initialized the Hamiltonian
     to modify the corresponding matrix elements
     :param system: Instance of System class or subclass derived from it
     :param probability: Parameter to specify probability of selecting an atom as
      an impurity. Defaults to 0.5
     :param energy: Value of on-site energy of impurities. Default to 2
    """
    if system.hamiltonian is None:
        print("Error: Hamiltonian must be initialized before calling introduce_impurities routine")
        sys.exit(1)

    prob_array = np.random.uniform(0, 1, system.natoms)
    selected_atoms_as_impurities = np.where(prob_array > probability)[0]

    for i in selected_atoms_as_impurities:
        atom_interval = np.arange(system.norbitals*i, system.norbitals*(i+1))
        system.hamiltonian[0][atom_interval, atom_interval] = energy * np.ones(system.norbitals)

    return system


def amorphize(system: System, spread=0.1, distribution="uniform", planar=False):
    """ Routine to amorphize a crystalline system. This routine takes each atom and displaces
    it with respect to its original position by an amount given by a distribution. Thus
    by specifying the spread of the distribution and the distribution itself, we can
    achieve higher or lesser degrees of amorphization.
    By default it will into account the size of the supercell, so that any atom that moves outside
    of it goes to the other side.
    NB: This routine DOES NOT modify the hopping parameters according to the displacements,
    this has to be done in the class defining the system
    :param system: Instance of System class or any derived subclass
    :param spread: Maximum distance the atoms can be displaced. Given in units of
    first neighbours (spread=1 means totally random positions). Defaults to 0.1
    :param distribution: Either "uniform" or "gaussian". By default distribution is
    always uniform, U(0, spread). If we choose gaussian, then the spread acts as the
    variance of the distribution in the radial direction N(0, spread),
    the angles are still given by uniform dist.
    Also "gaussian_cartesian", which adds gaussian noise to each cartesian component directly
    :param planar: Optional parameter to enforce atom displacement happening only in the
    plane defining a 2D system. """

    first_neighbour_distance = system.compute_first_neighbour_distance()
    max_displacement = spread * first_neighbour_distance
    if distribution is "uniform":
        radius_array = np.random.uniform(0, max_displacement, system.natoms)
    elif distribution is "gaussian":
        radius_array = np.random.normal(0, max_displacement, system.natoms)
    elif distribution is "gaussian_cartesian":
        # To be rewritten, two return points for function are bad practice
        x = np.random.normal(0, spread, system.natoms)
        y = np.random.normal(0, spread, system.natoms)
        z = np.random.normal(0, spread, system.natoms)
        displacements = np.array([x, y, z]).T

        new_motif = np.asfarray(np.copy(system.motif))
        new_motif[:, :3] += displacements
        system.motif = new_motif

        return system
    else:
        print("Error: Incorrect distribution specified in amorphize, it has to be " +
              "either uniform or gaussian")
        sys.exit(1)
    phi_array = np.random.uniform(0, 2 * np.pi, system.natoms)
    if not planar:
        theta_array = np.random.uniform(0, np.pi, system.natoms)
    else:
        theta_array = np.ones(system.natoms) * np.pi/2

    x = radius_array * np.sin(theta_array) * np.cos(phi_array)
    y = radius_array * np.sin(theta_array) * np.sin(phi_array)
    z = radius_array * np.cos(theta_array)
    displacements = np.array([x, y, z]).T

    new_motif = np.asfarray(np.copy(system.motif))
    new_motif[:, :3] += displacements
    system.motif = new_motif

    return system


def alloy(system: System, *concentrations: float):
    """ Routine to alloy a material with two or more chemical species. In practice this means that each atom
    is reasigned to a random chemical species, while keeping its position.
    For two chemical species, is is only necessary to specify the concentration x of the first one, Ax B1-x.
    For n species, one has to specify n-1, so that An1 Bn2 Cn3 ... Mnn-1 """

    if len(concentrations) != system.species - 1:
        raise ValueError("Must give nspecies - 1 concentrations")

    natoms_species = [int(system.natoms*x) for x in concentrations]
    remaining_indices = np.arange(system.natoms)
    final_species = np.zeros([system.natoms])
    for index, natoms in enumerate(natoms_species):
        indices = np.random.choice(remaining_indices, natoms, replace=False)
        final_species[indices] = index
        remaining_indices = np.delete(remaining_indices, indices)
    
    final_species[remaining_indices] = len(concentrations)
    system.motif[:, 3] = final_species

    return system


# ----------------------------- Listing routines -----------------------------
def remove_atoms(system: System, indices):
    """ Routine to remove atoms from the system motif (i.e. create vacancies) according to
     a list of indices provided
      :param indices: List with indices of those atoms we want to remove, following the same
      ordering as those atoms in the motif
      :param system: Crystal class or subclass derived from it (typically System) """

    all_atoms = list(range(len(system.natoms)))
    remaining_atoms = [index for index in all_atoms if index not in indices]
    system.motif = system.motif[remaining_atoms]

    return system


def set_impurities(system: System, indices, energy=0.1):
    """ Routine to set impurities on the system on the atoms specified by the list provided
     :param system: System class or derived subclass. Must have Hamiltonian initialized
     :param indices: List with indices of those atoms we want to transform into impurities. The indices
     are referred to the order in the motif
     :param energy: On-site energy for the impurities. Defaults to 0.1 """
    if system.hamiltonian is None:
        print("Error: Hamiltonian must be initialized before calling introduce_impurities routine")
        sys.exit(1)

    for i in indices:
        atom_interval = np.arange(system.norbitals*i, system.norbitals*(i+1))
        system.hamiltonian[0][atom_interval, atom_interval] = energy * np.ones(system.norbitals)

    return system


def change_species(system: SlaterKoster, indices):
    """ Routine to change the species of the atoms of the motif according to 
    the array indices, which contains the new species for each atom of the motif.
    :param system: SlaterKoster object
    :param indices: List of indices denoting the chemical species of each atom. Must have length equal
    to system.natoms """

    if(len(indices) != system.natoms):
        raise ValueError("Indices array must have the same length as system.natoms")
    if np.any(indices >= system.species):
        raise ValueError('Indices array must not contain indices equal or higher than the number of species')
    if np.any(indices < 0):
        raise ValueError('Indices must be positive numbers')

    system.motif[:, 3] = indices

    return system







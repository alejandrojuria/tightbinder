"""
Modifications of the system such as strain or electric field.
"""

from shutil import ExecError
import numpy as np
from tightbinder.models import SlaterKoster
from tightbinder.system import System
from tightbinder.fileparse import mix_parameters

def apply_constant_electric_field(system: System, intensity: float, axis: str) -> System:
    """ 
    Routine to apply a constant electric field to a Hamiltonian, along the specified axis. Modifies
    on-site energies of the atoms of the system according to their positions.

    :param system: System object
    :param intensity: Amplitude of applied electric field
    :param axis: String
    :returns: System object 
    """
    
    pass

def strain(system: SlaterKoster, strain: float, n: int = 8) -> SlaterKoster:
    """ 
    Routine to simulate the effect of strain on a given system. Modifies all SK amplitudes following 
    a polynomial law to incorporate the effect of a lower bond length.

    :param system: SlaterKoster object
    :param strain: Value of strain, 0 <= strain < 1
    :param n: Power of
    :return: Strained SlaterKoster object. 
    """
    
    if strain >= 1 or strain < 0:
        raise ValueError('strain must be a number in [0, 1)')

    system.system_name = 'Strained' + system.system_name
    factor = (1 - strain)**(-n)
    for neighbour in system.configuration['SK amplitudes']:
        for species, coefs in system.configuration['SK amplitudes'][neighbour]:
            system.configuration['SK amplitudes'][neighbour][species] = [coef*factor for coef in coefs]

    return system

def saturate_bonds(system: SlaterKoster, onsite: float = 0, vss: float = 0, vsp: float = 0, edge_atoms: list = None) -> SlaterKoster:
    """ 
    Routine to saturate dangling bonds in materials by adding hydrogen atoms which are
    connected to the atoms at the edge of the system.
    The new species corresponds to hydrogen atoms (or equivalently some species with only s orbitals).
    Note that the specified hopping will be mixed with the hoppings provided for the other species.

    :param system: SlaterKoster model to saturate 
    :param onsite: Value of the onsite energy for H atoms.
    :param vss: Value of the SK amplitude for s-s orbitals. It will be mixed with the hopping of the
        species that participate. 
    :param vsp: Value of the SK amplitude for s-p orbitals. It will be mixed with the hopping of the
        species that participate. 
    :param edge_atoms: Optional list of atoms to which we connect the H atoms
    :return: Modified SlaterKoster object. 
    """

    if not system.hamiltonian:
        raise ExecError("System must be initialized first")

    if edge_atoms is None:
        edge_atoms = system.find_lowest_coordination_atoms()
    new_species = system.nspecies
    nn_distance = system.compute_first_neighbour_distance()
    motif_center = system.crystal_center()
    cell = [0., 0., 0.]
    for atom_index in edge_atoms:
        atom_position = system.motif[atom_index, :3]
        difference_position = atom_position - motif_center
        hydrogen_position = difference_position/np.linalg.norm(difference_position)*nn_distance + atom_position
        system.add_atom(hydrogen_position, new_species)
        system.add_bond(atom_index, system.natoms - 1, cell)
        system.add_bond(system.natoms - 1, atom_index, cell)
    
    if system.configuration["Spin"] == True:
        system.configuration["Filling"].append(1)
    else:
        system.configuration["Filling"].append(0.5)
    system.configuration["Species"] += 1
    system.configuration["Orbitals"].append(['s'])
    system.configuration["Onsite energy"].append([onsite])
    system.configuration["Spin-orbit coupling"].append(0)

    new_species = str(new_species) + str(new_species)
    amplitudes = [vss, vsp, 0, 0, 0, 0, 0, 0, 0, 0] # Hardcoded, maybe define some routine to perform this transformation
    system.configuration["SK amplitudes"]['1'][new_species] = amplitudes
    mix_parameters(system.configuration)

    return system

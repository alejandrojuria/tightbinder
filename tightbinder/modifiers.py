import numpy as np
from tightbinder.system import System

def apply_constant_electric_field(system: System, intensity, axis) -> System:
    """ Routine to apply a constant electric field to a Hamiltonian, along the specified axis. Modifies
    on-site energies of the atoms of the system according to their positions.
    :param system: System object
    :param intensity: double
    :param axis: String
    :returns: System object """
    
    pass

def strain(system: System, strain, n=8) -> System:
    """ Routine to simulate the effect of strain on a given system. Modifies all SK amplitudes following 
    a polynomial law to incorporate the effect of a lower bond length.
    :param system: System object
    :param strain: double, 0 <= strain < 1
    :param axis: String
    :return: System object """
    
    if strain >= 1 or strain < 0:
        raise ValueError('strain must be a number in [0, 1)')

    system.system_name = 'Strained' + system.system_name
    factor = (1 - strain)**(-n)
    for neighbour in system.configuration['SK amplitudes']:
        for species, coefs in system.configuration['SK amplitudes'][neighbour]:
            system.configuration['SK amplitudes'][neighbour][species] = [coef*factor for coef in coefs]

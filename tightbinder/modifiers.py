import numpy as np
from tightbinder.system import System

def apply_constant_electric_field(system, intensity, axis) -> System:
    """ Routine to apply a constant electric field to a Hamiltonian, along the specified axis. Modifies
    on-site energies of the atoms of the system according to their positions.
    :param system: System object
    :param intensity: double
    :param axis: String
    :returns: System object """
    
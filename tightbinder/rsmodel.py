import numpy as np
from system import System
import itertools


class RSmodel(System):
    """ Class to construct toy models, in which one sets the hoppings manually. This models
     are by default OBC; by setting one hopping between different unit cells it automatically becomes
     PBC. """
    def __init__(self, system_name=None, bravais_lattice=None, motif=None):
        super().__init__(system_name=system_name, bravais_lattice=bravais_lattice, motif=motif)
        self.hoppings = None
        self.boundary = "OBC"

    def add_atom(self, position, species=0):
        """ Method to add one atom from some numered species into a specific position.
        Parameters:
            array position: len(3)
            int species: Used to number the species. Defaults to 0 """
        atom = np.append(position, species)
        self.motif = np.concatenate([self.motif, atom], axis=0)

    def add_atoms(self, atoms, species=None):
        """ Method to add a list of atoms of specified species at some positions.
         Built on top of method add_atom.
         Parameters:
             matrix atoms: natoms x 3 (each row are the coordinates of an atom)
             list species: list of size natoms """
        if species is None:
            species = np.zeros(len(atoms))
        for n, atom in enumerate(atoms):
            self.add_atom(atom, species[n])

    def add_hopping(self, hopping, initial, final, cell=(0., 0., 0.)):
        """ Method to add a hopping between two atoms of the motif.
        NB: The hopping has a specified direction, from initial to final. Since the final
        Hamiltonian is computed taking into account hermiticity, it is not necessary to specify the hopping
        in the other direction.
         Parameters:
             complex hopping
             int initial, final: Indices of the atoms in the motif
             array cell: Bravais vector connecting the cells of the two atoms. Defaults to zero """
        hopping_info = [hopping, initial, final, cell]
        self.hoppings.append(hopping_info)

    def add_hoppings(self, hoppings, initial, final, cells=None):
        """ Same method as add_hopping but we input a list of hoppings at once.
        Parameters:
             list hoppings: list of size nhop
             list initial, final: list of indices
             matrix cells: Each row denotes the Bravais vector connecting two cells. Defaults to None """
        if cells is None:
            cells = np.zeros([len(hoppings), 3])
        else:
            cells = np.array(cells)
        for n, hopping in enumerate(hoppings):
            self.add_hopping(hopping, initial[n], final[n], cells[n, :])

    def initialize_hamiltonian(self):
        """ Method to set up the matrices that compose the Hamiltonian, either the Bloch Hamiltonian
        or the real-space one """
        for hopping in self.hoppings:
            cell = hopping[3]
            if np.linalg.norm(cell) != 0:
                self.boundary = "PBC"
                break

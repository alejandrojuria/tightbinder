# Implementation of System class, which incorporates both Hamiltonian and Crystal classes.
# Specifically, System includes the Crystal as a class attribute.
# The reasoning behind this, is to make available the crystal specific routines and attributes to the system,
# while being explicit about its ambit and without the crystal being a separate entity.
# As for the Hamiltonian, its functionality is directly available from System due to the inheritance.
# Base Hamiltonian class implementation. It will be used for the other hamiltonians to derive
# from it
# System has the basic functionality for the other models to derive from it.

import numpy as np
from crystal import Crystal
import result
from utils import angle_between
import sys


class System(Crystal):
    def __init__(self, system_name=None, bravais_lattice=None, motif=None, crystal=None):
        if crystal is None:
            super().__init__(bravais_lattice, motif)
        else:
            assert type(crystal) == Crystal
            if bravais_lattice is not None or motif is not None:
                print("Warning: Provided values for Bravais lattice or motif are disregarded" +
                      "in presence of crystal object")
            super().__init__(crystal.bravais_lattice, crystal.motif)
        self.system_name = system_name
        self._norbitals = None
        self._basisdim = None

    @property
    def norbitals(self):
        return self._norbitals

    @norbitals.setter
    def norbitals(self, norbitals):
        assert type(norbitals) == int
        self._norbitals = norbitals

    def reduce(self, **ncells):
        """ Routine to reduce the dimensionality of the System object along the specified
         directions, by repeating unit cells along those directions until a given size
         (number of unit cells) is reached. Thus we make the original system finite along those
         directions.
         Input: int n1, n2 or n3 """
        if len(ncells) == 0:
            print("Error: Reduce method must be called with at least one parameter (nx, ny or nz), exiting...")
            sys.exit(1)
        key_to_index = {"n1": 0, "n2": 1, "n3": 2}
        for key in ncells.keys():
            if key not in ["n1", "n2", "n3"]:
                print("Error: Invalid input (must be n1, n2 or n3), exiting...")
                sys.exit(1)

            new_motif = self.motif
            for n in range(1, ncells[key]):
                motif_copy_displaced = np.copy(self.motif)
                motif_copy_displaced[:, :3] += n * self.bravais_lattice[key_to_index[key]]
                new_motif = np.append(new_motif, motif_copy_displaced, axis=0)

            self.motif = new_motif
        indices = [index for index in list(range(self.ndim)) \
                   if index not in [key_to_index[key] for key in ncells.keys()]]
        self.bravais_lattice = self.bravais_lattice[indices]

        return self

    def ribbon(self, width, orientation="horizontal"):
        """ Routine to generate a ribbon for a 2D crystal"""
        if self.ndim != 2:
            print(f"Ribbons can not be generated for {self.ndim}D structures")
            sys.exit(1)
        else:
            mesh_points = []
            for i in range(self.ndim):
                mesh_points.append(list(range(-1, 2)))

            combinations = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.ndim)
            rectangular_basis = np.zeros([2, 3])
            all_possible_atoms = []
            motif = []

            # Calculate rectangular lattice basis vectors
            for n, m in combinations:
                vector = n * self.bravais_lattice[0] + m * self.bravais_lattice[1]
                for atom in self.motif:
                    new_atom = np.copy(atom)
                    new_atom[:3] += vector
                    all_possible_atoms.append(new_atom)

                if n == m == 0:
                    continue
                vector_angle = np.arctan2(vector[1], vector[0])

                if abs(vector_angle - 0.0) < 1E-10:
                    rectangular_basis[0, :] = vector
                elif abs(vector_angle - np.pi/2) < 1E-10:
                    rectangular_basis[1, :] = vector

            if not np.linalg.norm(rectangular_basis, axis=1).any():
                print("Could not generate a rectangular lattice, exiting...")
                sys.exit(1)

            # Calculate new motif
            for i, atom in enumerate(all_possible_atoms):
                if (0 <= atom[0] < np.linalg.norm(rectangular_basis[0, :])) and \
                   (0 <= atom[1] < np.linalg.norm(rectangular_basis[1, :])):
                    motif.append(atom)

            # Update system attributes
            self.motif = motif
            self.bravais_lattice = rectangular_basis

            # Reduce system dimensionality
            if orientation == "horizontal":
                self.reduce(n1=width)
            elif orientation == "vertical":
                self.reduce(n2=width)
            else:
                print("Error: Wrong orientation provided, must be either vertical or horizontal, exiting...")
                sys.exit(1)

            return self

    def _restrict_lattice2rectangle(self, vectors):
        atoms = []
        mesh_points = []
        for i in range(self.ndim):
            mesh_points.append(list(range(-1, 2)))

        combinations = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.ndim)
        for n, m in combinations:
            vector = n*self.bravais_lattice[0] + m*self.bravais_lattice[1]
            for position in self.motif:
                atom_position = vector + position



    def hamiltonian_k(self, k):
        """ Generic implementation of hamiltonian_k H(k). To be overwritten
         by specific implementations of System """
        pass

    def solve(self, kpoints):
        """ Diagonalize the Hamiltonian to obtain the band structure and the eigenstates """
        nk = len(kpoints)
        eigen_energy = np.zeros([self._basisdim, nk])
        eigen_states = []
        for n, k in enumerate(kpoints):
            hamiltonian = self.hamiltonian_k(k)
            results = np.linalg.eigh(hamiltonian)
            eigen_energy[:, n] = results[0]
            eigen_states.append(results[1])

        return result.Result(eigen_energy, np.array(eigen_states), kpoints)


class FrozenClass:
    """ Class to enforce immutable attributes """
    _is_frozen = False

    def __setattr__(self, key, value):
        if self._is_frozen:
            raise Exception("Predefined model can not be modified")
        super().__setattr__(key, value)

    def _freeze(self):
        self._is_frozen = True



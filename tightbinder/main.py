# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from . import fileparse, hamiltonian, crystal
import time


def main():
    """ Main routine """
    # ------------------------------ Argument & file parsing ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='input file for tight-binding model')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()

    try:
        file = open(args.file, 'r')
    except IOError:
        print('Error: Input file does not exist')
        sys.exit(1)

    configuration = fileparse.parse_config_file(file)

    reciprocal_basis = crystal.reciprocal_lattice(configuration['Bravais lattice'])
    # kpoints = crystal.brillouin_zone_mesh(configuration['Mesh'], reciprocal_basis)

    crystal_group = crystal.crystallographic_group(configuration['Bravais lattice'])
    special_points, labels = crystal.high_symmetry_points(crystal_group, reciprocal_basis)
    kpoints = crystal.high_symmetry_path(special_points, 100)

    square = hamiltonian.Hamiltonian(configuration)
    square.initialize_hamiltonian()

    results = square.solve(kpoints)
    results.plot_along_path(labels)


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

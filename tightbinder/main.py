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
    special_points = crystal.high_symmetry_points(crystal_group, reciprocal_basis)
    special_points = crystal.reorder_special_points(special_points, ["M", r"$\Gamma$", "K", "M"])
    special_points, labels = crystal.split_labels_from_special_points(special_points)
    kpoints = crystal.high_symmetry_path(special_points, configuration['Mesh'][0])

    bi = hamiltonian.Hamiltonian(configuration)
    bi.initialize_hamiltonian()

    results = bi.solve(kpoints)
    results.plot_along_path(labels)


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from tightbinder import fileparse, hamiltonian, crystal


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

    verbose = args.verbose

    configuration = fileparse.parse_config_file(file)

    reciprocal_basis = crystal.reciprocal_lattice(configuration['Bravais lattice'])

    chain = hamiltonian.Hamiltonian(configuration)
    chain.initialize_hamiltonian()

    kpoints = crystal.brillouin_zone_mesh([11], reciprocal_basis)

    results = chain.solve(kpoints)

    print(results[0])


if __name__ == "__main__":
    main()

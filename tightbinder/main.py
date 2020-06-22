# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals, interaction values)

import argparse, sys
from tightbinder import fileparse, hamiltonian, crystal
from numpy import np


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

    bravais_lattice = configuration['Bravais lattice']
    reciprocal_basis = crystal.reciprocal_lattice(bravais_lattice)
    
    kpoints = crystal.brillouin_zone_mesh([2, 3], reciprocal_basis)
    group = crystal.crystallographic_group(bravais_lattice)

    motif = configuration['Motif']
    hamiltonian.first_neighbours(motif, bravais_lattice, "minimal", r=1)

    sk_amplitudes = configuration["SK amplitudes"]
    orbitals = hamiltonian.transform_orbitals_to_string(configuration["Orbitals"])

    basis = hamiltonian.create_atomic_orbital_basis(motif, orbitals)

    k = 0.5
    h = hamiltonian.initialize_hamiltonian(k, configuration)


if __name__ == "__main__":
    main()

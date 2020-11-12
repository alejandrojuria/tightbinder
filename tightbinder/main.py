# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
import fileparse, hamiltonian, crystal, topology
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

    lattice = crystal.Crystal(configuration)
    # lattice.plot_crystal(cell_number=1)
    labels = ["M", r"$\Gamma$", "K", "M"]
    lattice.high_symmetry_path(200, labels)

    bi = hamiltonian.Hamiltonian(configuration)
    bi.initialize_hamiltonian()

    results = bi.solve(lattice.kpoints)
    results.plot_along_path(labels)

    wccs = topology.calculate_wannier_centre_flow(bi, lattice, filling=10, number_of_points=50)
    topology.plot_wannier_centre_flow(wccs)


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

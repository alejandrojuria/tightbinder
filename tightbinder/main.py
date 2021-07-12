# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
import fileparse, crystal, topology
import time
from models import SKModel


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
    labels = ["M", "G", "K", "M"]
    lattice.high_symmetry_path(200, labels)

    bi = SKModel(configuration)
    bi.initialize_hamiltonian()

    results = bi.solve(lattice.kpoints)
    results.plot_along_path(labels)

    wcc_flow = topology.calculate_wannier_centre_flow(bi, number_of_points=10)
    topology.plot_wannier_centre_flow(wcc_flow, show_midpoints=True)
    print(f"Chern number: {topology.calculate_chern_number(wcc_flow)}")
    print(f"Z2 invariant: {topology.calculate_z2_invariant(wcc_flow)}")

    topology.plot_wannier_centre_flow(wcc_flow)
    topology.calculate_chern(wcc_flow)


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

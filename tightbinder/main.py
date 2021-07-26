# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import calculate_wannier_centre_flow, calculate_z2_invariant, plot_wannier_centre_flow
import time
from tightbinder.models import SKModel
import matplotlib.pyplot as plt


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

    configuration = parse_config_file(file)
    bi = SKModel(configuration)
    # lattice.plot_crystal(cell_number=1)
    labels = ["M", "G", "K", "M"]
    kpoints = bi.high_symmetry_path(200, labels)
    bi.initialize_hamiltonian()

    results = bi.solve(kpoints)
    results.plot_along_path(labels)

    bi.filling = 5./8.
    wcc_flow = calculate_wannier_centre_flow(bi, number_of_points=10)
    plot_wannier_centre_flow(wcc_flow, show_midpoints=True)
    print(f"Z2 invariant: {calculate_z2_invariant(wcc_flow)}")

    plot_wannier_centre_flow(wcc_flow)
    plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

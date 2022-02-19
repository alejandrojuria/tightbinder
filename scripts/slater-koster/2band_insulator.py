# Script to simulate a two-band insulator using the SK approximation. The material
# is based on a square lattice with two orbitals, s and pz.

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
    model = SKModel(configuration)
    # lattice.plot_crystal(cell_number=1)
    labels = ["M", "G", "K", "M"]
    # labels = ["K", "G", "K"]
    kpoints = model.high_symmetry_path(200, labels)
    model.initialize_hamiltonian()

    results = model.solve(kpoints)
    results.plot_along_path(labels)
    plt.grid()
    model.export_model("2band_insulator_model.txt")

    plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

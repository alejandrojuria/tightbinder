# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import calculate_wannier_centre_flow, calculate_z2_invariant, plot_wannier_centre_flow
import time
from tightbinder.models import SlaterKoster
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
    if 'Radius' in configuration:
        model = SlaterKoster(configuration, mode='radius', r=configuration['Radius'])
    else:
        model = SlaterKoster(configuration)

    # model.visualize() # Has to be fixed
    labels = configuration['High symmetry points']
    kpoints = model.high_symmetry_path(configuration['Mesh'][0], labels)
    model.initialize_hamiltonian()
    results = model.solve(kpoints)
    results.plot_along_path(labels)

    plt.show()

if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

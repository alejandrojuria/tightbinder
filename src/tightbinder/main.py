#!/usr/bin/python

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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def main():
    """ Main routine """
    # ------------------------------ Argument & file parsing ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='input file for tight-binding model')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-cv', '--crystalview', action='store_true', help='3D view of the crystal')
    parser.add_argument('-z2', '--z2invariant', action='store_true', help='Compute Z2 invariant')
    parser.add_argument('-z2p', '--z2plot', action='store_true', help='Plot Wannier charge centers used for Z2 computation')
    parser.add_argument('-e', '--export', action='store_true', help='Export model Hamiltonian to file')
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

    # Possible system modifications before initialize_hamiltonian
    model.initialize_hamiltonian()
    print(len(model.bonds))

    if args.crystalview:
        model.visualize()

    if args.z2invariant:
        wcc = calculate_wannier_centre_flow(model, 20)
        z2  = calculate_z2_invariant(wcc)
        print(f'Z2 invariant is: {z2}')

        if args.z2plot:
            plot_wannier_centre_flow(wcc, show_midpoints=True)

    if args.export:
        modelfile = args.file
        # Remove extension if present and blanks
        modelfile = modelfile.replace(".txt", '').strip() + ".model"
        model.export_model(modelfile, fmt="%10.6f")
        print(f"Written model to file {modelfile}")

    labels = configuration['High symmetry points']
    kpoints = model.high_symmetry_path(configuration['Mesh'][0], labels)

    results = model.solve(kpoints)
    results.plot_along_path(labels, title=f'{model.system_name}')

    plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

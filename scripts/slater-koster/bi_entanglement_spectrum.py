# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms.
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
import time
from tightbinder.models import SKModel
import matplotlib.pyplot as plt
import numpy as np


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
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 6), dpi=100)
    fontsize = 24

    # First compute the HWCC flow for topological Bi(111) with non-zero SOC
    bi = SKModel(configuration)
    bi.motif = np.array(bi.motif)
    bi = bi.reduce(n1=10)
    bi.initialize_hamiltonian()

    plane = [0, 1, 0, np.max(bi.motif[:, 1]) / 2]
    # bi.visualize()
    labels = ["K", "G", "K"]
    kpoints = bi.high_symmetry_path(200, labels)

    bi.filling = 5./8.
    spectrum = entanglement_spectrum(bi, plane, kpoints)
    plot_entanglement_spectrum(spectrum, bi, ax[0], fontsize=fontsize)

    # Now compute the HWCC flow for trivial Bi(111) (zero SOC)
    configuration["Spin-orbit coupling"] = 0.0

    bi = SKModel(configuration)
    labels = ["M", "G", "K", "M"]
    kpoints = bi.high_symmetry_path(200, labels)
    bi.initialize_hamiltonian()
    results = bi.solve(kpoints)

    bi.filling = 5. / 8.
    spectrum = entanglement_spectrum(bi, plane, kpoints)
    plot_entanglement_spectrum(spectrum, bi, ax[1], fontsize=fontsize)

    # Add text to the subplots, (a) and (b)
    ax[0].text(0.9, 0.75, "(a)", fontsize=fontsize)
    ax[1].text(0.9, 0.75, "(b)", fontsize=fontsize)

    plt.subplots_adjust(wspace=0.15)
    # plt.savefig("wcc_flow.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms.
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
from tightbinder.topology import specify_partition_shape, specify_partition_plane
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
    for axis in ax:
        for side in ['top', 'bottom', 'left', 'right']:
            axis.spines[side].set_linewidth(2)

    fontsize = 24

    # First compute the HWCC flow for topological Bi(111) with non-zero SOC
    bi = SKModel(configuration)
    bi.motif = np.array(bi.motif)
    bi = bi.reduce(n1=20).supercell(n1=3)
    bi.filling = 5. / 8.
    bi.ordering = "atomic"
    partition_center = np.array([np.max(bi.motif[:, 0])/2, np.max(bi.motif[:, 1])/2, 0])
    print(np.max(bi.motif[:, 1]))
    radius = np.min([np.max(bi.motif[:, 0]), np.max(bi.motif[:, 1])])/4
    print(radius)
    partition = specify_partition_shape(bi, shape='circle', center=partition_center, r=radius)
    plane = np.array([1, 0, 0, np.max(bi.motif[:, 0] / 2)])
    partition = specify_partition_plane(bi, plane)
    print(partition)
    print()
    labels = ["K", "G", "K"]
    kpoints = bi.high_symmetry_path(21, labels)
    bi.initialize_hamiltonian()

    entanglement = entanglement_spectrum(bi, partition, kpoints=kpoints)
    plot_entanglement_spectrum(entanglement, bi, ax=ax[0], fontsize=fontsize)

    # Now compute the HWCC flow for trivial Bi(111) (zero SOC)
    bi.configuration["Spin-orbit coupling"] = 0.0
    bi.initialize_hamiltonian()

    entanglement = entanglement_spectrum(bi, partition, kpoints=kpoints)
    plot_entanglement_spectrum(entanglement, bi, ax=ax[1], fontsize=fontsize)

    # Add text to the subplots, (a) and (b)
    ax[0].text(0.9, 0.75, "(a)", fontsize=fontsize)
    ax[1].text(0.9, 0.75, "(b)", fontsize=fontsize)

    plt.subplots_adjust(wspace=0.15)
    # plt.savefig("bi_entanglement_periodic_w50.png", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

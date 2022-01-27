from tightbinder.models import SKModel
from tightbinder.fileparse import parse_config_file

# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms.
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals,
# interaction values)

import argparse
import sys
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
from tightbinder.topology import specify_partition_plane
import time
from tightbinder.models import SKModel
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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
    fontsize = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)

    # First compute the HWCC flow for topological Bi(111) with non-zero SOC
    bi = SKModel(configuration)
    bi.motif = np.array(bi.motif)
    bi = bi.reduce(n1=20)
    bi.filling = 5. / 8.
    bi.ordering = "atomic"
    bi.boundary = "PBC"
    labels = ["K", "G", "K"]
    kpoints = bi.high_symmetry_path(201, labels)
    bi.initialize_hamiltonian()
    results = bi.solve(kpoints)
    filling = bi.filling * bi.basisdim
    results.plot_along_path(labels, ax=ax, filling=filling, edge_states=True, fontsize=fontsize, y_values=[-1, 1])

    plt.savefig("bi_ribbon_bands.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

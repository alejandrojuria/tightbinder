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
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 6), dpi=100)
    fontsize = 24

    # First compute the HWCC flow for topological Bi(111) with non-zero SOC
    bi = SKModel(configuration)
    print(bi.norbitals)
    # bi = bi.ribbon(width=10)
    bi.initialize_hamiltonian()
    # bi.visualize()
    labels = ["K", "G", "K"]
    kpoints = bi.high_symmetry_path(200, labels)

    results = bi.solve(kpoints)

    bi.filling = 5./8.
    wcc_flow = calculate_wannier_centre_flow(bi, number_of_points=10)
    print(f"Z2 invariant: {calculate_z2_invariant(wcc_flow)}")
    plot_wannier_centre_flow(wcc_flow, show_midpoints=True, ax=ax[0], fontsize=fontsize)

    # Now compute the HWCC flow for trivial Bi(111) (zero SOC)
    configuration["Spin-orbit coupling"] = 0.0

    bi = SKModel(configuration)
    labels = ["M", "G", "K", "M"]
    kpoints = bi.high_symmetry_path(200, labels)
    bi.initialize_hamiltonian()
    results = bi.solve(kpoints)

    bi.filling = 5. / 8.
    wcc_flow = calculate_wannier_centre_flow(bi, number_of_points=10)
    print(f"Z2 invariant: {calculate_z2_invariant(wcc_flow)}")
    plot_wannier_centre_flow(wcc_flow, show_midpoints=True, ax=ax[1], fontsize=fontsize)

    # Add text to the subplots, (a) and (b)
    ax[0].text(0.9, 0.75, "(a)", fontsize=fontsize)
    ax[1].text(0.9, 0.75, "(b)", fontsize=fontsize)

    plt.subplots_adjust(wspace=0.15)
    for axis in ax:
        for border in ['top', 'bottom', 'left', 'right']:
            axis.spines[border].set_linewidth(2)

    plt.savefig("wcc_flow.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

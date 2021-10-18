# Script to plot the phase diagram for the Wilson-Dirac Fermion model in
# its crystalline phase for both 3D and 2D

import numpy as np
import matplotlib.pyplot as plt


def main():
    fontsize = 20
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 4), dpi=100)

    # First plot for 3D Wilson
    # Plot vertical lines to indicate gap closings
    ax[0].vlines([0, 2, 4, 6], 0, 1, linestyle='dashed')
    ax[0].set_ylim([0, 1])
    ax[0].tick_params(left=False)
    ax[0].set(yticklabels=[])
    ax[0].set_ylabel("3D Wilson", fontsize=fontsize)

    # Fill areas between gap closings with different colours to denote topological phases
    ax[0].fill_between([-2, 0], 0, 1, facecolor="blue", alpha=0.5)
    ax[0].text(-1, 0.5, "Trivial", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[0].fill_between([0, 2], 0, 1, facecolor="green", alpha=0.5)
    ax[0].text(1, 0.5, "STI", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[0].fill_between([2, 4], 0, 1, facecolor="yellow", alpha=0.4)
    ax[0].fill_between([2, 4], 0, 1, facecolor="black", alpha=0.1)
    ax[0].text(3, 0.5, "WTI", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[0].fill_between([4, 6], 0, 1, facecolor="green", alpha=0.5)
    ax[0].text(5, 0.5, "STI", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[0].fill_between([6, 8], 0, 1, facecolor="blue", alpha=0.5)
    ax[0].text(7, 0.5, "Trivial", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    # Then plot for 2D Wilson
    # Plot vertical lines to indicate gap closings
    ax[1].vlines([1, 3, 5], 0, 1, linestyle='dashed')
    ax[1].set_ylim([0, 1])
    ax[1].tick_params(left=False)
    ax[1].set(yticklabels=[])
    ax[1].set_ylabel("2D Wilson", fontsize=fontsize)
    ax[1].tick_params(axis='x', labelsize=fontsize)

    # Fill areas between gap closings with different colours to denote topological phases
    ax[1].fill_between([-2, 1], 0, 1, facecolor="blue", alpha=0.5)
    ax[1].text(-0.5, 0.5, "Trivial", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[1].fill_between([1, 3], 0, 1, facecolor="green", alpha=0.5)
    ax[1].text(2, 0.5, "TI", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[1].fill_between([3, 5], 0, 1, facecolor="green", alpha=0.5)
    ax[1].text(4, 0.5, "TI", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[1].fill_between([5, 8], 0, 1, facecolor="blue", alpha=0.5)
    ax[1].text(6.5, 0.5, "Trivial", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    ax[1].set_xlim([-2, 8])
    ax[1].set_xlabel(rf"$M (eV)$", fontsize=fontsize)
    plt.subplots_adjust(hspace=0.1)

    # plt.show()
    plt.savefig("wilson_phases.png", bbox_inches='tight')


if __name__ == "__main__":
    main()

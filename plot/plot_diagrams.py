# Script to plot the gap diagram as a function of two parameters for a given model
# This script is intended to work with the following files (simultaneously):
# - wilson_2d_diagram_L20_gap_r11_5samples
# - wilson_2d_diagram_L20_gap_r14_5samples
# To run it:
# python plot_diagrams.py file1 file2

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib


def transform_data(x, y):
    npoints = len(x[:, 0])
    dx = (x[0, 1] - x[0, 0]) / 2
    x -= dx
    extended_x = np.zeros([npoints + 1, npoints + 1])
    extended_x[:-1, :-1] = x
    extended_x[-1, :-1] = x[-1, :]
    extended_x[:, -1] = x[-1, -1] + dx * 2
    x = extended_x

    extended_y = np.zeros([npoints + 1, npoints + 1])
    dy = y[1, 0] - y[0, 0]
    y -= dy
    extended_y[:-1, :-1] = y
    extended_y[-1, :] = y[-1, 0] + dy * 2
    extended_y[:-1, -1] = y[:, -1]
    y = extended_y

    return x, y


def extract_data_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    mass, disorder, gap = [], [], []
    for line in lines:
        line = [float(value) for value in line.split('\t')[:-1]]
        mass.append(line[0])
        disorder.append(line[1])
        gap.append(line[2:])

    npoints = int(np.sqrt(len(mass)))
    mass = np.array(mass).reshape(npoints, npoints)
    disorder = np.array(disorder).reshape(npoints, npoints)

    average_gap = []
    for gap_data in gap:
        average_gap.append(np.average(gap_data))
    average_gap = np.array(average_gap).reshape(npoints, npoints)
    average_gap[average_gap < 1E-4] = 1E-4

    return mass, disorder, average_gap


def main():
    nfiles = len(sys.argv) - 1
    fig, ax = plt.subplots(1, nfiles, sharey=True, figsize=(15, 6), dpi=100)
    fontsize = 24
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)
    for i in range(1, nfiles + 1):
        filename = sys.argv[i]
        mass, disorder, gap = extract_data_from_file(filename)
        mass, disorder = transform_data(mass, disorder)

        pc = ax[i - 1].pcolor(mass, disorder, gap, vmin=0, vmax=gap.max())
        # plt.pcolor(mass, disorder, average_gap, norm=LogNorm(vmin=average_gap.min(), vmax=average_gap.max()))
        ax[i - 1].set_xlabel(r"M (eV)", fontsize=fontsize)
        ax[i - 1].set_xlim([0, 6])
        ax[i - 1].tick_params(axis='x', labelsize=fontsize)

    ax[0].tick_params(axis='y', labelsize=fontsize)
    ax[0].set_ylim([0, 0.5])
    ax[0].set_ylabel(r"$\Delta r$ $(\AA)$", fontsize=fontsize)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    cbar = fig.colorbar(pc, ax=ax, pad=0.01)
    cbar.set_label(r"Gap $(eV)$", fontsize=fontsize)

    plt.savefig("gap_diagrams.png", bbox_inches='tight')


if __name__ == "__main__":
    main()


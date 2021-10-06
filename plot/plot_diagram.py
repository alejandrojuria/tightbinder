# Script to plot the gap diagram as a function of two parameters for a given model

import numpy as np
import sys
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) > 2:
        raise SyntaxError("Script expects only one file, exiting...")

    file = open(sys.argv[1], "r")
    lines = file.readlines()
    mass, disorder, gap = [], [], []
    for line in lines:
        print(line.split('\t')[:-1])
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

    plt.pcolor(mass, disorder, average_gap)
    plt.show()


if __name__ == "__main__":
    main()


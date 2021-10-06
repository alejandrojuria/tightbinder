# Script to plot the gap diagram as a function of two parameters for a given model

import numpy as np
import sys
import matplotlib.pyplot as plt


def main():
    nargs = len(sys.argv)
    for i in range(1, nargs):
        filename = sys.argv[i]
        x, y, z = extract_data(filename)
        plot_data(x, y, z)

    plt.show()


def extract_data(filename):
    file = open(filename, "r")

    # Check if first line is a str
    first_line = file.readline().split('\t')
    data = []

    try:
        first_value = float(first_line[0])
        data.append([float(value) for value in first_line])
    except ValueError:
        print('Skipping header line...')

    lines = file.readlines()
    for line in lines:
        data.append([float(value) for value in line.split('\t')])

    data = np.array(data)
    x, y, z = data.T # Unpack is done by columns
    npoints = len(x)
    x = x.reshape(npoints, npoints)
    y = y.reshape(npoints, npoints)
    z = z.reshape(npoints, npoints)

    return x, y, z


def plot_data(x, y, z, ax=None):
    if ax is None:
        fig, ax = plt.figure()
    fig.pcolor(x, y, z)


if __name__ == "__main__":
    main()


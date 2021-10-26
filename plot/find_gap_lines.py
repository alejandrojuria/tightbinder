# Script to obtain the curves that define the closings of the gap
# in the m-displacement diagram.

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plot.plot_diagrams import extract_data_from_file, transform_data


def find_gap_closing(gap_array, m_array):
    """ For fixed disorder, find the gap closings in an array of gaps for
     all m values """
    gap_closings = []
    previous_gap = 100
    gap_cutoff = 2E-1
    for n, gap in enumerate(gap_array):
        mass = m_array[n]
        if gap < gap_cutoff < previous_gap:
            gap_closings.append(mass)  # Append value of m
        elif gap > gap_cutoff > previous_gap:
            if m_array[n - 1] != gap_closings[-1]:
                print(m_array[n - 1], gap_closings[-1])
                gap_closings.append(m_array[n - 1])  # Append previous m value
        previous_gap = gap

    return gap_closings


def main():
    filename = sys.argv[1]
    mass, disorder, gap = extract_data_from_file(filename)
    mass_values = mass[0, :]
    disorder_values = disorder[:, 0]
    all_gap_closings = []
    for gap_array in gap:
        all_gap_closings.append(find_gap_closing(gap_array, mass_values))
    print(all_gap_closings)
    all_gap_closings = np.array(all_gap_closings).T

    fig = plt.figure()
    mass, disorder = transform_data(mass, disorder)
    plt.pcolor(mass, disorder, gap, norm=LogNorm(vmin=1E-1, vmax=gap.max()))
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()

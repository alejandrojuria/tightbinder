# Script to plot the gap for different system sizes, the edge occupation averaged between the
# lowest energy eigenstates and the coordination number for for both spread=0 and 0.5, all
# of them vs M
# All data is taken from text files, meaning it was previously computed. Therefore this scripts
# only handles reading from file and plotting the data


import numpy as np
import matplotlib.pyplot as plt
from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize


def extract_edge_occupation(filename):
    file = open(filename, 'r')
    spread, mass, occupations = [], [], []
    all_occupations, all_mass = [], []
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        if len(line) == 1:
            if line[0] == "#":
                all_occupations.append(occupations)
                all_mass.append(mass)
                occupations = []
                mass = []
            else:
                spread.append(float(line[0]))
        else:
            line = [float(value) for value in line]
            mass.append(line[0])
            occupations.append(line[1])

    data = {
        'spread': spread,
        'mass': all_mass,
        'occupation': all_occupations
    }

    file.close()
    return data


def plot_edge_occupation_data(data, ax=None, fontsize=10):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    nseries = len(data['spread'])
    for n in range(nseries):
        mass = data['mass'][n]
        occupation = data['occupation'][n]
        ax.plot(mass, occupation, '-', linewidth=3)

    ax.legend([rf"$\Delta r=${n}" for n in data['spread']], fontsize=fontsize)
    ax.tick_params('both', labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][0]))
    ax.set_xticks([-1, 1, 3, 5, 7])


def extract_gap(filename):
    file = open(filename, 'r')
    cellsize, mass, gap, gap_std = [], [], [], []
    all_gap, all_gap_std, all_mass = [], [], []
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        if len(line) == 1:
            if line[0] == "#":
                all_gap.append(gap)
                all_gap_std.append(gap_std)
                all_mass.append(mass)
                gap, gap_std, mass = [], [], []
            else:
                cellsize.append(float(line[0]))
        else:
            line = [float(value) for value in line]
            mass.append(line[0])
            gap.append(line[1])
            gap_std.append(line[2])

    data = {
        'gap': all_gap,
        'gap_std': all_gap_std,
        'mass': all_mass,
        'cellsize': cellsize
    }

    file.close()
    return data


def plot_gap_data(data, ax=None, fontsize=10):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplot(111)

    nseries = len(data['cellsize'])
    for n in range(nseries):
        gap = data['gap'][n]
        std = data['gap_std'][n]
        mass = data['mass'][n]
        ax.plot(mass, gap)

    ax.legend([f"N={n}" for n in data['cellsize']])
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][1]))


def extract_coordination(filename):
    file = open(filename, 'r')
    spread, all_coordinations, all_mass, all_std = [], [], []
    coordination, mass, std = [], [], []
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        if len(line) == 1:
            if line[0] == "#":
                all_coordinations.append(coordination)
                all_std.append(std)
                all_mass.append(mass)
                coordination, std, mass = [], [], []
            else:
                spread.append(float(line[0]))
        else:
            line = [float(value) for value in line]
            mass.append(line[0])
            coordination.append(line[1])
            std.append(line[2])

    data = {
        'spread': spread,
        'coordination': all_coordinations,
        'mass': all_mass,
        'coord_std': all_std
    }

    file.close()
    return data


def plot_coordination(data, ax=None, fontsize=10):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplot(111)

    nseries = len(data['cellsize'])
    for n in range(nseries):
        coordination = data['coordination'][n]
        std = data['coord_std'][n]
        mass = data['mass'][n]
        ax.plot(mass, coordination)

    ax.legend([rf"$\Delta r=${n}" for n in data['spread']])
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][1]))


def main():
    fig, ax = plt.subplots(2, 2, figsize=(15, 15), dpi=100)
    fontsize = 24

    # |x-|
    # |--|
    # Plot average gap for different system sizes vs M for fixed spread and cutoff

    # data = extract_gap(gapfile)
    # plot_gap_data(data, ax[0, 0], fontsize)
    #
    # # |-x|
    # # |--|
    # # Plot average gap for different system sizes vs M for fixed spread and cutoff
    # data = extract_coordination(cooordinationfile)
    # plot_coordination(data, ax[0, 0], fontsize)

    # |--|
    # |xy|
    # Plot edge occupation vs M for fixed spread and cutoff
    files = [
        './data/edge_occupation_r11_L20',
        './data/edge_occupation_r14_L20'
    ]
    for i, file in enumerate(files):
        data = extract_edge_occupation(files[i])
        plot_edge_occupation_data(data, ax[1, i], fontsize)

    plt.show()


if __name__ == "__main__":
    main()

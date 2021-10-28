# Script to plot the gap for different system sizes, the edge occupation averaged between the
# lowest energy eigenstates and the coordination number for for both spread=0 and 0.5, all
# of them vs M
# All data is taken from text files, meaning it was previously computed. Therefore this scripts
# only handles reading from file and plotting the data


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize


def extract_edge_occupation(filename):
    file = open(filename, 'r')
    spread, mass, occupations, std = [], [], [], []
    all_occupations, all_mass, all_std = [], [], []
    lines = file.readlines()
    for line in lines:
        line = line.split()
        if len(line) == 1:
            if line[0] == "#":
                all_occupations.append(occupations)
                all_mass.append(mass)
                all_std.append(std)
                occupations = []
                mass = []
                std = []
            else:
                spread.append(float(line[0]))
        else:
            line = [float(value) for value in line]
            mass.append(line[0])
            occupations.append(line[1])
            std.append(line[2])

    data = {
        'spread': spread,
        'mass': all_mass,
        'occupation': all_occupations,
        'std': all_std
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
        occupation = np.array(data['occupation'][n])
        std = np.array(data['std'][n])
        ax.plot(mass, occupation, '-', linewidth=3)
        ax.fill_between(mass, occupation-std, occupation+std, alpha=0.5)

    ax.legend([rf"$\Delta r=${n}" for n in data['spread']],
              fontsize=fontsize*3/4, frameon=True).set_zorder(2)
    ax.tick_params('both', labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][0]))
    ax.set_xticks([-1, 1, 3, 5, 7])
    ax.set_ylabel("Edge occupation", fontsize=fontsize)
    ax.set_xlabel("M (eV)", fontsize=fontsize)


def extract_gap(filename):
    file = open(filename, 'r')
    cellsize, mass, gap, gap_std = [], [], [], []
    all_gap, all_gap_std, all_mass = [], [], []
    lines = file.readlines()
    for line in lines:
        line = line.split()
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
    axins = ax.inset_axes([0.5, 0.3, 0.4, 0.4])
    x1, x2, y1, y2 = 1.5, 4.5, 0, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)

    nseries = len(data['cellsize'])
    cmap = matplotlib.cm.get_cmap("viridis")
    colors = [cmap(n) for n in np.linspace(0, 1, nseries)]
    for n in range(nseries):
        gap = np.array(data['gap'][n])
        std = np.array(data['gap_std'][n])
        mass = data['mass'][n]
        ax.plot(mass, gap, linewidth=3, c=colors[n])
        axins.plot(mass, gap, linewidth=1.5, c=colors[n])
        ax.fill_between(mass, gap-std, gap+std, alpha=0.5)
        # ax.errorbar(mass, gap, yerr=std, linewidth=3, elinewidth=1.5, capsize=5)

    ax.legend([f"N={int(n)}" for n in data['cellsize']],
              fontsize=fontsize*3/4, frameon=True, loc="upper left").set_zorder(2)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][1]))
    ax.set_xticks([0, 2, 4, 6])
    ax.set_ylim(np.min(gap), np.max(gap))
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_ylabel("Gap (eV)", fontsize=fontsize)
    ax.set_xlabel("M (eV)", fontsize=fontsize)

    axins.tick_params(axis="both", labelsize=fontsize/2)


def extract_coordination(filename):
    file = open(filename, 'r')
    spread, all_coordinations, all_mass, all_std = [], [], [], []
    coordination, mass, std = [], [], []
    lines = file.readlines()
    for line in lines:
        line = line.split()
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

    nseries = len(data['spread'])
    for n in range(nseries):
        coordination = np.array(data['coordination'][n])
        std = np.array(data['coord_std'][n])
        mass = np.array(data['mass'][n])
        ax.plot(mass, coordination, linewidth=2)
        ax.fill_between(mass, coordination-std, coordination+std, alpha=0.5)

    ax.legend([r"$R_{cutoff}=$" + str(n) for n in data['spread']],
              fontsize=fontsize*3/4, frameon=False)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][1]))
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_xlabel(r"$\Delta$r $(\AA)$", fontsize=fontsize)
    ax.set_ylabel("Coordination", fontsize=fontsize)


def main():
    fig, ax = plt.subplots(2, 2, figsize=(15, 12), dpi=100)
    fontsize = 24
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0, 0].spines[axis].set_linewidth(2)
        ax[1, 0].spines[axis].set_linewidth(2)
        ax[0, 1].spines[axis].set_linewidth(2)
        ax[1, 1].spines[axis].set_linewidth(2)

    # |x-|
    # |--|
    # Plot average gap for different system sizes vs M for fixed spread and cutoff
    gapfile = "./data/gap_r11_spread_03"
    gapdata = extract_gap(gapfile)
    plot_gap_data(gapdata, ax[0, 0], fontsize=fontsize)

    # # |-x|
    # # |--|
    # # Plot average gap for different system sizes vs M for fixed spread and cutoff
    coordinationfile = "./data/coordination_L30"
    data = extract_coordination(coordinationfile)
    print(data["coordination"])
    plot_coordination(data, ax[0, 1], fontsize)

    # |--|
    # |xy|
    # Plot edge occupation vs M for fixed spread and cutoff
    files = [
        './data/edge_occupation_r11_L30',
        './data/edge_occupation_r14_L30'
    ]
    for i, file in enumerate(files):
        data = extract_edge_occupation(files[i])
        plot_edge_occupation_data(data, ax[1, i], fontsize)

    plt.savefig("gap_edge_coord.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

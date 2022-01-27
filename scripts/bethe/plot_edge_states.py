# Script to plot one topological edge state, one trivial state and the
# edge occupation as a function of m for different bond lengths
# for the Bethe lattice with the Wilson-fermion model

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tightbinder.models import WilsonAmorphous, bethe_lattice
from tightbinder.result import State


def plot_state(mass, length, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplot(111)

    model = WilsonAmorphous(m=mass)
    model.motif, model.bonds = bethe_lattice(z=3, depth=8, length=length)
    model.motif = np.array(model.motif)
    model.boundary = "OBC"
    print(f"Coordination number: {model.coordination_number()}")
    model.initialize_hamiltonian(find_bonds=False)
    results = model.solve()

    energies = np.abs(results.eigen_energy)
    print(f"Energy of edge state: {np.min(energies)}")
    index = np.where(energies == np.min(energies))[0][0]
    edge_state = State(results.eigen_states[0][:, index], model)
    edge_state.plot_amplitude(ax)


def extract_edge_occupation(filename):
    file = open(filename, 'r')
    length, mass, occupations = [], [], []
    all_occupations, all_mass = [], []
    lines = file.readlines()
    for line in lines:
        line = line.split()
        if len(line) == 1:
            if line[0] == "#":
                all_occupations.append(occupations)
                all_mass.append(mass)
                occupations = []
                mass = []
            else:
                length.append(float(line[0]))
        else:
            line = [float(value) for value in line]
            mass.append(line[0])
            occupations.append(line[1])

    data = {
        'length': length,
        'mass': all_mass,
        'occupation': all_occupations,
    }

    file.close()
    return data


def plot_edge_occupation_data(data, ax=None, fontsize=10):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    nseries = len(data['length'])
    cmap = matplotlib.cm.get_cmap("viridis")
    colors = [cmap(n) for n in np.linspace(0, 1, nseries)]

    for n in range(nseries):
        mass = data['mass'][n]
        occupation = np.array(data['occupation'][n])
        ax.plot(mass, occupation, '-', linewidth=3, c=colors[n])

    ax.legend([rf"{n}" for n in data['length']],
              fontsize=fontsize*2/4, frameon=True)
    ax.tick_params('both', labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][0]))
    ax.set_xticks([-1, 1, 3, 5, 7])
    ax.set_ylabel("Edge occupation", fontsize=fontsize)
    ax.set_xlabel("M (eV)", fontsize=fontsize)


def main():
    fig = plt.figure(figsize=(18, 9), dpi=100)
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax = [ax1, ax2]
    fontsize = 24

    # |x-|
    # Plot topological edge state
    plot_state(mass=2.5, length=0.7, ax=ax[0])
    ax[0].text(0, 0.9, '(a)', horizontalalignment='center',
               verticalalignment='center', transform=ax[0].transAxes, fontsize=fontsize)

    # |-x|
    # Plot trivial state
    plot_state(mass=1, length=1.4, ax=ax[1])
    ax[1].text(0, 0.9, '(b)', horizontalalignment='center',
               verticalalignment='center', transform=ax[1].transAxes, fontsize=fontsize)

    # Plot edge occupation as inset of trivial state
    axins = ax[1].inset_axes([0.1, 0.05, 0.4, 0.4])
    x1, x2, y1, y2 = 1.5, 4.5, 0, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)
    filename = "./data/bethe_edge_occupation"
    occupationdata = extract_edge_occupation(filename)
    plot_edge_occupation_data(occupationdata, ax=axins, fontsize=fontsize)

    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.savefig("bethe_edge_states_test.png", bbox_inches='tight')


if __name__ == "__main__":
    main()

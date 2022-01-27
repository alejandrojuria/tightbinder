# Script to plot one topological edge state, one trivial state and the
# edge occupation as a function of m for different bond lengths
# for the Bethe lattice with the Wilson-fermion model

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tightbinder.models import WilsonAmorphous, bethe_lattice
from tightbinder.result import State


def plot_state(mass, length, ax=None, linewidth=1.5):
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
    edge_state.plot_amplitude(ax, linewidth=linewidth)


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
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    nseries = len(data['length'])
    cmap = matplotlib.cm.get_cmap('viridis')
    colors = [cmap(n) for n in np.linspace(0, 1, nseries)]
    for n in range(nseries):
        mass = data['mass'][n]
        occupation = np.array(data['occupation'][n])
        ax.plot(mass, occupation, '-', c=colors[n], linewidth=3)

    ax.legend([rf"l={n}" for n in data['length']],
              fontsize=fontsize, frameon=False).set_zorder(2)
    ax.tick_params('both', labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][0]))
    ax.set_xticks([-1, 1, 3, 5, 7])
    ax.set_ylabel("Edge occupation", fontsize=fontsize)
    ax.set_xlabel("M (eV)", fontsize=fontsize)
    # ax.set_aspect(8)


def main():
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111)
    fontsize = 24

    filename = "./data/bethe_edge_occupation"
    occupationdata = extract_edge_occupation(filename)
    plot_edge_occupation_data(occupationdata, ax=ax, fontsize=fontsize)

    # Topological state inset
    axins = ax.inset_axes([-0.05, 0.5, 0.5, 0.5])
    dummy_axins = ax.inset_axes([0, 0.5, 0.3, 0.45])
    dummy_axins.axis('off')
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)
    plot_state(mass=2.2, length=0.75, ax=axins)
    ax.indicate_inset([2.2, 0.8, 0.1, 0.025], dummy_axins, linestyle="-", linewidth=3, edgecolor="black")

    axins.set_aspect('equal', 'box')

    # Trivial state inset
    axins2 = ax.inset_axes([0.55, 0.5, 0.5, 0.5])
    dummy_axins = ax.inset_axes([0.64, 0.65, 0.32, 0.35])
    dummy_axins.axis('off')
    for axis in ['top', 'bottom', 'left', 'right']:
        axins2.spines[axis].set_linewidth(2)
    plot_state(mass=5, length=1.25, ax=axins2)
    axins2.set_aspect('equal', 'box')
    ax.indicate_inset([5, 0.065, 0.1, 0.025], dummy_axins, linestyle="-", linewidth=3, edgecolor="black")

    plt.tight_layout()
    plt.savefig("bethe_states_insets.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

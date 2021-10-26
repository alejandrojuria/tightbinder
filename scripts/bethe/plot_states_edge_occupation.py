# Script to plot one topological edge state, one trivial state and the
# edge occupation as a function of m for different bond lengths
# for the Bethe lattice with the Wilson-fermion model

import numpy as np
import matplotlib.pyplot as plt
from tightbinder.models import WilsonAmorphous, bethe_lattice
from tightbinder.result import State


def plot_state(mass, length, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplot(111)

    model = WilsonAmorphous(m=mass)
    model.motif, model.bonds = bethe_lattice(z=3, depth=7, length=length)
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
    for n in range(nseries):
        mass = data['mass'][n]
        occupation = np.array(data['occupation'][n])
        ax.plot(mass, occupation, '-', linewidth=3)

    ax.legend([rf"$\Delta r=${n}" for n in data['length']],
              fontsize=fontsize*3/4, frameon=False)
    ax.tick_params('both', labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(data['mass'][0]), np.max(data['mass'][0]))
    ax.set_xticks([-1, 1, 3, 5, 7])
    ax.set_ylabel("Edge occupation", fontsize=fontsize)
    ax.set_xlabel("M (eV)", fontsize=fontsize)


def main():
    fig = plt.figure(figsize=(12, 18), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax = [ax1, ax2, ax3]
    fontsize = 24

    # |xx|
    # |--|
    # Plot topological edge state
    plot_state(mass=2, length=0.8, ax=ax[0])

    # |--|
    # |x-|
    # Plot trivial state
    plot_state(mass=1, length=1.4, ax=ax[1])

    # |--|
    # |-x|
    # Plot edge occupation
    filename = "./data/bethe_edge_occupation"
    occupationdata = extract_edge_occupation(filename)
    plot_edge_occupation_data(occupationdata, ax=ax[2], fontsize=fontsize)

    plt.show()

if __name__ == "__main__":
    main()

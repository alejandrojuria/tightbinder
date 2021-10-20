from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import numpy as np
import matplotlib.pyplot as plt


def find_lowest_eigenstates(n, results):
    energies = np.abs(results.eigen_energy)
    # Find lowest eigenstate
    min_energy = np.min(energies)
    index_min_energy = np.where(energies == min_energy)[0][0]
    states = [results.eigen_states[0][index_min_energy, :]]
    for i in range(1, n//2 + 1):
        states.append(results.eigen_states[0][:, index_min_energy + i])
        states.append(results.eigen_states[0][:, index_min_energy - i])

    return states


def main():
    """ Initialize WilsonFermions model for a M value and compute the average IPR for the 10 lowest eigenstates,
    for all given M values so that we plot average IPR vs M """

    m_values = np.linspace(-1, 7, 20)
    disorder_values = [0.0, 0.5]
    cutoff = 1.4
    cellsize = 20
    n = 21
    wilson = WilsonAmorphous(m=m_values[0], r=cutoff).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    wilson.boundary = "OBC"
    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    ax = [ax1, ax2]
    colors = ['tab:red', 'tab:blue']
    for i, spread in enumerate(disorder_values):
        wilson = amorphize(wilson, spread=spread)
        edge_atoms = wilson.identify_edges()
        edge_occupation = []
        for mass in m_values:
            print(f"Computing IPR for M={mass}...")
            wilson.m = mass
            wilson.initialize_hamiltonian()
            wilson.remove_disconnected_atoms()
            # edge_atoms = wilson.find_lowest_coordination_atoms()
            results = wilson.solve()
            states = find_lowest_eigenstates(n, results)
            edge_occupation.append(
                np.sum(results.calculate_specific_occupation(edge_atoms, states))/n
            )

        ax[i].plot(m_values, edge_occupation, '-', color=colors[i], label=rf"$\Delta r$={spread}")

    plt.title(rf"Edge occupation vs M ($R={cutoff}$)")
    fig.legend(loc="upper right")
    ax1.set_ylabel("Edge occupation", color=colors[0])
    ax2.set_ylabel("Edge occupation", color=colors[1])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    plt.xlabel(r"$M (eV)$")
    plt.show()


if __name__ == "__main__":
    main()

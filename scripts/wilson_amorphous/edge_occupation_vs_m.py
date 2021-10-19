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

    m_values = np.linspace(-3, 10, 30)
    disorder_values = [0.0, 0.5]
    cutoff = 1.1
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
        ipr = []
        for mass in m_values:
            print(f"Computing IPR for M={mass}...")
            wilson.m = mass
            wilson.initialize_hamiltonian()
            results = wilson.solve()
            states = find_lowest_eigenstates(n, results)
            ipr.append(results.calculate_average_ipr(states))

        ax[i].plot(m_values, ipr, '-', color=colors[i], label=rf"$\Delta r$={spread}")

    plt.title(rf"IPR vs M ($R={cutoff}$)")
    fig.legend(loc="upper right")
    ax1.set_ylabel("IPR", color=colors[0])
    ax2.set_ylabel("IPR", color=colors[1])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    plt.xlabel(r"$M (eV)$")
    plt.show()


if __name__ == "__main__":
    main()

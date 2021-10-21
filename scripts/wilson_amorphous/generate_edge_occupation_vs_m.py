from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import numpy as np


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

    file = open('edge_occupation_r11_L20', 'w')
    m_values = np.linspace(-1, 7, 51)
    disorder_values = [0.0, 0.5]
    cutoff = 1.1
    cellsize = 20
    n = 21
    wilson = WilsonAmorphous(m=m_values[0], r=cutoff).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    wilson.boundary = "OBC"
    for i, spread in enumerate(disorder_values):
        file.write(f"{spread}\n")
        wilson = amorphize(wilson, spread=spread)
        # edge_atoms = wilson.identify_edges()
        edge_occupation = []
        for mass in m_values:
            print(f"Computing IPR for M={mass}...")
            wilson.m = mass
            wilson.initialize_hamiltonian()
            wilson.remove_disconnected_atoms()
            edge_atoms = wilson.find_lowest_coordination_atoms()
            results = wilson.solve()
            states = find_lowest_eigenstates(n, results)
            occupation = np.sum(results.calculate_specific_occupation(edge_atoms, states))/n
            edge_occupation.append(occupation)
            file.write(f"{mass}\t{occupation}\n")
        file.write('#\n')

    file.close()


if __name__ == "__main__":
    main()

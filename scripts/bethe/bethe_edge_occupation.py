#!/home/minyn03/alejandro/miniconda3/envs/cc_env/bin/python3 -u

# Script to generate the edge occupation of the Bethe lattice
# as a function of the mass for different bond lengths

from models import bethe_lattice, WilsonAmorphous
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
    file = open("bethe_edge_occupation", "w")
    
    length_values = [1]
    mass_array = np.linspace(-1, 7, 11)
    n = 21
    for length in length_values:
        file.write(f"{length}\n")
        for mass in mass_array:
            print(f"Mass: {mass}, length: {length}")
            model = WilsonAmorphous(m=mass)
            model.motif, model.bonds = bethe_lattice(z=3, depth=8, length=length)
            model.boundary = "OBC"
            model.motif = np.array(model.motif)
            edge_atoms = model.find_lowest_coordination_atoms()
            model.initialize_hamiltonian(find_bonds=False)
            results = model.solve()
            states = find_lowest_eigenstates(n, results)
            edge_occupation = np.sum(
                results.calculate_specific_occupation(edge_atoms, states)) / n

            print(f"Edge occupation: {edge_occupation}\n")
            file.write(f"{mass}\t{edge_occupation}\n")
        file.write("#\n")

    file.close()


if __name__ == "__main__":
    main()

from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np


def main():
    mass = [0.2, 0.4, 0.6, 0.9]
    disorder = [0.5]
    cellsize = 20
    radius = 1.4
    basemodel = WilsonAmorphous(m=mass[0], r=radius).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    basemodel.boundary = "OBC"
    fig, ax = plt.subplots(1, 4, sharey=True)
    for spread in disorder:
        wilson = amorphize(basemodel, spread=spread, planar=True)
        edge_atoms_indices = wilson.identify_edges()
        for j, m in enumerate(mass):
            wilson.m = m
            wilson.initialize_hamiltonian()
            wilson.remove_disconnected_atoms()
            print(f"Coordination number: {wilson.coordination_number()}")
            # edge_atoms_indices = wilson.find_lowest_coordination_atoms()
            result = wilson.solve()
            occupation = result.calculate_specific_occupation(edge_atoms_indices)
            result.plot_quantity(occupation, name=fr"Edge occupation M={m}, $\Delta r$={spread}", ax=ax[j])

    ax[0].set_ylim([0, 1])
    plt.show()


if __name__ == "__main__":
    main()

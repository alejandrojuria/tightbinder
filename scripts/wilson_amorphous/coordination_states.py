from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np


def main():
    mass = [0.5, 1.5, 2.5, 3]
    disorder = [0.0, 0.5]
    cellsize = 20
    radius = 1.1
    basemodel = WilsonAmorphous(m=mass[0], r=radius).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    basemodel.boundary = "OBC"
    fig, ax = plt.subplots(2, 4)
    for i, spread in enumerate(disorder):
        wilson = amorphize(basemodel, spread=spread, planar=True)
        for j, m in enumerate(mass):
            wilson.m = m
            wilson.initialize_hamiltonian()
            print(f"Coordination number: {wilson.coordination_number()}")
            edge_atoms_indices = wilson.find_lowest_coordination_atoms()
            result = wilson.solve()
            occupation = result.calculate_specific_occupation(edge_atoms_indices)
            result.plot_quantity(occupation, name=fr"Edge occupation M={m}, $\Delta r$={spread}", ax=ax[i, j])

    plt.show()

if __name__ == "__main__":
    main()
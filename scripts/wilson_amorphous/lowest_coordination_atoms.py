from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np


def main():
    mass = 2
    disorder = 0.1
    cellsize = 20
    radius = 1.1
    wilson = WilsonAmorphous(m=mass, r=radius).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    wilson = amorphize(wilson, spread=disorder, planar=True)
    wilson.boundary = "OBC"
    wilson.initialize_hamiltonian()
    print(f"Coordination number: {wilson.coordination_number()}")
    edge_atoms_indices = wilson.find_lowest_coordination_atoms()
    edge_atoms = np.array([wilson.motif[i, :] for i in edge_atoms_indices])
    atoms = wilson.motif[:, :3]
    # plt.scatter(atoms[:, 0], atoms[:, 1])
    fig, ax = plt.subplots()
    for n, neighbours in enumerate(wilson.neighbours):
        x0, y0 = atoms[n, :2]
        for atom in neighbours:
            xneigh, yneigh = atoms[atom[0], :2]
            ax.plot([x0, xneigh], [y0, yneigh], "-k")
    ax.scatter(edge_atoms[:, 0], edge_atoms[:, 1], c="b", alpha=0.5)
    ax.axis('off')

    plt.show()



if __name__ == "__main__":
    main()
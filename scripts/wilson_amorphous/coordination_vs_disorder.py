from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np


def main():
    mass = 1
    npoints = 100
    disorder = np.linspace(0, 0.5, npoints)
    samples = 20
    cellsize = 20
    radius = 1.1
    coordination = np.zeros((npoints, 2))
    for n, spread in enumerate(disorder):
        samples_coordination = []
        average_coordination = 0
        deviation = 0
        for _ in range(samples):
            # Important: model has to be reinitialized each time
            # since Python does not make deep copies by default but re-references

            wilson = WilsonAmorphous(m=mass, r=radius).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
            wilson.boundary = "OBC"
            wilson = amorphize(wilson, spread=spread, planar=True)
            wilson.initialize_hamiltonian()
            samples_coordination.append(wilson.coordination_number())
        average_coordination = np.sum(samples_coordination)/samples
        deviation = np.std(samples_coordination)
        coordination[n, :] = [average_coordination, deviation]

    print(len(disorder), len(coordination[:, 0]))
    # plt.errorbar(disorder, coordination[:, 0], yerr=coordination[:, 1], ecolor='k', capsize=3, elinewidth=1)
    plt.plot(disorder, coordination[:, 0], 'k-')
    plt.fill_between(disorder, coordination[:, 0]-coordination[:, 1],
                     coordination[:, 0]+coordination[:, 1], alpha=0.5)
    plt.title("Coordination number vs disorder")
    plt.xlabel(rf"$\Delta r$ ($\AA$)")
    plt.ylabel(rf"Coordination number")
    plt.show()


if __name__ == "__main__":
    main()
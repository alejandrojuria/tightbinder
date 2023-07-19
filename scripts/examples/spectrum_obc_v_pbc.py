from tightbinder.models import AgarwalaChern
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np

def main():

    ncells = 6
    disorder = 0.2
    m = -1 # Regulates the topological behaviour
    cutoff = 1.3 # Cutoff distance to identify neighbours
    np.random.seed(1)

    # Initialize model and amorphize the supercell
    model = AgarwalaChern(m=m, r=cutoff).supercell(n1=ncells, n2=ncells)
    model = amorphize(model, disorder)

    # Obtain the spectrum of the model as it is (PBC)
    model.initialize_hamiltonian()
    results_pbc = model.solve()

    # Change the boundary condition and obtain the OBC spectrum
    model.boundary = "OBC"
    model.initialize_hamiltonian()
    results_obc = model.solve()

    # Plot both spectra into the same Axes object
    fig, ax = plt.subplots(1, 1)
    n = range(model.basisdim)
    ax.scatter(n, results_pbc.eigen_energy, marker="o", edgecolors="black", label="PBC", zorder=2.2)
    ax.scatter(n, results_obc.eigen_energy, marker="o", edgecolors="black", label="OBC", zorder=2.1)
    ax.grid("on")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$E$ (eV)")
    ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
from tightbinder.models import SlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.observables import dos_kpm
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # Set seed for reproducibility
    np.random.seed(1)

    # Parse configuration file and init. model
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)

    ncells = 25
    model = SlaterKoster(config).supercell(n1=ncells, n2=ncells)

    # KPM is intended to be used with sparse matrices.
    # With this the Hamiltonian is stored as a sparse matrix.
    model.matrix_type = "sparse"
    model.initialize_hamiltonian()

    # The density of states computed with the KPM uses the Hamiltonian directly,
    # without need to diagonalize the system first.
    density, energies = dos_kpm(model, nmoments=150, npoints=400, r=30)
    
    # Plot DoS
    fig, ax = plt.subplots(1, 1)
    ax.plot(energies, density, "b-")
    ax.set_ylabel(r"DoS")
    ax.set_xlabel(r"$E$ (eV)")
    ax.grid("on")
    ax.set_xlim([-10, 10])

    # Check normalization
    area = np.trapz(density, energies)
    print(f"Area: {area:.4f}")


if __name__ == "__main__":
    main()
    plt.show()
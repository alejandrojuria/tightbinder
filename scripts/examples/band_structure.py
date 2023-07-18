# Script to obtain and plot the band structure of a Slater-Koster model for hBN.
# To be executed from the root of the repository.

from tightbinder.models import SlaterKoster
from tightbinder.fileparse import parse_config_file
import matplotlib.pyplot as plt

def main():

    # Open and read configuration file
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)

    # Initialize model
    model = SlaterKoster(config)

    # Generate reciprocal path to evaluate the bands
    nk = 100
    labels = config["High symmetry points"]
    kpoints = model.high_symmetry_path(nk, labels)

    # Initialize and solve Bloch Hamiltonian
    model.initialize_hamiltonian()
    results = model.solve(kpoints)

    # Plot band structure
    results.plot_along_path(labels)


if __name__ == "__main__":
    main()
    plt.show()
from tightbinder.models import SlaterKoster
from tightbinder.fileparse import parse_config_file
import matplotlib.pyplot as plt

def main():

    # Parse configuration file
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)

    # Init. model and consider a finite supercell along one Bravais vector
    width = 15
    model = SlaterKoster(config).reduce(n1=width)

    # Create k point mesh
    nk = 100
    labels = ["K", "G", "K"]
    kpoints = model.high_symmetry_path(nk, labels)

    # Initialize Bloch Hamiltonian and obtain the band structure
    model.initialize_hamiltonian()
    results = model.solve(kpoints)
    
    # Plot bands of the ribbon; restrict energy window to [-2, 2] interval.
    results.plot_along_path(labels, e_values=[-2, 2])


if __name__ == "__main__":
    main()
    plt.show()
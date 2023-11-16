from tightbinder.models import SlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import specify_partition_plane, entanglement_spectrum, plot_entanglement_spectrum
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():

    # Parse configuration file
    path = Path(__file__).parent / ".." / "examples" / "inputs" / "Bi111.yaml"
    config = parse_config_file(path)
    
    # Init. model and build ribbon
    width = 7
    model = SlaterKoster(config).reduce(n1=width)
    
    # Declare the plane which defines the partitions for the entanglement spectrum
    plane = [0, 1, 0, np.max(model.motif[:, 1])/2]
    partition = specify_partition_plane(model, plane)

    # Generate k points
    nk = 50
    labels = ["K", "G", "K"]
    kpoints = model.high_symmetry_path(nk, labels)

    # Compute the entanglement spectrum as a function of k
    model.initialize_hamiltonian()
    es = entanglement_spectrum(model, partition, kpoints)

    # Plot the spectrum
    fig, ax = plt.subplots(1, 1)
    plot_entanglement_spectrum(es, model, ax=ax)
    ax.grid("on")


if __name__ == "__main__":
    main()
    plt.show()
from models import WilsonAmorphous
from state import State
from disorder import amorphize
import numpy as np
import topology
import matplotlib.pyplot as plt
import viewer


def main():
    print("Initializing Wilson Amorphous model")
    wilson = WilsonAmorphous(m=2, r=1.4)
    wilson = wilson.reduce(n3=0)
    wilson = wilson.supercell(n1=20, n2=20)
    wilson = amorphize(wilson, spread=0.5)
    # wilson.initialize_hamiltonian()
    # wcc = topology.calculate_wannier_centre_flow(wilson, number_of_points=10, nk_subpath=10)
    # topology.plot_wannier_centre_flow(wcc, show_midpoints=True)
    # z2 = topology.calculate_z2_invariant(wcc)
    # print(f"Invariant is: {z2}")

    # wilson.boundary = "OBC"
    wilson.initialize_hamiltonian()
    wilson.visualize()
    results = wilson.solve()
    edge_states = results.identify_edge_states(wilson)
    try:
        energies = np.abs(results.eigen_energy[edge_states])
        print(f"Energy of edge state: {np.min(energies)}")
        index = np.where(energies == np.min(energies))[0][0]
        edge_state = State(results.eigen_states[0][:, edge_states[index]], wilson)
        edge_state.plot_amplitude()
    except IndexError:
        print("No edge states found")
    print("Finished")

    plt.show()


if __name__ == "__main__":
    main()

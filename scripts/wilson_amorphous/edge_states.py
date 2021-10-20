from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
from tightbinder.result import State
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Initializing Wilson Amorphous model")
    plot_energy_spectrum = False
    wilson = WilsonAmorphous(m=0.9,  r=1.4)
    wilson = wilson.reduce(n3=0)
    wilson = wilson.supercell(n1=20, n2=20)
    wilson = amorphize(wilson, spread=0.5)
    wilson.visualize()

    # wcc = topology.calculate_wannier_centre_flow(wilson, number_of_points=10, nk_subpath=10)
    # topology.plot_wannier_centre_flow(wcc, show_midpoints=True)
    # z2 = topology.calculate_z2_invariant(wcc)
    # print(f"Invariant is: {z2}")

    wilson.boundary = "OBC"
    wilson.initialize_hamiltonian()

    # wilson.visualize()
    results_obc = wilson.solve()

    if plot_energy_spectrum:
        wilson.boundary = "PBC"
        wilson.initialize_hamiltonian()
        results_pbc = wilson.solve([[0., 0., 0.]])
        eigenval_pbc = results_pbc.eigen_energy.reshape(-1, 1)
        eigenval_obc = results_obc.eigen_energy.reshape(-1, 1)
        plt.figure()
        plt.plot(eigenval_obc, 'g+')
        plt.plot(eigenval_pbc, 'b+')
        plt.legend(["OBC", "PBC"])

    # edge_states = results_obc.identify_edge_states(wilson)
    ipr = results_obc.calculate_ipr()
    results_obc.plot_quantity(ipr, sort=True)
    try:
        energies = np.abs(results_obc.eigen_energy)
        print(f"Energy of edge state: {np.min(energies)}")
        index = np.where(energies == np.min(energies))[0][0]
        edge_state = State(results_obc.eigen_states[0][:, index], wilson)
        print(f"IPR={edge_state.compute_ipr()}")
        edge_state.plot_amplitude()
    except IndexError:
        print("No edge states found")
        raise
    except ValueError:
        print("No edge states found")
        raise
    print("Finished")

    plt.show()


if __name__ == "__main__":
    main()

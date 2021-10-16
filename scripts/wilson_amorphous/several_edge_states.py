from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
from tightbinder.result import State
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Initializing Wilson Amorphous model")
    plot_energy_spectrum = False
    [mass, cutoff, cellsize, spread] = [0, 1.1, 20, 0.5]
    wilson = WilsonAmorphous(m=mass,  r=cutoff).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    wilson = amorphize(wilson, spread=spread)
    wilson.boundary = "OBC"
    wilson.initialize_hamiltonian()
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

    fig, ax = plt.subplots(2, 4)
    energies = np.abs(results_obc.eigen_energy)
    print(f"Energy of edge state: {np.min(energies)}")
    index = np.where(energies == np.min(energies))[0][0]
    for i in range(4):
        state_1 = State(results_obc.eigen_states[0][:, index + 2*i], wilson)
        state_2 = State(results_obc.eigen_states[0][:, index - 2*i - 2], wilson)
        ipr1 = state_1.compute_ipr()
        ipr2 = state_2.compute_ipr()
        state_1.plot_amplitude(ax[0, i])
        state_2.plot_amplitude(ax[1, i])
        ax[0, i].set_title(rf"IPR={ipr1}", fontsize=10)
        ax[1, i].set_title(rf"IPR={ipr2}", fontsize=10)

    print("Finished")
    fig.suptitle(rf"$M$={mass}, $\Delta r$={spread}, $R$={cutoff}")
    fig.set_size_inches(18.5, 10.5)
    plt.show()


if __name__ == "__main__":
    main()

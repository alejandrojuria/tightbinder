from tightbinder.models import bethe_lattice, WilsonAmorphous
from tightbinder.result import State
import matplotlib.pyplot as plt
import numpy as np


def main():
    length = 0.8
    mass = 2
    motif, bonds = bethe_lattice(z=3, depth=8, length=length)
    model = WilsonAmorphous(m=mass)
    model.motif = motif
    model.bonds = bonds
    model.boundary = "OBC"

    model.initialize_hamiltonian(find_bonds=False)
    results = model.solve()
    results.plot_spectrum(title=f"Bethe lattice M={mass}")
    abs_energy = np.abs(results.eigen_energy)
    min_energy = min(abs_energy)
    print(min_energy)
    edge_state = np.where(abs_energy == np.min(abs_energy))[0][0]
    state = State(results.eigen_states[0][:, edge_state], model)
    state.plot_amplitude(title=f"Bethe lattice M={mass}, bond length={length}")
    plt.show()


if __name__ == "__main__":
    main()

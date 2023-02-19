from tightbinder.models import SlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.observables import transmission
import numpy as np
import matplotlib.pyplot as plt

def main():
    fig, ax = plt.subplots(1, 2)
    ncell = 4
    file = open("examples/chain.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).reduce(n1=ncell)
    model.initialize_hamiltonian()
    trans, energies = transmission(model, [0], [model.natoms - 1], 0, 3, 200, t=-1, delta=1E-7)
    result = model.solve()
    result.plot_spectrum(ax=ax[0])

    
    ax[1].plot(energies, trans)


if __name__ == "__main__":
    main()
    plt.show()
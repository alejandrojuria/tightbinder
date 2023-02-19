from tightbinder.models import AmorphousSlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.disorder import amorphize, alloy
from tightbinder.observables import integrated_ldos
from tightbinder.modifiers import saturate_bonds
import matplotlib.pyplot as plt
import numpy as np


def main():
    dpi = 500
    file = open("examples/Bi(111).txt", "r")
    config = parse_config_file(file)
    

    ncell = 6
    niter = 50
    for j in range(niter):
        model = AmorphousSlaterKoster(config, r=10)
        model.decay_amplitude = 1
        model.initialize_hamiltonian(override_bond_lengths=True) # First init to find crystalline bond length
        model = model.ribbon(width=ncell)
        model.motif = model.motif[1:model.natoms-1, :] # Create supercell and amorphize
        model = model.reduce(n1=ncell + 2)
        model.motif = model.motif[1:model.natoms-1, :]
        model = amorphize(model, spread=0.4)
        model.initialize_hamiltonian()
        result = model.solve()
        result.rescale_bands_to_fermi_level()

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=dpi)

        all_ildos = []

        for i in range(model.natoms):
            all_ildos.append(integrated_ldos(result, i, -0.1, 0.1))
        
        if j == 0:
            all_ildos_total = np.array(all_ildos)/max(all_ildos)
        else:
            all_ildos_total += np.array(all_ildos)/max(all_ildos)
            
        all_ildos_total /= niter

    ax.scatter(model.motif[:, 0], model.motif[:, 1], c=all_ildos_total, cmap="copper", s=dpi/4)
    ax.axis('equal')


if __name__ == "__main__":
    main()
    plt.savefig("ildos.png", bbox_tight="inches")

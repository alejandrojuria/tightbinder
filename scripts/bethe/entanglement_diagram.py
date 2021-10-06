# Script to generate the gap diagram of the Bethe lattice
# as a function of the hopping parameter t (or equivalently the neighbour distance)
# and the mass parameter of the Wilson-fermion model

from tightbinder.models import bethe_lattice, WilsonAmorphous
from tightbinder.topology import entanglement_spectrum
import numpy as np


def main():
    file = open("bethe_entanglement_diagram", "w")
    length_array = np.linspace(0.5, 1.5, 31)
    mass_array = np.linspace(-1, 7, 31)
    for length in length_array:
        for mass in mass_array:
            print(f"Mass: {mass}, length: {length}")
            bethe = WilsonAmorphous(m=mass)
            bethe.motif, bethe.bonds = bethe_lattice(z=3, depth=8, length=length)
            bethe.motif = np.array(bethe.motif)
            bethe.boundary = "OBC"
            bethe.ordering = "atomic"
            bethe.initialize_hamiltonian(find_bonds=False)

            eps = 1E-3
            plane = [1, 0, 0, np.max(bethe.motif[:, 0]/2) + eps]
            entanglement = entanglement_spectrum(bethe, plane, kpoints=[[0., 0., 0.]])
            file.write(f"{mass}\t{length}\t")
            entanglement = entanglement.reshape(-1, )
            for point in entanglement:
                file.write(f"{point}\t")
            file.write("\n")
    file.close()


if __name__ == "__main__":
    main()

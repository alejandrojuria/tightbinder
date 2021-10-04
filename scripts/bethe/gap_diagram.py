# Script to generate the gap diagram of the Bethe lattice
# as a function of the hopping parameter t (or equivalently the neighbour distance)
# and the mass parameter of the Wilson-fermion model

from tightbinder.models import bethe_lattice, WilsonAmorphous
import numpy as np


def main():
    file = open("bethe_gap_diagram", "w")
    file.write("mass\tlength\tgap\n")
    length_array = np.linspace(0.5, 1.5, 31)
    mass_array = np.linspace(-1, 7, 31)
    for length in length_array:
        for mass in mass_array:
            model = WilsonAmorphous(m=mass)
            model.motif, model.bonds = bethe_lattice(z=3, depth=8, length=length)
            model.boundary = "OBC"

            model.initialize_hamiltonian(find_bonds=False)
            results = model.solve()
            filling = int(model.natoms * model.norbitals * model.filling)
            gap = results.calculate_gap(filling)
            file.write(f"{mass}\t{length}\t{gap}\n")

    file.close()


if __name__ == "__main__":
    main()

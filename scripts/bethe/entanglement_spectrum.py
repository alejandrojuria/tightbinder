from tightbinder.models import WilsonAmorphous, bethe_lattice
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
import matplotlib.pyplot as plt
import numpy as np

mass = 2.2
length = 0.7666666666666666
bethe = WilsonAmorphous(m=mass)
bethe.motif, bethe.bonds = bethe_lattice(z=3, depth=8, length=length)
bethe.motif = np.array(bethe.motif)
bethe.boundary = "OBC"
bethe.ordering = "atomic"
bethe.initialize_hamiltonian(find_bonds=False)

eps = 1E-3
plane = [1, 0, 0, np.max(bethe.motif[:, 0]/2) + eps]
entanglement = entanglement_spectrum(bethe, plane, kpoints=[[0., 0., 0.]])
plot_entanglement_spectrum(entanglement, bethe)

plt.show()

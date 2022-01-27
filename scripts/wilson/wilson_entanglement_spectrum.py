from tightbinder.models import WilsonAmorphous
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np

mass = 1.85
spread = 0.5
r = 1.1
model = WilsonAmorphous(m=mass, r=r)
model = model.reduce(n3=0).supercell(n1=30, n2=30)
model = amorphize(model, spread=spread)
model.ordering = "atomic"
model.boundary = "PBC"
model.initialize_hamiltonian()
plane = [0, 1, 0, np.max(model.motif[:, 1])/2]
kpoints = [[0., 0., 0.]]
entanglement = entanglement_spectrum(model, plane, kpoints)
plot_entanglement_spectrum(entanglement, model, title=rf"M={mass}, $\Delta r=${spread}")

plt.savefig(f"entspec_m{mass}_Dr{spread}_r{r}.png")

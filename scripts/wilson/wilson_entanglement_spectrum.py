from tightbinder.models import WilsonAmorphous
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
import matplotlib.pyplot as plt
import numpy as np

model = WilsonAmorphous(m=1.5, r=1.1)
model.boundary = "PBC"
model.initialize_hamiltonian()
results = model.solve()
results.plot_spectrum()

model = model.reduce(n3=0).supercell(n1=20, n2=20)
model.ordering = "atomic"
model.boundary = "OBC"
model.initialize_hamiltonian()
plane = [0, 1, 0, np.max(model.motif[:, 1])/2]
entanglement = entanglement_spectrum(model, plane)
plot_entanglement_spectrum(entanglement, model)

plt.show()

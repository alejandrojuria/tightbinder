# Script to plot the entanglement spectrum for both periodic and open boundary conditions, to check
# whether there is any relevant difference (meaning I have to redo everything)

from tightbinder.models import WilsonAmorphous
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum, specify_partition_plane
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 3))

# Compute and plot PBC entanglement
model = WilsonAmorphous(m=2, r=1.1)
model = model.reduce(n3=0).supercell(n1=30, n2=30)
model.ordering = "atomic"
model.boundary = "PBC"
model.initialize_hamiltonian()
kpoints = [[0., 0., 0.]]

plane = [0, 1, 0, np.max(model.motif[:, 1])/2]
partition = specify_partition_plane(model, plane)
pbc_entanglement = entanglement_spectrum(model, partition, kpoints)
plot_entanglement_spectrum(pbc_entanglement, model, ax, color="g")

# Compute and plot OBC entanglement
model.boundary = "OBC"
model.initialize_hamiltonian()

plane = [0, 1, 0, np.max(model.motif[:, 1])/2]
partition = specify_partition_plane(model, plane)
obc_entanglement = entanglement_spectrum(model, partition)
plot_entanglement_spectrum(obc_entanglement, model, ax)

# Label plot
fig.legend(["PBC", "OBC"], loc="upper right")
plt.show()

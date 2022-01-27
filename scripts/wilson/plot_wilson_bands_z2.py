from tightbinder.models import WilsonAmorphous
from tightbinder.topology import calculate_wannier_centre_flow, calculate_z2_invariant, plot_wannier_centre_flow
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np


masses = [0.5, 2, 3.5, 5.5]
r = 1.1
fig, ax = plt.subplots(2, len(masses), figsize=(12, 6), dpi=100, sharey='row')
fontsize = 20

for index, m in enumerate(masses):
    # Init model
    model = WilsonAmorphous(m=m, r=r)
    model = model.reduce(n3=0)
    model.ordering = "atomic"
    model.boundary = "PBC"
    model.initialize_hamiltonian()

    # Solve model and plot bands
    labels = ["M", "K", "G", "M"]
    kpoints = model.high_symmetry_path(100, labels)
    results = model.solve(kpoints)
    results.plot_along_path(labels, ax=ax[0, index], fontsize=fontsize)

    # Obtain Z2 index from HWCC evolution
    hwcc = calculate_wannier_centre_flow(model, 20, nk_subpath=50)
    plot_wannier_centre_flow(hwcc, show_midpoints=True, ax=ax[1, index], fontsize=24)
    print(f"Mass: {m}, Z2 invariant: {calculate_z2_invariant(hwcc)}\n")

# Setup the plot correctly
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for n, axis in enumerate(ax.flatten()):
    axis.set_ylabel("")
    for border in ['top', 'bottom', 'left', 'right']:
        axis.spines[border].set_linewidth(2)
    axis.text(0.05, 0.9, '(' + letters[n] + ')', transform=axis.transAxes, fontsize=3*fontsize/4)

for n in range(4):
    ax[0, n].legend([f"M={masses[n]}"], loc='upper right', fontsize=fontsize*2/4)

ax[0, 0].set_ylabel(r'$\epsilon$ (eV)', fontsize=fontsize)
ax[1, 0].set_ylabel(r'$\hat{x}_n$', fontsize=fontsize)

plt.savefig(f"wilson_bands_hwcc.png", bbox_inches='tight')

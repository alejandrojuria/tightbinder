from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import matplotlib.pyplot as plt
import numpy as np

disorder = np.linspace(0, 0.5, 30)
mass = np.linspace(-1, 7, 30)
nsamples = 100
systemsize = 20
filling = int(0.5 * systemsize ** 2)
file = open("wilson_gap", "w")
for m in mass:
    for spread in disorder:
        gaps = []
        print(f"M: {m}, sigma: {spread}\n")
        for i in range(nsamples):
            print(f"Sample {i}")
            model = WilsonAmorphous(m=m, r=1.1).reduce(n3=0)
            model = model.supercell(n1=systemsize, n2=systemsize)
            model = amorphize(model, spread=spread, planar=True)
            model.ordering = "atomic"
            model.boundary = "PBC"
            model.initialize_hamiltonian()
            results = model.solve()
            gap = results.calculate_gap(filling=filling)
            gaps.append(gap)

        average_gap = np.average(gaps)
        variance_gap = np.var(gaps)
        file.write(f"{m}\t{spread}\t{average_gap}\t{variance_gap}\n")

plt.show()

from tightbinder.models import WilsonAmorphous
import matplotlib.pyplot as plt

model = WilsonAmorphous(m=1, r=1.1).reduce(n3=0)
model.initialize_hamiltonian()
labels = ["M", "G", "K", "M"]
kpoints = model.high_symmetry_path(200, labels)
results = model.solve(kpoints)
results.plot_along_path(labels)
plt.show()

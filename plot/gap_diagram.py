import matplotlib.pyplot as plt
import numpy as np


file = open("wilson_2d_diagram_L20_gap_r11", "r")
lines = file.readlines()
mass = []
disorder = []
gap = []
for line in lines:
    line = [float(number) for number in line.split()]
    mass.append(line[0])
    disorder.append(line[1])
    gap.append(line[2])

# Adapt mass and disorder arrays to match pcolor
n = int(np.sqrt(len(mass)))
mass = np.array(mass).reshape(n, n)
dm = mass[0, 1] - mass[0, 0]
mass -= dm/2
extra_mass = np.ones((n, 1))*(mass[0, -1] + dm)
mass = np.append(mass, extra_mass, axis=1)
mass = np.append(mass, mass[:1, :], axis=0)

disorder = np.array(disorder).reshape(n, n)
dd = disorder[1, 0] - disorder[0, 0]
disorder -= dd/2
extra_disorder = np.ones((1, n))*(disorder[-1, 0] + dd)
disorder = np.append(disorder, extra_disorder, axis=0)
disorder = np.append(disorder, disorder[:, :1], axis=1)

gap = np.array(gap).reshape(n, n)

plt.pcolor(mass, disorder, gap)
plt.colorbar()
plt.title("Gap diagram")
plt.xlabel(r"$M$ ($eV$)")
plt.ylabel(r"$\Delta r$ ($\AA$)")
plt.show()

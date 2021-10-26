#! /home/minyn03/alejandro/miniconda3/bin/python3 -u

import tightbinder.models as models
import time
from tightbinder.disorder import amorphize
import numpy as np


def main():
    """ Main routine """
    textfile = open("wilson_2d_diagram_L20_gap_r11", "w")
    mass_parameter_values = np.linspace(-1, 7, 30)
    disorder_values = np.linspace(0, 0.3, 30)
    num_samples = 1
    instances = range(num_samples)
    labels = ["M", "K", "G", "M"]
    for spread in disorder_values:
        for mass in mass_parameter_values:
            print(f"Mass: {mass}, disorder:{spread}")
            header = str(mass) + "\t" + str(spread) + "\t"
            textfile.write(header)
            average_gap = 0
            for i in instances:
                print(f"Instance: {i}")
                wilson = models.WilsonAmorphous(t=1, m=mass, r=1.1)
                wilson = wilson.reduce(n3=0)
                wilson = wilson.supercell(n1=20, n2=20)
                wilson = amorphize(wilson, spread=spread, planar=True)
                wilson.initialize_hamiltonian()
                kpoints = [[0., 0., 0.]]
                results = wilson.solve(kpoints)
                # results.plot_along_path(labels)
                # plt.show()
               
                filling = int(wilson.filling * wilson.natoms * wilson.norbitals)
                gap = results.calculate_gap(filling)
                print(f"Gap is: {gap}")
                average_gap += gap
                string = str(gap) + "\t"
                textfile.write(string)
                print("-------------------------------------------")

            average_gap /= num_samples
            textfile.write(str(average_gap) + "\n")
    textfile.close()


if __name__ == "__main__":
    initial_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - initial_time}s')

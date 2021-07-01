
import matplotlib.pyplot as plt
import numpy as np
from utils import condense_vector


class State:
    def __init__(self, eigenvector, system):
        self.eigenvector = eigenvector
        self.motif = system.motif
        self.norbitals = system.norbitals
        self.hoppings = system.neighbours

    def atomic_amplitude(self):
        """ Method to obtain the probability amplitude corresponding to each atom.
         Returns:
             array amplitude [len(eigenvector)/norbitals] """
        amplitude = np.abs(self.eigenvector) ** 2
        return condense_vector(amplitude, self.norbitals)

    def plot_amplitude(self):
        """ Method to plot the atomic amplitude of the state on top of the crystalline positions"""
        amplitude = np.array(self.atomic_amplitude())
        atoms = self.motif[:, :3]
        # plt.scatter(atoms[:, 0], atoms[:, 1])
        plt.figure()
        for n, neighbours in enumerate(self.hoppings):
            x0, y0 = atoms[n, :2]
            for atom in neighbours:
                xneigh, yneigh = atoms[atom[0], :2]
                plt.plot([x0, xneigh], [y0, yneigh], "-k")
        scaled_amplitude = amplitude*len(amplitude)**1.5
        plt.scatter(atoms[:, 0], atoms[:, 1], c="b", alpha=0.5, s=scaled_amplitude)
        plt.axis('off')
        plt.show()

    def compute_spin_projection(self, axis):
        """ Method to compute the expected value of any of the three spin operators Sx, Sy or Sz.
        All calculations are done supposing that the original atomic basis for the tight-binding is written
        using the z projection of the spin (if the calculation is spinful).
        Parameters:
            char axis: 'x', 'y' or 'z'
        Returns:
             float spin  """

        axis_list = ["x", "y", "z"]
        if axis not in axis_list:
            raise KeyError("Axis must be x, y or z")

        pass  # //////// TO DO






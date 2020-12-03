# Module with all the routines necessary to extract information
# about the crystal from the configuration file: reciprocal space,
# symmetry group and high symmetry points

# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import LinAlgError
import math
import sys

# --------------- Constants ---------------
PI = 3.141592653589793238
EPS = 0.001

# --------------- Routines ---------------


class Crystal:
    def __init__(self, configuration):
        self.name = configuration['System name']
        self.bravais_lattice = configuration['Bravais lattice']
        self.motif = configuration['Motif']
        self.dimension = configuration['Dimensionality']

        # To be initialized in methods
        self.group = None
        self.reciprocal_basis = None
        self.kpoints = None
        self.high_symmetry_points = None

        # Init methods
        self.crystallographic_group()
        self.reciprocal_lattice()
        self.determine_high_symmetry_points()

    def crystallographic_group(self):
        """ Determines the crystallographic group associated to the material from
        the configuration file. NOTE: Only the group associated to the Bravais lattice,
        reduced symmetry due to presence of motif is not taken into account. Its purpose is to
        provide a path in k-space to plot the band structure.
        Workflow is the following: First determine length and angles between basis vectors.
        Then we separate by dimensionality: first we check angles, and then length to
        determine the crystal group. """

        basis_norm = [np.linalg.norm(vector) for vector in self.bravais_lattice]
        group = ''

        basis_angles = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                if j <= i: continue
                v1 = self.bravais_lattice[i]
                v2 = self.bravais_lattice[j]
                basis_angles.append(math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))

        if self.dimension == 2:
            if abs(basis_angles[0] - PI / 2) < EPS:
                if abs(basis_norm[0] - basis_norm[1]) < EPS:
                    group += 'Square'
                else:
                    group += 'Rectangular'

            elif (abs(basis_angles[0] - PI / 3) < EPS or
                  abs(basis_angles[0] - 2 * PI / 3) < EPS) and abs(basis_norm[0] - basis_norm[1]) < EPS:
                group += 'Hexagonal'

            else:
                if abs(basis_norm[0] - basis_norm[1]) < EPS:
                    group += 'Centered rectangular'
                else:
                    group += 'Oblique'

        elif self.dimension == 3:
            print('Work in progress')

        self.group = group

    def reciprocal_lattice(self):
        """ Routine to compute the reciprocal lattice basis vectors from
        the Bravais lattice basis. The algorithm is based on the fact that
        a_i\dot b_j=2PI\delta_ij, which can be written as a linear system of
        equations to solve for b_j. Resulting vectors have 3 components independently
        of the dimension of the vector space they span."""

        reciprocal_basis = np.zeros([self.dimension, 3])
        coefficient_matrix = np.array(self.bravais_lattice)

        if self.dimension == 1:
            reciprocal_basis = 2*PI*coefficient_matrix/(np.linalg.norm(coefficient_matrix)**2)

        else:
            coefficient_matrix = coefficient_matrix[:, 0:self.dimension]
            for i in range(self.dimension):
                coefficient_vector = np.zeros(self.dimension)
                coefficient_vector[i] = 2*PI
                try:
                    reciprocal_vector = np.linalg.solve(coefficient_matrix, coefficient_vector)
                except LinAlgError:
                    print('Error: Bravais lattice basis is not linear independent')
                    sys.exit(1)

                reciprocal_basis[i, 0:self.dimension] = reciprocal_vector

        self.reciprocal_basis = reciprocal_basis

    def brillouin_zone_mesh(self, mesh):
        """ Routine to compute a mesh of the first Brillouin zone using the
        Monkhorst-Pack algorithm. Returns a list of k vectors. """

        if len(mesh) != self.dimension:
            print('Error: Mesh does not match dimension of the system')
            sys.exit(1)

        kpoints = []
        mesh_points = []
        for i in range(self.dimension):
            mesh_points.append(list(range(0, mesh[i] + 1)))

        mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.dimension)

        for point in mesh_points:
            kpoint = 0
            for i in range(self.dimension):
                kpoint += (2.*point[i] - mesh[i])/(2*mesh[i])*self.reciprocal_basis[i]
            kpoints.append(kpoint)

        self.kpoints = np.array(kpoints)

    def determine_high_symmetry_points(self):
        """ Routine to compute high symmetry points depending on the dimension of the system.
        These symmetry points will be used to plot the band structure along the main
        reciprocal paths of the system (irreducible BZ). Returns a dictionary with pairs
        {letter denoting high symmetry point: its coordinates} """

        norm = np.linalg.norm(self.reciprocal_basis[0])
        special_points = {"G": np.array([0., 0., 0.])}
        if self.dimension == 1:
            special_points.update({'K': self.reciprocal_basis[0]/2})

        elif self.dimension == 2:
            special_points.update({'M': self.reciprocal_basis[0]/2})
            if self.group == 'Square':
                special_points.update({'K', self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})

            elif self.group == 'Rectangle':
                special_points.update({'M*', self.reciprocal_basis[1]/2})
                special_points.update({'K',  self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})
                special_points.update({'K*', -self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})

            elif self.group == 'Hexagonal':
                # special_points.update({'K',: reciprocal_basis[0]/2 + reciprocal_basis[1]/2})
                special_points.update({'K': norm/math.sqrt(3)*(self.reciprocal_basis[0]/2 - self.reciprocal_basis[1]/2)/(
                                             np.linalg.norm(self.reciprocal_basis[0]/2 - self.reciprocal_basis[1]/2))})

            else:
                print('High symmetry points not implemented yet')

        else:
            special_points.update({'M': self.reciprocal_basis[0]/2})
            if self.group == 'Cube':
                special_points.update({'M*': self.reciprocal_basis[1]/2})
                special_points.update({'K': self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})

            else:
                print('High symmetry points not implemented yet')

        self.high_symmetry_points = special_points

    def __reorder_high_symmetry_points(self, labels):
        """ Routine to reorder the high symmetry points according to a given vector of labels
         for later representation.
         DEPRECATED with the dictionary update """
        points = []
        for label in labels:
            appended = False
            for point in self.high_symmetry_points:
                if appended:
                    continue
                if label == point[0]:
                    points.append(point)
                    appended = True

        self.high_symmetry_points = points

    def __strip_labels_from_high_symmetry_points(self):
        """ Routine to output both labels and array corresponding to high symmetry points
         DEPRECATED with the dictionary update """

        for n, point in enumerate(self.high_symmetry_points):
            self.high_symmetry_points[n] = point[1]

    def high_symmetry_path(self, nk, points):
        """ Routine to generate a path in reciprocal space along the high symmetry points """

        # self.__reorder_high_symmetry_points(points)
        # self.__strip_labels_from_high_symmetry_points()

        kpoints = []
        number_of_points = len(points)
        interval_mesh = int(nk/number_of_points)
        previous_point = self.high_symmetry_points[points[0]]
        for point in points[1:]:
            next_point = self.high_symmetry_points[point]
            kpoints += list(np.linspace(previous_point, next_point, interval_mesh, endpoint=False))
            previous_point = next_point
        kpoints.append(self.high_symmetry_points[points[-1]])

        self.kpoints = kpoints

    def plot_crystal(self, cell_number=1):
        """
        Method to visualize the crystalline structure (Bravais lattice + motif).
        Parameters:
            (optional) cell_number: Number of cells appended in each direction with respect to the origin
            By default, cell_number = 1 (zero = only motif)
        """

        bravais_lattice = np.array(self.bravais_lattice)
        motif = np.array(self.motif)

        fig = plt.figure()
        ax = Axes3D(fig)
        bravais_vectors_mesh = []
        for i in range(self.dimension):
            bravais_vectors_mesh.append(list(range(-cell_number, cell_number + 1)))
        bravais_vectors_mesh = np.array(np.meshgrid(*bravais_vectors_mesh)).T.reshape(-1, self.dimension)

        bravais_vectors = np.zeros([len(bravais_vectors_mesh[:, 0]), 3])
        for n, coefs in enumerate(bravais_vectors_mesh):
            for i, coef in enumerate(coefs):
                bravais_vectors[n, :] += coef * bravais_lattice[i]

        all_atoms_list = np.zeros([len(bravais_vectors_mesh)*len(motif), 3])
        iterator = 0
        for vector in bravais_vectors:
            for atom in motif:
                all_atoms_list[iterator, :] = atom[:3] + vector
                iterator += 1

        [min_axis, max_axis] = [np.min(all_atoms_list), np.max(all_atoms_list)]

        for atom in all_atoms_list:
            ax.scatter(atom[0], atom[1], atom[2], color='y', s=50)

        ax.set_xlabel('x (A)')
        ax.set_ylabel('y (A)')
        ax.set_zlabel('z (A)')
        ax.set_title(self.name + ' crystal')
        ax.set_xlim3d(min_axis, max_axis)
        ax.set_ylim3d(min_axis, max_axis)
        ax.set_zlim3d(min_axis, max_axis)
        plt.show()




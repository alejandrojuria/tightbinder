# Module with all the routines necessary to extract information
# about the crystal from the configuration file: reciprocal space,
# symmetry group and high symmetry points

# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

from typing import final
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import LinAlgError
import math
import sys
from .utils import generate_basis_combinations


# --------------- Constants ---------------
PI = 3.141592653589793238
EPS = 0.001


# --------------- Routines ---------------
class Crystal:
    """ Implementation of Crystal class. Defaults to square lattice if no input is given. """

    def __init__(self, bravais_lattice=None, motif=None):
        # To be initialized in methods
        self.group = None
        self.reciprocal_basis = None
        self.high_symmetry_points = None

        self._ndim = None
        self.natoms = None
        self._bravais_lattice = None
        self.bravais_lattice = bravais_lattice
        self._motif = None
        self.motif = motif

    @property
    def ndim(self) -> int:
        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        assert type(ndim) == int
        if ndim > 3 or ndim < (-1):
            print("Incorrect dimension for system, exiting...")
            sys.exit(1)
        self._ndim = ndim

    @property
    def bravais_lattice(self) -> np.ndarray:
        return self._bravais_lattice

    @bravais_lattice.setter
    def bravais_lattice(self, bravais_lattice):
        if bravais_lattice is None:
            self._bravais_lattice = None
        else:
            assert type(bravais_lattice) == list or type(bravais_lattice) == np.ndarray
            bravais_lattice = np.array(bravais_lattice)
            if bravais_lattice.shape[1] != 3:
                raise Exception("Vectors must have three components")
            self._bravais_lattice = bravais_lattice
            self._ndim = len(bravais_lattice)
            self.update()

    @property
    def motif(self):
        return self._motif

    @motif.setter
    def motif(self, motif):
        if motif is None:
            self._motif = None
        else:
            assert type(motif) == list or type(motif) == np.ndarray
            for atom in motif:
                if len(atom) != 4:
                    raise Exception("Each position must have three components")
            self._motif = motif
            self.natoms = len(motif)

    def add_atom(self, position, species=0):
        """ Method to add one atom from some numbered species into a specific position.
        Parameters:
            array position: len(3)
            int species: Used to number the species. Defaults to 0 """
        assert type(position) == list or type(position) == np.ndarray
        if len(position) != 3:
            raise Exception("Vector must have three components")

        self.motif.append(position)

    def add_atoms(self, atoms, species=None):
        """ Method to add a list of atoms of specified species at some positions.
         Built on top of method add_atom.
         Parameters:
             matrix atoms: natoms x 3 (each row are the coordinates of an atom)
             list species: list of size natoms """
        if species is None:
            species = np.zeros(len(atoms))
        for n, atom in enumerate(atoms):
            self.add_atom(atom, species[n])

    def remove_atom(self, index):
        if type(self.motif) == list:
            self.motif.pop(index)
        elif type(self.motif) == np.ndarray:
            self.motif = np.delete(self.motif, index, 0)

    def remove_atoms(self, indices):
        indices = np.sort(indices)
        for n, index in enumerate(indices):
            self.remove_atom(index - n)

    def update(self):
        """ Routine to initialize or update the intrinsic attributes of the Crystal class whenever
        the Bravais lattice is changed """
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
        for i in range(self.ndim):
            for j in range(self.ndim):
                if j <= i: continue
                v1 = self.bravais_lattice[i]
                v2 = self.bravais_lattice[j]
                basis_angles.append(math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))

        if self.ndim == 2:
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

        elif self.ndim == 3:
            if (basis_angles[0] - PI/2) < EPS and (basis_angles[1] - PI/2) < EPS and (basis_angles[2] - PI/2) < EPS:
                if abs(basis_norm[0] - basis_norm[1]) < EPS and abs(basis_norm[0] - basis_norm[2]) < EPS and abs(basis_norm[1] - basis_norm[2]) < EPS:
                    group += 'Cube'

            else:
                print('Work in progress')

        self.group = group

    def reciprocal_lattice(self):
        """ Routine to compute the reciprocal lattice basis vectors from
        the Bravais lattice basis. The algorithm is based on the fact that
        a_i\dot b_j=2PI\delta_ij, which can be written as a linear system of
        equations to solve for b_j. Resulting vectors have 3 components independently
        of the dimension of the vector space they span."""

        if self.bravais_lattice is None:
            self.reciprocal_basis = None
            return

        reciprocal_basis = np.zeros([self.ndim, 3])
        coefficient_matrix = np.array(self.bravais_lattice)

        if self.ndim == 1:
            reciprocal_basis = 2*PI*coefficient_matrix/(np.linalg.norm(coefficient_matrix)**2)

        else:
            coefficient_matrix = coefficient_matrix[:, :self.ndim]
            for i in range(self.ndim):
                coefficient_vector = np.zeros(self.ndim)
                coefficient_vector[i] = 2*PI
                try:
                    reciprocal_vector = np.linalg.solve(coefficient_matrix, coefficient_vector)
                except LinAlgError:
                    print('Error: Bravais lattice basis is not linear independent')
                    sys.exit(1)

                reciprocal_basis[i, 0:self.ndim] = reciprocal_vector

        self.reciprocal_basis = reciprocal_basis

    def brillouin_zone_mesh(self, mesh):
        """ Routine to compute a mesh of the first Brillouin zone using the
        Monkhorst-Pack algorithm. Returns a list of k vectors. """

        if len(mesh) != self.ndim:
            print('Error: Mesh does not match dimension of the system')
            sys.exit(1)

        kpoints = []
        mesh_points = []
        for i in range(self.ndim):
            mesh_points.append(list(range(0, mesh[i] + 1)))

        mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, self.ndim)

        for point in mesh_points:
            kpoint = 0
            for i in range(self.ndim):
                kpoint += (2.*point[i] - mesh[i])/(2*mesh[i])*self.reciprocal_basis[i]
            kpoints.append(kpoint)

        return np.array(kpoints)

    def determine_high_symmetry_points(self):
        """ Routine to compute high symmetry points depending on the dimension of the system.
        These symmetry points will be used to plot the band structure along the main
        reciprocal paths of the system (irreducible BZ). Returns a dictionary with pairs
        {letter denoting high symmetry point: its coordinates} """

        norm = np.linalg.norm(self.reciprocal_basis[0])
        special_points = {r"$\Gamma$": np.array([0., 0., 0.])}
        if self.ndim == 1:
            special_points.update({'K': self.reciprocal_basis[0]/2})

        elif self.ndim == 2:
            special_points.update({'M': self.reciprocal_basis[0]/2})
            if self.group == 'Square':
                special_points.update({'K': self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})

            elif self.group == 'Rectangular':
                special_points.update({'X': special_points['M']})
                special_points.update({'Y': self.reciprocal_basis[1]/2})
                special_points.update({'S': self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})

            elif self.group == 'Hexagonal':
                # special_points.update({'K',: reciprocal_basis[0]/3 + reciprocal_basis[1]/3})
                special_points.update({'K': norm/math.sqrt(3)*(self.reciprocal_basis[0]/2 - self.reciprocal_basis[1]/2)/(
                                             np.linalg.norm(self.reciprocal_basis[0]/2 - self.reciprocal_basis[1]/2))})

            elif self.group == 'Oblique' or self.group == 'Centered rectangular':
                if np.linalg.norm(self.bravais_lattice[0]) > np.linalg.norm(self.bravais_lattice[1]):
                    a = np.linalg.norm(self.bravais_lattice[0])
                    b = np.linalg.norm(self.bravais_lattice[1])
                else:
                    a = np.linalg.norm(self.bravais_lattice[1])
                    b = np.linalg.norm(self.bravais_lattice[0])
                gamma = math.acos(np.dot(self.bravais_lattice[0], 
                                         self.bravais_lattice[1])/(a*b))
                eta = (1 - b*math.cos(gamma)/a)/(2*(math.sin(gamma)**2))
                nu = 0.5 - eta*a*math.cos(gamma)/b

                special_points.update({'X': special_points['M']})
                special_points.update({'Y': self.reciprocal_basis[1]/2})
                special_points.update({'C': self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})
                special_points.update({'H': eta*self.reciprocal_basis[0] + (1 - nu)*self.reciprocal_basis[1]})
                special_points.update({r"$H_1$": (1- eta)*self.reciprocal_basis[0] + nu*self.reciprocal_basis[1]})

            else:
                print('High symmetry points not implemented yet')

        else:
            if self.group == 'Cube':
                special_points.update({'X': self.reciprocal_basis[0]/2})
                special_points.update({'M': self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2})
                special_points.update({'R': self.reciprocal_basis[0]/2 + self.reciprocal_basis[1]/2 +
                                            self.reciprocal_basis[2]/2})

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

        # Transform possible Gamma point to latex
        for n, point in enumerate(points):
            if point == "G":
                points[n] = r"$\Gamma$"
            elif point == "H1":
                points[n] = r"$H_1$"

        kpoints = []
        number_of_points = len(points)
        interval_mesh = int(nk/(number_of_points - 1))
        previous_point = self.high_symmetry_points[points[0]]
        for point in points[1:]:
            next_point = self.high_symmetry_points[point]
            kpoints += list(np.linspace(previous_point, next_point, interval_mesh, endpoint=False))
            previous_point = next_point
        kpoints.append(self.high_symmetry_points[points[-1]])

        return kpoints

    def identify_edges(self, loop=6):
        """ Method to obtain the atoms that correspond to the edges of the system.
         This method uses directly the ConvexHull method from Scipy to identify the outmost points
         of the motif. Compute the ConvexHull three times, each time removing the points detected in the
         previous iteration.
          Returns: Indices of atoms belonging to edges of motif
           NB: If given 3d points, they must not be coplanar for the algorithm
           to work. If they are, then the points must be given as 2d """

        atoms_coordinates = np.copy(self.motif[:, :2])
        hull = ConvexHull(atoms_coordinates)
        edge_indices = hull.vertices
        max_x = np.max(atoms_coordinates[:, 0])
        min_x = np.min(atoms_coordinates[:, 0])
        max_y = np.max(atoms_coordinates[:, 1])
        min_y = np.min(atoms_coordinates[:, 1])
        midpoint = np.array([(max_x - min_x)/2, (max_y - min_y)/2])
        for index in edge_indices:
            atoms_coordinates[index, :] = midpoint

        # Repeat this procedure more times
        for _ in range(loop):
            hull = ConvexHull(atoms_coordinates)
            edge_indices = np.concatenate((edge_indices, hull.vertices))
            for index in hull.vertices:
                atoms_coordinates[index, :] = midpoint

        return edge_indices

    def plot_crystal(self, cell_number=1, crystal_name=''):
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
        for i in range(self.ndim):
            bravais_vectors_mesh.append(list(range(-cell_number, cell_number + 1)))
        bravais_vectors_mesh = np.array(np.meshgrid(*bravais_vectors_mesh)).T.reshape(-1, self.ndim)

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
        ax.set_title(crystal_name + ' crystal')
        ax.set_xlim3d(min_axis, max_axis)
        ax.set_ylim3d(min_axis, max_axis)
        ax.set_zlim3d(min_axis, max_axis)
        plt.axis('off')
        plt.show()

    def visualize(self):
        CrystalView(self).visualize()


class CrystalView:
    vp = __import__('vpython')
    def __init__(self, crystal):
        self.bravais_lattice = crystal.bravais_lattice
        self.motif = np.array(crystal.motif)
        self.ndim = crystal.ndim
        self.atoms = []
        self.extra_atoms = []
        self.bonds = []
        self.extra_bonds = []
        self.atom_radius = None

        try:
            self.edge_atoms = crystal.find_lowest_coordination_atoms()
            self.neighbours = crystal.bonds
        except IndexError or AttributeError as e:
            print('visualize(): System must be initialized first')
            raise

        self.vp.scene.visible = False
        self.vp.scene.background = self.vp.color.black
        self.__compute_atom_radius()
        self.__create_atoms()
        self.__create_bonds()

        # Canvas attributes
        self.drag = False
        self.initial_position = None
        self.cell_vectors = None
        self.cell_boundary = None

    def __compute_atom_radius(self):
        first_neigh_distance = 0
        for initial_atom, final_atom, cell, _ in self.neighbours:
            bond_length = np.linalg.norm(self.motif[initial_atom, :3] - self.motif[final_atom, :3] - cell)
            if (bond_length < first_neigh_distance) or first_neigh_distance == 0:
                first_neigh_distance = bond_length

        self.atom_radius = first_neigh_distance/7

    def __create_atoms(self):
        # Origin cell
        mesh_points = generate_basis_combinations(self.ndim)
        color_list = [self.vp.color.yellow, self.vp.color.red]
        for position in self.motif:
            species = int(position[3])
            atom = self.vp.sphere(radius=self.atom_radius, color=color_list[species])
            atom.pos.x, atom.pos.y, atom.pos.z = position[:3]
            self.atoms.append(atom)

        # Super cell
        if self.bravais_lattice is not None:
            for point in mesh_points:
                unit_cell = np.array([0., 0., 0.])
                if np.linalg.norm(point) == 0:
                    continue
                for n in range(self.ndim):
                    unit_cell += point[n] * self.bravais_lattice[n]
                for position in self.motif:
                    species = int(position[3])
                    atom = self.vp.sphere(radius=self.atom_radius, color=color_list[species])
                    atom.visible = False
                    atom.pos.x, atom.pos.y, atom.pos.z = (position[:3] + unit_cell)
                    self.extra_atoms.append(atom)

    def __create_bonds(self):
        mesh_points = generate_basis_combinations(self.ndim)
        try:
            # Origin cell
            for initial_atom, final_atom, cell, nn in self.neighbours:
                unit_cell = self.vp.vector(*cell)
                bond = self.vp.curve(self.atoms[initial_atom].pos, self.atoms[final_atom].pos + unit_cell)
                self.bonds.append(bond)
                if np.linalg.norm(cell) > 1E-4:
                    continue
                    reverse_bond = self.vp.curve(self.atoms[final_atom].pos, self.atoms[initial_atom].pos - unit_cell)
                    self.bonds.append(reverse_bond)

            # Super cell
            for point in mesh_points:
                unit_cell = np.array(([0., 0., 0.]))
                for n in range(self.ndim):
                    unit_cell += point[n] * self.bravais_lattice[n]
                unit_cell = self.vp.vector(unit_cell[0], unit_cell[1], unit_cell[2])
                for bond in self.bonds:
                    supercell_bond = self.vp.curve(bond.point(0)["pos"] + unit_cell, bond.point(1)["pos"] + unit_cell)
                    supercell_bond.visible = False
                    self.extra_bonds.append(supercell_bond)

        except AttributeError:
            print("Warning: To visualize bonds, Viewer must receive an initialized System")

    def visualize(self):
        self.vp.scene.visible = True
        self.vp.button(text="supercell", bind=self.__add_unit_cells)
        self.vp.button(text="primitive unit cell", bind=self.__remove_unit_cells)
        self.vp.button(text="remove bonds", bind=self.__remove_bonds)
        self.vp.button(text="show bonds", bind=self.__show_bonds)
        self.vp.button(text="draw cell boundary", bind=self.__draw_boundary)
        self.vp.button(text="remove cell boundary", bind=self.__remove_boundary)
        self.vp.button(text="highlight edge atoms", bind=self.__highlight_edge)
        self.vp.scene.bind("mousedown", self.__mousedown)
        self.vp.scene.bind("mouseup", self.__mouseup)

    def __mousedown(self):
        self.initial_position = self.vp.scene.mouse.pos
        self.drag = True

    def __mouseup(self):
        if self.drag:
            increment = self.vp.scene.mouse.pos - self.initial_position
            self.initial_position = self.vp.scene.mouse.pos
            self.vp.scene.camera.pos -= increment
        self.drag = False

    def __add_unit_cells(self):
        for atom in self.extra_atoms:
            atom.visible = True

    def __show_bonds(self):
        for bond in self.bonds:
            bond.visible = True
        if self.extra_atoms[0].visible:
            for bond in self.extra_bonds:
                bond.visible = True

    def __remove_unit_cells(self):
        for atom in self.extra_atoms:
            atom.visible = False
        if self.extra_bonds[0].visible:
            for bond in self.extra_bonds:
                bond.visible = False

    def __remove_bonds(self):
        for bond in self.bonds:
            bond.visible = False
        for bond in self.extra_bonds:
            bond.visible = False

    def __draw_boundary(self):
        self.cell_vectors = []
        mid_point = np.array([0., 0., 0.])
        size_vector = [0., 0., 0.]
        for n in range(self.ndim):
            mid_point += self.bravais_lattice[n] / 2
            size_vector[n] = self.bravais_lattice[n][n]
        self.cell_boundary = self.vp.box(pos=self.vp.vector(mid_point[0], mid_point[1], mid_point[2]),
                                    size=self.vp.vector(size_vector[0], size_vector[1], size_vector[2]),
                                    opacity=0.5)
        for basis_vector in self.bravais_lattice:
            unit_cell = self.vp.vector(basis_vector[0], basis_vector[1], basis_vector[2])
            vector = self.vp.arrow(pos=self.vp.vector(0, 0, 0),
                              axis=unit_cell,
                              color=self.vp.color.green,
                              shaftwidth=0.1)
            self.cell_vectors.append(vector)

    def __remove_boundary(self):
        self.cell_boundary.visible = False
        for vector in self.cell_vectors:
            vector.visible = False

    def __highlight_edge(self):
        for atom_index in self.edge_atoms:
            self.atoms[atom_index].color = self.vp.color.green





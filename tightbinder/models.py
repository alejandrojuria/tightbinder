# Module which incorporates several well-known toy models.
# Specifically, we can find implementations of: BHZ model, Wilson-fermions model,
# amorphous Wilson-fermions model
# Note that this models are hard-coded, except for parameters of their Hamiltonians

import numpy as np
import math
import cmath
from system import System, FrozenClass
from crystal import Crystal

# Module-level variables
sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=np.complex_)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=np.complex_)
sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=np.complex_)


class BHZ(System, FrozenClass):
    """
    Implementation of generalized BHZ model. The model is based on a
    2D square lattice of side a=1, and is a four band model. The model takes three parameters:
    g = mixing term
    u = sublattice potential
    c = coupling operator "amplitude"
    """
    def __init__(self, g, u, c):
        super().__init__(system_name="BHZ model", crystal=Crystal([[1, 0, 0], [0, 1, 0]],
                                                                  motif=[[0, 0, 0, 0]]))
        self.norbitals = 4
        self.filling = 0.5
        self.basisdim = self.norbitals
        self.g = g
        self.u = u
        self.c = c

        self._freeze()

    def hamiltonian_k(self, k):
        global sigma_x, sigma_y, sigma_z
        coupling = self.c*sigma_y

        id2 = np.eye(2)
        kx, ky = k[0], k[1]

        hamiltonian = (np.kron(id2, (self.u + np.cos(kx) + np.cos(ky))*sigma_z + np.sin(ky)*sigma_y) +
                       np.kron(sigma_z, np.sin(kx)*sigma_x) + np.kron(sigma_x, coupling) +
                       self.g*np.kron(sigma_z, sigma_y)*(np.cos(kx) + np.cos(7*ky) - 2))

        return hamiltonian


class WilsonFermions2D(System, FrozenClass):
    """
    Implementation of Wilson-fermions model. This model takes a 2D square lattice of side a, it is
    a four band model.
    The hamiltonian takes the following two parameters:
    t = hopping amplitude
    m = mass term
    All parameters default to 1 if not specified otherwise
    """
    def __init__(self, side=1, t=1, m=1):
        super().__init__(system_name="Wilson-fermions 2D", crystal=Crystal([[side, 0, 0], [0, side, 0]],
                                                                           motif=[[0, 0, 0, 0]]))
        self.system_name = "Wilson-fermions model"
        self.norbitals = 4
        self._basisdim = self.norbitals * 1
        self.filling = 0.5

        self.a = side
        self.t = t
        self.m = m

        self._freeze()

    def hamiltonian_k(self, k):
        global sigma_x, sigma_y, sigma_z

        alpha_x = np.kron(sigma_x, sigma_x)
        alpha_y = np.kron(sigma_x, sigma_y)
        beta = np.kron(sigma_z, np.eye(2, dtype=np.complex_))

        hamiltonian = self.t*(np.sin(k[0] * self.a) * alpha_x + np.sin(k[1] * self.a) * alpha_y) + (
                      np.cos(k[0] * self.a) + np.cos(k[1] * self.a) + self.m - 3) * beta

        return hamiltonian


class WilsonFermions3D(System, FrozenClass):
    """
    Implementation of Wilson-fermions model. This model takes a 3D square lattice of side a, it is
    a four band model.
    The hamiltonian takes the following two parameters:
    t = hopping amplitude
    m = mass term
    All parameters default to 1 if not specified otherwise
    """
    def __init__(self, side=1, t=1, m=1):
        super().__init__(system_name="Wilson-fermions 3D",
                         crystal=Crystal([[side, 0, 0], [0, side, 0], [0, 0, side]],
                                         motif=[[0, 0, 0, 0]]))
        self.num_orbitals = 4
        self._basisdim = self.num_orbitals * self.natoms

        self.a = side
        self.t = t
        self.m = m

        self._freeze()

    def hamiltonian_k(self, k):
        global sigma_x, sigma_y, sigma_z

        alpha_x = np.kron(sigma_x, sigma_x)
        alpha_y = np.kron(sigma_x, sigma_y)
        alpha_z = np.kron(sigma_x, sigma_z)
        beta = np.kron(sigma_z, np.eye(2, dtype=np.complex_))

        hamiltonian = (self.t*(np.sin(k[0] * self.a) * alpha_x + np.sin(k[1] * self.a) * alpha_y +
                       np.sin(k[2] * self.a) * alpha_z) +
                       (np.cos(k[0] * self.a) + np.cos(k[1] * self.a) + np.cos(k[2] * self.a) +
                       self.m - 3) * beta)

        return hamiltonian


class WilsonAmorphous(System):
    """
    Implementation of 3D Wilson-fermion model in real-space. This models is a generalization of the
    standard Wilson-fermion model, which uses a cubic unit cell. Instead, this allows for any spatial
    distribution of the atoms, and since the Hamiltonian is built in real-space any number of atoms
    can be considered.
    Parameters:
    side: first-neighbour distance in the crystalline system
    t: Hopping parameter
    m: Mass term
    r: cutoff distance for neighbour interaction
    All parameters default to 1 if not specified otherwise
    """
    def __init__(self, side=1, t=1, m=1, r=1.1):
        super().__init__(system_name="Wilson Amorphous model",
                         crystal=Crystal([[side, 0, 0], [0, side, 0], [0, 0, side]],
                                         motif=[[0, 0, 0, 0]]))

        self.filling = 0.5
        self.norbitals = 4
        self._basisdim = self.norbitals * len(self.motif)
        self.boundary = "PBC"

        self.a = side
        self.t = t
        self.m = m
        self.r = r
        self.parameters = {"C": self.t, "a": self.a, "M": self.m}

    @staticmethod
    def _hopping_matrix(initial_position, final_position, parameters):
        """ Computes hopping matrix according to Wilson-fermion model for two
          given atomic positions.
          :param initial_position: Array 1x3
          :param final_position: Array 1x3
          :param parameters: Dictionary{C, a, M}"""

        x, y, z = final_position - initial_position
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi = math.atan2(y, x)
        theta = math.acos(z / r)

        hopping = np.zeros([4, 4], dtype=np.complex_)
        np.fill_diagonal(hopping, [1, 1, -1, -1])
        offdiag_block = np.array([[-1j * math.cos(theta), -1j * cmath.exp(-1j * phi) * math.sin(theta)],
                                  [-1j * cmath.exp(1j * phi) * math.sin(theta), 1j * math.cos(theta)]],
                                 dtype=np.complex_)
        hopping[0:2, 2:4] = offdiag_block
        hopping[2:4, 0:2] = offdiag_block
        hopping *= 0.5
        hopping *= parameters["C"]*math.exp(1 - r/parameters["a"])

        return hopping

    def initialize_hamiltonian(self):
        """ Routine to initialize the matrices that compose the Bloch Hamiltonian """
        print("Computing neighbours...")
        self.first_neighbours(mode="radius", r=self.r)
        self._determine_connected_unit_cells()

        hamiltonian = []
        for _ in self._unit_cell_list:
            hamiltonian.append(np.zeros(([self._basisdim, self._basisdim]), dtype=np.complex_))

        for n, atom in enumerate(self.motif):
            hamiltonian_atom_block = np.diag(np.array([-3 + self.m, -3 + self.m,
                                                       3 - self.m, 3 - self.m]))
            hamiltonian[0][self.norbitals * n:self.norbitals * (n + 1),
                           self.norbitals * n:self.norbitals * (n + 1)] = hamiltonian_atom_block

        for i, atom in enumerate(self.motif):
            for neighbour in self.neighbours[i]:

                atom_position = atom[:3]
                neigh_index = neighbour[0]
                neigh_unit_cell = list(neighbour[1])
                neigh_position = np.array(self.motif[neigh_index][:3]) + np.array(neigh_unit_cell)
                h_cell = self._unit_cell_list.index(neigh_unit_cell)
                hamiltonian[h_cell][4*i:4*(i + 1),
                                    4*neigh_index:4*(neigh_index + 1)] = self._hopping_matrix(atom_position,
                                                                                              neigh_position,
                                                                                              self.parameters)

        self.hamiltonian = hamiltonian

    def hamiltonian_k(self, k):
        """
        Routine to evaluate the Bloch Hamiltonian at a given k point. It adds the k dependency of the Bloch Hamiltonian
        through the complex exponentials.

        :param k: k vector (Array 1x3)
        :param conditions: defaults to PBC. Can be either PBC or OBC
        :return: Bloch Hamiltonian matrix
        """

        dimension = len(self.hamiltonian[0])
        hamiltonian_k = np.zeros([dimension, dimension], dtype=np.complex_)
        for cell_index, cell in enumerate(self._unit_cell_list):
            hamiltonian_k += (self.hamiltonian[cell_index] * cmath.exp(1j * np.dot(k, cell)))

        return hamiltonian_k



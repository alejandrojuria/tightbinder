# Module which incorporates several well-known toy models.
# Specifically, we can find implementations of: BHZ model, Wilson-fermions model,
# amorphous Wilson-fermions model
# Note that this models are hard-coded, except for parameters of their Hamiltonians

import numpy as np
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
        self.num_orbitals = 4
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
    """
    def __init__(self, side=1, t=1, m=1):
        super().__init__(system_name="Wilson-fermions 2D", crystal=Crystal([[side, 0, 0], [0, side, 0]],
                                                                           motif=[[0, 0, 0, 0]]))
        self.system_name = "Wilson-fermions model"
        self.num_orbitals = 4

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
    """
    def __init__(self, side=1, t=1, m=1):
        super().__init__(system_name="Wilson-fermions 3D",
                         crystal=Crystal([[side, 0, 0], [0, side, 0], [0, 0, side]],
                                         motif=[[0, 0, 0, 0]]))
        self.num_orbitals = 4

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


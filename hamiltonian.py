# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

import numpy as np
import sys

# --------------- Constants ---------------
PI = 3.14159265359



# --------------- Routines ---------------
''' Routine to compute the reciprocal lattice basis vectors from
the Bravais lattice basis. The algorithm is based on the fact that
a_i\dot b_j=2PI\delta_ij, which can be written as a linear system of
equations to solve for b_j '''
def reciprocal_lattice(bravais_lattice):

    dimension = len(bravais_lattice)
    reciprocal_basis = np.zeros(dimension, dimension)
    coefficient_matrix = np.array(bravais_lattice)

    for i in range(dimension):
        coefficient_vector = np.zeros(dimension)
        coefficient_vector[i] = 2*PI
        try:
            reciprocal_basis_vector = np.linalg.solve(coefficient_matrix, coefficient_vector)
        except:
            print('Error: Bravais vectors are not linear indepedent')
            sys.exit(1)
        reciprocal_basis[i, :] = reciprocal_basis_vector
    
    return reciprocal_basis







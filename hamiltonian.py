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
equations to solve for b_j. Resulting vectors have 3 components independently
of the dimension of the vector space they span.'''
def reciprocal_lattice(bravais_lattice):

    dimension = len(bravais_lattice)
    reciprocal_basis = np.zeros([dimension, 3])
    coefficient_matrix = np.array(bravais_lattice)

    if(dimension == 1):
        reciprocal_basis = 2*PI*coefficient_matrix/(np.linalg.norm(coefficient_matrix)**2)
    
    else:
        coefficient_matrix = coefficient_matrix[:, 0:dimension]
        for i in range(dimension):
            coefficient_vector = np.zeros(dimension)
            coefficient_vector[i] = 2*PI
            try:
                reciprocal_vector = np.linalg.solve(coefficient_matrix, coefficient_vector)
            except:
                print('Error: Bravais lattice basis is not linear independent')
                sys.exit(1)
            
            reciprocal_basis[i, 0:dimension] = reciprocal_vector

    return reciprocal_basis


''' Routine to compute a mesh of the first Brillouin zone using the
Monkhorst-Pack algorithm. Returns a list of k vectors. '''
def brillouin_zone_mesh(mesh, reciprocal_basis):

    dimension = len(reciprocal_basis)
    if(len(mesh) != dimension):
        print('Error: Mesh does not match dimension of the system')
        sys.exit(1)

    kpoints = []
    mesh_points = []
    for i in range(dimension):
        mesh_points.append(list(range(1, mesh[i] + 1)))

    mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1,dimension)

    for point in mesh_points:
        kpoint = 0
        for i in range(dimension):
            kpoint += (2.*point[i] - mesh[i])/(2*mesh[i])*reciprocal_basis[i]
        kpoints.append(kpoint)

    return np.array(kpoints)

''' Routine to compute high symmetry points depending on the dimension of the system.
These symmetry points will be used in order to plot the band structure along the main
reciprocal paths of the system. Returns a vector with the principal high symmetry points,
and the letter used to name them. '''
def high_symmetry_points(reciprocal_basis):

    dimension = len(reciprocal_basis)
    special_points = [['Gamma', np.array([0.,0.,0.])]]
    if(dimension == 1):
         special_points.append(['K', reciprocal_basis[0]/2])



        

    return special_points





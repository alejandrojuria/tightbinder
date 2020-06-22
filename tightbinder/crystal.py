# Module with all the routines necessary to extract information
# about the crystal from the configuration file: reciprocal space,
# symmetry group and high symmetry points

# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

import numpy as np
from numpy.linalg import LinAlgError
import math
import sys

# --------------- Constants ---------------
PI = 3.141592653589793238
EPS = 1E-16

# --------------- Routines ---------------


def crystallographic_group(bravais_lattice):
    """ Determines the crystallographic group associated to the material from
    the configuration file. NOTE: Only the group associated to the Bravais lattice,
    reduced symmetry due to presence of motif is not taken into account. Its purpose is to
    provide a path in k-space to plot the band structure.
    Workflow is the following: First determine length and angles between basis vectors.
    Then we separate by dimensionality: first we check angles, and then length to
    determine the crystal group. """

    dimension = len(bravais_lattice)
    basis_norm = [np.linalg.norm(vector) for vector in bravais_lattice]
    group = ''

    basis_angles = []
    for i in range(dimension):
        for j in range(dimension):
            if j <= i: continue
            v1 = bravais_lattice[i]
            v2 = bravais_lattice[j]
            basis_angles.append(math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
    
    if dimension == 2:
        if abs(basis_angles[0] - PI / 2) < EPS:
            if abs(basis_norm[0] - basis_norm[1]) < EPS:
                group += 'Square'
            else:
                group += 'Rectangular'
        
        elif (abs(basis_angles[0] - PI / 3) < EPS or abs(basis_angles[0] - 2 * PI / 3) < EPS) and abs(basis_norm[0] - basis_norm[1]) < EPS:
            group += 'Hexagonal'
        
        else:
            if abs(basis_norm[0] - basis_norm[1]) < EPS:
                group += 'Centered rectangular'
            else:
                group += 'Oblique'
        
    elif dimension == 3:
        print('Work in progress')

    return group


def reciprocal_lattice(bravais_lattice):
    """ Routine to compute the reciprocal lattice basis vectors from
    the Bravais lattice basis. The algorithm is based on the fact that
    a_i\dot b_j=2PI\delta_ij, which can be written as a linear system of
    equations to solve for b_j. Resulting vectors have 3 components independently
    of the dimension of the vector space they span."""

    dimension = len(bravais_lattice)
    reciprocal_basis = np.zeros([dimension, 3])
    coefficient_matrix = np.array(bravais_lattice)

    if dimension == 1:
        reciprocal_basis = 2*PI*coefficient_matrix/(np.linalg.norm(coefficient_matrix)**2)
    
    else:
        coefficient_matrix = coefficient_matrix[:, 0:dimension]
        for i in range(dimension):
            coefficient_vector = np.zeros(dimension)
            coefficient_vector[i] = 2*PI
            try:
                reciprocal_vector = np.linalg.solve(coefficient_matrix, coefficient_vector)
            except LinAlgError:
                print('Error: Bravais lattice basis is not linear independent')
                sys.exit(1)
            
            reciprocal_basis[i, 0:dimension] = reciprocal_vector

    return reciprocal_basis


def brillouin_zone_mesh(mesh, reciprocal_basis):
    """ Routine to compute a mesh of the first Brillouin zone using the
    Monkhorst-Pack algorithm. Returns a list of k vectors. """

    dimension = len(reciprocal_basis)
    if len(mesh) != dimension:
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


def high_symmetry_points(group, reciprocal_basis):
    """ Routine to compute high symmetry points depending on the dimension of the system.
    These symmetry points will be used in order to plot the band structure along the main
    reciprocal paths of the system. Returns a vector with the principal high symmetry points,
    and the letter used to name them. """

    dimension = len(reciprocal_basis)
    special_points = [['Gamma', np.array([0.,0.,0.])]]
    if dimension == 1:
        special_points.append(['K', reciprocal_basis[0]/2])
        
    elif dimension == 2:
        special_points.append(['M', reciprocal_basis[0]/2])
        if group == 'Square':
            special_points.append(['K', reciprocal_basis[0]/2 + reciprocal_basis[1]/2])
        
        elif group == 'Rectangle':
            special_points.append(['M*', reciprocal_basis[1]/2])
            special_points.append(['K',  reciprocal_basis[0]/2 + reciprocal_basis[1]/2])
            special_points.append(['K*',-reciprocal_basis[0]/2 + reciprocal_basis[1]/2])
        
        elif group == 'Hexagonal':
            special_points.append(['K',  reciprocal_basis[0]/2 + reciprocal_basis[1]/2])
            special_points.append(['K*',-reciprocal_basis[0]/2 + reciprocal_basis[1]/2])
        
        else:
            print('High symmetry points not implemented yet')
    
    else:
        special_points.append(['M', reciprocal_basis[0]/2])
        if group == 'Cube':
            special_points.append(['M*', reciprocal_basis[1]/2])
            special_points.append(['K', reciprocal_basis[0]/2 + reciprocal_basis[1]/2])

        else:
            print('High symmetry points not implemented yet')

    return special_points


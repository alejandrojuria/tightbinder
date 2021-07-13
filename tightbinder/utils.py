import numpy as np


def generate_basis_combinations(ndim):
    """ Routine to generate the coefficients corresponding to linear combinations of
     some basis vectors. Possible coefficients for each vector are -1, 0 and 1.
     Parameters:
         int ndim: size of basis
     Returns: matrix 3^ndim x ndim. Each row is [c1, c2, ...] such that
     a linear combination of the basis vectors is given by:
     v = c1*basis1 + c2*basis2 + ... """
    mesh_points = []
    for i in range(ndim):
        mesh_points.append(list(range(-1, 2)))
    mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, ndim)

    return mesh_points


def condense_vector(vector, step):
    """ Routine to reduce the dimensionality of a given vector by summing each consecutive n (step) numbers.
     Parameters:
         array vector lx1: array to be reduced
         int step: reduction step
     Returns:
           """
    if len(vector) % step != 0:
        raise ArithmeticError("Step must divide vector length")
    if step > len(vector):
        raise ArithmeticError("Step can not be larger than vector length")

    reduced_vector = []
    for i in range(len(vector)//step):
        atom_amplitudes = vector[i*step:(i+1)*step]
        reduced_vector.append(np.sum(atom_amplitudes))

    return np.array(reduced_vector)


def scale_array(array, factor=10):
    """ Routine to scale a vector by a factor max_value/max(vector), where max_value is the new maximum value.
    :param array
    :param factor (double). Defaults to 10
    :return scaled_vector """

    n = len(array)
    array = np.array(array) * n * factor

    return array

"""
Miscellaneous routines used in the other modules.
"""


from typing import Callable, Union
import numpy as np
from scipy.spatial import Delaunay


def generate_basis_combinations(ndim: int) -> np.ndarray:
    """ 
    Routine to generate the coefficients corresponding to linear combinations of
    some basis vectors. Possible coefficients for each vector are -1, 0 and 1.

    :param ndim: Dimension of the basis.
    :return: Matrix where each row are the coefficients for a given
        combination of the basis vectors, i.e. [c1, c2, ...] such that
        v = c1*basis_1 + c2*basis_2 + ... 
    """

    mesh_points = []
    if ndim == 0:
        return mesh_points
    for i in range(ndim):
        mesh_points.append(list(range(-1, 2)))
    mesh_points = np.array(np.meshgrid(*mesh_points)).T.reshape(-1, ndim)

    return mesh_points


def condense_vector(vector: Union[list, np.ndarray], step: int) -> np.ndarray:
    """ 
    Routine to reduce the dimensionality of a given vector by summing each consecutive n (step) numbers.

    :param vector: Array to be reduced
    :param step: Number of components to sum
    :returns: Condensed array
    """
    
    if (len(vector) % step) != 0:
        raise ArithmeticError("Step must divide vector length")
    if step > len(vector):
        raise ArithmeticError("Step can not be larger than vector length")

    reduced_vector = []
    for i in range(len(vector)//step):
        atom_amplitudes = vector[i*step:(i+1)*step]
        reduced_vector.append(np.sum(atom_amplitudes))

    return np.array(reduced_vector)


def scale_array(array: Union[list, np.ndarray], factor: int = 10) -> np.ndarray:
    """ 
    Routine to scale a vector by a factor max_value/max(vector), where max_value is the new maximum value.

    :param array: Array to scale.
    :param factor: Factor to multiply the array. Defaults to 10
    :return: Scaled array.
    """

    n = len(array)
    array = np.array(array) * n * factor

    return array


def pretty_print_dictionary(d: dict, indent: int = 0) -> None:
    """
    Routine to pretty print the contents of a dictionary.
    Note that this is mainly used to indent recursively all dict. contents, so by
    default it should be left at zero.

    :param d: Dictionary to pretty print
    :param indent: Number of tabs to indent the contents of the dictionary.
    """

    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_dictionary(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def overrides(interface_class: type) -> Callable:
    """
    Decorator to print whether a method is being overwritten or not.

    :param interface_class: Base class where the method to be overwritten is present.
    :return: Decorator.
    """
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def alpha_shape_2d(points: np.ndarray, alpha: float, only_outer: bool = True):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    From: https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points

    :param points: np.array of shape (n, 2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

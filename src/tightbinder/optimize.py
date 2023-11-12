"""
The optimize module provides several routines to fit the parameters of a SlaterKoster model
to reproduce some given bands, usually from DFT. The optimization is done using the
minimize function from SciPy.
"""


from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize
from tightbinder.models import SlaterKoster
from tightbinder.fileparse import transform_sk_coefficients


def extract_model_parameters(model: SlaterKoster) -> List[float]:
    """
    Routine to obtain the parameters of a Slater-Koster model as a list.
    
    :param model: Model whose number of parameters we want to know.
    :return: Number of parameters.
    """

    parameters = np.array([])
    for onsite_energies in model.configuration["Onsite energy"]:
        parameters = np.concatenate([parameters, onsite_energies])
    
    for distance in model.configuration["SK"].keys():
        for _, coefs in model.configuration["SK"][distance].items():
            parameters = np.concatenate([parameters, coefs])
    
    if model.configuration["Spin"]:
        parameters = np.concatenate([parameters, model.configuration["Spin-orbit coupling"]])

    return parameters


def map_parameters_to_model(model: SlaterKoster, parameters: list) -> None:
    """ 
    Routine to modify the trainable parameters of a SlaterKoster model to those
    given here. Adjustable parameters are onsite energy, slater-koster amplitudes and
    spin-orbit coupling. 
    
    :param model: SlaterKoster model to modify.
    :param parameters: List with the new parameters.
    """

    original_parameters = extract_model_parameters(model)
    expected_paramters = len(original_parameters)

    if len(parameters) != expected_paramters:
        raise ValueError(f"Invalid number of parameters provided: given {len(parameters)}, expected {expected_paramters}")

    counter = 0
    for i, onsite_energies in enumerate(model.configuration["Onsite energy"]):
        nparameters = len(onsite_energies)
        model.configuration["Onsite energy"][i] = parameters[counter:counter+nparameters]
        counter += nparameters
    
    for distance in model.configuration["SK"].keys():
        for species, coefs in model.configuration["SK"][distance].items():
            nparameters = len(coefs)
            model.configuration["SK"][distance][species] = parameters[counter:counter+nparameters]
            counter += nparameters
    
    if model.configuration["Spin"]:
        nparameters = len(model.configuration["Spin-orbit coupling"])
        model.configuration["Spin-orbit coupling"] = parameters[counter:counter+nparameters]

    model.configuration["SK amplitudes"] = model.configuration["SK"]
    transform_sk_coefficients(model.configuration)


def loss_init(model: SlaterKoster, kpoints: np.ndarray, energy: np.ndarray, penalization: np.ndarray = None, map: Callable = None) -> Callable:
    """ 
    Routine to generate the loss function (least squares) in terms of the
    parameters of a given model, evaluated over a specific k path and compared with some bands, 
    typically from DFT, to fit the model bands to 
    
    :param model: SlaterKoster model. Must implement the method solve() to evaluate
        the model.
    :param kpoints: Array of shape (nk, 3) with the kpoints.
    :param energy: Matrix (n, nk) with the energy bands we want to adjust.
    :param penalization: Matrix (n, nk) to penalize specific terms in the loss.
    :return: Loss function to optimize.
    """

    if penalization is None:
        penalization = np.ones(energy.shape)

    def loss(parameters: list) -> float:
        """
        Loss function to fit model bands to DFT bands.
        
        :param parameters: List of parameters of the subyacent model.
        :return: Loss value
        """

        if map is None:
            map_parameters_to_model(model, parameters)
        else:
            map(model, parameters)
        model.initialize_hamiltonian(find_bonds=False, verbose=False)
        results = model.solve(kpoints)
        loss_value = np.sum((results.eigen_energy - energy)**2*penalization)

        return loss_value
    
    return loss



def fit(model: SlaterKoster, kpoints: np.ndarray, energy: np.ndarray, penalization: np.ndarray = None, 
        map: Callable = None, x0: list = None, method: str = None, step: float = 1e-2) -> SlaterKoster:
    """
    Routine to fit the parameters of a SlaterKoster model to reproduce some given bands.

    :param model: SlaterKoster model to fit.
    :param kpoints: Array (nk, 3) with the kpoints where the known bands are given.
    :param energy: Matrix (n, nk) with the energy bands. n must match the dimension of the 
        hamiltonian of the model.
    :param penalization: Matrix (n, nk) to penalize specific bands or kpoints in the loss.
    :param map: Custom function to map the parameters to the model. Defaults to None.
    :param x0: Initial guess for optimization. Defaults to automatic parameter extraction if None.
    :param method: Specify optimization method. To see available options, see scipy.minimize documentation.
    :param step: Value of change of parameters when minimizing. Defaults to 1e-2.
    :return: Model with the fitted parameters.
    """

    loss: Callable = loss_init(model, kpoints, energy, penalization, map=map)
    if x0 is None:
        x0 = extract_model_parameters(model)
    optimization = minimize(loss, x0, method=method, options={'eps': step})
    fitted_parameters = optimization.x
    print(f"{optimization.message}")
    print(f"Fitted parameters: {fitted_parameters}")

    if map is None:
        map_parameters_to_model(model, fitted_parameters)
    else:
        map(model, fitted_parameters)

    return model


def read_bands_from_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Routine to read some bands from a file.
    
    :param filename: Route to file relative to current working directory.
    :return: Array (nk, 3) with the kpoints, and matrix (n, nk) with the
        energy bands.
    """

    file = open(filename, "r")
    lines = file.readlines()
    iterator = iter(lines)
    kpoints, bands = [], []
    for line in iterator:
        kpoint = [float(number) for number in line.split()]
        energies = [float(number) for number in next(iterator).split()]

        kpoints.append(kpoint)
        bands.append(energies)

    kpoints = np.array(kpoints)
    bands = np.array(bands)

    return kpoints, bands


from tightbinder.models import SlaterKoster
from tightbinder.disorder import amorphize, introduce_vacancies, introduce_impurities
from tightbinder.fileparse import parse_config_file
from tightbinder.utils import hash_numpy_array
import numpy as np
import math
from pathlib import Path

INPUT_FILE_DIR = Path(__file__).parent / ".." / "examples" / "inputs"
"""
Tests to check functionality of the disorder module.
"""


def test_amorphize():
    """
    Tests that amorphize can be applied to a model.
    """
    
    np.random.seed(1)
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)

    model = SlaterKoster(config).supercell(n1=2, n2=2)
    model = amorphize(model, 0.2)

    motif_hash = hash_numpy_array(model.motif)
    
    assert math.isclose(motif_hash, 322.32157833573854)
        

def test_vacancies():
    """
    Tests that introduce_vacancies can be applied to a model.
    """
    
    np.random.seed(1)
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)

    model = SlaterKoster(config).supercell(n1=2, n2=2)
    model = introduce_vacancies(model, 0.2)

    motif_hash = hash_numpy_array(model.motif)
    
    assert math.isclose(motif_hash, 171.9359264901)
    

def test_impurities():
    """
    Tests that introduce_impurities can be applied to a model.
    """
    
    np.random.seed(1)
    
    path = INPUT_FILE_DIR / "hBN.yaml"
    config = parse_config_file(path)

    model = SlaterKoster(config).supercell(n1=2)
    model.initialize_hamiltonian()

    model = introduce_impurities(model, 10, 0.9)

    hamiltonian_hash = np.real(hash_numpy_array(model.hamiltonian[0]))
    
    assert math.isclose(hamiltonian_hash, 431.73999999999995)
    
    

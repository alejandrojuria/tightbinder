from tightbinder.models import SlaterKoster
from tightbinder.disorder import amorphize, introduce_vacancies, introduce_impurities
from tightbinder.fileparse import parse_config_file
from tightbinder.utils import hash_numpy_array
import numpy as np

"""
Tests to check functionality of the disorder module.
"""


def test_amorphize():
    """
    Tests that amorphize can be applied to a model.
    """
    
    np.random.seed(1)
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config).supercell(n1=2, n2=2)
    model = amorphize(model, 0.2)
    
    motif_hash = hash_numpy_array(model.motif)
        
    assert motif_hash == "82e4913237df4fa0a84244c5857d443e"
        

def test_vacancies():
    """
    Tests that introduce_vacancies can be applied to a model.
    """
    
    np.random.seed(1)
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config).supercell(n1=2, n2=2)
    model = introduce_vacancies(model, 0.2)
    
    motif_hash = hash_numpy_array(model.motif)
        
    assert motif_hash == "1adc6630cb9136d79aa6efafe86a8387"
    

def test_impurities():
    """
    Tests that introduce_impurities can be applied to a model.
    """
    
    np.random.seed(1)
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config).supercell(n1=2)
    model.initialize_hamiltonian()
    
    model = introduce_impurities(model, 10, 0.9)
    
    hamiltonian_hash = hash_numpy_array(model.hamiltonian[0])
        
    assert hamiltonian_hash == "93609010a8c3a2f2ebae75938aa75e9a"
    
    
    
    
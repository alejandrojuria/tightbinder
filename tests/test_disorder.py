from tightbinder.models import SlaterKoster
from tightbinder.disorder import amorphize, introduce_vacancies, introduce_impurities
from tightbinder.fileparse import parse_config_file
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
        
    expected_motif = np.array([[-1.96545960e-01,  1.48990382e-01,  6.55557867e-02,  0.00000000e+00],
                               [ 2.19673100e+00, -1.04642199e-01, -1.66581569e+00,  0.00000000e+00],
                               [ 3.92584389e+00,  2.26661453e+00,  6.32944216e-05,  0.00000000e+00],
                               [ 6.50041277e+00,  2.16755798e+00, -1.43467506e+00,  0.00000000e+00],
                               [ 3.94072365e+00, -2.21612250e+00, -7.27792726e-02,  0.00000000e+00],
                               [ 6.54716456e+00, -2.27049876e+00, -1.64122624e+00,  0.00000000e+00],
                               [ 7.94529402e+00,  1.62597119e-02,  6.30498800e-02,  0.00000000e+00],
                               [ 1.03855896e+01, -1.52747786e-01, -1.70513512e+00,  0.00000000e+00]])
    
    assert np.allclose(model.motif, expected_motif)
    

def test_vacancies():
    """
    Tests that introduce_vacancies can be applied to a model.
    """
    
    np.random.seed(1)
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config).supercell(n1=2, n2=2)
    model = introduce_vacancies(model, 0.2)
            
    expected_motif = np.array([[ 0.,       0.,       0.,       0.,    ],
                               [ 2.61724,  0.,      -1.585,    0.     ],
                               [ 6.54311,  2.2666,  -1.585,    0.     ],
                               [10.46898,  0.,      -1.585,    0.     ]])
    
    assert np.allclose(model.motif, expected_motif)
    

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
        
    expected_hamiltonian = [[10. +0.j, -2.3+0.j,  0. +0.j,  0. +0.j],
                            [-2.3+0.j, 10. +0.j, -2.3+0.j,  0. +0.j],
                            [ 0. +0.j, -2.3+0.j, 10. +0.j, -2.3+0.j],
                            [ 0. +0.j,  0. +0.j, -2.3+0.j, 10. +0.j]]
                
    assert np.allclose(model.hamiltonian[0], expected_hamiltonian)
    
    
    
    
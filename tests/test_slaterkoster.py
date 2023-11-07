from tightbinder.models import SlaterKoster
from tightbinder.fileparse import parse_config_file
import numpy as np

"""
This file constains tests for both the System class and the SlaterKoster class, which inherits from System.
The approach taken here is to test the SlaterKoster class, and then to test the System class by extension.
These constitute functional tests of the library. 
"""

def test_configuration_parsing():
    """
    Function to test the parsing of a configuration file.
    """

    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)

    assert config['Dimensionality'] == 2
    
    assert len(config['Bravais lattice']) == 2
    assert config['Bravais lattice'][0] == [2.16506, 1.25, 0.0]
    assert config['Bravais lattice'][1] == [2.16506, -1.25, 0.0]
    
    assert config['Species'] == 2
    
    assert len(config['Motif']) == 2
    assert config['Motif'][0] == [0, 0, 0, 0]
    assert config['Motif'][1] == [1.443376, 0, 0, 1]
    
    assert len(config['Filling']) == 2
    assert config['Filling'][0] == 0.5
    assert config['Filling'][1] == 0.5
    
    assert len(config['Orbitals']) == 2
    assert config['Orbitals'][0] == ['s']
    assert config['Orbitals'][1] == ['s']
    
    assert len(config['Onsite energy']) == 2
    assert config['Onsite energy'][0] == [3.625]
    assert config['Onsite energy'][0] == [3.625]
    
    assert len(config['SK amplitudes'].keys()) == 1
    assert config['SK amplitudes']['1']['01'] == [-2.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    assert config['Spin'] == False 
    
    assert len(config['Spin-orbit coupling']) == 2
    assert config['Spin-orbit coupling'][0] == 0
    assert config['Spin-orbit coupling'][1] == 0
    
    assert len(config['Mesh']) == 2
    assert config['Mesh'][0] == 100
    assert config['Mesh'][1] == 100
    
    assert config['High symmetry points'] == ['G', "K", 'M', "G"]
    

def build_slaterkoster() -> SlaterKoster:
    """
    Function to build a SlaterKoster model from the configuration file.
    """
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config)
    
    return model


def test_sk_initialization():
    
    model = build_slaterkoster()
    

def test_supercell():
    """
    Routine to test the supercell method from System with a basic SlaterKoster model.
    """
    
    model = build_slaterkoster().supercell(n1=2, n2=2)
    
    assert model.boundary == "PBC"
    
    assert model.natoms == 8    
    assert np.allclose(model.bravais_lattice[0], [4.330128, 2.5, 0.0])
    assert np.allclose(model.bravais_lattice[1], [4.330128, -2.5, 0.0])    
    
    expected_motif = np.array([[0, 0, 0, 0],
                                [1.443376, 0, 0, 1],
                                [2.16506, 1.25, 0.0, 0],
                                [3.608436, 1.25, 0.0, 1],
                                [2.16506, -1.25, 0.0, 0],
                                [3.608436, -1.25, 0.0, 1],
                                [4.330128, 0, 0, 0],
                                [5.773504, 0, 0, 1]])
        
    assert np.allclose(model.motif, expected_motif)


def test_reduce():
    """
    Tests reduce method of System for generation of supercells with open boundary conditions.
    """
    
    model = build_slaterkoster().reduce(n1=2, n2=2)
    
    assert model.boundary == "OBC"    
    assert model.natoms == 8    
    
    expected_motif = np.array([[0, 0, 0, 0],
                                [1.443376, 0, 0, 1],
                                [2.16506, 1.25, 0.0, 0],
                                [3.608436, 1.25, 0.0, 1],
                                [2.16506, -1.25, 0.0, 0],
                                [3.608436, -1.25, 0.0, 1],
                                [4.330128, 0, 0, 0],
                                [5.773504, 0, 0, 1]])
        
    assert np.allclose(model.motif, expected_motif)
    
    
def test_add_bonds():
    """
    Tests the method add_bonds from System.
    """
    
    model = build_slaterkoster()
    model.add_bonds([0, 0], [0, 1], model.bravais_lattice)
    
    assert model.bonds[0][0] == 0
    assert model.bonds[0][1] == 0
    assert np.allclose(model.bonds[0][2], model.bravais_lattice[0])
    
    assert model.bonds[1][0] == 0
    assert model.bonds[1][1] == 1
    assert np.allclose(model.bonds[1][2], model.bravais_lattice[1])
        
    model.add_bond(1, 1, model.bravais_lattice[0] + model.bravais_lattice[1], '2')
    
    assert model.bonds[2][0] == 1
    assert model.bonds[2][1] == 1
    assert np.allclose(model.bonds[2][2], model.bravais_lattice[0] + model.bravais_lattice[1])
    assert model.bonds[2][3] == '2'
    

def test_neighbours():
    """
    Tests the method find_neighbours from System.
    """
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config)
    model.find_neighbours()
    
    assert model.bonds[0][0] == 0
    assert model.bonds[0][1] == 1
    assert np.allclose(model.bonds[0][2], np.array([-2.16506, -1.25   ,  0.     ]))
    
    assert model.bonds[1][0] == 0
    assert model.bonds[1][1] == 1
    assert np.allclose(model.bonds[1][2], np.array([-2.16506,  1.25   ,  0.     ]))
    
    assert model.bonds[2][0] == 0
    assert model.bonds[2][1] == 1
    assert np.allclose(model.bonds[2][2], np.array([0., 0., 0.]))
    
    assert model.bonds[3][0] == 1
    assert model.bonds[3][1] == 0
    assert np.allclose(model.bonds[3][2], np.array([0., 0., 0.]))
    
    assert model.bonds[4][0] == 1
    assert model.bonds[4][1] == 0
    assert np.allclose(model.bonds[4][2], np.array([2.16506, -1.25   , 0.     ]))
    
    assert model.bonds[5][0] == 1
    assert model.bonds[5][1] == 0
    assert np.allclose(model.bonds[5][2], np.array([2.16506, 1.25   , 0.     ]))
    

def test_high_symmetry_path():
    """
    Tests that the high symmetry path specified is correctly generated.
    """
    
    model = build_slaterkoster()
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(10, ['G', 'K', 'M', 'G'])
        
    assert np.allclose(kpoints[0], np.array([0., 0., 0.]))
    assert np.allclose(kpoints[1], np.array([0.        , 0.55850559, 0.        ])) 
    assert np.allclose(kpoints[2], np.array([0.        , 1.11701117, 0.        ]))
    assert np.allclose(kpoints[3], np.array([0.        , 1.67551676, 0.        ]))
    assert np.allclose(kpoints[4], np.array([0.24184031, 1.53589019, 0.        ]))
    assert np.allclose(kpoints[5], np.array([0.48368061, 1.39626363, 0.        ]))
    assert np.allclose(kpoints[6], np.array([0.72552092, 1.25663706, 0.        ]))
    assert np.allclose(kpoints[7], np.array([0.48368061, 0.83775804, 0.        ]))
    assert np.allclose(kpoints[8], np.array([0.24184031, 0.41887902, 0.        ]))
    assert np.allclose(kpoints[9], np.array([0., 0., 0.]))    
        
        
def test_bloch_hamiltonian():
    """
    Method to test the generation of the Bloch Hamiltonian from the results of the diagonalization.
    """
    
    model = build_slaterkoster()
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(10, ['G', 'K', 'M', 'G'])
    results = model.solve(kpoints)
    
    assert np.allclose(results.eigen_energy[0], 
                       np.array([-7.79426873, -6.85983334, -4.76896809, -3.625,      -3.82600578, -4.1510816,                 
                        -4.29309038, -5.38615122, -7.08312255, -7.79426873]))
    assert np.allclose(results.eigen_energy[1], 
                       np.array([7.79426873,  6.85983334,  4.76896809,  3.625,       3.82600578,  4.1510816,
                        4.29309038,  5.38615122,  7.08312255,  7.79426873]))    
    
    
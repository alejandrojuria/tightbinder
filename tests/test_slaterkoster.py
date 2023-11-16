from tightbinder.models import SlaterKoster, AmorphousSlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.disorder import amorphize
from tightbinder.utils import hash_numpy_array
import numpy as np
import math
from pathlib import Path

INPUT_FILE_DIR = Path(__file__).parent / ".." / "examples" / "inputs"
"""
This file constains tests for the main classes of the library, i.e. SlaterKoster and AmorphousSlaterKoster.
This also serves to test the base classes System and Crystal, which are inherited by the former.
All tests are functional, as they intend to check specific functionalities of the library and not 
specific routines individually.
"""

def test_configuration_parsing():
    """
    Function to test the parsing of a configuration file.
    """

    path = INPUT_FILE_DIR / "hBN.yaml"
    config = parse_config_file(path)

    assert config['Dimensions'] == 2
    
    assert len(config['Lattice']) == 2
    assert config['Lattice'][0] == [2.16506, 1.25, 0.0]
    assert config['Lattice'][1] == [2.16506, -1.25, 0.0]
    
    assert len(config['Species']) == 2
    
    assert len(config['Motif']) == 2
    assert config['Motif'][0] == [0, 0, 0, 0]
    assert config['Motif'][1] == [1.443376, 0, 0, 1]
    
    assert len(config['Filling']) == 2
    assert config['Filling'][0] == 0.5
    assert config['Filling'][1] == 0.5
    
    assert len(config['Orbitals']) == 2
    assert config['Orbitals'][0] == ['s']
    assert config['Orbitals'][1] == ['s']
    
    assert len(config['OnsiteEnergy']) == 2
    assert config['OnsiteEnergy'][0] == [3.625]
    assert config['OnsiteEnergy'][0] == [3.625]
    
    assert len(config['SKAmplitudes'].keys()) == 1
    assert config['SKAmplitudes']['1']['01'] == [-2.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    assert config['Spin'] == False 
    
    assert len(config['SOC']) == 2
    assert config['SOC'][0] == 0
    assert config['SOC'][1] == 0
    
    assert len(config['Mesh']) == 2
    assert config['Mesh'][0] == 100
    assert config['Mesh'][1] == 100
    
    assert config['SymmetryPoints'] == ['G', "K", 'M', "G"]
    

def build_slaterkoster() -> SlaterKoster:
    """
    Function to build a SlaterKoster model from the configuration file.
    """
    
    path = INPUT_FILE_DIR / "hBN.yaml"
    config = parse_config_file(path)
        
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
    
    motif_hash = hash_numpy_array(model.motif)
    
    assert math.isclose(motif_hash, 99.833119893184)
        

def test_reduce():
    """
    Tests reduce method of System for generation of supercells with open boundary conditions.
    """
    
    model = build_slaterkoster().reduce(n1=2, n2=2)
    
    assert model.boundary == "OBC"    
    assert model.natoms == 8    
    
    motif_hash = hash_numpy_array(model.motif)
        
    assert math.isclose(motif_hash, 99.833119893184)
    
    
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
    

def test_neighbour_distances():
    """
    Tests that neighbour distances are computed correctly.
    """    
    
    model = build_slaterkoster().supercell(n1=4, n2=4)
    model.initialize_hamiltonian()
    
    distances = model.compute_neighbour_distances(5)
        
    assert np.allclose(distances, [1.44, 2.5,  2.89, 3.82, 4.33])
    

def test_neighbours():
    """
    Tests the method find_neighbours from System.
    """
    
    model = build_slaterkoster()
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
    
    kpoints_hash = hash_numpy_array(np.array(kpoints))
        
    assert math.isclose(kpoints_hash, 12.243192500457077)
        
        
def test_bloch_hamiltonian():
    """
    Method to test the generation of the Bloch Hamiltonian from the results of the diagonalization.
    """
    
    model = build_slaterkoster()
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(10, ['G', 'K', 'M', 'G'])
    results = model.solve(kpoints)
    
    energies_hash = hash_numpy_array(results.eigen_energy)
        
    assert math.isclose(energies_hash, 667.8478374529238)
    

def test_fermi_energy():
    """
    Tests that the Fermi energy is computed properly.
    """
    
    model = build_slaterkoster()
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(100, ['G', 'K', 'M', 'G'])
    results = model.solve(kpoints)
    
    fermi_energy = results.calculate_fermi_energy(model.filling)
    
    assert math.isclose(fermi_energy, -3.625)
    
    
def test_fermi_energy_supercell():
    """
    Tests that the Fermi energy is computed properly when working with supercells.
    """
    
    model = build_slaterkoster().supercell(n1=2, n2=2)
    model.initialize_hamiltonian()
    
    labels = ['G', 'K', 'M', 'G']
    kpoints = model.high_symmetry_path(100, labels)
    results = model.solve(kpoints)
    
    fermi_energy = results.calculate_fermi_energy(model.filling)
    
    assert math.isclose(fermi_energy, -3.625)
    

def test_rescale_bands_to_fermi_energy():
    """
    Tests method to set the Fermi energy to zero.
    """
    
    model = build_slaterkoster()
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(100, ['G', 'K', 'M', 'G'])
    results = model.solve(kpoints)
    
    results.rescale_bands_to_fermi_level()
    fermi_energy = results.calculate_fermi_energy(model.filling)
    
    assert math.isclose(fermi_energy, 0)
    
    
def test_soc_bands():
    """
    Tests that SOC is correctly implemented.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)

    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(10, ['G', 'K', 'M', 'G'])
    results = model.solve(kpoints)
        
    results.rescale_bands_to_fermi_level()
    
    energies_hash = hash_numpy_array(results.eigen_energy)
        
    assert math.isclose(energies_hash, 5155.652053621002)
    

def test_edge_bands():
    """
    Tests that edge bands appear on the band structure of the ribbon of a topological insulator.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
    
    model = SlaterKoster(config).reduce(n1=5)
    model.initialize_hamiltonian()
    
    kpoints = model.high_symmetry_path(10, ['K', 'G', 'K'])
    results = model.solve(kpoints)
        
    results.rescale_bands_to_fermi_level()
    
    energies_hash = hash_numpy_array(results.eigen_energy)
            
    assert math.isclose(energies_hash, 29453.85050171607)
    

def test_amorphous_slater_koster():
    """
    Tests that the AmorphousSlaterKoster model can be initialized, disordered and the band structure can be obtained correctly.
    """
    
    np.random.seed(1)
    
    # Parameters of the calculation
    ncells = 4
    disorder = 0.1

    # Parse configuration file
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)

    # Init. model and construct supercell
    first_neighbour_distance = np.linalg.norm(config["Motif"][1][:3])
    cutoff = first_neighbour_distance * 1.4
    
    model = AmorphousSlaterKoster(config, r=cutoff).supercell(n1=ncells, n2=ncells)
    model.decay_amplitude = 1

    model = amorphize(model, disorder)
    model.initialize_hamiltonian(override_bond_lengths=True)

    results = model.solve()
    
    energies_hash = hash_numpy_array(results.eigen_energy)
                
    assert math.isclose(energies_hash, 8595.611419662699)
    
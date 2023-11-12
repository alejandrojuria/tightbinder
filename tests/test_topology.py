from tightbinder.topology import calculate_wannier_centre_flow, calculate_z2_invariant, calculate_chern_number, chern_marker, specify_partition_plane, entanglement_spectrum, plot_wannier_centre_flow
from tightbinder.models import SlaterKoster, AgarwalaChern
from tightbinder.fileparse import parse_config_file
from tightbinder.utils import hash_numpy_array
import numpy as np
import matplotlib.pyplot as plt

"""
Module with tests to check the topology module.
"""

def test_wannier_centre_flow():
    """
    Tests the calculation of the wannier centre flow for a SlaterKoster model.
    """
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 5, refine_mesh=False)
    
    wcc_hash = hash_numpy_array(wcc)
        
    assert wcc_hash == "e0880afb289b010bf41270bf32cdf954"
    

def test_wannier_centre_flow_refined():
    """
    Tests the calculation of the wannier centre flow but with a refined mesh.
    """
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 5)
    
    wcc_hash = hash_numpy_array(wcc)
        
    assert wcc_hash == "6fbdd484ca9e10531b960358df2f667c"
    

def test_z2_invariant():
    """
    Tests the z2 invariant calculation algorithm.
    """
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, refine_mesh=False)
    z2 = calculate_z2_invariant(wcc)
    
    assert z2 == 1
    

def test_z2_phase_diagram():
    """
    Tests that the z2 invariant calculation algorithm works to determine a phase diagram, changing the spin-orbit coupling.
    """
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, refine_mesh=False)
    z2 = calculate_z2_invariant(wcc)
    
    assert z2 == 1
    
    model.configuration["Spin-orbit coupling"][0] = 0.5
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, refine_mesh=False)
    z2 = calculate_z2_invariant(wcc)
    
    assert z2 == 0
    
    
def test_chern_wcc():
    """
    Tests calculation of Wannier centers for a Chern insulator.
    """
    
    model = AgarwalaChern(m=-1)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, full_BZ=True, refine_mesh=False)
    
    wcc_hash = hash_numpy_array(wcc)
        
    assert wcc_hash == "1040b618750741aaef82451ecd13a0bb"
    

def test_chern_number():
    """
    Tests calculation of Chern number.
    """
    
    model = AgarwalaChern(m=-1)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, full_BZ=True)
    chern = calculate_chern_number(wcc)
    
    assert chern == 0.9804999999999999
    

def test_chern_marker():
    """
    Tests calculation of Chern marker.
    """
    
    model = AgarwalaChern(m=-1).reduce(n1=10, n2=10)
    model.boundary = "OBC"
    model.initialize_hamiltonian()
    
    results = model.solve()
    
    marker = chern_marker(model, results).reshape(-1, 2)
    marker = np.sum(marker, axis=1)
    
    marker_hash = hash_numpy_array(marker)
        
    assert marker_hash == "46aa8dcdc353933baaf9f47cf0e04b01"
        

def test_partition_plane():
    """
    Tests generation of two partitions of the system according to a given plane.
    """
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config).reduce(n1=10)
    model.initialize_hamiltonian()
    
    plane = [0, 1, 0, np.max(model.motif[:, 1]/2)]
    partition = specify_partition_plane(model, plane)
        
    assert np.allclose(partition, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    

def test_entanglement_spectrum():
    """
    Tests calculation of the entanglement spectrum from a spatial partition of a system.
    """
    
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)
    
    model = SlaterKoster(config).reduce(n1=5)
    model.initialize_hamiltonian()
    
    plane = [0, 1, 0, np.max(model.motif[:, 1]/2)]
    partition = specify_partition_plane(model, plane)
    entanglement = entanglement_spectrum(model, partition)
    
    entanglement_hash = hash_numpy_array(entanglement)
        
    assert entanglement_hash == "11e0c56d514bb782933cfc46dd9c7f4a"
    
    
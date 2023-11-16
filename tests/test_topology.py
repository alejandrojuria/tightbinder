from tightbinder.topology import calculate_wannier_centre_flow, calculate_z2_invariant, calculate_chern_number, chern_marker, specify_partition_plane, entanglement_spectrum, plot_wannier_centre_flow
from tightbinder.models import SlaterKoster, AgarwalaChern
from tightbinder.fileparse import parse_config_file
from tightbinder.utils import hash_numpy_array
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

INPUT_FILE_DIR = Path(__file__).parent / ".." / "examples" / "inputs"
"""
Module with tests to check the topology module.
"""

def test_wannier_centre_flow():
    """
    Tests the calculation of the wannier centre flow for a SlaterKoster model.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 5, refine_mesh=False)
    
    wcc_hash = hash_numpy_array(wcc)
        
    assert math.isclose(wcc_hash, 9.241406000000001)
    

def test_wannier_centre_flow_refined():
    """
    Tests the calculation of the wannier centre flow but with a refined mesh.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
        
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 5)
    
    wcc_hash = hash_numpy_array(wcc)
    
    assert math.isclose(wcc_hash, 22.862526000000003)
            

def test_z2_invariant():
    """
    Tests the z2 invariant calculation algorithm.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, refine_mesh=False)
    z2 = calculate_z2_invariant(wcc)
    
    assert z2 == 1
    

def test_z2_phase_diagram():
    """
    Tests that the z2 invariant calculation algorithm works to determine a phase diagram, changing the spin-orbit coupling.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
    
    model = SlaterKoster(config)
    model.initialize_hamiltonian()
    
    wcc = calculate_wannier_centre_flow(model, 10, refine_mesh=False)
    z2 = calculate_z2_invariant(wcc)
    
    assert z2 == 1
    
    model.configuration["SOC"][0] = 0.5
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
    
    assert math.isclose(wcc_hash,  2.62954)
            

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
        
    assert math.isclose(marker_hash, 197.6131102153834)
        

def test_partition_plane():
    """
    Tests generation of two partitions of the system according to a given plane.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
    
    model = SlaterKoster(config).reduce(n1=10)
    model.initialize_hamiltonian()
    
    plane = [0, 1, 0, np.max(model.motif[:, 1]/2)]
    partition = specify_partition_plane(model, plane)
        
    assert np.allclose(partition, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    

def test_entanglement_spectrum():
    """
    Tests calculation of the entanglement spectrum from a spatial partition of a system.
    """
    
    path = INPUT_FILE_DIR / "Bi111.yaml"
    config = parse_config_file(path)
    
    model = SlaterKoster(config).reduce(n1=5)
    model.initialize_hamiltonian()
    
    plane = [0, 1, 0, np.max(model.motif[:, 1]/2)]
    partition = specify_partition_plane(model, plane)
    entanglement = entanglement_spectrum(model, partition)
    
    entanglement_hash = hash_numpy_array(entanglement)
    
    assert math.isclose(entanglement_hash, 19.351744089214876)
            
    
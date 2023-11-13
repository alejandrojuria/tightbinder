from tightbinder.models import WilsonAmorphous, HaldaneModel, BHZ, AgarwalaChern
from tightbinder.result import State
from tightbinder.utils import hash_numpy_array
import numpy as np
import matplotlib.pyplot as plt
import math

"""
This scripts contains tests to verify the functionality of the predefined models.
"""

def test_wilson_amorphous():
    """
    Tests that the WilsonAmorphous model can be initialized and obtain the band structure correctly.
    """
    
    model = WilsonAmorphous(m=4, r=1.4)
    model.initialize_hamiltonian()
    
    labels = ["G", "X", "M", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
    
    energies_hash = hash_numpy_array(results.eigen_energy)
    
    assert math.isclose(energies_hash, 856.0)
            

def test_haldane_model():
    """
    Tests that the HaldaneModel model can be initialized and obtain the band structure correctly.
    """
    
    model = HaldaneModel()
    model.initialize_hamiltonian()
    
    labels = ["G", "M", "K", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
    
    energies_hash = hash_numpy_array(results.eigen_energy)
        
    assert math.isclose(energies_hash, 76.56623716571589)
    

def test_bhz():
    """
    Tests that the BHZ model can be initialized and obtain the band structure correctly.
    """
    
    model = BHZ(g=1, u=0.0, c=0.0)
    model.initialize_hamiltonian()
    
    labels = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
    
    energies_hash = hash_numpy_array(results.eigen_energy)
        
    assert math.isclose(energies_hash, 304.0)

def test_agarwala_chern():
    """
    Tests that the AgarwalaChern model can be initialized and obtain the band structure correctly.
    """
    
    model = AgarwalaChern()
    model.initialize_hamiltonian()
    
    labels = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
    
    energies_hash = hash_numpy_array(results.eigen_energy)
        
    assert math.isclose(energies_hash, 206.0)
    

def test_edge_state_amplitude():
    """
    Checks that the amplitude of the edge state is correctly computed.
    """
    
    # Declaration of parameters of the model
    m = 4
    cutoff = 1.4
    disorder = 0.2
    ncells = 12

    # Init. model, create supercell with OBC and amorphize
    model = WilsonAmorphous(m=m, r=cutoff).reduce(n1=ncells, n2=ncells)

    # Obtain the eigenstates of the system
    model.initialize_hamiltonian()
    results = model.solve()

    # Identify a edge state (zero mode) of the system
    state_index = np.argmin(np.abs(results.eigen_energy))
    edge_state = State(results.eigen_states[0][:, state_index], model)
            
    assert np.allclose(np.sort(edge_state.amplitude)[-5:], 
                       [0.01730692, 0.01730692, 0.01730692, 0.01730692, 0.01730692])
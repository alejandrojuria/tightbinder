from tightbinder.models import WilsonAmorphous, HaldaneModel, BHZ, AgarwalaChern
from tightbinder.result import State
import numpy as np
import matplotlib.pyplot as plt

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
    
    assert np.allclose(results.eigen_energy[0],
                       [-7., -6.244998, -4.35889894, -3., -2.64575131, 
                        -1.73205081, -1., -2.64575131, -5.56776436, -7.])
    

def test_haldane_model():
    """
    Tests that the HaldaneModel model can be initialized and obtain the band structure correctly.
    """
    
    model = HaldaneModel()
    model.initialize_hamiltonian()
    
    labels = ["G", "M", "K", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
        
    assert np.allclose(results.eigen_energy[0],
                       [-3.0, -2.64575131, -1.73205081, -1.00000000, -8.79385242e-01,
                        -5.32088886e-01, 0.0, -1.34729636, -2.53208889, -3.0])
    

def test_bhz():
    """
    Tests that the BHZ model can be initialized and obtain the band structure correctly.
    """
    
    model = BHZ(g=1, u=0.0, c=0.0)
    model.initialize_hamiltonian()
    
    labels = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
            
    assert np.allclose(results.eigen_energy[0],
                       [-2., -2.28736766, -4.08609256, -4.47213595, -4.61651143,
                        -3.40295857, -2., -1.80277564, -1.80277564, -2.])
    

def test_agarwala_chern():
    """
    Tests that the AgarwalaChern model can be initialized and obtain the band structure correctly.
    """
    
    model = AgarwalaChern()
    model.initialize_hamiltonian()
    
    labels = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(10, labels)
    results = model.solve(kpoints)
    
    print(results.eigen_energy[0])
        
    assert np.allclose(results.eigen_energy[0],
                       [-1., -2.34520788, -4.18330013, -5., -4.58257569,
                        -3.60555128, -3., -2.64575131, -1.73205081, -1.])
    

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
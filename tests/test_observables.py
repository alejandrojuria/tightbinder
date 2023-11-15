from tightbinder.observables import dos, dos_kpm, TransportDevice
from tightbinder.fileparse import parse_config_file
from tightbinder.models import SlaterKoster
from tightbinder.utils import hash_numpy_array
import numpy as np
import math
from pathlib import Path

"""
Module with tests to verify the functionality of the observables module.
"""


def test_dos():
    """
    Tests calculation of the density of states.
    """
    
    path = Path(__file__).parent / ".." / "examples" / "inputs" / "hBN.txt"
    with open(path) as fp:
        config = parse_config_file(fp)
        model = SlaterKoster(config)

        nk = 20
        kpoints = model.brillouin_zone_mesh([nk, nk])

        model.initialize_hamiltonian()
        results = model.solve(kpoints)

        density, energies = dos(results, delta=0.05, npoints=200)
        
        density_hash = hash_numpy_array(np.array(density))
        
        assert math.isclose(density_hash, 5.678625492981987)
            

def test_dos_kpm():
    """
    Tests calculation of the density of states using the KPM.
    """
    
    np.random.seed(1)

    path = Path(__file__).parent / ".." / "examples" / "inputs" / "hBN.txt"
    with open(path) as fp:
        config = parse_config_file(fp)

        ncells = 10
        model = SlaterKoster(config).supercell(n1=ncells, n2=ncells)

        model.matrix_type = "sparse"
        model.initialize_hamiltonian()

        density, energies = dos_kpm(model, nmoments=150, npoints=400, r=30)
        
        density_hash = hash_numpy_array(np.array(density))
        
        assert math.isclose(density_hash, 0.7510637669286024)
            

def test_transport_device():
    """
    Tests initialization of a transport device.
    """
    
    length, width = 10, 5

    path = Path(__file__).parent / ".." / "examples" / "inputs" / "chain.txt"
    with open(path) as fp:
        config = parse_config_file(fp)

        model = SlaterKoster(config)

        model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
        model = model.reduce(n2=width)

        left_lead = np.copy(model.motif)
        left_lead[:, :3] -= model.bravais_lattice[0]

        right_lead = np.copy(model.motif)
        right_lead[: , :3] += length * model.bravais_lattice[0]

        period = model.bravais_lattice[0, 0]

        model = model.reduce(n1=length)
        
        device = TransportDevice(model, left_lead, right_lead, period, "default")
            
    
def test_transmission():
    """
    Tests calculation of the transmission.
    """
    
    length, width = 7, 4

    path = Path(__file__).parent / ".." / "examples" / "inputs" / "chain.txt"
    with open(path) as fp:
        config = parse_config_file(fp)

        model = SlaterKoster(config)

        model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
        model = model.reduce(n2=width)

        left_lead = np.copy(model.motif)
        left_lead[:, :3] -= model.bravais_lattice[0]

        right_lead = np.copy(model.motif)
        right_lead[: , :3] += length * model.bravais_lattice[0]

        period = model.bravais_lattice[0, 0]

        model = model.reduce(n1=length)
        
        device = TransportDevice(model, left_lead, right_lead, period, "default")
        trans, energy = device.transmission(-5, 5, 50)
        
        transmission_hash = hash_numpy_array(np.array(trans))
        
        assert math.isclose(transmission_hash, 243.99797881099673)
            

def test_conductance():
    """
    Tests calculation of the conductance.
    """
    
    length, width = 7, 4

    path = Path(__file__).parent / ".." / "examples" / "inputs" / "chain.txt"
    with open(path) as fp:
        config = parse_config_file(fp)

        model = SlaterKoster(config)

        model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
        model = model.reduce(n2=width)

        left_lead = np.copy(model.motif)
        left_lead[:, :3] -= model.bravais_lattice[0]

        right_lead = np.copy(model.motif)
        right_lead[: , :3] += length * model.bravais_lattice[0]

        period = model.bravais_lattice[0, 0]

        model = model.reduce(n1=length)
        
        device = TransportDevice(model, left_lead, right_lead, period, "default")
        conductance = device.conductance()
        
        assert np.allclose(conductance, 6.240297897758343)


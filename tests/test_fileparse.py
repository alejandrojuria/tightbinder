# Script to run tests on fileparse module

from pathlib import Path
import pytest
from tightbinder.fileparse import parse_config_file

path = Path(__file__).parent.parent.resolve()
files = [
    path / "examples/test/square.txt",
    path / "examples/test/bi(111).txt"
]


def check_configuration(configuration, expected_configuration):
    keys = expected_configuration.keys()
    for key in keys:
        try:
            assert configuration[key] == expected_configuration[key]
        except AssertionError as e:
            pytest.fail(f"Configuration mismatch for {key}, " +
                        f"{configuration[key]} differs from {expected_configuration[key]}")


def test_parse_square():
    file = open(files[0], "r")
    configuration = parse_config_file(file)
    expected_configuration = {
        "System name": "Test square",
        "Dimensionality": 2,
        "Bravais lattice": [[1, 0, 0],
                            [0, 1, 0]],
        "Species": 1,
        "Motif": [[0, 0, 0, 0]],
        "Orbitals": [1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Onsite energy": [[1]],
        "SK amplitudes": [[-0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "Spin": False,
        "Spin-orbit coupling": 0,
        "Mesh": [100, 100]
    }
    check_configuration(configuration, expected_configuration)


def test_parse_bi_bilayer():
    file = open(files[1], "r")
    configuration = parse_config_file(file)
    expected_configuration = {
        "System name": "Bi(111) bilayer w/ SOC",
        "Dimensionality": 2,
        "Bravais lattice": [[3.92587, 2.2666, 0.0],
                            [3.92587, -2.2666, 0.0]],
        "Species": 1,
        "Motif": [[0, 0, 0, 0],
                  [2.61724, 0, -1.585, 0]],
        "Orbitals": [1, 1, 1, 1, 0, 0, 0, 0, 0],
        "Onsite energy": [[-10.906, -0.486]],
        "SK amplitudes": [[-0.608, 1.320, 1.854, -0.600, 0, 0, 0, 0, 0, 0]],
        "Spin": True,
        "Spin-orbit coupling": 0.3,
        "Mesh": [200, 200]
    }
    check_configuration(configuration, expected_configuration)


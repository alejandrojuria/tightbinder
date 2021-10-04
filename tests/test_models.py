# Script to run tests on SK model

from pathlib import Path
from tightbinder.fileparse import parse_config_file
from tightbinder.models import SKModel, WilsonAmorphous
import numpy as np
import pytest

path = Path(__file__).parent.parent.resolve()
files = [
    path / "examples/test/chain.txt",
    path / "examples/test/bi(111).txt"
]


class SKModelTest:

    @staticmethod
    def initialize_model(path_to_file):
        file = open(path_to_file, "r")
        configuration = parse_config_file(file)
        model = SKModel(configuration)
        model.initialize_hamiltonian()

        return model

    def bonds(self, path_to_file, expected_bonds):
        model = self.initialize_model(path_to_file)
        nbonds, nebonds = len(model.bonds), len(expected_bonds)

        # Check number of bonds
        if nbonds != nebonds:
            pytest.fail(f"Mismatch between number of bonds: "
                        f"found {nbonds}, expected {nebonds}")

        # Check exact match of bonds
        # Type annotation for the cell arrays to avoid IDE warnings when comparing
        cell: np.ndarray
        ecell: np.ndarray
        for n, bond in enumerate(model.bonds):
            initial, final, cell = bond
            einitial, efinal, ecell = expected_bonds[n]  # e stands for expected
            if initial != einitial:
                pytest.fail(f"Mismatch between initial atoms: "
                            f"initial {initial}, expected {einitial}")
            elif final != efinal:
                pytest.fail(f"Mismatch between final atoms: "
                            f"final {final}, expected {efinal}")
            elif (cell != ecell).any():
                pytest.fail(f"Mismatch between cells: "
                            f"cell {cell}, expected {ecell}")

    def bands(self, path_to_file, path_labels, e_max_energy, e_min_energy, expected_gap):
        model = self.initialize_model(path_to_file)
        kpoints = model.high_symmetry_path(200, path_labels)
        result = model.solve(kpoints)

        max_energy = np.round(np.max(result.eigen_energy), 2)
        min_energy = np.round(np.min(result.eigen_energy), 2)
        print(max_energy, min_energy)
        if max_energy != e_max_energy:
            pytest.fail(f"Mismatch between maximum energy {max_energy} and expected one {e_max_energy}")
        if min_energy != e_min_energy:
            pytest.fail(f"Mismatch between minimum energy {min_energy} and expected one {e_min_energy}")

        filling = int(5./8*model.basisdim)
        gap = result.calculate_gap(filling)
        if np.round(gap, 2) != np.round(expected_gap, 2):
            pytest.fail(f"Mismatch between gap {gap} and expected one {expected_gap}")


class TestChain(SKModelTest):

    def test_bonds(self):
        expected_bonds = [[0, 0, [-1, 0, 0]]]
        self.bonds(files[0], expected_bonds)

    def test_bands(self):
        labels = ["K", "G", "K"]
        e_max_energy, e_min_energy = 1.6, 0.4
        self.bands(files[0], labels, e_max_energy, e_min_energy, expected_gap=0)

        assert np.max(result.eigen_energy) == 1

class TestBiBilayer(SKModelTest):

    def test_bonds(self):
        expected_bonds = [[0, 1, [-3.92587, -2.2666, 0.]],
                          [0, 1, [-3.92587, 2.2666, 0.]],
                          [0, 1, [0., 0., 0.]]]
        self.bonds(files[1], expected_bonds)

    def test_bands(self):
        labels = ["M", "G", "K", "M"]
        e_max_energy, e_min_energy = 2.83, -13.07
        expected_gap = 0.6097006206944817
        self.bands(files[1], labels, e_max_energy, e_min_energy, expected_gap)


class TestAmorphous:

    @staticmethod
    def initialize_model():
        pass

    def test_neighbours(self):
        pass

    def test_bands(self):
        pass

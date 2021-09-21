# Script to run tests on SK model

from pathlib import Path
from tightbinder.fileparse import parse_config_file
from tightbinder.models import SKModel, WilsonAmorphous

path = Path(__file__).parent.parent.resolve()
files = [
    path / "examples/test/square.txt",
    path / "examples/test/bi(111).txt"
]


class TestSquare:

    @staticmethod
    def initialize_model():
        file = open(files[0], "r")
        configuration = parse_config_file(file)
        model = SKModel(configuration)
        model.initialize_hamiltonian()

    def test_neighbours(self, model):
        assert model

    def test_bands(self, model):
        pass


class TestBiBilayer:

    def test_neighbours(self):
        pass

    def test_bands(self):
        pass

class TestAmorphous:

    def test_neighbours(self):
        pass

    def test_bands(self):

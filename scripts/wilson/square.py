
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tightbinder.models import SKModel
from tightbinder.fileparse import parse_config_file


def main():
    path = Path(__file__).parent.parent.resolve()
    filename = path / "examples/test/chain.txt"
    file = open(filename, "r")
    configuration = parse_config_file(file)
    model = SKModel(configuration)
    model.filling = 5./8
    model.initialize_hamiltonian()
    print(model.bonds)
    labels = ["K", "G", "K"]
    kpoints = model.high_symmetry_path(400, labels)
    result = model.solve(kpoints)
    result.plot_along_path(labels)

    plt.show()


if __name__ == "__main__":
    main()


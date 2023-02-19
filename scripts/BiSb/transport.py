from tightbinder.models import AmorphousSlaterKoster
from tightbinder.disorder import alloy, amorphize
from tightbinder.fileparse import parse_config_file
import numpy as np
import matplotlib.pyplot as plt

def main():
    file = open("examples/Bi(111).txt", "r")
    config = parse_config_file(file)
    model = AmorphousSlaterKoster(config, r=3.2).ribbon(width=3)


if __name__ == "__main__":
    main()
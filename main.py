# Implementation of general tight-binding model based on the Slater-Koser approximation to parametrize
# the hopping amplitudes in term of the angles and orbitals between atoms. 
# Requires an input file with the data regarding the model (dimensionality, number of atoms in motif, orbitals, interaction values)

import argparse, sys
import fileparse
import hamiltonian
import numpy as np

''' Main routine '''
def main():
    # ------------------------------ Argument & file parsing ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='input file for tight-binding model')
    args = parser.parse_args()

    try:
        file = open(args.file, 'r')
    except:
        print('Error: Input file does not exist')
        sys.exit(1)
    
    configuration = fileparse.parseConfigFile(file)

    bravais_lattice = configuration['Bravais lattice']
    reciprocal_basis = hamiltonian.reciprocal_lattice(bravais_lattice)
    print(reciprocal_basis)

if __name__ == "__main__":
    main()

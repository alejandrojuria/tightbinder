# Definition of all routines to build and solve the tight-binding hamiltonian
# constructed from the parameters of the configuration file

import numpy as np
import sys

# --------------- Constants ---------------
PI = 3.14159265359



# --------------- Routines ---------------



''' Given a list of atoms (motif), it returns a list in which each 
index corresponds to a list of atoms that are first neighboirs to that index
on the initial atom list.
I.e.: Atom list -> List of neighbours/atom.
By default it will look for the minimal distance between atoms to determine first neighbours.
For amorphous systems the option radius is available to determine neighbours within a given radius R'''
def first_neighbours(motif, bravais_lattice, mode="minimal", R=None):

    motif = np.array(motif)
    bravais_lattice = np.array(bravais_lattice)

    # Determine neighbour distance from one fixed atom
    if(mode == "minimal"):
        d0 = 1E100
        for i in range(len(motif)):
            if(i == 0):
                continue
            distance = np.linalg.norm(motif[i] - motif[0])
            if(distance < d0):
                d0 = distance
        for n in range(len(bravais_lattice)):
            for i in range(len(motif)):
                distance = np.linalg.norm(motif[i, :3] + bravais_lattice[n]- motif[i, :3])
                if(distance < d0):
                    d0 = distance
    elif(mode == "radius"):
        if R == None:
            print('Radius not defined in "radius" mode, exiting...')
            sys.exit(1)
        auxArray = np.zeros([len(bravais_lattice) + 1, 3])
        auxArray[1:, :] = bravais_lattice
        bravais_lattice = auxArray
        neighbours = []
        for i in range(len(motif)):
            atom = motif[i, :3]
            actualNeighbours = []
            for n in range(len(bravais_lattice)):
                for j in range(len(motif)):
                    distance = np.linalg.norm(atom - motif[j, :3] - bravais_lattice[n])
                    if(distance < R):
                        actualNeighbours.append([j, n])
            neighbours.append(actualNeighbours)
    else:
        print('Incorrect mode option. Exiting... ')
        sys.exit(1)
            
    print(d0)

    return 
            

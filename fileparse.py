# Implementation of routines to parse the tight-binding parameters written in
# the input file. Config file can have blank spaces or comments starting with "!" 
# (without commas) which are ommited at parsing

import sys, re

''' Routine to parse the arguments raw (without any treatment) from the input file '''
def parseRawArguments(file):

    arguments = {}
    value = []

    lines = file.readlines()
    for line in lines:

        # Identifier
        if line[0] == '#':
            try:
                arguments.update({newArgument : value})
                value = []
            except:
                pass
            newArgument = line[1:].strip()

        # Blank line
        elif line[0] in ['\n', '\r\n', '!', ' ']:
            continue

        # Argument values
        else:
            value.append(line.strip())
    arguments.update({newArgument : value})
            
    return arguments

''' Routine to check whether all required fields for model are present in the input file '''
def checkArguments(arguments, requiredArguments):
    for arg in requiredArguments:
        if arg not in arguments.keys():
            print(f'{arg} not present in input file')
            sys.exit(1)

''' Routine to rewrite correctly the arguments parsed (raw) from the input file '''
def shapeArguments(arguments):

    for arg in arguments:
        if arg == 'System name':
            try:
                arguments[arg] = arguments[arg][0]
            except:
                print('Error: No system name given')
                sys.exit(1)

        elif arg in ['Dimensionality', 'Species']:
            try:
                arguments[arg] = int(arguments[arg][0])
            except:
                print(f'Error: {arg} has to be an integer')
                sys.exit(1)

        elif arg == 'Bravais lattice':
            auxArray = []
            for line in arguments[arg]:
                auxArray.append([float(num) for num in re.split(' |, |,', line)])
            arguments[arg] = auxArray

        elif arg == 'Motif':
            auxArray = []
            for n, line in enumerate(arguments[arg]):
                auxArray.append([float(num) for num in re.split(' |, |,|; |;', line)])
                if(arguments['Species'] == 1):
                    try:
                        auxArray[n][3] = 1 # Default value to 1
                    except:
                        auxArray[n].append(1)
            arguments[arg] = auxArray

        elif arg == 'Orbitals':
            possibleOrbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2-y2', 'd3z2-r2']
            auxArray = []
            try:
                orbitals = re.split(' |, |,', arguments[arg][0])
            except:
                print('Error: No orbitals included')
                sys.exit(1)
            else:
                # Check that all are correctly written
                for orbital in orbitals:
                    if(orbital not in possibleOrbitals):
                        print('Error: Incorrect orbital specified')
                        sys.exit(1)
                # Label with true or false
                for orbital in possibleOrbitals:
                    if orbital in orbitals:
                        auxArray.append(1)
                    else:
                        auxArray.append(0)
                arguments[arg] = auxArray
            
        elif arg == 'Onsite energy':
            try:
                arguments[arg] = [float(num) for num in re.split(' |, |,', arguments[arg][0])]
            except:
                print('Error: No onsite energies included')
                sys.exit(1)
                
        elif arg == 'SK amplitudes':
            auxArray = []
            for line in arguments[arg]:
                try:
                    auxArray.append([float(num) for num in re.split(' |, |,|; |;', line)])
                except:
                    print('Error: No Slater-Koster amplitudes given')
                    sys.exit(1)
            arguments[arg] = auxArray

        elif arg == 'Spin-orbit coupling':
            try:
                arguments[arg] = float(arguments[arg][0])
            except:
                arguments[arg] = 0.0
        
    return arguments

''' Routine to check that the present arguments are coherent among them (numbers basically) '''
def checkCoherence(arguments):

    # Check dimensions
    if(arguments['Dimensionality'] > 3 or arguments['Dimensionality'] < 0):
        print('Error: Invalid dimension!')
        sys.exit(1)
    
    # Check species
    if(arguments['Species'] < 0):
        print('Error: Species has to be a positive number (1 or 2)')
        sys.exit(1)

    # Check vector basis
    if (arguments['Dimensionality'] != len(arguments['Bravais lattice'])):
        print('Error: Dimension and number of basis vectors do not match')
        sys.exit(1)
    
    # Check that motif has elements specified if num. of species = 2
    if (arguments['Species'] == 2):
        for atom in arguments['Motif']:
            try:
                atomElement = atom[3] # Try to access
            except: 
                print('Error: Atomic species not specified in motif')
                sys.exit(1)

            if(atomElement > arguments['Species']):
                print('Error: Incorrect species labeling in motif')
                sys.exit(1)

    # Check SK coefficients are present for all species
    if (len(arguments['SK amplitudes']) != arguments['Species']):
        print('Error: Missing SK coefficients for both atomic species')
    
    # ------------ Orbital consistency ------------
    diffOrbitals = 0
    neededSKcoefs = 0
    if True == arguments['Orbitals'][0]:   # s
        diffOrbitals += 1
        neededSKcoefs += 1
    if True in arguments['Orbitals'][1:4]: # p 
        diffOrbitals += 1
        neededSKcoefs += 3
    if True in arguments['Orbitals'][4:]:  # d
        diffOrbitals += 1
        neededSKcoefs += 6

    # Check onsite energy for all orbitals
    if(diffOrbitals != len(arguments['Onsite energy'])):
        print('Error: Mismatch between number of onsite energies and orbitals')
        sys.exit(1)

    # Check whether all necessary SK coefs are present for all orbitals
    for speciesCoefs in arguments['SK amplitudes']:
        if(len(speciesCoefs) != neededSKcoefs):
            print('Error: Mismatch between orbitals and required SK amplitudes')
            sys.exit(1)
    
    return 0

''' Routine to obtain all the information from the configuration file, already shaped and verified '''
def parseConfigFile(file):

    configuration = shapeArguments(parseRawArguments(file))
    requiredArguments = ['System name', 'Dimensionality', 'Bravais lattice', 'Species',
                         'Motif', 'Orbitals', 'Onsite energy', 'SK amplitudes']
    checkArguments(configuration, requiredArguments)
    checkCoherence(configuration)

    return configuration
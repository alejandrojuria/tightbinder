# Implementation of routines to parse the tight-binding parameters written in
# the input file. Config file can have blank spaces or comments starting with "!" 
# (without commas) which are ommited at parsing

import sys
import re
import numpy as np


def parse_raw_arguments(file):
    """ Routine to parse the arguments raw (without any treatment) from the input file """

    arguments = {}
    value = []

    lines = file.readlines()
    for line in lines:
        # Identifier
        if line[0] == '#':
            try:
                arguments.update({new_argument: value})
                value = []
            except UnboundLocalError:
                pass
            new_argument = line[1:].strip()
        # Blank line
        elif line[0] in ['\n', '\r\n', '!', ' ']:
            continue
        # Argument values
        else:
            value.append(line.strip())
    arguments.update({new_argument: value})

    return arguments


def check_arguments(arguments, required_arguments):
    """ Routine to check whether all required fields for model are present in the input file """

    for arg in required_arguments:
        if arg not in arguments.keys():
            print(f'{arg} not present in input file')
            sys.exit(1)


def shape_arguments(arguments):
    """ Routine to rewrite correctly the arguments parsed (raw) from the input file """

    for arg in arguments:
        if arg == 'System name':
            try:
                arguments[arg] = arguments[arg][0]
            except IndexError as e:
                print(f'{type(e).__name__}: No system name given')
                sys.exit(1)

        elif arg in ['Dimensionality', 'Species']:
            try:
                arguments[arg] = int(arguments[arg][0])
            except IndexError as e:
                print(f'{type(e).__name__}: No {arg} given')
                sys.exit(1)
            except ValueError as e:
                print(f'{type(e).__name__}: {arg} has to be an integer')
                sys.exit(1)

        elif arg == 'Bravais lattice':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append([float(num) for num in re.split(' |, |,', line)])
                except IndexError as e:
                    print(f'{type(e).__name__}: No {arg} vectors given given')
                    sys.exit(1)
                except ValueError as e:
                    print(f'{type(e).__name__}: {arg} vectors have to be numbers')
                    sys.exit(1)
            arguments[arg] = aux_array

        elif arg == 'Motif':
            aux_array = []
            for n, line in enumerate(arguments[arg]):
                aux_array.append([float(num) for num in re.split(' |, |,|; |;', line)])
                if arguments['Species'] == 1:
                    try:
                        aux_array[n][3] = 0  # Default value to 1
                    except IndexError:
                        aux_array[n].append(0)
            arguments[arg] = aux_array

        elif arg == 'Orbitals':
            possible_orbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'd3z2-r2']
            aux_array = []
            try:
                orbitals = re.split(' |, |,', arguments[arg][0])
            except IndexError as e:
                print(f'{type(e).__name__}: No orbitals included')
                sys.exit(1)
            else:
                # Check that all are correctly written
                for orbital in orbitals:
                    if orbital not in possible_orbitals:
                        print('Error: Incorrect orbital specified')
                        sys.exit(1)
                # Label with true or false
                for orbital in possible_orbitals:
                    if orbital in orbitals:
                        aux_array.append(1)
                    else:
                        aux_array.append(0)
                arguments[arg] = aux_array
            
        elif arg == 'Onsite energy':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append([float(num) for num in re.split(' |, |,', line)])
                except IndexError as e:
                    print(f'{type(e).__name__}: No onsite energies included')
                    sys.exit(1)
                except ValueError as e:
                    print(f'{type(e).__name__}: Onsite energies must be numbers')
                    sys.exit(1)
            arguments[arg] = aux_array

        elif arg == 'SK amplitudes':
            aux_array = []
            for n, line in enumerate(arguments[arg]):
                try:
                    aux_array.append([float(num) for num in re.split(' |, |,|; |;', line)])
                except IndexError as e:
                    print(f'{type(e).__name__}: No Slater-Koster amplitudes given')
                    sys.exit(1)
                except ValueError as e:
                    print(f'{type(e).__name__}: Slater-Koster amplitudes must be numbers')
                    sys.exit(1)
            arguments[arg] = aux_array

        elif arg == 'Spin':
            if arguments[arg][0] == "True":
                arguments[arg] = True
            elif arguments[arg][0] == "False":
                arguments[arg] = False
            else:
                print('Error: Spin parameter must be True or False (or 1 or 0 respectively)')
                sys.exit(1)
            #except IndexError as e:
            #    print('Warning: No spin parameter given, defaulting to spinless')
            #except ValueError as e:
            #    print(f'{type(e).__name__}: Spin parameter must be True or False (or 1 or 0 respectively)')
            #    sys.exit(1)

        elif arg == 'Spin-orbit coupling':
            try:
                arguments[arg] = float(arguments[arg][0])
            except IndexError as e:
                print(f'Warning: No spin-orbit coupling given, defaulting to 0...')
                arguments[arg] = 0.0
            except ValueError as e:
                print(f'{type(e).__name__}: Spin-orbit coupling must be a number')
                sys.exit(1)

        elif arg == 'Mesh':
            try:
                arguments[arg] = [int(num) for num in re.split(' |, |,', arguments[arg][0])]
            except IndexError as e:
                print(f'{type(e).__name__}: No mesh given')
                sys.exit(1)
            except ValueError as e:
                print(f'{type(e).__name__}: Mesh must be integer numbers')
                sys.exit(1)

    return arguments


def check_coherence(arguments):
    """ Routine to check that the present arguments are coherent among them (numbers basically) """

    # --------------- Model ---------------
    # Check dimensions
    if arguments['Dimensionality'] > 3 or arguments['Dimensionality'] < 0:
        print('Error: Invalid dimension!')
        sys.exit(1)
    
    # Check species
    if arguments['Species'] < 0:
        print('Error: Species has to be a positive number (1 or 2)')
        sys.exit(1)
    elif arguments['Species'] > 2:
        print('Error: Species has to be a positive number (1 or 2)')
        sys.exit(1)

    # Check vector basis
    if arguments['Dimensionality'] != len(arguments['Bravais lattice']):
        print('Error: Dimension and number of basis vectors do not match')
        sys.exit(1)

    # Check length of vector basis
    for vector in arguments['Bravais lattice']:
        if len(vector) != 3:
            print('Error: Bravais vectors must have three components')
            sys.exit(1)
    
    # Check that motif has elements specified if num. of species = 2
    if arguments['Species'] == 2:
        for atom in arguments['Motif']:
            try:
                atom_element = atom[3]  # Try to access
            except IndexError:
                print('Error: Atomic species not specified in motif')
                sys.exit(1)

            if atom_element > arguments['Species']:
                print('Error: Incorrect species labeling in motif')
                sys.exit(1)

    # Check onsite energies are present for all species
    if len(arguments['Onsite energy']) != arguments['Species']:
        print('Error: Missing onsite energies for both atomic species')
        sys.exit(1)

    # Check SK coefficients are present for all species
    if len(arguments['SK amplitudes']) < arguments['Species']:
        print('Error: Missing SK coefficients for both atomic species')
        sys.exit(1)
    elif len(arguments['SK amplitudes']) > arguments['Species']:
        print('Error: Expected only one row of SK coefficients')
        sys.exit(1)

    if arguments['Spin-orbit coupling'] != 0 and not arguments['Spin']:
        print('Warning: Spin-orbit coupling is non-zero but spin was set to False. ')
        arguments['Spin'] = True
    
    # ------------ Orbital consistency ------------
    diff_orbitals = 0
    needed_SK_coefs = 0
    if arguments['Orbitals'][0]:    # s
        diff_orbitals += 1
        needed_SK_coefs += 1
    if True in arguments['Orbitals'][1:4]:  # p
        diff_orbitals += 1
        needed_SK_coefs += 2
    if True in arguments['Orbitals'][4:]:   # d
        diff_orbitals += 1
        needed_SK_coefs += 3
    if arguments['Orbitals'][0] and True in arguments['Orbitals'][1:4]:
        needed_SK_coefs += 1
    if arguments['Orbitals'][0] and True in arguments['Orbitals'][4:]:
        needed_SK_coefs += 1
    if True in arguments['Orbitals'][1:4] and True in arguments['Orbitals'][4:]:
        needed_SK_coefs += 2

    # Check onsite energy for all orbitals
    for onsite_energies in arguments['Onsite energy']:
        if diff_orbitals != len(onsite_energies):
            print('Error: Mismatch between number of onsite energies and orbitals')
            sys.exit(1)

    # Check whether all necessary SK coefs are present for all orbitals
    for species_coefs in arguments['SK amplitudes']:
        if len(species_coefs) != needed_SK_coefs:
            print('Error: Mismatch between orbitals and required SK amplitudes')
            sys.exit(1)

    # ---------------- Simulation ----------------
    # Check mesh matches dimension
    if len(arguments['Mesh']) != arguments['Dimensionality']:
        print('Error: Mesh dimension does not match system dimension')
        sys.exit(1)
    
    return None


def transform_sk_coefficients(configuration):
    """ Routine to transform SK coefficients into standardized form
    for later manipulation in hamiltonian """

    orbitals = configuration['Orbitals']
    amplitudes = [0]*10  # Number of possible SK amplitudes
    amplitudes_list = []
    for coefs in configuration['SK amplitudes']:
        if orbitals[0] and True in orbitals[1:4] and True in orbitals[4:]:
            amplitudes = coefs
        elif orbitals[0] and True in orbitals[1:4]:
            amplitudes[0:4] = coefs[0:4]
        elif orbitals[0] and True in orbitals[4:]:
            amplitudes[0] = coefs[0]
            amplitudes[4:] = coefs[1:]
        elif True in orbitals[1:4] and True in orbitals[4:]:
            amplitudes[2:4] = coefs[0:2]
            amplitudes[5:] = coefs[2:]
        elif orbitals[0]:  # s
            amplitudes[0] = coefs[0]
        elif True in orbitals[1:4]:  # p
            amplitudes[2:4] = coefs
        elif True in orbitals[4:]:  # d
            amplitudes[7:] = coefs
        else:
            print('Error: No orbitals given')
            sys.exit(1)
        amplitudes_list.append(amplitudes)
    
    configuration['SK amplitudes'] = amplitudes_list

    return None


def parse_config_file(file):
    """ Routine to obtain all the information from the configuration file, already shaped and verified """

    configuration = shape_arguments(parse_raw_arguments(file))
    required_arguments = ['System name', 'Dimensionality', 'Bravais lattice', 'Species',
                          'Motif', 'Orbitals', 'Onsite energy', 'SK amplitudes']
    check_arguments(configuration, required_arguments)
    check_coherence(configuration)

    transform_sk_coefficients(configuration)

    return configuration

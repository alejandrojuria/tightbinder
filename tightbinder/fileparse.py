# Implementation of routines to parse the tight-binding parameters written in
# the input file. Config file can have blank spaces or comments starting with "!" 
# (without commas) which are ommited at parsing

import re


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
            raise KeyError(f"{arg} not present in input file")


def shape_arguments(arguments):
    """ Routine to rewrite correctly the arguments parsed (raw) from the input file """

    for arg in arguments:
        if arg == 'System name':
            try:
                arguments[arg] = arguments[arg][0]
            except IndexError as e:
                print(f'{type(e).__name__}: No system name given')
                raise

        elif arg in ['Dimensionality', 'Species']:
            try:
                arguments[arg] = int(arguments[arg][0])
            except IndexError as e:
                print(f'{type(e).__name__}: No {arg} given')
                raise
            except ValueError as e:
                print(f'{type(e).__name__}: {arg} has to be an integer')
                raise

        elif arg == 'Bravais lattice':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append([float(num) for num in re.split(' |, |,', line)])
                except IndexError as e:
                    print(f'{type(e).__name__}: No {arg} vectors given given')
                    raise
                except ValueError as e:
                    print(f'{type(e).__name__}: {arg} vectors have to be numbers')
                    raise
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
            for line in arguments[arg]:
                try:
                    orbitals = re.split(' |, |,', line)
                except IndexError as e:
                    print(f'{type(e).__name__}: No orbitals included')
                    raise
                else:
                    # Check that all are correctly written
                    for orbital in orbitals:
                        if orbital not in possible_orbitals:
                            print(orbital)
                            print('Error: Incorrect orbital specified')
                            raise ValueError("Specified orbitals are not correct")
                    aux_array.append(orbitals)
            arguments[arg] = aux_array

        elif arg == 'Onsite energy':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append([float(num) for num in re.split(' |, |,', line)])
                except IndexError as e:
                    print(f'{type(e).__name__}: No onsite energies included')
                    raise
                except ValueError as e:
                    print(f'{type(e).__name__}: Onsite energies must be numbers')
                    raise
            arguments[arg] = aux_array

        elif arg == 'SK amplitudes':
            dictionary = {}
            for n, line in enumerate(arguments[arg]):
                    try:
                        aux_array = [float(num) for num in re.split(' |, |,|; |;', line)]
                        if arguments["Species"] == 1 and len(aux_array) == len(arguments["Orbitals"][0]):
                            aux_array = "00" + aux_array
                        orbitals = str(int(aux_array[0])) + str(int(aux_array[1]))
                        dictionary[orbitals] = aux_array[2:]
                    except IndexError as e:
                        print(f'{type(e).__name__}: No Slater-Koster amplitudes given')
                        raise
                    except ValueError as e:
                        print(f'{type(e).__name__}: Slater-Koster amplitudes must be numbers')
                        raise
            arguments[arg] = dictionary

        elif arg == 'Spin':
            if arguments[arg][0] == "True":
                arguments[arg] = True
            elif arguments[arg][0] == "False":
                arguments[arg] = False
            else:
                print('Error: Spin parameter must be True or False (or 1 or 0 respectively)')
                raise
            #except IndexError as e:
            #    print('Warning: No spin parameter given, defaulting to spinless')
            #except ValueError as e:
            #    print(f'{type(e).__name__}: Spin parameter must be True or False (or 1 or 0 respectively)')
            #    sys.exit(1)

        elif arg == 'Spin-orbit coupling':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append(float(arguments[arg][0]))
                except IndexError as e:
                    print(f'Warning: No spin-orbit coupling given, defaulting to 0...')
                    arguments[arg] = 0.0
                except ValueError as e:
                    print(f'{type(e).__name__}: Spin-orbit coupling must be a number')
                    raise
            arguments[arg] = aux_array

        elif arg == 'Mesh':
            try:
                arguments[arg] = [int(num) for num in re.split(' |, |,', arguments[arg][0])]
            except IndexError as e:
                print(f'{type(e).__name__}: No mesh given')
                raise
            except ValueError as e:
                print(f'{type(e).__name__}: Mesh must be integer numbers')
                raise

        elif arg == 'Radius':
            try:
                arguments[arg] = float(arguments[arg][0])
            except IndexError as e:
                print(f'{type(e).__name__}: No {arg} given')
                raise
            except ValueError as e:
                print(f'{type(e).__name__}: {arg} has to be a float')
                raise


        elif arg == 'High symmetry points':
            try:
                arguments[arg] = [str(label) for label in re.split(' |, |,', arguments[arg][0])]
            except IndexError as e:
                print(f'{type(e).__name__}: No high symmetry points given')
                raise


    return arguments


def check_coherence(arguments):
    """ Routine to check that the present arguments are coherent among them (numbers basically) """

    # --------------- Model ---------------
    # Check dimensions
    if arguments['Dimensionality'] > 3 or arguments['Dimensionality'] < 0:
        raise ValueError('Error: Invalid dimension!')

    # Check species
    if arguments['Species'] < 0:
        raise ValueError('Error: Species has to be a positive number (1 or 2)')
    elif arguments['Species'] > 2:
        raise ValueError('Error: Species has to be a positive number (1 or 2)')

    # Check vector basis
    if arguments['Dimensionality'] != len(arguments['Bravais lattice']):
        raise AssertionError('Error: Dimension and number of basis vectors do not match')

    # Check length of vector basis
    for vector in arguments['Bravais lattice']:
        if len(vector) != 3:
            raise ValueError('Error: Bravais vectors must have three components')

    # Check that motif has elements specified if num. of species = 2
    if arguments['Species'] == 2:
        for atom in arguments['Motif']:
            try:
                atom_element = atom[3]  # Try to access
            except IndexError:
                print('Error: Atomic species not specified in motif')
                raise

            if atom_element > arguments['Species']:
                raise AssertionError('Error: Incorrect species labeling in motif')

    # Check onsite energies are present for all species
    if len(arguments['Onsite energy']) != arguments['Species']:
        raise AssertionError('Error: Missing onsite energies for both atomic species')

    # Check orbitals present for all species
    if (len(arguments['Orbitals']) != arguments['Species']):
        raise AssertionError("Error: Orbitals must be specified for each species")

    # Check number of SK coefficients
    if len(arguments['SK amplitudes']) > arguments['Species']*(arguments['Species'] + 1)/2:
        raise AssertionError('Error: Expected only one row of SK coefficients')

    # Check number of SOC strenghts
    if arguments['Spin'] and len(arguments['Spin-orbit coupling']) != arguments["Species"]:
        raise AssertionError("Error: Values for SOC strength must match number of species")

    # Check if SOC are non-zero
    for soc in arguments['Spin-orbit coupling']:
        if soc != 0 and not arguments['Spin']:
            print('Warning: Spin-orbit coupling is non-zero but spin was set to False. ')
            arguments['Spin'] = True
    
    # ------------ Orbital consistency ------------
    # Check onsite energy per orbital per species
    for n, onsite_energies in enumerate(arguments["Onsite energy"]):
        if len(onsite_energies) != len(arguments["Orbitals"][n]):
            raise AssertionError("Error: Each orbital requires one onsite energy value")

    # Check SK amplitudes match present orbitals
    for item in arguments["SK amplitudes"].items():
        species = item[0]
        coefs    = item[1]
        needed_SK_coefs = 0

        first_orbitals = arguments["Orbitals"][int(species[0])]
        second_orbitals = arguments["Orbitals"][int(species[1])]

        first_orbital_list = []
        for orbital in first_orbitals:
            if orbital[0] not in first_orbital_list:
                first_orbital_list.append(orbital[0])
        
        second_orbital_list = []
        for orbital in second_orbitals:
            if orbital[0] not in second_orbital_list:
                second_orbital_list.append(orbital[0])
        
        amplitudes_per_orbitals = {"ss": 1, "sp": 1, "sd": 1, "pp": 2, "pd": 2, "dd":3}
        for key in amplitudes_per_orbitals.keys():
            if((key[0] in first_orbital_list and key[1] in second_orbital_list)
                or 
               (key[1] in first_orbital_list and key[0] in second_orbital_list)
              ):
                needed_SK_coefs += amplitudes_per_orbitals[key]

        if needed_SK_coefs != len(coefs):
            raise AssertionError("Error: Wrong number of SK amplitudes for given orbitals")


        # Check whether all necessary SK coefs are present for all orbitals
        if len(coefs) != needed_SK_coefs:
            raise AssertionError('Error: Mismatch between orbitals and required SK amplitudes')

    # ---------------- Simulation ----------------
    # Check mesh matches dimension
    if len(arguments['Mesh']) != arguments['Dimensionality']:
        raise AssertionError('Error: Mesh dimension does not match system dimension')

    return None


def transform_sk_coefficients(configuration):
    """ Routine to transform SK coefficients into standardized form
    for later manipulation in hamiltonian """

    dict = {}
    for species, coefs in configuration['SK amplitudes'].items():
        amplitudes = [0]*10  # Number of possible SK amplitudes
        first_orbitals = configuration["Orbitals"][int(species[0])]
        second_orbitals = configuration["Orbitals"][int(species[1])]

        first_orbital_list = []
        for orbital in first_orbitals:
            if orbital[0] not in first_orbital_list:
                first_orbital_list.append(orbital[0])
        
        second_orbital_list = []
        for orbital in second_orbitals:
            if orbital[0] not in second_orbital_list:
                second_orbital_list.append(orbital[0])
        
        if "d" in second_orbital_list and "d" not in first_orbital_list:
            aux_array = second_orbital_list
            second_orbital_list = first_orbital_list
            first_orbital_list = aux_array
        if "d" not in first_orbital_list and "d" not in second_orbital_list:
            if "p" in second_orbital_list and "p" not in first_orbital_list:
                aux_array = second_orbital_list
                second_orbital_list = first_orbital_list
                first_orbital_list = aux_array

        it = 0
        if "s" in first_orbital_list:
            if "s" in second_orbital_list:
                amplitudes[0] = coefs[it]
                it += 1
        if "p" in first_orbital_list:
            if "s" in second_orbital_list:
                amplitudes[1] = coefs[it]
                it += 1
            if "p" in second_orbital_list:
                amplitudes[2:4] = coefs[it:it+2]
                it += 2
        if "d" in first_orbital_list:
            if "s" in second_orbital_list:
                amplitudes[4] = coefs[it]
                it += 1
            if "p" in second_orbital_list:
                amplitudes[5:7] = coefs[it:it+2]
                it += 2
            if "d" in second_orbital_list:
                amplitudes[7:] = coefs[it:]

        dict[species] = amplitudes
    
    configuration['SK amplitudes'] = dict

    return None


def parse_config_file(file):
    """ Routine to obtain all the information from the configuration file, already shaped and verified """

    print("Parsing configuration file... ")
    configuration = shape_arguments(parse_raw_arguments(file))
    required_arguments = ['System name', 'Dimensionality', 'Bravais lattice', 'Species',
                          'Motif', 'Orbitals', 'Onsite energy', 'SK amplitudes']
    check_arguments(configuration, required_arguments)
    check_coherence(configuration)

    transform_sk_coefficients(configuration)
    print("Done\n")

    return configuration

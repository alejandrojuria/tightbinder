""" 
Routines to parse the tight-binding parameters written in the input file.
"""

from io import TextIOWrapper
import re
from tightbinder.utils import pretty_print_dictionary
from typing import List

def parse_raw_arguments(file: TextIOWrapper) -> dict:
    """ 
    Routine to parse the arguments raw (without any treatment) from the input file, following
    the rules defined for configuration files.

    :param file: Pointer to configuration file, obtained from call to open().
    :return: Dictionary with the content corresponding to each flag of the config. file. 
    """

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


def check_arguments(arguments: dict, required_arguments: List[str]) -> None:
    """ 
    Routine to check whether all required fields for model are present in the input file.

    :param arguments: Dictionary whose keys are the arguments to compare.
    :param required_arguments: List with expected arguments.
    :raises KeyError: Raise exception if not all required arguments are present in the arguments. 
    """

    for arg in required_arguments:
        if arg not in arguments.keys():
            raise KeyError(f"{arg} not present in input file")


def shape_arguments(arguments: dict) -> dict:
    """ Routine to rewrite correctly the arguments parsed (raw) from the input file.

    :param arguments: Dictionary with the raw contents of the config. file.
    :return: Dictionary with the contents of the config. file in numerical format. 
    :raises IndexError: Raised if some arguments are missing
    :raises ValueError: Raised if some arguments have incorrect values (e.g. string instead of numbers).
    :raises SyntaxError: Raised for the SK amplitudes if not using correctly the brackets.
    :raises NotImplementedError: Raised if there is an unexpected argument present. 
    """

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
                if not line:
                    raise SyntaxError('No Slater-Koster amplitudes given')
                else:
                    # First try to parse square brakets
                    if ((line.find('[') != -1 and line.find(']') == -1) 
                        or 
                        (line.find('[') == -1 and line.find(']') != -1)):
                        raise SyntaxError('SK amplitudes: Square brackets must be closed')
                    line = list(filter(None, re.split(r'\[|\]', line)))
                    if len(line) == 1 and len(arguments[arg]) == 1:
                        species_neigh = ['0', '0', '1']
                    elif len(line) == 1 and len(arguments[arg]) != 1:
                        raise SyntaxError(f'{arg}: Only one line can be present if not specifying species or neighbours')
                    elif len(line) == 2:
                        species_neigh = re.split(' |, |,|; |;', line[0])
                        if len(species_neigh) == 1:
                            species_neigh = ['0', '0'] + species_neigh
                        elif len(species_neigh) == 2:
                            species_neigh = species_neigh + ['1']
                        elif len(species_neigh) > 3:
                            raise SyntaxError(f'{arg}: More values than expected in braket')     
                    else:
                        raise SyntaxError(f'{arg}: Too many square brackets found')

                    neigh = species_neigh[2]                
                    if neigh not in dictionary:
                        dictionary[species_neigh[2]] = {}

                    amplitudes = list(filter(None, re.split(' |, |,|; |;', line[-1].strip())))
                    try:
                        amplitudes = [float(num) for num in amplitudes]
                        species = species_neigh[0] + species_neigh[1]
                        if species in dictionary[neigh]:
                            print(f'\033[93m Warning: Overwriting SK amplitudes for species pair ' + species + '\033[0m')
                        dictionary[neigh][species] = amplitudes
                    except ValueError as e:
                        raise Exception(f'Slater-Koster amplitudes must be numbers ({amplitudes})') from e

            arguments[arg] = dictionary

        elif arg == 'Spin':
            if arguments[arg][0] == "True":
                arguments[arg] = True
            elif arguments[arg][0] == "False":
                arguments[arg] = False
            else:
                print('Error: Spin parameter must be True or False (or 1 or 0 respectively)')
                raise

        elif arg == 'Spin-orbit coupling':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append(float(line))
                except IndexError as e:
                    print(f'Warning: No spin-orbit coupling given, defaulting to 0...')
                    arguments[arg] = 0.0
                except ValueError as e:
                    print(f'{type(e).__name__}: Spin-orbit coupling must be a number')
                    raise
            arguments[arg] = aux_array

        elif arg == 'Filling':
            aux_array = []
            for line in arguments[arg]:
                try:
                    aux_array.append(float(line))
                except ValueError as e:
                    print(f'{type(e).__name__}: Filling must be a number')
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

        elif arg in ['Radius', 'Mixing']:
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

        else:
            raise NotImplementedError(f'{arg} is not a valid parameter')

    return arguments


def check_coherence(arguments: dict) -> None:
    """ 
    Routine to check that the present arguments are coherent among them.

    :param arguments: Dictionary with the config. file content already processed with shape_arguments(). 
    """

    # --------------- Model ---------------
    # Check dimensions
    if arguments['Dimensionality'] > 3 or arguments['Dimensionality'] < 0:
        raise ValueError('Error: Invalid dimension!')

    # Check species
    if arguments['Species'] < 0:
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
                raise AssertionError('Incorrect species labeling in motif')

    # Check onsite energies are present for all species
    if len(arguments['Onsite energy']) > arguments['Species']:
        raise AssertionError('Too many onsite energies for given species')
    elif len(arguments['Onsite energy']) < arguments['Species']:
        raise AssertionError('Missing onsite energies for both atomic species')

    # Check orbitals present for all species
    if (len(arguments['Orbitals']) != arguments['Species']):
        raise AssertionError("Orbitals must be specified for each species")

    # Check number of SK coefficients
    for neighbour in arguments["SK amplitudes"].keys():
        nspecies_pairs = len(arguments['SK amplitudes'][neighbour].keys())
        if nspecies_pairs > arguments['Species']*(arguments['Species'] + 1)/2:
            raise AssertionError('Too many rows of SK coefficients')

    # Check number of SOC strenghts
    if arguments['Spin'] and len(arguments['Spin-orbit coupling']) != arguments["Species"]:
        raise AssertionError("Values for SOC strength must match number of species")

    # Check if SOC are non-zero
    for soc in arguments['Spin-orbit coupling']:
        if soc != 0 and not arguments['Spin']:
            print('Warning: Spin-orbit coupling is non-zero but spin was set to False. ')
            arguments['Spin'] = True
    
    # Check if mixing has a valid value if it is present
    if 'Mixing' in arguments:
        if arguments["Mixing"] < 0 or arguments["Mixing"] > 1:
            raise AssertionError("Mixing must be a value between 0 and 1")

        # TODO: Check if this is necessary
        #if not all([len(orbitals)==len(arguments["Orbitals"][0]) for orbitals in arguments["Orbitals"]]):
        #    raise AssertionError("Mixing can only be used with isoelectronic species")
    
    if 'Radius' in arguments:
        if arguments['Radius'] < 0:
            raise AssertionError("Radius must be a positive number")

    if 'Filling' in arguments:
        if len(arguments['Filling']) != arguments['Species']:
            raise AssertionError("Must provide number of electrons of each chemical species.")

        total_electrons = 0.0
        for atom in arguments['Motif']:
            species = int(atom[3])
            total_electrons += arguments['Filling'][species]
        if not total_electrons.is_integer():
            raise AssertionError('Total number of electrons of the system must be a positive integer.')

    # ------------ Orbital consistency ------------
    # Check onsite energy per orbital per species
    for n, onsite_energies in enumerate(arguments["Onsite energy"]):
        if len(onsite_energies) != len(arguments["Orbitals"][n]):
            raise AssertionError("Error: Each orbital requires one onsite energy value")

    # Check SK amplitudes match present orbitals
    for neighbour in arguments["SK amplitudes"].keys():
        for species, coefs in arguments["SK amplitudes"][neighbour].items():
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
                raise AssertionError("Wrong number of SK amplitudes for given orbitals")


            # Check whether all necessary SK coefs are present for all orbitals
            if len(coefs) != needed_SK_coefs:
                raise AssertionError('Mismatch between orbitals and required SK amplitudes')

    # ---------------- Simulation ----------------
    # Check mesh matches dimension
    if len(arguments['Mesh']) != arguments['Dimensionality']:
        raise AssertionError('Mesh dimension does not match system dimension')

    if 'Radius' in arguments and len(arguments['SK amplitudes'].keys()) > 1:
        raise AssertionError('Must not specify neighbours in radius mode')



def sk_coefficients_standard_form(coefficients: list, orbitals: list) -> list:
    """ 
    Routine to transform the SK coefficients from the parsed form to the
    standard form.

    :param coefficients: List with the parsed coefficients
    :param orbitals: List with orbitals corresponding to the given SK amplitudes
    :return: List with SK amplitudes in standard form. 
    """

    pass


def transform_sk_coefficients(configuration: dict) -> None:
    """ 
    Routine to transform SK coefficients into standardized form
    for later manipulation in hamiltonian.

    :param configuration: Dictionary with the contents of the config. file already shaped. 
    """

    # First store untransformed amplitudes.
    configuration['SK'] = configuration["SK amplitudes"]

    dict = {}
    for neighbour in configuration["SK amplitudes"].keys():
        dict[neighbour] = {}
        for species, coefs in configuration["SK amplitudes"][neighbour].items():
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

            dict[neighbour][species] = amplitudes
        
    configuration['SK amplitudes'] = dict



def mix_parameters(configuration: dict) -> None:
    """ 
    Method to mix the given parameters in case that some are missing in presence of multiple species.

    :param configuration: Dictionary with the contents of the config. file already shaped. 
    """

    if 'Mixing' not in configuration:
        mixing = 0.5
    else:
        mixing = configuration['Mixing']

    for i in range(configuration['Species']):
        for j in range(i + 1, configuration['Species']):
            species   = str(i) + str(j)
            species_i = str(i) + str(i)
            species_j = str(j) + str(j)
            for neighbour in configuration['SK amplitudes'].keys():
                SK_dictionary = configuration['SK amplitudes'][neighbour]
                if species not in SK_dictionary and species_i in SK_dictionary and species_j in SK_dictionary:
                    SK_dictionary[species] = [SK_dictionary[species_i][n]*mixing + SK_dictionary[species_j][n]*(1 - mixing) for n in range(len(SK_dictionary[species_i]))]


def parse_config_file(file: TextIOWrapper) -> dict:
    """ 
    Routine to obtain all the information from the configuration file, already shaped and verified.

    :param file: Python pointer to config. file as returned by open().
    :return: Dictionary with the contents of the config. file in standarized form, ready to be used by
        the class SlaterKoster. 
    """

    print("Parsing configuration file... ", end='')
    configuration = shape_arguments(parse_raw_arguments(file))
    required_arguments = ['System name', 'Dimensionality', 'Bravais lattice', 'Species',
                          'Motif', 'Orbitals', 'Onsite energy', 'SK amplitudes']
    check_arguments(configuration, required_arguments)
    check_coherence(configuration)

    mix_parameters(configuration)
    transform_sk_coefficients(configuration)
    
    print("Done\n")

    return configuration

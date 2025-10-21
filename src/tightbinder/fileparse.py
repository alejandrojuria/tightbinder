""" 
Routines to parse the tight-binding parameters written in the input file.
"""

from io import TextIOWrapper
import re
from typing import List
import yaml
from numbers import Number

from tightbinder.utils import pretty_print_dictionary

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
        
        if arg == 'Species':
            if type(arguments[arg]) is str:
                arguments[arg] = [arguments[arg]]
        
        elif arg in ['Filling', 'SOC']:
            if isinstance(arguments[arg], Number):
                arguments[arg] = [float(arguments[arg])]
                
        elif arg == 'Mesh':
            if isinstance(arguments[arg], Number):
                arguments[arg] = [int(arguments[arg])]
                
        elif arg == 'OnsiteEnergy':
            for n, line in enumerate(arguments[arg]):
                if isinstance(line, Number):
                    arguments[arg][n] = [float(line)] 

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

        elif arg == 'SKAmplitudes':
            dictionary = {}
            for n, line in enumerate(arguments[arg]):
                if not line:
                    raise SyntaxError('No Slater-Koster amplitudes given')
                else:
                    # First try to parse square brakets
                    if ((line.find('(') != -1 and line.find(')') == -1) 
                        or 
                        (line.find('(') == -1 and line.find(')') != -1)):
                        raise SyntaxError('SK amplitudes: Square brackets must be closed')
                    line = list(filter(None, re.split(r'\(|\)', line)))
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

        elif arg == 'SymmetryPoints':
            try:
                arguments[arg] = [str(label) for label in re.split(' |, |,', arguments[arg])]
            except IndexError as e:
                print(f'{type(e).__name__}: No high symmetry points given')
                raise

    return arguments


def check_coherence(arguments: dict) -> None:
    """ 
    Routine to check that the present arguments are coherent among them.

    :param arguments: Dictionary with the config. file content already processed with shape_arguments(). 
    """

    # --------------- Model ---------------
    # Check dimensions
    if arguments['Dimensions'] > 3 or arguments['Dimensions'] < 0:
        raise ValueError('Error: Invalid dimension!')

    # Check species
    if len(arguments['Species']) < 0:
        raise ValueError('Error: Species has to be a positive number (1 or 2)')

    # Check vector basis
    if arguments['Dimensions'] != len(arguments['Lattice']):
        raise AssertionError('Error: Dimension and number of basis vectors do not match')

    # Check length of vector basis
    for vector in arguments['Lattice']:
        if len(vector) != 3:
            raise ValueError('Error: Bravais vectors must have three components')

    # Check that motif has elements specified if num. of species > 2
    if len(arguments['Species']) >= 2:
        for atom in arguments['Motif']:
            try:
                atom_element = atom[3]  # Try to access
            except IndexError:
                print('Error: Atomic species not specified in motif')
                raise

            if atom_element > len(arguments['Species']):
                raise AssertionError('Incorrect species labeling in motif')

    # Check onsite energies are present for all species
    if len(arguments['OnsiteEnergy']) > len(arguments['Species']):
        raise AssertionError('Too many onsite energies for given species')
    elif len(arguments['OnsiteEnergy']) < len(arguments['Species']):
        raise AssertionError('Missing onsite energies for both atomic species')

    # Check orbitals present for all species
    if (len(arguments['Orbitals']) != len(arguments['Species'])):
        raise AssertionError("Orbitals must be specified for each species")

    # Check number of SK coefficients
    for neighbour in arguments["SKAmplitudes"].keys():
        nspecies_pairs = len(arguments['SKAmplitudes'][neighbour].keys())
        if nspecies_pairs > len(arguments['Species'])*(len(arguments['Species']) + 1)/2:
            raise AssertionError('Too many rows of SK coefficients')

    # Check number of SOC strenghts
    if arguments['Spin'] and len(arguments['SOC']) != len(arguments['Species']):
        raise AssertionError("Values for SOC strength must match number of species")

    # Check if SOC are non-zero
    for soc in arguments['SOC']:
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
        if len(arguments['Filling']) != len(arguments['Species']):
            raise AssertionError("Must provide number of electrons of each chemical species.")

        total_electrons = 0.0
        for atom in arguments['Motif']:
            species = int(atom[3])
            total_electrons += arguments['Filling'][species]
        if not total_electrons.is_integer():
            raise AssertionError('Total number of electrons of the system must be a positive integer.')

    # ------------ Orbital consistency ------------
    # Check onsite energy per orbital per species
    for n, onsite_energies in enumerate(arguments["OnsiteEnergy"]):
        if len(onsite_energies) != len(arguments["Orbitals"][n]):
            raise AssertionError("Error: Each orbital requires one onsite energy value")

    # Check SK amplitudes match present orbitals
    for neighbour in arguments["SKAmplitudes"].keys():
        for species, coefs in arguments["SKAmplitudes"][neighbour].items():
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
    if len(arguments['Mesh']) != arguments['Dimensions']:
        raise AssertionError('Mesh dimension does not match system dimension')

    if 'Radius' in arguments and len(arguments['SKAmplitudes'].keys()) > 1:
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
    configuration['SK'] = configuration["SKAmplitudes"]

    dict = {}
    for neighbour in configuration["SKAmplitudes"].keys():
        dict[neighbour] = {}
        for species, coefs in configuration["SKAmplitudes"][neighbour].items():
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
        
    configuration['SKAmplitudes'] = dict



def mix_parameters(configuration: dict) -> None:
    """ 
    Method to mix the given parameters in case that some are missing in presence of multiple species.

    :param configuration: Dictionary with the contents of the config. file already shaped. 
    """

    if 'Mixing' not in configuration:
        mixing = 0.5
    else:
        mixing = configuration['Mixing']

    for i in range(len(configuration['Species'])):
        for j in range(i + 1, len(configuration['Species'])):
            species   = str(i) + str(j)
            species_i = str(i) + str(i)
            species_j = str(j) + str(j)
            for neighbour in configuration['SKAmplitudes'].keys():
                SK_dictionary = configuration['SKAmplitudes'][neighbour]
                if species not in SK_dictionary and species_i in SK_dictionary and species_j in SK_dictionary:
                    SK_dictionary[species] = [SK_dictionary[species_i][n]*mixing + SK_dictionary[species_j][n]*(1 - mixing) for n in range(len(SK_dictionary[species_i]))]


# def parse_config_file(file: TextIOWrapper) -> dict:
#     """ 
#     Routine to obtain all the information from the configuration file, already shaped and verified.

#     :param file: Python pointer to config. file as returned by open().
#     :return: Dictionary with the contents of the config. file in standarized form, ready to be used by
#         the class SlaterKoster. 
#     """

#     print("Parsing configuration file... ", end='')
#     configuration = shape_arguments(parse_raw_arguments(file))
#     required_arguments = ['System name', 'Dimensions', 'Bravais lattice', 'Species',
#                           'Motif', 'Orbitals', 'Onsite energy', 'SK amplitudes']
#     check_arguments(configuration, required_arguments)
#     check_coherence(configuration)

#     mix_parameters(configuration)
#     transform_sk_coefficients(configuration)
    
#     print("Done\n")

#     return configuration


def parse_config_file(filename: str) -> dict:
    """
    Routine to parse the YAML configuration file, and extract all the information already shaped and verified.
    
    :param filename: Path of the YAML configuration file.
    :return: Dictionary with the contents of the config. file in standarized form, ready to be used by
        the class SlaterKoster.
    """
    
    with open(filename, 'r') as file:
        configuration = shape_arguments(yaml.safe_load(file))
        
        required_arguments = ['SystemName', 'Dimensions', 'Lattice', 'Species',
                          'Motif', 'Orbitals', 'OnsiteEnergy', 'SKAmplitudes']
        check_arguments(configuration, required_arguments)
        check_coherence(configuration)
        
        mix_parameters(configuration)
        transform_sk_coefficients(configuration)
        
        return configuration

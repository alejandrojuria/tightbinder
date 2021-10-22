# Script to read the entanglement file, repair it by computing the missing
# entanglement entropies and also writing the data to a .csv file

import numpy as np
import pandas as pd
from tightbinder.models import bethe_lattice, WilsonAmorphous
from tightbinder.topology import entanglement_spectrum


def bethe_wilson_model_entanglement(mass, length):
    model = WilsonAmorphous(m=mass, t=1, r=1.1)
    model.motif, model.bonds = bethe_lattice(z=3, depth=8, length=length)
    model.motif = np.array(model.motif)
    model.boundary = "OBC"
    model.ordering = "atomic"
    model.initialize_hamiltonian(find_bonds=False)
    eps = 1E-3

    plane = [1, 0, 0, np.max(model.motif[:, 0] / 2) + eps]
    entanglement = entanglement_spectrum(model, plane, kpoints=[[0., 0., 0.]])
    entanglement = list(entanglement.reshape(-1))

    return entanglement


def find_missing_points_in_data(data, gridshape):
    """ Note that data is a dictionary, so it is mutable
     This routine mutes data """
    min_mass = np.min(data["Mass"])
    max_mass = np.max(data["Mass"])
    min_length = np.min(data["Length"])
    max_length = np.max(data["Length"])
    mass_array = np.linspace(min_mass, max_mass, gridshape[0])
    length_array = np.linspace(min_length, max_length, gridshape[1])
    full_mass, full_length = np.meshgrid(mass_array, length_array)
    full_mass = full_mass.reshape(-1)
    full_length = full_length.reshape(-1)
    points = []
    for i, mass in enumerate(full_mass):
        if mass != data["Mass"][i]:
            data["Mass"].insert(i, mass)
            data["Length"].insert(i, full_length[i])
            data["Spectrum"].insert(i, [])
            points.append([i, mass, full_length[i]])

    return points


def complete_data(points, data):
    for index, mass, length in points:
        entanglement = bethe_wilson_model_entanglement(mass, length)
        data["Spectrum"][index] = entanglement

    return data


def find_mid_spectrum(spectrum):
    for n, value in enumerate(spectrum):
        if value >= 0.5 > spectrum[n - 1]:
            midpoint = n
            break

    return midpoint


def truncate_spectrum(spectrum, npoints=200):
    midpoint = find_mid_spectrum(spectrum)
    truncated_spectrum = spectrum[midpoint - npoints//2:midpoint + npoints//2]

    return truncated_spectrum


def truncate_spectrum_from_data(data, n=200):
    for i, spectrum in enumerate(data["Spectrum"]):
        data["Spectrum"][i] = truncate_spectrum(spectrum, npoints=n)

    return data


def main():
    filename = "./data/bethe_entanglement_diagram"
    file = open(filename, "r")
    lines = file.readlines()
    missing_spectres = 0
    mass = []
    length = []
    all_spectrum = []
    for line in lines:
        line = line.split()
        if len(line) == 1 and line[0] == "NaN":
            missing_spectres += 1
            continue
        line = [float(value) for value in line]
        mass.append(line[0])
        length.append(line[1])
        all_spectrum.append(line[2:])

    data = {
        "Mass": mass,
        "Length": length,
        "Spectrum": all_spectrum
    }
    npoints = int(np.sqrt(len(mass) + missing_spectres))
    gridshape = [npoints, npoints]
    points = find_missing_points_in_data(data, gridshape)
    data = complete_data(points, data)
    data = truncate_spectrum_from_data(data, n=200)

    df = pd.DataFrame.from_dict(data)
    df.columns = ["Mass", "Length", "Entanglement Eigenvalues"]
    df.to_csv("bethe_entanglement.csv", index=False)


if __name__ == "__main__":
    main()

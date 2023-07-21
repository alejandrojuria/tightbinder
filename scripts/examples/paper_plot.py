from tightbinder.models import SlaterKoster, AmorphousSlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import calculate_wannier_centre_flow, plot_wannier_centre_flow, calculate_z2_invariant
from tightbinder.disorder import amorphize
from tightbinder.result import State
from tightbinder.observables import TransportDevice
import matplotlib.pyplot as plt
import numpy as np

def ribbon(ax):

    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)

    ncells = 5

    model = SlaterKoster(config).reduce(n1=ncells)
    labels = ["K", "G", "K"]
    kpoints = model.high_symmetry_path(100, labels)

    model.initialize_hamiltonian()
    results = model.solve(kpoints)

    results.plot_along_path(labels, ax=ax, edge_states=True, e_values=[-2, 2])

def wannier(ax):

    # Parse configuration file
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)

    # Init. model and Hamiltonian
    model = SlaterKoster(config)
    model.initialize_hamiltonian()

    # Compute Wannier charge centre evolution
    nk = 20
    wcc = calculate_wannier_centre_flow(model, nk, refine_mesh=False)

    # Plot evolution of the charge centers and compute the Z2 invariant
    plot_wannier_centre_flow(wcc, ax=ax)

def phase_diagram(ax):
    # Parse configuration file
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)

    # Init. model
    model = SlaterKoster(config)

    # Generate different values for the spin-orbit coupling and iterate over them
    # At every iteration change SOC value of the model and compute the invariant
    nk = 20
    z2_values = []
    soc_values = np.linspace(0.1, 2, 10)
    for soc in soc_values:

        model.configuration["Spin-orbit coupling"][0] = soc
        model.initialize_hamiltonian()

        # Compute and store Z2 invariant
        wcc = calculate_wannier_centre_flow(model, nk, refine_mesh=False)
        z2 = calculate_z2_invariant(wcc)
        z2_values.append(z2)
    
    ax.scatter(soc_values, z2_values, marker="o", edgecolors="black", zorder=2.1)
    ax.plot(soc_values, z2_values, "k-", linewidth=2)
    ax.set_ylabel(r"$\mathbb{Z}_2$ invariant")
    ax.set_xlabel(r"$\lambda$ (eV)")
    ax.grid("on")

def edge_state(ax):

    # Declaration of parameters of the model
    np.random.seed(1)
    disorder = 0.1
    ncells = 12

    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)

    # Init. model
    model = AmorphousSlaterKoster(config, r=4.2837).ribbon(width=int(ncells/1.5))
    model.motif = np.array(model.motif)
    model.motif = model.motif[1:-1, :]
    model = model.reduce(n1=ncells)
    model.motif = np.array(model.motif)
    model.motif = model.motif[1:-1, :]
    model.decay_amplitude = 1
    model.initialize_hamiltonian(override_bond_lengths=True)
    model = amorphize(model, disorder)
    model.initialize_hamiltonian()

    # Obtain the eigenstates of the system
    
    results = model.solve()
    results.rescale_bands_to_fermi_level()

    # Identify a edge state (zero mode) of the system
    state_index = np.argmin(np.abs(results.eigen_energy))
    edge_state = State(results.eigen_states[0][:, state_index], model)

    # Plot real-space probability density
    edge_state.plot_amplitude(ax=ax)


def transport(ax):

    # Declare parameters
    length, width = 10, 5


    # Parse configuration file
    file = open("./examples/Bi111.txt", "r")
    config = parse_config_file(file)

    # Init. model
    model = SlaterKoster(config).ribbon(width=width)

    # Compute bands of the ribbon to compare later with the transmission
    nk, labels = 100, ["K", "G", "K"]
    kpoints = model.high_symmetry_path(nk, labels)
    model.initialize_hamiltonian()
    results = model.solve(kpoints)

    # Compute Fermi energy to center transmission later
    ef = results.calculate_fermi_energy(model.filling)

    # Plot bands

    # For the transmission, dirst define the positions of the unit cell of the leads 
    # and their periodicity
    left_lead = np.copy(model.motif)
    left_lead[:, :3] -= model.bravais_lattice[0]

    right_lead = np.copy(model.motif)
    right_lead[: , :3] += length * model.bravais_lattice[0]

    period = model.bravais_lattice[0, 0]

    # Finally set a finite ribbon to which we attach the leads
    model = model.reduce(n1=length)

    # Create the transport device and compute the transmission
    device = TransportDevice(model, left_lead, right_lead, period, "default")
    trans, energy = device.transmission(-5, 5, 100)

    # Plot transmission and device
    ax[0].plot(trans, energy - ef, 'k-')
    device.visualize_device(ax=ax[1])
    ax[0].set_ylim([-1.5, 1.5])
    ax[0].set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax[1].axis("off")
    ax[0].set_xlabel(r"$T(E)$")



if __name__ == "__main__":

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ribbon(ax[0, 0])
    wannier(ax[0, 1])
    phase_diagram(ax[0, 2])
    #edge_state(ax[1, 0])
    transport(ax[1, 1:2])

    


    plt.show()
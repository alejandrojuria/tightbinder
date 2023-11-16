from tightbinder.models import SlaterKoster, AmorphousSlaterKoster
from tightbinder.fileparse import parse_config_file
from tightbinder.topology import calculate_wannier_centre_flow, plot_wannier_centre_flow, calculate_z2_invariant
from tightbinder.disorder import amorphize
from tightbinder.result import State
from tightbinder.observables import TransportDevice
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from pathlib import Path

USE_LATEX = False
if USE_LATEX:
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
fontsize = 20

def ribbon(ax):

    path = Path(__file__).parent / ".." / "examples" / "inputs" / "Bi111.yaml"
    config = parse_config_file(path)

    ncells = 12

    model = SlaterKoster(config).reduce(n1=ncells)
    labels = ["K", "G", "K"]
    kpoints = model.high_symmetry_path(100, labels)

    model.initialize_hamiltonian()
    results = model.solve(kpoints)

    results.plot_along_path(labels, ax=ax, edge_states=True, e_values=[-2, 2], fontsize=fontsize, edgecolor="yellowgreen")

def wannier(ax):

    # Parse configuration file
    path = Path(__file__).parent / ".." / "examples" / "inputs" / "Bi111.yaml"
    config = parse_config_file(path)

    # Init. model and Hamiltonian
    model = SlaterKoster(config)
    model.initialize_hamiltonian()

    # Compute Wannier charge centre evolution
    nk = 20
    wcc = calculate_wannier_centre_flow(model, nk, refine_mesh=False)

    # Plot evolution of the charge centers and compute the Z2 invariant
    plot_wannier_centre_flow(wcc, ax=ax, fontsize=fontsize)
    ax.set_ylabel(r"$\hat{x}_n$", labelpad=-10)

    ax.legend()

def phase_diagram(ax):
    # Parse configuration file
    path = Path(__file__).parent / ".." / "examples" / "inputs" / "Bi111.yaml"
    config = parse_config_file(path)

    # Init. model
    model = SlaterKoster(config)

    # Generate different values for the spin-orbit coupling and iterate over them
    # At every iteration change SOC value of the model and compute the invariant
    nk = 20
    z2_values = []
    soc_values = np.linspace(0.0, 2, 20)
    for soc in soc_values:

        model.configuration["SOC"][0] = soc
        model.initialize_hamiltonian()

        # Compute and store Z2 invariant
        wcc = calculate_wannier_centre_flow(model, nk, refine_mesh=False)
        z2 = calculate_z2_invariant(wcc)
        z2_values.append(z2)

    z2_values[0] = 0

    # Compute transition point
    index = (np.where(np.array(z2_values) == 1)[0])[0]
    
    transition_point = (soc_values[index] + soc_values[index - 1])/2.
    
    ax.scatter(soc_values, z2_values, marker="o", edgecolors="black", zorder=2.1)
    ax.plot(soc_values, z2_values, "k-", linewidth=2)
    ax.set_ylabel(r"$\mathbb{Z}_2$ invariant", fontsize=fontsize)
    ax.set_xlabel(r"$\lambda$ (eV)", fontsize=fontsize)
    ax.grid("on")

    ax.vlines(transition_point, -0.1, 1.1, linestyle="dashed")
    ax.fill_between([0, transition_point], -0.1, 1.1, facecolor='green', alpha=0.2)
    ax.fill_between([transition_point, 2], -0.1, 1.1, facecolor='blue', alpha=0.2)

    ax.set_xlim([0, 2])
    ax.set_ylim([-0.1, 1.1])

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=fontsize)

    ax.text(0.2, 0.5, "Trivial", transform=ax.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)
    ax.text(0.7, 0.5, "Topological", transform=ax.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)


def edge_state(ax):

    # Declaration of parameters of the model
    np.random.seed(1)
    disorder = 0.1
    ncells = 10 # Set to 15 to reproduce paper figure

    path = Path(__file__).parent / ".." / "examples" / "inputs" / "Bi111.yaml"
    config = parse_config_file(path)

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
    edge_state.plot_amplitude(ax=ax, factor=300)


def transport(ax1, ax2):

    # Declare parameters
    length, width = 10, 6

    # Parse configuration file
    path = Path(__file__).parent / ".." / "examples" / "inputs" / "Bi111.yaml"
    config = parse_config_file(path)

    # Init. model
    model = SlaterKoster(config).ribbon(width=width, orientation="vertical")

    # Compute bands of the ribbon to compare later with the transmission
    nk, labels = 100, ["K", "G", "K"]
    kpoints = model.high_symmetry_path(nk, labels)
    model.initialize_hamiltonian()
    results = model.solve(kpoints)

    # Compute Fermi energy to center transmission later
    ef = results.calculate_fermi_energy(model.filling)
    print(ef)

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
    model.matrix_type = "sparse"

    # Create the transport device and compute the transmission
    device = TransportDevice(model, left_lead, right_lead, period, "default")
    # trans, energy = device.transmission(-2, 2, 200)
    
    pathtransmission = Path(__file__).parent / "transmission"
    pathenergy = Path(__file__).parent / "energy"

    trans = np.loadtxt(pathtransmission)
    energy = np.loadtxt(pathenergy)

    ax2.plot(energy - ef, trans, 'k-')
    ax2.set_ylabel(r"$T(E)$", fontsize=fontsize)
    ax2.set_xlabel(r"$E$ (eV)", fontsize=fontsize)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_xlim([-1, 1])
    ax2.tick_params(labelsize=fontsize)

    ax2.hlines(2, -1, 1, linestyle="dashed", color="blue", alpha=0.5)
    ax2.set_yticks([0, 2, 10])

    inset = ax2.inset_axes([0.1, 0.45, 0.86, 0.52])

    inset.patch.set_alpha(0.9)
    # inset.axis("off")

    # Plot transmission and device
    # ax1.plot(trans, energy - ef, 'k-')
    device.visualize_device(ax=inset)
    # ax1.set_ylim([-1.5, 1.5])
    # ax1.set_xticks([0, 2, 4, 6, 8, 10, 12])

    inset.spines['bottom'].set_color('white')
    inset.spines['top'].set_color('white') 
    inset.spines['right'].set_color('white')
    inset.spines['left'].set_color('white')
    

    # Get rid of the ticks
    inset.set_xticks([]) 
    inset.set_yticks([]) 
    

    # ax1.set_xlabel(r"$T(E)$")


if __name__ == "__main__":

    gs = GridSpec(2, 3, height_ratios=[1, 1.1])
    fig = plt.figure(figsize=(10, 6))
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:])
    ax = [ax0, ax1, ax2, ax4]
    ribbon(ax0)
    wannier(ax1)
    phase_diagram(ax2)
    edge_state(ax3)
    transport("ff", ax4)


    linewidth = 1.5
    for axis in ax:
        for side in ['top','bottom','left','right']:
            axis.spines[side].set_linewidth(linewidth)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # plt.tight_layout()

    ax0.text(0.1, 0.75, "(a)", transform=ax0.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)
    
    ax1.text(0.1, 0.9, "(b)", transform=ax1.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)
    
    ax2.text(0.1, 0.9, "(c)", transform=ax2.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)
    
    ax3.text(-0.1, 0.9, "(d)", transform=ax3.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)
    
    ax4.text(0.05, 0.9, "(e)", transform=ax4.transAxes, horizontalalignment='center',
     verticalalignment='center', fontsize=fontsize/1.5)
    
    # plt.savefig("paper_plot.png", dpi=500, bbox_inches="tight")

    plt.show()



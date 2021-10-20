# Script for paper: plots a edge state together with the energy spectrum for
# both OBC and PBC, and the entanglement spectrum

import numpy as np
import matplotlib.pyplot as plt
from tightbinder.models import WilsonAmorphous
from tightbinder.topology import entanglement_spectrum, plot_entanglement_spectrum
from tightbinder.result import State
from tightbinder.disorder import amorphize


def main():
    # ----------------------- Parameters -----------------------
    mass = 3
    cutoff = 1.1
    spread = 0.5
    cellsize = 30
    fontsize = 24
    markersize = 14

    # ----------------------- Plot init. -----------------------
    fig = plt.figure(figsize=(18, 12), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax = [ax1, ax2, ax3]

    # Change subplot box linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[1].spines[axis].set_linewidth(2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[2].spines[axis].set_linewidth(2)

    # ----------------------- Model init. & solving -----------------------
    model = WilsonAmorphous(m=mass, r=cutoff).reduce(n3=0)
    model = model.supercell(n1=cellsize, n2=cellsize)
    model = amorphize(model, spread=spread)
    model.boundary = "PBC"
    model.initialize_hamiltonian()
    results_pbc = model.solve()

    model.boundary = "OBC"
    model.initialize_hamiltonian()
    results_obc = model.solve()

    # ----------------------- Plot data -----------------------
    # Edge state
    energies = np.abs(results_obc.eigen_energy)
    print(f"Energy of edge state: {np.min(energies)}")
    index = np.where(energies == np.min(energies))[0][0]
    edge_state = State(results_obc.eigen_states[0][:, index + 6], model)
    edge_state.plot_amplitude(ax[0])

    # Energy spectrum
    eigenval_pbc = results_pbc.eigen_energy.reshape(-1, 1)
    eigenval_obc = results_obc.eigen_energy.reshape(-1, 1)
    ax[1].plot(eigenval_pbc, 'b.', markersize=markersize)
    ax[1].plot(eigenval_obc, 'g-', linewidth=markersize/4)
    ax[1].set_xlim(0, len(eigenval_obc))
    ax[1].set_ylim(np.min(eigenval_obc), np.max(eigenval_obc))
    ax[1].legend(["PBC", "OBC"], fontsize=fontsize, frameon=False)
    # ax[1].set_xlabel("States", fontsize=fontsize)
    ax[1].set_ylabel(rf"$E_n$ $(eV)$", fontsize=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize, direction="in")
    ax[1].set_box_aspect(0.8)

    # Inset for energy spectrum
    axins = ax[1].inset_axes([0.55, 0.05, 0.4, 0.4])
    axins.plot(eigenval_pbc, 'b.', markersize=markersize)
    axins.plot(eigenval_obc, 'g-', linewidth=markersize/4)
    x1, x2, y1, y2 = len(eigenval_obc)/2 - 300, len(eigenval_obc)/2 + 300, -0.25, 0.25
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax[1].indicate_inset_zoom(axins, edgecolor="black")
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(1.5)

    # Entanglement spectrum
    model.ordering = "atomic"
    model.initialize_hamiltonian()

    plane = [1, 0, 0, np.max(model.motif[:, 0] / 2)]
    entanglement = entanglement_spectrum(model, plane, kpoints=[[0., 0., 0.]])
    plot_entanglement_spectrum(entanglement, model, ax[2],
                               fontsize=fontsize, markersize=markersize)
    axins_ent = ax[2].inset_axes([0.02, 0.5, 0.4, 0.4])
    axins_ent.plot(entanglement, 'g.')
    x1, x2, y1, y2 = len(entanglement) / 2 - 100, len(entanglement) / 2 + 100, 0, 1
    axins_ent.set_xlim(x1, x2)
    axins_ent.set_xticks([x1, x2])
    axins_ent.set_ylim(y1, y2)
    axins_ent.set_yticklabels('')
    axins_ent.tick_params(axis="x", labelsize=fontsize/2)
    for axis in ['top', 'bottom', 'left', 'right']:
        axins_ent.spines[axis].set_linewidth(1.5)
    # ax[2].indicate_inset_zoom(axins_ent, edgecolor="black")
    ax[2].set_box_aspect(0.8)
    bbox = ax[2].get_position()
    bbox.y0 = bbox.y0 + 0.05
    bbox.y1 = bbox.y1 + 0.05
    ax[2].set_position(bbox)

    # plt.tight_layout()
    plt.savefig("edge_state_w_spectrums_r11.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

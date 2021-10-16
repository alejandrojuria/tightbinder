from tightbinder.models import WilsonAmorphous
from tightbinder.disorder import amorphize
import numpy as np
import matplotlib.pyplot as plt


def main():
    """ Initialize one WilsonFermions model, compute its IPR and then reinit for a different M value
     and compute its IPR, to plot both simultaneously """
    print("Initializing Wilson Amorphous model")
    m_topo, m_trivial = 2.5, 0.5
    cutoff = 1.1
    cellsize = 20
    spread = 0.0
    wilson = WilsonAmorphous(m=m_topo,  r=cutoff).reduce(n3=0).supercell(n1=cellsize, n2=cellsize)
    wilson = amorphize(wilson, spread=spread)
    wilson.boundary = "OBC"
    wilson.initialize_hamiltonian()
    results_topo = wilson.solve()

    wilson.m = m_trivial
    wilson.initialize_hamiltonian()
    results_trivial = wilson.solve()

    ipr_topo = results_topo.calculate_ipr()
    ipr_trivial = results_trivial.calculate_ipr()
    # ipr_topo, ipr_trivial = np.sort(ipr_topo), np.sort(ipr_trivial)
    x = np.arange(0, len(ipr_trivial))

    plt.figure()
    plt.bar(x, ipr_topo, width=1, color="blue", alpha=1)
    plt.bar(x, ipr_trivial, width=1, color="green", alpha=0.5)
    plt.legend([rf"$M={m_topo}$", rf"$M={m_trivial}$"])
    plt.title(rf"IPR $\Delta r={spread}$, $R={cutoff}$")
    plt.ylabel("IPR")
    plt.xlabel("Sorted states by IPR")
    plt.show()


if __name__ == "__main__":
    main()

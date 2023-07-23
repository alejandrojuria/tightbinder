Topological characterization
================================

The library provides different quantities to determine the topological behaviour of materials. The main one is based on the calculation of
Wilson loops, from which we obtain the Wannier charge centers. From their evolution one can determine both the :math:`\mathbb{Z}_2` invariant or the 
Chern number :math:`C`. Alternatively, the library also allows to computer the Chern marker, from which one can also extract the Chern number. The spatial 
entanglement spectrum can be also computed as it may provide information about the topology of the system.

Calculation of Wannier charge centers
---------------------------------------
The simplest way to determine if a crystalline system is topological or not is to extract the corresponding topological invariant from the
evolution of the Wannier charge centers. This can be done with the methods defined inside the module :mod:`tightbinder.topology`, specifically
via :meth:`tightbinder.topology.calculate_wannier_centre_flow()`

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    from tightbinder.topology import calculate_wannier_centre_flow
    
    file = open("/examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).reduce(n1=5, n2=5)

    model.initialize_hamiltonian()

    nk = 20
    wcc = calculate_wannier_centre_flow(model, nk)

.. note::
    The calculation of the evolution of the WCC can be relatively complex in some systems, meaning that it might be difficult to
    assert the topological invariant just from looking at the evolution if the mesh used is not dense enough. To compensante this, the method 
    :meth:`tightbinder.topology.calculate_wannier_centre_flow()` incorporates an algorithm to iteratively refine the mesh until the desired 
    accuracy, making it easier to track the evolution of the centers. We refer to the documentation of the method for more information 
    on the different parameters it can take.


Z2 invariant
---------------------------------------
Once we have the Wannier charge centers, we can plot their evolution to see if there is charge pumping or not. In a time-reversal topological 
insulator, one has to inspect the evolution over half BZ (in 2D), since the other half is the same because of time-reveral symmetry. With 
:meth:`tightbinder.topology.calculate_z2_invariant()` we extract the invariant from the WCCs, whereas :meth:`tightbinder.topology.plot_wannier_centre_flow` allows 
us to examine the evolution to ensure the invariant is correct.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    from tightbinder.topology import calculate_wannier_centre_flow, plot_wannier_centre_flow, calculate_z2_invariant
    
    file = open("/examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).reduce(n1=5, n2=5)

    model.initialize_hamiltonian()

    nk = 20
    wcc = calculate_wannier_centre_flow(model, nk)

    z2 = calculate_z2_invariant(wcc)
    plot_wannier_centre_flow(wcc)


Chern number 
---------------------------------------
Alternatively, we might want to determine the Chern number of a Chern insulator (i.e. topological insulator without time-reveral symmetry). The procedure
is analogue to that of the :math:`\mathbb{Z}_2` invariant, although in this case we have to look at the evolution of the WCC along the whole BZ for 2D materials, which
can be done with the option `full_BZ=True` in :meth:`tightbinder.topology.calculate_wannier_centre_flow()`.
Then, inspecting their evolution we can determine the Chern number.

.. code-block:: python

    from tightbinder.models import HaldaneModel
    from tightbinder.topology import calculate_wannier_centre_flow, plot_wannier_centre_flow

    model = HaldaneModel()
    model.initialize_hamiltonian()

    nk = 20
    wcc = calculate_wannier_centre_flow(model, nk, full_BZ=True)
    plot_wannier_centre_flow(wcc, full_BZ=True)


Chern marker 
---------------------------------------
For finite systems we can not compute the Chern number from the evolution of the WCC, since this is done in reciprocal space. Instead, we can compute the
Chern marker which is evaluated over each spatial position. From its value at the bulk of the finite system, one can also estimate the value of the Chern number,
which is particularly useful for non-crystalline systems.

.. code-block:: python

    from tightbinder.models import AgarwalaModel
    from tightbinder.topology import chern_marker

    model = AgarwalaModel().reduce(n1=4, n2=4)
    model.initialize_hamiltonian()

    results = model.solve()

    marker = chern_marker(model, results)


Entanglement spectrum
---------------------------------------
Another quantity that was shown to be related to the topological behaviour of the system is its entanglement spectrum. Namely, the spectrum of the
reduced density matrix restricted to one partition of the system reflects whether the system is topological or not. To compute it, first we need to 
specify a partition of our system, usually via a plane that divides the material into two halves. Then, the entanglement spectrum can be built.

.. code-block:: python

    from tightbinder.models import AgarwalaModel
    from tightbinder.topology import specify_partition_plane, entanglement_spectrum
    import numpy as np

    model = AgarwalaModel().reduce(n1=4, n2=4)
    model.initialize_hamiltonian()

    plane = specify_partition_plane([0., 1., 0., -np.max(model.motif[:, 1]])
    entanglement = entanglement_spectrum(model, plane)


.. note::
    The entanglement spectrum can also be evaluated as a function of :math:`k`. The current implementation only allows computing the entanglement spectrum 
    as a function of :math:`k` for 1D systems such as ribbons; for 2D or higher results might be incorrect.

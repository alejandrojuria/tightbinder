Defining custom models
=============================

The library has been built with a particular focus on Slater-Koster models. However, sometimes it is useful to define instead a specific model instead of 
using a Slater-Koster parametrization. For instance, we may have a toy model whose band structure can be obtained analitically, or the Bloch Hamiltonian
can be written directly. In those cases, it is more convenient to hard-code directly the Bloch Hamiltonian instead of constructing it following the SK approach.
The library currently implements several models, which can be seen in the :mod:`tightbinder.models` section. Here, we show how to define 
one of this models.

Any new model has to extend the :class:`tightbinder.system.System` class. Then, the minimum requirements to be able to instanciate the new class is 
to implement the constructor, ``__init__()`` and the Bloch Hamiltonian via ``hamiltonian_k()``. For instance, next we show the implementation of the 
BHZ model, which is a toy model known to describe time-reversal topological insulators [Bernevig2013]_

.. code-block:: python

    from tightbinder.system import System, FrozenClass
    import numpy as np

    class BHZ(System, FrozenClass):

        def __init__(self, g, u, c):
            super().__init__(system_name="BHZ model",
                            crystal=Crystal([[1, 0, 0], [0, 1, 0]],
                                            motif=[[0, 0, 0, 0]]))
            self.norbitals = [4]
            self.filling = 2
            self.basisdim = self.norbitals[0]
            self.g = g
            self.u = u
            self.c = c

            self._freeze()

        def hamiltonian_k(self, k):
            
            sigma_x = np.array([[0, 1],
                                [1, 0]], dtype=np.complex_)
            sigma_y = np.array([[0, -1j],
                                [1j, 0]], dtype=np.complex_)
            sigma_z = np.array([[1, 0],
                                [0, -1]], dtype=np.complex_)

            coupling = self.c * sigma_y

            id2 = np.eye(2)
            kx, ky = k[0], k[1]

            hamiltonian = (np.kron(id2, (self.u + np.cos(kx) + np.cos(ky)) * sigma_z + np.sin(ky) * sigma_y) +
                        np.kron(sigma_z, np.sin(kx) * sigma_x) + np.kron(sigma_x, coupling) +
                        self.g * np.kron(sigma_z, sigma_y) * (np.cos(kx) + np.cos(7 * ky) - 2))

            return hamiltonian

Note that the attributes of the model have to comply with those of :class:`tightbinder.system.System`, so we refer to the documentation for details on the 
expected values. This is the very minimum needed to define a new model, but one can also define more complex models where the Bloch Hamiltonian is not 
known from the beginning, but has to be built with a call to :meth:`tightbinder.system.System.initialize_hamiltonian()`. This is useful if as with a SK model, 
before getting the spectrum we want to modify the unit cell in different ways.

.. [Bernevig2013] Bernevig, B. A. (2013). Topological insulators and topological superconductors. Princeton university press.
Obtaining the band structure
==================================

Hamiltonian initialization
----------------------------------
Once the model is built and prepared as desired (e.g. modifying the unit cell), the first thing to do is to initialize the Hamiltonian, which 
is done with :meth:`tightbinder.system.System.initialize_hamiltonian()`.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    model.initialize_hamiltonian()

Generate kpoints
----------------------------------
Usually, we want to look at the spectrum of the Bloch Hamiltonian :math:`H(k)` at k points along high symmetry paths in the irreducible Brillouin zone (IBZ). 
We can define this reciprocal path by specifying the high symmetry points that compose it, with the method :meth:`tightbinder.crystal.Crystal.high_symmetry_path()`.
For instance, since hBN has a triangular lattice, we define the path in the IBZ to be :math:`\Gamma - K - M - \Gamma`:

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    nk = 100
    points = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(nk, points)

.. note::
    The method to generate the high symmetry path currently works only with 1D and 2D systems, since the high symmetry points have been specified only
    for the possible Bravais lattice in those dimensions. For 3D the library still works, but one has to specify manually this high symmetry path.

Obtaining the spectrum
----------------------------------
Once the model and its Hamiltonian have been initialized, and we know over which k points we are going to evaluate the Bloch Hamiltonian, we can 
obtain the spectrum calling the :meth:`tightbinder.system.System.solve()` method, which returns a :class:`tightbinder.result.Result` object.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    nk = 100
    points = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(nk, points)

    model.initialize_hamiltonian()

    result = model.solve(kpoints)


Plotting the bands
----------------------------------
Once we have the ``Result`` object with the eigenvalues and eigenvectors of the system, we can plot the eigenvalues to analyze the bands of the system. 
This is done with the :meth:`tightbinder.result.Result.plot_along_path()` method of the `Result` class.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    nk = 100
    points = ["G", "K", "M", "G"]
    kpoints = model.high_symmetry_path(nk, points)

    model.initialize_hamiltonian()

    result = model.solve(kpoints)
    result.plot_along_path(points)


Spectrum of a finite system
----------------------------------
So far we have shown how to obtain the band structure of a crystalline system. In case that we want to get the spectrum of a finite system, the procedure 
is simpler since we do not need to generate the k points first. When the spectrum is not a function of :math:`k`, it can be plotted instead using the method
:meth:`tightbinder.result.Result.plot_spectrum()`.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).reduce(n1=5, n2=5)

    model.initialize_hamiltonian()

    result = model.solve()
    result.plot_spectrum()

.. note::
    The same code works for a periodic system, but in that case the call to ``solve()`` diagonalizes the Bloch Hamiltonian at :math:`k=0` only.


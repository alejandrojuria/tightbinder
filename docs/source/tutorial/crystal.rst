Building the crystal
=======================================

Model creation
---------------------------------------
The crystalline structure of the system is automatically built from the configuration file when we initialize the :class:`tightbinder.models.SlaterKoster` model. 
By default the models are built with periodic boundary conditions (PBC), specially if the configuration file has Bravais vectors defined.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

Then, the information about crystal can be accessed from the model via its attributes :attr:`Crystal.bravais_lattice` and :attr:`Crystal.motif`:

.. code-block:: python

    model.bravais_lattice
    model.motif


Supercell
---------------------------------------
Once the model has been initialized for the first time, we can make changes to the crystal before initializing the Hamiltonian to obtain the energy spectrum.
For instance, we can change the unit cell to describe instead a supercell made of 2x2 original unit cells:

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).supercell(n1=2, n2=2)

Note that ``supercell()`` is a method from ``System``, which is the base class for all model classes, meaning that building supercells is not restricted to SlaterKoster
but is a general feature of any model.


Reducing dimensionality
---------------------------------------
In some cases we might be interested in removing the periodic boundary conditions after building the supercell, for instance if we want to look at the spectrum
of a semi-infinite system such as a ribbon or a slab, or directly to study a finite system.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).supercell(n1=2).reduce(n2=2)

The ``reduce()`` method first builds a supercell in the directions specified, and then removes the Bravais vectors in those directions effectively switching
to OBC along those boundaries. In the example shown here, we first double the unit cell along one direction (still periodic), and then the unit cell is doubled in the
other direction but the periodicity is removed.
Note that one can go from a fully periodic system to a finite one (PBC to OBC) either using the ``reduce()`` method,
or simply changing the ``boundary`` attribute of model:

.. code-block:: python

    model.boundary = "OBC"


Amorphous supercell
---------------------------------------
The description of amorphous solids can be regarded as an extension of the supercell procedure. To capture accurately the physics of amorphous solids, we usually
want to have a big supercell so that there is enough variation inside of it. Therefore, the procedure is to first produce a supercell starting from the crystalline
system, and then move the positions of the atoms sampling the displacements from statistical distributions (either a uniform or gaussian distribution) with the 
``amorphize()`` method.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    from tightbinder.disorder import amorphize
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).supercell(n1=2, n2=2)

    disorder = 0.1
    model = amorphize(model, disorder)

Adding vacancies
---------------------------------------
Instead of amorphizing the crystalline supercell, one could want instead to introduce vacancies in the system, i.e. to remove atoms. This can be done in an 
analogous way to the ``amorphize()`` method with ``introduce_vacancies()``:

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    from tightbinder.disorder import introduce_vacancies
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config).supercell(n1=2, n2=2)

    probability = 0.1
    model = introduce_vacancies(model, probability)


Crystal visualization
---------------------------------------
After initializing the model, it is useful to visualize the model to ensure that the subyacent crystal lattice was also appropiately built, specially if
we have performed any manipulation of the lattice (see e.g supercell or disorder). The library currently provides two different ways to present the crystal lattice:
either plotting the positions of the atoms via Matplotlib, or with a 3D render using VPython. Both cases are illustrated below:

.. code-block:: python

    model.visualize() # Uses VPython to render the crystal

Note that the ``visualize()`` method uses VPython, which opens a browser window if run the code is ran from terminal. For a simpler 
way of plotting the crystal, one can use the method ``plot_crystal()``:

.. code-block:: python

    model.plot_crystal() # Plots the crystal projection onto the xy plane.

Note that both methods are actually implemented by the base class ``Crystal`` and so are not specific to any model class.


Bonds between atoms
---------------------------------------
The bonds between atoms are usually determined when the Hamiltonian of the system is built with the method ``initialize_hamiltonian()``. However, in 
some cases we might be interested in knowing them beforehand, either to check whether they are correct or to modify the list of bonds. 
This can be done calling the method ``find_neighbours()``, and they can be visualized with ``plot_wireframe()``.

.. code-block:: python

    from tightbinder.fileparse import parse_config_file
    from tightbinder.models import SlaterKoster
    from tightbinder.disorder import introduce_vacancies
    
    file = open("./examples/hBN.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    model.find_neighbours()
    model.plot_wireframe()

.. note::
    In general, the use of explicit use of ``find_neighbours()`` is not advised since it is already used in ``initialize_hamiltonian()``, which 
    also takes care of building the Hamiltonian from those bonds. Therefore, one can also use ``plot_wireframe()`` after ``initialize_hamiltonian()``
    instead.

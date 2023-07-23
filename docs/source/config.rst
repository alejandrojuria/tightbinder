Configuration files
=============================

For the description of Slater-Koster models, the library uses configuration files where the user has to write all the information regarding the material 
in question, following the spirit of DFT codes. A complete configuration file requires specifying the crystalline structure (Bravais lattice and motif), the 
different chemical species present, the orbitals for each different type of atom and the onsite energies and hoppings between the orbitals. Multiple examples
of valid configuration files can be found under the folder ``/examples`` in the repository, and are also present in the :doc:`Materials <../materials/index>` section. 
Here we go over the different fields expected in the configuration files, as well as the syntaxis used.

The configuration files are defined in terms of fields. Each field begins with the # character, and must be within those specified next. In fields with 
multiple values (e.g. vectors), the numbers must be separated by a single space, or instead by a comma or a semicolon.

.. tip::

    In general, the body of every field is parsed to ensure that its content has the expected shape and values, and to raise an error describing 
    the source of the problem in case it was incorrectly filled. Nevertheless, it is possible that unseen errors arise, which is why we recommend 
    to follow and modify the already existing configuration files to minimize the possibility of error.

The expected input depends on the specific field, which we describe in the following. If a field is optional, it is denoted as ``[field]``:

.. code-block:: 

    System name

Name of the system, used mainly to write the system name in the title of plots.

.. code-block:: 

    Dimensionality

Dimension of the system, must be a number between 0 and 3. It must match the number of Bravais vectors present.

.. code-block:: 

    Bravais lattice

Each row corresponds to one vector of the basis (vectors have to be in :math:`\mathbb{R}^3`).
For 1D and 2D systems, the z (third) component must be zero (z should always be regarded as height component). Number of vectors must match the 
dimension of the system.

.. code-block:: 

    Species

Number of different chemical species.

.. code-block:: 

    Motif

Each row corresponds to one vector of the basis, where the first three numbers are the coordinates and the fourth number denotes the atomic species.
Numbering of species starts at 0.

.. code-block:: 

    Orbitals

The Slater-Koster description is based on cubic harmonic orbitals. The possible orbitals are: s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2.
They can be specified in any order, and each row specifies the orbitals for each chemical species.

.. code-block::

    Filling

Specify number of electrons per chemical species; one line per species. 
Fillings can be fractional, but the total number of electons in unit cell must be an integer.

.. code-block::

    Onsite energy

We must specify the onsite energy of each orbital.
In case of having several atomic species multiple line must be specified, one for each element.

.. code-block::
    
    SK amplitudes
 
As opposed to the orbitals, for the hopping or Slater-Koster amplitudes the order is fixed: 
Vsss, Vsps, Vpps, Vppp, Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd (separated by spaces).
Not present amplitudes can be omitted but ordering has te be respected always (e.g. there are no d orbitals so we omit all the amplituds involving d orbitals).
The syntax for SK amplitudes allows to specify between which species are the hoppings, as well as which is the corresponding neighbour, so in principle
one can write hoppings up to arbitrary number of neighbours.
Syntax is: [species1, species2; n-th neighbour] V1 V2 ...

.. code-block::

    [Spin]

Determine whether the model is spinless (spin polarized) or spinful. Must be either True (spinful) or False (spinless); if left blank it will default to False.
NOTE: True or False must be capitalized to follow Python naming.


.. code-block:: 
    
    [Spin-orbit coupling]

Using a non-zero value of spin-orbit coupling will automatically produce a spinful model, i.e. it sets 
the ``spin`` option to ``True`` even if it was set to ``False`` before. The amplitudes must be specified for all species, one line per species.


.. code-block::
    
    [Mesh]

Number of :math:`k` points in each direction. Syntax is Nx [Ny Nz]. It suffices to provide the required number of points depending on the system's dimension.
This option is only used if the ``tightbinder/main.py`` is called to plot the band structure from the configuration file. When using the API,
the number of :math:`k` points has to be specified manually (which can be a reference to the numbers specified here also).

.. code-block::

    [High symmetry points]

Label of points which make the path to evalute the bands of the system. Only used automatically when plotting the bands with ``tightbinder/main.py``.
As with ``Mesh``, one has to specify manually the high symmetry points when using the library; nevertheless in this case it is usually useful to write them 
in the configuration file and simply read them when generating the reciprocal path.

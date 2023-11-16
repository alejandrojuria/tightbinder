Configuration files
=============================

For the description of Slater-Koster models, the library uses configuration files where the user has to write all the information regarding the material 
in question, following the spirit of DFT codes. A complete configuration file requires specifying the crystalline structure (Bravais lattice and motif), the 
different chemical species present, the orbitals for each different type of atom and the onsite energies and hoppings between the orbitals. Multiple examples
of valid configuration files can be found under the folder ``/examples`` in the repository, and are also present in the :doc:`Materials <../materials/index>` section. 
Here we go over the different fields expected in the configuration files, as well as the syntaxis used.

The configuration files are defined using the `YAML<https://en.wikipedia.org/wiki/YAML>`_ format. This format allows to write the configuration files with a structure
matching the interal storage of the information (this is, a dictionary), also in a readable way for humans. 
Each field is defined by a keyword, followed by a colon and the corresponding value(s).
When a field denotes some kind of list, there are two forms to write it: either by enclosing the elements in square brackets and separating them by commas, or by
writing each element on a different line, starting with a hyphen.
Note that since this is YAML file, there are several ways to write the same information (as we saw now for arrays). We recommend to follow the example inputs provided
to build the configuration files, and optionally to read how the YAML format works to have a deeper understanding.

.. warning::

    The YAML format only works with spaces, and not with tabs. Files can have any form of indentation, as long as it is consistent and only uses spaces.

.. tip::

    In general, the body of every field is parsed to ensure that its content has the expected shape and values, and to raise an error describing 
    the source of the problem in case it was incorrectly filled. Nevertheless, it is possible that unseen errors arise, which is why we recommend 
    to follow and modify the already existing configuration files to minimize the possibility of error.

The expected input depends on the specific field, which we describe in the following. If a field is optional, it is denoted as ``[field]``:

.. code-block:: 

    SystemName

Name of the system, used mainly to write the system name in the title of plots.

.. code-block:: 

    Dimensions

Dimension of the system, must be a number between 0 and 3. It must match the number of Bravais vectors present.

.. code-block:: 

    Lattice

Bravais lattice of the system; each row corresponds to one vector of the basis (vectors have to be in :math:`\mathbb{R}^3`).
Each vector has to be given in form ``[x, y, z]``, with the three components separated by commas and enclosed in square brackets.
For 1D and 2D systems, the z (third) component must be zero (z should always be regarded as height component). Number of vectors must match the 
dimension of the system.

.. code-block:: 

    Species

Symbols used to denote the different chemical species present in the system, e.g. ``[B, N]``. They must be separated by commas and enclosed by brackets.
If there is only one, the square brackets can be omitted.

.. code-block:: 

    Motif

Each row corresponds to one vector of the basis, where the first three numbers are the coordinates and the fourth number denotes the atomic species.
Numbering of species starts at 0. As with the Bravais lattice, each vector has to be given in form ``[x, y, z, species]``, with the four components separated by commas and enclosed in square brackets.

.. code-block:: 

    Orbitals

The Slater-Koster description is based on cubic harmonic orbitals. The possible orbitals are: s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2.
They can be specified in any order, and each row specifies the orbitals for each chemical species. The orbitals can be separated using spaces, commas or both.

.. code-block::

    Filling

Specify number of electrons per chemical species; one value per species following the YAML syntax
for lists. If there is only one, it can be written directly.
Fillings can be fractional, but the total number of electons in unit cell must be an integer.

.. code-block::

    OnsiteEnergy

Used to specify the onsite energy of each orbital of each species.
In case of having several atomic species multiple lines must be specified, one per species.
For each line, the onsite energies are specified following the YAML format for lists.
If there is only one value per species, it can be written without brackets.

.. code-block::
    
    SKAmplitudes
 
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
    
    [SOC]

Field for spin-orbit coupling. Using a non-zero value of spin-orbit coupling will automatically produce a spinful model, i.e. it sets 
the ``spin`` option to ``True`` even if it was set to ``False`` before. The amplitudes must be specified for all species, one value per species following the YAML syntax
for lists.
If there is only one value, it can be written without brackets.


.. code-block::
    
    [Mesh]

Number of :math:`k` points in each direction. Syntax is Nx [Ny Nz]. It suffices to provide the required number of points depending on the system's dimension.
This option is only used if the ``tightbinder/main.py`` is called to plot the band structure from the configuration file. When using the API,
the number of :math:`k` points has to be specified manually (which can be a reference to the numbers specified here also).

.. code-block::

    [SymmetryPoints]

Label of points which make the path to evalute the bands of the system. Only used automatically when plotting the bands with ``tightbinder/main.py``.
As with ``Mesh``, one has to specify manually the high symmetry points when using the library; nevertheless in this case it is usually useful to write them 
in the configuration file and simply read them when generating the reciprocal path. As opposed to the ``Species`` field, the high symmetry points 
do not have to be separated by commas and enclosed in square brackets; they have to be written in the same line separated by spaces, commas or both.

MoS2
=========================
To describe MoS:math:`_2`` we use a Slater-Koster model consisting of :math:`d_{xy}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{3z^2-r^2}` orbitals for the Mo atom, and :math:`p_x, p_y, p_z` orbitals for
the S atoms. The model also includes spin-orbit coupling, which produces a splitting of around 100 meV of the valence band at the :math:`\bold{K}` point. The crystal can
be regarded as a honeycomb lattice, although it is non planar since each atom of the unit cell lies in a different plane, following the order S-Mo-S. The model includes
hoppings between the same and different chemical species, Mo-Mo, S-S and Mo-S, up to first neighbours in all cases. Depending on how the SK parameters were fitted
to the DFT bands, the resulting model can vary. We show two different SK models for MoS:math:`_2` [Ridolfi2015]_ [Silva-Guillén2016]_.

* Model [Ridolfi2015]_

The band structure is:

.. image:: ../plots/mos2_ridolfi_bands.png
    :align: center

Do note that the model has more bands than those showed; however the relevant ones are the ones close to the Fermi energy. This model provides a good fit 
of the first 6 conduction bands and the two first valence bands; the rest of them can be considered unphysical.

The configuration file is:

.. code-block::
    :caption: examples/MoS2_Ridolfi.txt

    ! --------------------- Config file formatting rules ---------------------
    ! Lines starting with ! are comments
    ! Lines with # denote argument types and must be present (removing one or wirting it wrogn will result in error)
    ! Blank spaces are irrelevant
    ! All numbers must be separated by either spaces, commas or semicolons
    ! Beware: Leaving a blank space at the beginning of a line will result in it being skipped
    ! -------------------------- Material parameters -------------------------
    #System name
    ! Model taken from doi:10.1088/0953-8984/27/36/365501
    MoS2

    #Dimensionality
    2

    #Bravais lattice
    ! Each row corresponds to one vector of the basis (vectors have to be in R^3)
    ! For 1D and 2D systems, the z (third) component must be zero (z should always be regarded as height component)
    3.16 0.0 0.0
    1.58 2.73664 0.0

    #Species
    ! Number of atomic species.
    2

    #Motif
    ! Each row corresponds to one vector of the basis; fourth number denotes atomic species
    ! Numbering of species starts at 0.
    0 0 0 0
    0.0 1.8246186736331629 1.5683120530778172 1
    0.0 1.8246186736331629 -1.5683120530778172 1

    #Orbitals
    ! Always use cubic harmonic orbitals
    ! Order is irrelevant
    ! Possible orbitals are: s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2
    dxy dyz dzx dx2-y2 d3z2-r2
    px py pz

    #Filling
    ! Specify number of electrons per chemical species; one line per species. 
    ! Fillings can be fractional, but total number of electons in unit cell must be an integer.
    6
    4

    #Onsite energy
    ! One for each different orbital defined above.
    ! In case of having several atomic species multiple line must be specified, one for each element.
    -0.352 -1.563 -1.563 -0.352 0.201
    -54.839 -54.839 -39.275

    #SK amplitudes
    ! Order is fixed: Vsss, Vsps, Vpps, Vppp, Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd (separated by spaces).
    ! Not present amplitudes can be omitted but ordering has te be respected always.
    ! The syntax for SK amplitudes allows to specify between which species are the hoppings, as well as
    ! which is the corresponding neighbour. 
    ! Syntax is: [species1, species2; n-th neighbour] V1 V2 ...
    [0, 0; 1] -1.153 0.612 0.086
    [0, 1; 1] -9.880 4.196
    [1, 1; 1] 12.734 -2.175

    #Spin
    ! Determine whether the model is spinless (spin polarized) or spinful.
    ! Must be either True (spinful) or False (spinless); if left blank it will default to False.
    ! NOTE: True or False must be capitalized.
    True

    #Spin-orbit coupling
    ! Note: Using a non-zero value will automatically produce a spinful model.
    ! Amplitude must be specified for all species; one line per species.
    0.1125
    0.078

    ! --------------------- Simulation parameters ---------------------
    #Radius
    ! If present, SlaterKoster model runs in "radius" mode, meaning that it will look
    ! for neighbours up to the given radius value
    3.16

    #Mesh
    ! Number of kpoints in each direction. Syntax is Nx Ny Nz
    ! It suffices to provide the required number of points depending on the system's dimension
    200 200

    #High symmetry points
    ! Label of points which make the path to evalute the bands of the system
    G K M G


* Model [Silva-Guillén2016]

The band structure is:

.. image:: ../plots/mos2_silva_bands.png
    :align: center

There are less conduction bands in this model compared with the previous model, but instead all the valence bands have been fitted to the DFT calculation so
they can be regarded as physical.

The configuration file is:

.. code-block::
    :caption: examples/MoS2_Silva.txt

    ! --------------------- Config file formatting rules ---------------------
    ! Lines starting with ! are comments
    ! Lines with # denote argument types and must be present (removing one or wirting it wrogn will result in error)
    ! Blank spaces are irrelevant
    ! All numbers must be separated by either spaces, commas or semicolons
    ! Beware: Leaving a blank space at the beginning of a line will result in it being skipped
    ! -------------------------- Material parameters -------------------------
    #System name
    ! Model taken from https://doi.org/10.3390/app6100284
    MoS2

    #Dimensionality
    2

    #Bravais lattice
    ! Each row corresponds to one vector of the basis (vectors have to be in R^3)
    ! For 1D and 2D systems, the z (third) component must be zero (z should always be regarded as height component)
    3.16 0.0 0.0
    1.58 2.73664 0.0

    #Species
    ! Number of atomic species.
    2

    #Motif
    ! Each row corresponds to one vector of the basis; fourth number denotes atomic species
    ! Numbering of species starts at 0.
    0 0 0 0
    0.0 1.8244 1.586 1
    0.0 1.8244 -1.586 1

    #Orbitals
    ! Always use cubic harmonic orbitals
    ! Order is irrelevant
    ! Possible orbitals are: s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2
    dxy dyz dzx dx2-y2 d3z2-r2
    px py pz

    #Filling
    ! Specify number of electrons per chemical species; one line per species. 
    ! Fillings can be fractional, but total number of electons in unit cell must be an integer.
    6
    4

    #Onsite energy
    ! One for each different orbital defined above.
    ! In case of having several atomic species multiple line must be specified, one for each element.
    -1.511 -0.050 -0.050 -1.511 -1.094
    -3.559 -3.559 -6.886

    #SK amplitudes
    ! Order is fixed: Vsss, Vsps, Vpps, Vppp, Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd (separated by spaces).
    ! Not present amplitudes can be omitted but ordering has te be respected always.
    ! The syntax for SK amplitudes allows to specify between which species are the hoppings, as well as
    ! which is the corresponding neighbour. 
    ! Syntax is: [species1, species2; n-th neighbour] V1 V2 ...
    [0, 0; 1] -0.895 0.252 0.228
    [0, 1; 1] 3.689 -1.241
    [1, 1; 1] 1.225 -0.467

    #Spin
    ! Determine whether the model is spinless (spin polarized) or spinful.
    ! Must be either True (spinful) or False (spinless); if left blank it will default to False.
    ! NOTE: True or False must be capitalized.
    True

    #Spin-orbit coupling
    ! Note: Using a non-zero value will automatically produce a spinful model.
    ! Amplitude must be specified for all species; one line per species.
    0.1125
    0.078

    ! --------------------- Simulation parameters ---------------------
    #Radius
    ! If present, SlaterKoster model runs in "radius" mode, meaning that it will look
    ! for neighbours up to the given radius value
    3.17

    #Mesh
    ! Number of kpoints in each direction. Syntax is Nx Ny Nz
    ! It suffices to provide the required number of points depending on the system's dimension
    200 200

    #High symmetry points
    ! Label of points which make the path to evalute the bands of the system
    G M K G



.. [Ridolfi2015] A tight-binding model for MoS2 monolayers, E Ridolfi et al, J. Phys.: Condens. Matter 27 365501 (2015)
.. [Silva-Guillén2016] Electronic Band Structure of Transition Metal Dichalcogenides from Ab Initio and Slater–Koster Tight-Binding Model, Silva-Guillén et al., Applied Sciences 6, no. 10: 284 (2016)



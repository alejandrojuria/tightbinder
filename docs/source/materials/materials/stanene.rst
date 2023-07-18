Stanene
=========================

Stanene is a two-dimensional material analogue to graphene, germanene or silicine. It consists of lead atoms on a buckled honeycomb lattice. The Slater-Koster model used for its description consists of :math:`s, p_x, p_y, p_z` orbitals together
with spin-orbit coupling, and the corresponding parameters are taken from [Hattori2017]_. The band structure obtained is:

.. image:: ../plots/stanene_bands.png
    :width: 640
    :align: center

The configuration file for this model is:

.. code-block::
    :caption: examples/stanene.txt

    ! --------------------- Config file formatting rules ---------------------
    ! Lines starting with ! are comments
    ! Lines with # denote argument types and must be present (removing one or wirting it wrogn will result in error)
    ! Blank spaces are irrelevant
    ! All numbers must be separated by either spaces, commas or semicolons
    ! Beware: Leaving a blank space at the beginning of a line will result in it being skipped
    ! -------------------------- Material parameters -------------------------
    #System name
    ! Parameters taken from 
    Stanene

    #Dimensionality
    2

    #Bravais lattice
    ! Each row corresponds to one vector of the basis (vectors have to be in R^3)
    ! For 1D and 2D systems, the z (third) component must be zero (z should always be regarded as height component)
    4.07032 2.35 0.0
    4.07032 -2.35 0.0

    #Species
    ! Number of atomic species.
    1

    #Motif
    ! Each row corresponds to one vector of the basis; fourth number denotes atomic species
    ! Numbering of species starts at 0.
    0 0 0 0
    2.713546 0 -0.8347958 0

    #Orbitals
    ! Always use cubic harmonic orbitals
    ! Order is irrelevant
    ! Possible orbitals are: s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2
    s, px, py, pz

    #Filling
    ! Specify number of electrons per chemical species; one line per species. 
    ! Fillings can be fractional, but total number of electons in unit cell must be an integer.
    4

    #Onsite energy
    ! One for each different orbital defined above.
    ! In case of having several atomic species multiple line must be specified, one for each element.
    -9.00 -3.39 -3.39 -3.39

    #SK amplitudes
    ! Order is fixed: Vsss, Vsps, Vpps, Vppp, Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd (separated by spaces).
    ! Not present amplitudes can be omitted but ordering has te be respected always.
    ! The syntax for SK amplitudes allows to specify between which species are the hoppings, as well as
    ! which is the corresponding neighbour. 
    ! Syntax is: [species1, species2; n-th neighbour] V1 V2 ...
    [0, 0] -2.6245 2.6504 1.4926 -0.7877

    #Spin
    ! Determine whether the model is spinless (spin polarized) or spinful.
    ! Must be either True (spinful) or False (spinless); if left blank it will default to False.
    ! NOTE: True or False must be capitalized.
    True

    #Spin-orbit coupling
    ! Note: Using a non-zero value will automatically produce a spinful model.
    ! Amplitude must be specified for all species; one line per species.
    1.2

    ! --------------------- Simulation parameters ---------------------
    #Mesh
    ! Number of kpoints in each direction. Syntax is Nx Ny Nz
    ! It suffices to provide the required number of points depending on the system's dimension
    200 200

    #High symmetry points
    ! Label of points which make the path to evalute the bands of the system.
    G M K G
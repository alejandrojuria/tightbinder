Tutorial
====================

Once we have defined a valid configuration file, following the instructions given in the :doc:`Configuration files <../config>` section, we can obtain the associated
band structure simply running from the terminal the following command, from the root of the repository:

.. code-block:: bash

    python -m tightbinder.main examples/inputs/[config_file]

For a more detailed exploration of the model defined, we have to use directly the library. Next we show some common situations that might appear when 
setting up the simulation of a given material:

.. toctree::
    :maxdepth: 2

    crystal
    band_structures
    topology
    transport
    models

The library presents more features than those showed here; we refer to the documentation of the library and to the :doc:`Examples <../examples/index>` section to see 
the more advanced capabilities of the package.

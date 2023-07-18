Materials 
=========================

We provide several configuration files corresponding to different Slater-Koster parametrizations of different materials, which can also be found in the ``/examples`` folder under the root of the repository.
These files can be used either for reference to define new systems, or as the starting point for further calculations. In all cases we show the resulting band structure as well as the corresponding configuration files, together
with the reference to the paper providing the SK parameters.

All the bands shown here can be reproduced with the corresponding configuration file running the following command from the root of the repository:

.. code-block:: bash

   python tightbinder/main.py [config_file]

.. toctree::
   :caption: List of materials

   materials/chain
   materials/hbn
   materials/bi111
   materials/sb111
   materials/graphene
   materials/mos2
   materials/germanene
   materials/stanene
   materials/ges
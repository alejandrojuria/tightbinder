Transport calculations
===============================
One relevant feature of the library is the ability to perform transport calculations of systems using the Landaur-Buttiker formalism. Next we describe how to 
set up a transport experiment.

Setting up the device
-------------------------------
The library only allows to perform two-terminal calculations. Given some material, the first thing to do define the right and left leads.
For example, in this case we start from a hydrogen chain to which we append another Bravais vector to describe instead a hydrogen on a square lattice 
(which could have been done also directly with the configuration file). From it we define a ribbon of a certain width and length. The leads 
are defined as a motif: they contain the positions of the unit cell of the lead.

.. code-block:: python

    from tightbinder.models import SlaterKoster
    from tightbinder.fileparse import parse_config_file
    from tightbinder.observables import TransportDevice
    import numpy as np

    length, width = 10, 4

    file = open("./examples/chain.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
    model = model.reduce(n2=width)

    left_lead = np.copy(model.motif)
    left_lead[:, :3] -= model.bravais_lattice[0]

    right_lead = np.copy(model.motif)
    right_lead[: , :3] += length * model.bravais_lattice[0]

    period = model.bravais_lattice[0, 0]

    model = model.reduce(n1=length)


Calculation of the transmission
-------------------------------
Once the leads and the material are defined, we simply pass them to :class:`tightbinder.observables.TransportDevice` to create the transport device. 
Calling :meth:`tightbinder.observables.TransportDevice.transmission()` we compute the transmission on the specified energy range. Note that in 
the creation of the device one also has to specify the Bravais lattice vector used to define the leads, i.e. the spacing between lead unit cells.

.. code-block:: python

    from tightbinder.models import SlaterKoster
    from tightbinder.fileparse import parse_config_file
    from tightbinder.observables import TransportDevice
    import numpy as np

    length, width = 10, 4
    
    file = open("./examples/chain.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
    model = model.reduce(n2=width)

    left_lead = np.copy(model.motif)
    left_lead[:, :3] -= model.bravais_lattice[0]

    right_lead = np.copy(model.motif)
    right_lead[: , :3] += length * model.bravais_lattice[0]

    period = model.bravais_lattice[0, 0]

    model = model.reduce(n1=length)

    device = TransportDevice(model, left_lead, right_lead, period, "default")
    trans, energy = device.transmission(-5, 5, 300)

Conductance
-------------------------------
Alternatively, we can compute directly the conductance of the system at :math:`T=0`, namely the transmission at the Fermi energy. This can be done 
with the :meth:`tightbinder.observables.TransportDevice.conductance()` method.

.. code-block:: python

    from tightbinder.models import SlaterKoster
    from tightbinder.fileparse import parse_config_file
    from tightbinder.observables import TransportDevice
    import numpy as np

    length, width = 10, 4
    
    file = open("./examples/chain.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
    model = model.reduce(n2=width)

    left_lead = np.copy(model.motif)
    left_lead[:, :3] -= model.bravais_lattice[0]

    right_lead = np.copy(model.motif)
    right_lead[: , :3] += length * model.bravais_lattice[0]

    period = model.bravais_lattice[0, 0]

    model = model.reduce(n1=length)

    device = TransportDevice(model, left_lead, right_lead, period, "default")
    G = device.conductance()

Visualizing the device
-------------------------------
Finally, it is also useful to plot the transport device to ensure that both the leads, the central sample and the bonds between all of them are correct.
This can be done calling the :meth:`tightbinder.observables.TransportDevice.visualize_device()` method.

.. code-block:: python

    from tightbinder.models import SlaterKoster
    from tightbinder.fileparse import parse_config_file
    from tightbinder.observables import TransportDevice
    import numpy as np
    import matplotlib.pyplot as plt

    length, width = 10, 4
    
    file = open("./examples/chain.txt", "r")
    config = parse_config_file(file)
    model = SlaterKoster(config)

    model.bravais_lattice = np.concatenate((model.bravais_lattice, np.array([[0., 1, 0]])))
    model = model.reduce(n2=width)

    left_lead = np.copy(model.motif)
    left_lead[:, :3] -= model.bravais_lattice[0]

    right_lead = np.copy(model.motif)
    right_lead[: , :3] += length * model.bravais_lattice[0]

    period = model.bravais_lattice[0, 0]

    model = model.reduce(n1=length)

    device = TransportDevice(model, left_lead, right_lead, period, "default")
    device.visualize_device()

    plt.show()
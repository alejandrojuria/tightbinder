Installation
=======================================

The library can be installed directly with:

.. code-block:: bash 

    pip install tightbinder

Alternatively, it can also be manually installed. First clone the repository:

.. code-block:: bash

    git clone https://github.com/alejandrojuria/tightbinder.git

Then, from the root folder of the repository install package with ``pip``, which will automatically handle dependency installation:

.. code-block:: bash

    cd /tightbinder
    pip install .

.. note::

    For developing the library, it is more convenient to install the package in *editable' mode:

    .. code-block:: bash

        pip install -e .
    
    which will handle automatically changes made to the library without having to reinstall the library each time.

.. note::

    To avoid possible incompatibilities between dependencies, the usage of a virtual environment (venv) is advised. To do so,
    create one either using ``conda`` (for which we refer to the `documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_) 
    or with python, which we describe here (extracted from the official `docs <https://docs.python.org/3/library/venv.html>`_):

    .. code-block:: bash

        python -m venv /path/to/venv 

    After the venv has been created, it can be activated on GNU/Linux running

    .. code-block:: bash

        source /path/to/venv/bin/activate

    Once we are using the venv, follow the above instructions to install the library.
.. _install:

Installation
============

crispy is available able on the Python package index (PyPI) and can be directly installed into your python environment using

.. code-block::

    python -m pip install sst-crispy

.. warning:: 
    On PyPI, the pacakge is called ``sst-crispy`` (``crispy`` was already taken :( ) but when importing the package and using it in your code, it is imported as ``crispy``.

Alternatively, the code can be installed from source from using git:

.. code-block::

    git clone https://github.com/bionic-toucan/crispy2.git ./crispy/
    cd crispy
    python -m pip install .
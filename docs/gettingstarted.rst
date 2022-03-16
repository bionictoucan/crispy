.. _quick-start:

Quick Start Guide to crispy
===========================

The following is a quick guide on how to get started with crispy. This will cover the basic data wrapper and its constituent properties. For a more in-depth look at the other data wrappers please refer to :ref:`data-wrap-ex`.

The first thing to do is to import the default data wrapper ``CRISP``. This is the base used for all of data objects in the package. ``CRISP`` assumes that your observations are either imaging spectroscopic or imaging spectropolarimetric (although it'll give its best at any 3D or 4D data) and is given either as a `FITS file <https://fits.gsfc.nasa.gov/fits_standard.html>`_, an `hdf5 file <https://www.hdfgroup.org/solutions/hdf5/>`_ or an object dictionary (see :ref:`utils`).

.. jupyter-execute::

    from crispy import CRISP

The only argument we need now to create our data object is the path to the file:

.. jupyter-execute::

   c_ex = CRISP("../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits")

Our example object ``c_ex`` is now a ``CRISP`` object containing this observation. We can get details of the observation doing the following

.. jupyter-execute::

    print(c_ex)

We can then explore this data quickly using the ``.intensity_map()`` and ``.plot_spectrum()`` methods.

All data objects in crispy can be indexed in a similar manner to `numpy arrays <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_. For example, our data above is sampled at 15 different wavelengths, say we only wanted to work with the fourth wavelength then we could create new object as such:

.. jupyter-execute::

    c_sub = c_ex[3] # remember Python indexing starts at 0!

And this ``c_sub`` object will contain the data for only the fourth wavelength in our original data. This kind of slicing is useful as the whole object is sliced rather than just the ``.data`` attribute allowing us to keep everything together. We can therefore using the plotting class methods with the sliced objects e.g.

.. jupyter-execute::
    :hide-output:

    c_sub.intensity_map()

.. figure:: 
    :align: center
    :figclass: align-center
    :width: 100%
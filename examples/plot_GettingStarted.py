"""
.. _quick-start:

Quick Start Guide to crispy
===========================
"""

# %%
# The following is a quick guide on how to get started with crispy. This will
# cover the basic data wrapper and its constituent properties. For a more
# in-depth look at the other data wrappers please refer to :ref:`data-wrap-ex`.

# %%
# The first thing to do is to import the default data wrapper ``CRISP``. This is
# the base used for alll of the data objects in the package. ``CRISP`` assumes
# that your observations are either imaging spectroscopic or imaging
# spectropolarimetric (although it'll give its best at any 3D or 4D data) and is
# given either as a `FITS file
# <https://fits.gsfc.nasa.gov/fits_standard.html>`_, a `zarr file
# <https://zarr.readthedocs.io/en/stable/>`_ or an object dictionary (see
# :ref:`utils`).

from crispy import CRISP
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# %%
# The only argument we need now to create our data object is the path to the
# file:

c_ex = CRISP("example_data/2014/crisp_l2_20140906_152724_6563_r00447.fits")

# %%
# Our example object ``c_ex`` is now a ``CRISP`` object containing this
# observation. We can get details of the following observation doing the
# following

print(c_ex)

# %%
# All data objects in crispy can be indexed in a similar manner to `numpy arrays
# <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_. For example,
# our data above is sampled at 15 different wavelengths, say we only wanted to
# work with the imaging data from the fourth wavelength then we could create a
# new object as such:

c_sub = c_ex[3]  # remember Python indexing starts at 0!

# %%
# And this ``c_sub`` object will contain the data for only the fourth wavelength
# in our original data. This kind of slicing is useful as the whole object is
# sliced rather than just the ``.data`` property allowing us to keep everything
# together.

# %%
# This is also how the plotting methods work, they expect a slice of the object
# otherwise an error will be thrown. For example,

c_sub.intensity_map()
plt.show()

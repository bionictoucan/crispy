"""
.. _data-wrap-ex:

Examples Using the Main Data Wrappers from crispy
=================================================
"""

# %%
# Here, we provide a brief introduction on how to use the main data structures
# defined in ``crispy.crisp``. Specifically, these examples will cover the
# ``CRISP``, ``CRISPNonU``, and ``CRISPWideband`` structures.

from crispy import CRISP, CRISPWideband, CRISPNonU
import warnings
warnings.filterwarnings("ignore")

# %%
# CRISP
# -----
# To demonstrate the CRISP class we use an example of a CRISP observation from
# the M1.1 solar flare SOL20140906T17:09 observed by SST/CRISP in
# H:math:`\alpha` and Ca II 8542. The data is publicly avaiable through the
# F-CHROMA project `here
# <https://star.pst.qub.ac.uk/wiki/doku.php/public/solarflares/start>`_. Below
# we show the data for H :math:`\alpha`.

crisp = CRISP("example_data/2014/crisp_l2_20140906_152724_6563_r00447.fits")
print(crisp)

# %%
# The data structures use slicing derived from astropy's `NDData
# <https://docs.astropy.org/en/stable/nddata/>`_ meaning the objects can be
# directly sliced to produce new objects of just the slice e.g. for only looking
# at the far blue wing data in our observation

print(crisp[0])

# %%
# This is also the easiest way to use the plotting methods. For this
# observation, to view an image at a specific wavelength we would use the
# ``intensity_map`` instance method as shown below for the line core of the
# H :math:`\alpha` observation

crisp[7].intensity_map()

# %%
# We can then use slicing and the ``plot_spectrum`` instance method to plot the
# spectrum at a certain spatial point. Firstly though to identify the slice we
# need we use the ``from_lonlat`` instance method, in this example we take the
# point :math:`(-720'', -310'')` in the Helioprojective plane:

# %%
# .. note:: 
#   There are complimentary instance methods ``from_lonlat`` and
#   ``to_lonlat`` which convert coordinates to/from the Helioprojective frame.
#   The format of the Helioprojective coordinates are **always** given in the
#   format (longitude, latitude) while image plane coordinates are **always**
#   given in (y,x) pixels. This is to allow the direct indexing of the objects
#   via the image plane corrdinates while maintaining the Helioprojective (and
#   other physical coordinate systems) convention of (longitude, latitude).

y, x = crisp.from_lonlat(-720, -310)
print(y,x)

# %%
# 

crisp[:,y,x].plot_spectrum()

# %%
# CRISPNonU
# ---------
# For the CRISPNonU class, we choose an imaging spectropolarimetric Ca II 8542
# observation of the X2.2 solar flare SOL20170906T09:10.

crispnonu = CRISPNonU("example_data/2017/ca_00001.zarr")
print(crispnonu)

# %%
# The ``intensity_map`` and ``plot_spectrum`` methods will also work here with
# the correct slicing -- that is, the object will need to be sliced twice for
# the ``intensity_map`` instance method and thrice for the ``plot_spectrum``
# instance method. The main difference from the CRISPNonU class can be seen in
# the wavelengths sampled section: the wavelengths are sampled non-uniformly but
# the class deals with this for us. Here, we will show the polarimetric instance
# methods (which also exist in the ``CRISP`` class). Firstly is ``stokes_map``
# for the line core:

crispnonu[:,5].stokes_map(stokes="all")

# %%
# The polarimetric plotting methods take a keyword argument ``stokes`` which is
# a string specifying which of the Stokes parameters the user would like to
# plot. In the example above we have used "all" to plot the maps of all of the
# Stokes parameters at line centre. However, if the user would like to only
# display Stokes I, Q and V this can be accomplished by setting ``stokes =
# "IQV"``.

# %%
# We can then identify a point to view the Stokes profiles using the
# ``from_lonlat`` instance method as before and plot the Stokes profiles using
# the ``plot_stokes`` instance method:

y, x = crispnonu.from_lonlat(510, -260)
print(y,x)

# %%
#

crispnonu[:,:,38,257].plot_stokes(stokes="all")

# %%
# CRISPWideband
# -------------
# For the CRISPWideband class, we use the complimentary wideband Ca II 8542 for
# the observation shown as an example for the CRISPNonU class, above.

crispwideband = CRISPWideband("example_data/2017/wideband/ca_00001.zarr")
print(crispwideband)

# %%
# The CRISPWideband class has one useful plotting instance method, that is
# ``intensity_map``:

crispwideband.intensity_map()

# %%
# CRISPWideband can also utilise the ``from_lonlat`` and ``to_lonlat`` instance
# methods.